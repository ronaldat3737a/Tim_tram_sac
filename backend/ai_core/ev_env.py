"""Gymnasium environment: dispatch one low-battery EV to a charging station.

Step granularity (PROJECT_SPEC.md section 17.1/18/21, Phase 0 decision D.1):
one env.step() = one charging-station decision for one EV. The simulator is
internally fast-forwarded, tick by tick, until that decision is resolved
(the EV starts charging, or fails) before control returns to the caller.
Every other EV keeps moving in the background during that fast-forward.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.config import DEFAULT_CONFIG, SimulationConfig
from backend.simulation.network_graph import get_station_nodes, shortest_path
from backend.simulation.simulator import Simulator
from backend.simulation.vehicle import VehicleState

# Per station: normalized POI (x, y), distance, travel_time, queue, available
# capacity, path traffic. Station positions are re-drawn per scenario seed
# (network_graph._pick_access_nodes), so they are real information, not a
# constant the network could ignore.
_OBSERVATION_FIELDS_PER_STATION = 7
_OBSERVATION_FIELDS_FOR_EGO = 5


class EVEnv(gym.Env):
    """Single-agent charging-station dispatch environment."""

    def __init__(
        self,
        config: SimulationConfig = DEFAULT_CONFIG,
        tick_hook: Callable[[Simulator], None] | None = None,
    ):
        super().__init__()
        self.config = config
        # Optional observer called after every internal simulator.tick(),
        # e.g. for evaluation code (Phase 6) to sample per-second station
        # queue/occupancy history. None by default: no effect on behavior.
        self._tick_hook = tick_hook

        obs_dim = _OBSERVATION_FIELDS_FOR_EGO + _OBSERVATION_FIELDS_PER_STATION * config.num_stations
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(config.num_stations)

        self.simulator: Simulator | None = None
        self._current_vehicle_id: int | None = None
        self._max_distance: float = 1.0
        self._max_travel_time: float = 1.0
        # Real OSM coordinates are raw (lng, lat) around (105.8, 21.0) that
        # only vary in the 3rd-4th decimal place across the map; fed raw they
        # would be near-constant, huge-magnitude inputs to the Q-network.
        # Every coordinate feature is min-max scaled to [0, 1] against the
        # map's own bounding box, (x_min, y_min, x_max, y_max), set in reset().
        self._bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
        # action index -> station_id, in the same order the observation lists
        # stations. Set in reset() from the simulator's real stations.
        self._station_ids: list[int] = []
        # Round-robin queue over EVs currently needing a decision. Without
        # this, always taking get_vehicles_needing_decision()[0] would let a
        # single vehicle whose best station is persistently unreachable (a
        # deterministic policy keeps re-picking the same invalid station for
        # it) monopolize every remaining decision slot in the episode,
        # starving every other pending EV until truncation.
        self._pending_queue: deque[int] = deque()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        resolved_seed = seed if seed is not None else self.config.random_seed

        self.simulator = Simulator(self.config, seed=resolved_seed)
        self._bounds = self.simulator.graph.graph["bounds"]
        self._station_ids = sorted(self.simulator.stations)
        if len(self._station_ids) != self.action_space.n:
            raise RuntimeError(
                f"simulator built {len(self._station_ids)} stations but action_space "
                f"expects {self.action_space.n} (config.num_stations)"
            )
        self._max_distance, self._max_travel_time = self._compute_normalization_constants()
        self._pending_queue = deque()

        terminated, truncated = self._advance_until_next_event()
        if terminated or truncated:
            raise RuntimeError(
                "No EV ever needs a charging decision for this scenario/seed. "
                "Check low_battery_threshold and battery_consumption_per_distance."
            )

        self._current_vehicle_id = self._pick_next_vehicle()
        observation = self._build_observation(self._current_vehicle_id)
        info = {
            "next_vehicle_id": self._current_vehicle_id,
            "simulation_time": self.simulator.simulation_time,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.simulator is not None, "call reset() before step()"
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} outside action_space {self.action_space}")

        vehicle_id = self._current_vehicle_id
        vehicle = self.simulator.vehicles[vehicle_id]
        requested_station_id = self.station_id_for_action(action)

        # Every step() fully resolves the decision for this EV, whatever the
        # action. An EV awaiting a decision is frozen in place by the
        # simulator (simulator._move_vehicles skips it), so an unresolved
        # decision would put the same EV straight back in front of the
        # agent -- which is exactly how a bad policy used to rack up
        # thousands of invalid actions in one episode.
        if not any(
            self.simulator.is_station_reachable(vehicle_id, sid) for sid in self._station_ids
        ):
            # No station is reachable with the remaining battery: no action
            # could have saved this EV, so it is not the agent's fault and
            # is not counted as an invalid action. It is stranded right away
            # instead of being frozen until max_episode_steps.
            vehicle.state = VehicleState.FAILED
            reward = self.config.battery_failure_penalty
            info: dict[str, Any] = {
                "vehicle_id": vehicle_id,
                "requested_station_id": requested_station_id,
                "station_id": None,
                "station_node_id": None,
                "invalid_action": False,
                "stranded": True,
                "travel_time": 0.0,
                "waiting_time": 0.0,
                "station_overloaded": False,
                "vehicle_failed": True,
            }
        else:
            invalid_action = not self.simulator.is_station_reachable(vehicle_id, requested_station_id)
            # An unreachable pick is overridden by the nearest reachable
            # station, so the EV still gets dispatched and simulated time
            # still moves forward.
            station_id = (
                self._nearest_reachable_station(vehicle_id) if invalid_action else requested_station_id
            )
            reward, info = self._dispatch_and_resolve(vehicle_id, station_id)
            if invalid_action:
                # On top of the fallback's own outcome cost, so an invalid
                # pick always scores strictly worse than picking that same
                # fallback station directly.
                reward += self.config.invalid_action_penalty
            info["requested_station_id"] = requested_station_id
            info["invalid_action"] = invalid_action
            info["stranded"] = False

        terminated, truncated = self._advance_until_next_event()
        info["simulation_time"] = self.simulator.simulation_time

        if terminated or truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["next_vehicle_id"] = None
        else:
            self._current_vehicle_id = self._pick_next_vehicle()
            observation = self._build_observation(self._current_vehicle_id)
            info["next_vehicle_id"] = self._current_vehicle_id

        reward *= self.config.reward_scale
        return observation, float(reward), terminated, truncated, info

    def _dispatch_and_resolve(self, vehicle_id: int, station_id: int) -> tuple[float, dict[str, Any]]:
        """Assign a reachable station, fast-forward until the EV starts
        charging or fails, and return (outcome reward, info)."""
        vehicle = self.simulator.vehicles[vehicle_id]
        station = self.simulator.stations[station_id]
        # Overload is evaluated at decision time (current occupancy plus the
        # EV about to join) rather than after the wait resolves, since by
        # then the congestion the agent caused may have already cleared.
        # This also matches exactly what the agent already sees via the
        # available-capacity observation feature for this station.
        overloaded = station.occupancy + 1 > station.capacity

        self.simulator.assign_station(vehicle_id, station_id)
        while (
            vehicle.state in (VehicleState.TRAVELING, VehicleState.WAITING)
            and self.simulator.simulation_time < self.config.max_episode_steps
        ):
            self._tick()

        travel_time = vehicle.time_since_station_assigned
        waiting_time = vehicle.waiting_time
        reward = -(
            self.config.reward_travel_weight * travel_time
            + self.config.reward_waiting_weight * waiting_time
        )
        if overloaded:
            reward += self.config.station_overload_penalty
        if vehicle.state == VehicleState.FAILED:
            reward = self.config.battery_failure_penalty

        info = {
            "vehicle_id": vehicle_id,
            "station_id": station_id,
            "station_node_id": station.node_id,
            "travel_time": travel_time,
            "waiting_time": waiting_time,
            "station_overloaded": overloaded,
            "vehicle_failed": vehicle.state == VehicleState.FAILED,
        }
        return float(reward), info

    def _nearest_reachable_station(self, vehicle_id: int) -> int:
        """Fallback for an invalid action: the reachable station with the
        shortest road distance (same notion of "nearest" as the
        nearest_station baseline). Caller guarantees one is reachable."""
        reachable = [
            sid for sid in self._station_ids if self.simulator.is_station_reachable(vehicle_id, sid)
        ]
        return min(reachable, key=lambda sid: self.simulator.energy_required(vehicle_id, sid))

    def station_id_for_action(self, action: int) -> int:
        """Map a Discrete action index (0..num_stations-1) to the real
        station_id it dispatches to; that station's graph node (its POI
        node on the OSM map) is simulator.stations[station_id].node_id."""
        return self._station_ids[int(action)]

    def _advance_until_next_event(self) -> tuple[bool, bool]:
        """Tick until a decision is available or the episode ends."""
        while True:
            if self.simulator.all_vehicles_done():
                return True, False
            if self.simulator.simulation_time >= self.config.max_episode_steps:
                return False, True
            if self.simulator.get_vehicles_needing_decision():
                return False, False
            self._tick()

    def _tick(self) -> None:
        self.simulator.tick()
        if self._tick_hook is not None:
            self._tick_hook(self.simulator)

    def _pick_next_vehicle(self) -> int:
        """Pop the next pending EV in round-robin order (see _pending_queue)."""
        pending_ids = {v.vehicle_id for v in self.simulator.get_vehicles_needing_decision()}
        self._pending_queue = deque(vid for vid in self._pending_queue if vid in pending_ids)
        newly_eligible = sorted(pending_ids - set(self._pending_queue))
        self._pending_queue.extend(newly_eligible)
        return self._pending_queue.popleft()

    def _compute_normalization_constants(self) -> tuple[float, float]:
        graph = self.simulator.graph
        station_nodes = get_station_nodes(graph)
        max_distance = 0.0
        max_travel_time = 0.0
        for node in graph.nodes():
            for station_node in station_nodes:
                _, distance, travel_time = shortest_path(
                    graph, node, station_node, weight=self.config.routing_weight
                )
                max_distance = max(max_distance, distance)
                max_travel_time = max(max_travel_time, travel_time)
        return max(max_distance, 1e-6), max(max_travel_time, 1e-6)

    def build_observation(self, vehicle_id: int) -> np.ndarray:
        """Public accessor for the same normalized observation reset()/step()
        use internally, for an arbitrary vehicle_id -- not just whichever one
        is currently up for decision. Used for read-only previews (e.g. the
        API's "what would the policy suggest for this vehicle right now"
        endpoint); never called from reset()/step() itself, so it changes no
        existing behavior."""
        return self._build_observation(vehicle_id)

    def _build_observation(self, vehicle_id: int) -> np.ndarray:
        graph = self.simulator.graph
        vehicle = self.simulator.vehicles[vehicle_id]

        current = graph.nodes[vehicle.current_node]
        destination = graph.nodes[vehicle.destination_node]

        features = [
            *self._normalize_xy(current["x"], current["y"]),
            vehicle.battery_level / vehicle.battery_capacity,
            *self._normalize_xy(destination["x"], destination["y"]),
        ]

        for station_id in self._station_ids:
            station = self.simulator.stations[station_id]
            station_node = graph.nodes[station.node_id]
            path, distance, travel_time = shortest_path(
                graph, vehicle.current_node, station.node_id, weight=self.config.routing_weight
            )
            queue_norm = min(len(station.queue) / station.capacity, 1.0)
            available_norm = min(max(station.capacity - station.occupancy, 0) / station.capacity, 1.0)
            traffic_norm = min(self._average_path_traffic_weight(graph, path) / self.config.max_traffic_weight, 1.0)

            features.extend(
                [
                    *self._normalize_xy(station_node["x"], station_node["y"]),
                    min(distance / self._max_distance, 1.0),
                    min(travel_time / self._max_travel_time, 1.0),
                    queue_norm,
                    available_norm,
                    traffic_norm,
                ]
            )

        # Every feature above is already scaled into [0, 1] by construction;
        # this is a last-line guard so a NaN/inf or any out-of-range outlier
        # can never reach the Q-network, whatever its source.
        observation = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(observation, 0.0, 1.0)

    def _normalize_xy(self, x: float, y: float) -> tuple[float, float]:
        """Min-max scale a real (lng, lat) into [0, 1] x [0, 1] using the
        map's bounding box (see self._bounds)."""
        x_min, y_min, x_max, y_max = self._bounds
        x_norm = (x - x_min) / max(x_max - x_min, 1e-9)
        y_norm = (y - y_min) / max(y_max - y_min, 1e-9)
        return min(max(x_norm, 0.0), 1.0), min(max(y_norm, 0.0), 1.0)

    @staticmethod
    def _average_path_traffic_weight(graph, path: list[int]) -> float:
        if len(path) < 2:
            return 0.0
        weights = [graph.edges[u, v]["traffic_weight"] for u, v in zip(path, path[1:])]
        return sum(weights) / len(weights)
