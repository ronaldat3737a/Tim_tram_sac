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

_OBSERVATION_FIELDS_PER_STATION = 5
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
        station_id = action

        if not self.simulator.is_station_reachable(vehicle_id, station_id):
            reward = self.config.invalid_action_penalty
            info: dict[str, Any] = {
                "vehicle_id": vehicle_id,
                "station_id": station_id,
                "invalid_action": True,
            }
            # Guarantees monotonic simulated-time progress even under an
            # always-invalid policy, so max_episode_steps truncation is
            # always reachable (see Phase 3 report for the full rationale).
            self._tick()
        else:
            station = self.simulator.stations[station_id]
            # Overload is evaluated at decision time (current occupancy plus
            # the EV about to join) rather than after the wait resolves,
            # since by then the congestion the agent caused may have already
            # cleared. This also matches exactly what the agent already sees
            # via the available-capacity observation feature for this station.
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
                "invalid_action": False,
                "travel_time": travel_time,
                "waiting_time": waiting_time,
                "station_overloaded": overloaded,
                "vehicle_failed": vehicle.state == VehicleState.FAILED,
            }

        terminated, truncated = self._advance_until_next_event()
        info["simulation_time"] = self.simulator.simulation_time

        if terminated or truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["next_vehicle_id"] = None
        else:
            self._current_vehicle_id = self._pick_next_vehicle()
            observation = self._build_observation(self._current_vehicle_id)
            info["next_vehicle_id"] = self._current_vehicle_id

        return observation, float(reward), terminated, truncated, info

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
        x_min, y_min, x_max, y_max = graph.graph["bounds"]
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_max - y_min, 1e-9)

        current = graph.nodes[vehicle.current_node]
        destination = graph.nodes[vehicle.destination_node]

        features = [
            (current["x"] - x_min) / x_span,
            (current["y"] - y_min) / y_span,
            vehicle.battery_level / vehicle.battery_capacity,
            (destination["x"] - x_min) / x_span,
            (destination["y"] - y_min) / y_span,
        ]

        for station_id in sorted(self.simulator.stations):
            station = self.simulator.stations[station_id]
            path, distance, travel_time = shortest_path(
                graph, vehicle.current_node, station.node_id, weight=self.config.routing_weight
            )
            queue_norm = min(len(station.queue) / station.capacity, 1.0)
            available_norm = min(max(station.capacity - station.occupancy, 0) / station.capacity, 1.0)
            traffic_norm = min(self._average_path_traffic_weight(graph, path) / self.config.max_traffic_weight, 1.0)

            features.extend(
                [
                    min(distance / self._max_distance, 1.0),
                    min(travel_time / self._max_travel_time, 1.0),
                    queue_norm,
                    available_norm,
                    traffic_norm,
                ]
            )

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _average_path_traffic_weight(graph, path: list[int]) -> float:
        if len(path) < 2:
            return 0.0
        weights = [graph.edges[u, v]["traffic_weight"] for u, v in zip(path, path[1:])]
        return sum(weights) / len(weights)
