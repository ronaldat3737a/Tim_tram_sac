"""Per-episode dispatch logic shared by the single-agent EVEnv and the
multi-agent MultiAgentEVEnv: observation features, the semi-MDP proxy
reward, dispatching one EV (with the invalid-pick fallback) and reporting
each dispatched trip once it resolves.

Reward is a per-decision proxy cost, charged to the decision that caused it
and to nothing else: -(expected travel time to the chosen station + expected
waiting time there), plus the invalid-pick and overload penalties. Expected
waiting counts the EVs charging at, queued at or driving to that station,
each taking one average charge time per charger. The EVs' real travel/
waiting times are still tracked and reported by collect_resolved() once each
one starts charging or fails; evaluation metrics use those, never the reward.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.config import SimulationConfig
from backend.simulation.network_graph import get_station_nodes, shortest_path
from backend.simulation.simulator import Simulator
from backend.simulation.traffic_logic import vehicle_effective_speed
from backend.simulation.vehicle import VehicleState

# Per station: normalized POI (x, y), distance, travel_time, queue, available
# capacity, path traffic. Station positions are re-drawn per scenario seed
# (network_graph._pick_access_nodes), so they are real information, not a
# constant the network could ignore.
OBSERVATION_FIELDS_PER_STATION = 7
OBSERVATION_FIELDS_FOR_EGO = 5
# Per station in the global state (MAPPO critic input): queue, available
# capacity, incoming EVs. Followed by global (non-station) fields: episode
# time progress.
STATE_FIELDS_PER_STATION = 3
STATE_GLOBAL_FIELDS = 1


def observation_dim(config: SimulationConfig) -> int:
    return OBSERVATION_FIELDS_FOR_EGO + OBSERVATION_FIELDS_PER_STATION * config.num_stations


def state_dim(config: SimulationConfig) -> int:
    return STATE_FIELDS_PER_STATION * config.num_stations + STATE_GLOBAL_FIELDS


class DispatchContext:
    """Everything one episode needs to observe, price and dispatch EVs on
    one Simulator. Built once per reset()."""

    def __init__(self, simulator: Simulator, config: SimulationConfig):
        self.simulator = simulator
        self.config = config
        # Real OSM coordinates are raw (lng, lat) around (105.8, 21.0) that
        # only vary in the 3rd-4th decimal place across the map; fed raw they
        # would be near-constant, huge-magnitude inputs to the network.
        # Every coordinate feature is min-max scaled to [0, 1] against the
        # map's own bounding box, (x_min, y_min, x_max, y_max).
        self.bounds: tuple[float, float, float, float] = simulator.graph.graph["bounds"]
        # action index -> station_id, in the same order the observation lists
        # stations.
        self.station_ids: list[int] = sorted(simulator.stations)
        if len(self.station_ids) != config.num_stations:
            raise RuntimeError(
                f"simulator built {len(self.station_ids)} stations but the action space "
                f"expects {config.num_stations} (config.num_stations)"
            )
        # Average time one EV occupies a charger: it asks for a station at
        # low_battery_threshold and charges back to full.
        self.expected_charge_time = (
            config.battery_capacity - config.low_battery_threshold
        ) / config.charging_rate
        self.max_distance, self.max_travel_time = self._compute_normalization_constants()
        # Dispatched EVs still driving to / queueing at their station:
        # vehicle_id -> station_id. Only used by collect_resolved().
        self.in_flight: dict[int, int] = {}

    # --- dispatch ------------------------------------------------------

    def dispatch(self, vehicle_id: int, requested_station_id: int) -> tuple[float, dict[str, Any]]:
        """Send one EV awaiting a decision to a station. Returns its unscaled
        reward and the decision's info fields."""
        vehicle = self.simulator.vehicles[vehicle_id]
        if not any(self.simulator.is_station_reachable(vehicle_id, sid) for sid in self.station_ids):
            # No station is reachable with the remaining battery: no action
            # could have saved this EV, so it is not the agent's fault and
            # is not counted as an invalid action. It fails right here
            # instead of being left standing on the road.
            self.simulator.fail_vehicle(vehicle)
            return self.config.battery_failure_penalty, {
                "station_id": None,
                "station_node_id": None,
                "invalid_action": False,
                "stranded": True,
                "station_overloaded": False,
            }

        invalid_action = not self.simulator.is_station_reachable(vehicle_id, requested_station_id)
        # An unreachable pick is overridden by the nearest reachable
        # station, so the EV is still dispatched and drives off at once.
        station_id = self.nearest_reachable_station(vehicle_id) if invalid_action else requested_station_id
        station = self.simulator.stations[station_id]
        # Overload is judged at decision time (queued + charging + EVs
        # already driving there, plus this one), matching the
        # available-capacity feature the agent saw. The EV still drives
        # all the way to the station and only joins its queue on physical
        # arrival (Simulator._handle_arrival) -- it never waits remotely.
        overloaded = station.occupancy + 1 > station.capacity
        # Priced before assign_station, so this EV is not counted among
        # the ones ahead of it.
        reward = -self.expected_dispatch_cost(vehicle_id, station_id)
        self.simulator.assign_station(vehicle_id, station_id)
        self.in_flight[vehicle_id] = station_id

        if invalid_action:
            reward += self.config.invalid_action_penalty
        if overloaded:
            reward += self.config.station_overload_penalty
        return reward, {
            "station_id": station_id,
            "station_node_id": station.node_id,
            "invalid_action": invalid_action,
            "stranded": False,
            "station_overloaded": overloaded,
        }

    def expected_dispatch_cost(self, vehicle_id: int, station_id: int) -> float:
        """Proxy cost of sending this EV to this station, known at decision
        time: weighted expected travel time (its shortest route, driven at
        its own traffic-limited speed, exactly as the simulator moves it)
        plus expected waiting time: every EV ahead of it (charging, queued
        or already driving there) takes one average charge, shared across
        the station's chargers."""
        graph = self.simulator.graph
        vehicle = self.simulator.vehicles[vehicle_id]
        station = self.simulator.stations[station_id]
        path, _, _ = shortest_path(
            graph, vehicle.current_node, station.node_id, weight=self.config.routing_weight
        )
        expected_travel_time = sum(
            graph.edges[u, v]["distance"] / vehicle_effective_speed(graph, u, v, vehicle.speed)
            for u, v in zip(path, path[1:])
        )
        evs_ahead = len(station.queue) + station.incoming_count + len(station.charging_vehicle_ids)
        expected_waiting_time = evs_ahead / max(station.num_chargers, 1) * self.expected_charge_time
        return (
            self.config.reward_travel_weight * expected_travel_time
            + self.config.reward_waiting_weight * expected_waiting_time
        )

    def nearest_reachable_station(self, vehicle_id: int) -> int:
        """Fallback for an invalid action: the reachable station with the
        shortest road distance (same notion of "nearest" as the
        nearest_station baseline). Caller guarantees one is reachable."""
        reachable = [sid for sid in self.station_ids if self.simulator.is_station_reachable(vehicle_id, sid)]
        return min(reachable, key=lambda sid: self.simulator.energy_required(vehicle_id, sid))

    def collect_resolved(self) -> list[dict[str, Any]]:
        """Retire in-flight EVs that started charging (or failed) and report
        their real travel/waiting times. Never feeds the reward."""
        resolved: list[dict[str, Any]] = []
        for vehicle_id, station_id in list(self.in_flight.items()):
            vehicle = self.simulator.vehicles[vehicle_id]
            if vehicle.state in (VehicleState.TRAVELING, VehicleState.WAITING):
                continue
            resolved.append(
                {
                    "vehicle_id": vehicle_id,
                    "station_id": station_id,
                    "travel_time": vehicle.time_since_station_assigned,
                    "waiting_time": vehicle.waiting_time,
                    "failed": vehicle.state == VehicleState.FAILED,
                }
            )
            del self.in_flight[vehicle_id]
        return resolved

    # --- features ------------------------------------------------------

    def build_observation(self, vehicle_id: int) -> np.ndarray:
        """One EV's local view: its own position, battery and destination,
        plus every station's position, route cost and load."""
        graph = self.simulator.graph
        vehicle = self.simulator.vehicles[vehicle_id]

        current = graph.nodes[vehicle.current_node]
        destination = graph.nodes[vehicle.destination_node]

        features = [
            *self._normalize_xy(current["x"], current["y"]),
            vehicle.battery_level / vehicle.battery_capacity,
            *self._normalize_xy(destination["x"], destination["y"]),
        ]

        for station_id in self.station_ids:
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
                    min(distance / self.max_distance, 1.0),
                    min(travel_time / self.max_travel_time, 1.0),
                    queue_norm,
                    available_norm,
                    traffic_norm,
                ]
            )

        return _sanitize(features)

    def global_state(self) -> np.ndarray:
        """Load of every station, independent of any one EV: queue,
        available capacity and EVs already driving there, each normalized by
        the station's capacity. Then the episode's time progress
        (simulation_time / max_episode_steps), so a critic can tell how much
        of the episode, and of its future decisions, is still ahead."""
        features: list[float] = []
        for station_id in self.station_ids:
            station = self.simulator.stations[station_id]
            features.extend(
                [
                    min(len(station.queue) / station.capacity, 1.0),
                    min(max(station.capacity - station.occupancy, 0) / station.capacity, 1.0),
                    min(station.incoming_count / station.capacity, 1.0),
                ]
            )
        features.append(self.simulator.simulation_time / max(self.config.max_episode_steps, 1))
        return _sanitize(features)

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

    def _normalize_xy(self, x: float, y: float) -> tuple[float, float]:
        """Min-max scale a real (lng, lat) into [0, 1] x [0, 1] using the
        map's bounding box (see self.bounds)."""
        x_min, y_min, x_max, y_max = self.bounds
        x_norm = (x - x_min) / max(x_max - x_min, 1e-9)
        y_norm = (y - y_min) / max(y_max - y_min, 1e-9)
        return min(max(x_norm, 0.0), 1.0), min(max(y_norm, 0.0), 1.0)

    @staticmethod
    def _average_path_traffic_weight(graph, path: list[int]) -> float:
        if len(path) < 2:
            return 0.0
        weights = [graph.edges[u, v]["traffic_weight"] for u, v in zip(path, path[1:])]
        return sum(weights) / len(weights)


def _sanitize(features: list[float]) -> np.ndarray:
    # Every feature is already scaled into [0, 1] by construction; this is a
    # last-line guard so a NaN/inf or any out-of-range outlier can never
    # reach a network, whatever its source.
    array = np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(array, 0.0, 1.0)
