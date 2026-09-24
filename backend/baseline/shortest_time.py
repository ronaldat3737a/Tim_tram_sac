"""Baseline 2 — Shortest Travel Time (PROJECT_SPEC.md section 27): argmin(travel_time).

Deliberately does not check battery reachability, matching the literal
spec wording (only Baseline 3 is reachability-aware).
"""

from __future__ import annotations

from backend.simulation.network_graph import shortest_path
from backend.simulation.simulator import Simulator


def choose_station(simulator: Simulator, vehicle_id: int) -> int:
    vehicle = simulator.vehicles[vehicle_id]
    travel_times = {
        station_id: shortest_path(
            simulator.graph,
            vehicle.current_node,
            station.node_id,
            weight=simulator.config.routing_weight,
        )[2]
        for station_id, station in simulator.stations.items()
    }
    return min(travel_times, key=travel_times.get)
