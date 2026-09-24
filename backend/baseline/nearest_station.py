"""Baseline 1 — Nearest Station (PROJECT_SPEC.md section 27): argmin(distance).

Deliberately does not check battery reachability, matching the literal
spec wording (only Baseline 3 is reachability-aware) and the naive
"pick the closest station" behavior described as the source of the
thundering-herd problem in section 3.1.
"""

from __future__ import annotations

from backend.simulation.network_graph import shortest_path
from backend.simulation.simulator import Simulator


def choose_station(simulator: Simulator, vehicle_id: int) -> int:
    vehicle = simulator.vehicles[vehicle_id]
    distances = {
        station_id: shortest_path(
            simulator.graph,
            vehicle.current_node,
            station.node_id,
            weight=simulator.config.routing_weight,
        )[1]
        for station_id, station in simulator.stations.items()
    }
    return min(distances, key=distances.get)
