"""Traffic-aware movement calculations for vehicles traversing an edge."""

from __future__ import annotations

import networkx as nx


def edge_effective_speed(graph: nx.Graph, u: int, v: int) -> float:
    """Road-only effective speed of edge (u, v): speed reduced by traffic."""
    edge = graph.edges[u, v]
    return edge["speed"] / (1.0 + edge["traffic_weight"])


def vehicle_effective_speed(graph: nx.Graph, u: int, v: int, vehicle_speed: float) -> float:
    """Actual speed of a specific vehicle on edge (u, v).

    A vehicle cannot exceed its own top speed nor the road's traffic-reduced
    effective speed, so the realized speed is the minimum of the two.
    """
    return min(vehicle_speed, edge_effective_speed(graph, u, v))


def distance_covered(effective_speed: float, time_step: float) -> float:
    """Distance a vehicle covers in one simulation tick at a given speed."""
    return effective_speed * time_step
