import networkx as nx
import pytest

from backend.simulation.traffic_logic import (
    distance_covered,
    edge_effective_speed,
    vehicle_effective_speed,
)


def _graph_with_one_edge(speed: float, traffic_weight: float) -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(0, 1, distance=10.0, speed=speed, traffic_weight=traffic_weight)
    return graph


def test_edge_effective_speed_reduced_by_traffic():
    graph = _graph_with_one_edge(speed=4.0, traffic_weight=1.0)

    assert edge_effective_speed(graph, 0, 1) == pytest.approx(2.0)


def test_edge_effective_speed_with_zero_traffic_equals_road_speed():
    graph = _graph_with_one_edge(speed=3.0, traffic_weight=0.0)

    assert edge_effective_speed(graph, 0, 1) == pytest.approx(3.0)


def test_vehicle_effective_speed_capped_by_vehicle_top_speed():
    graph = _graph_with_one_edge(speed=4.0, traffic_weight=0.0)

    assert vehicle_effective_speed(graph, 0, 1, vehicle_speed=1.5) == pytest.approx(1.5)


def test_vehicle_effective_speed_capped_by_road_speed():
    graph = _graph_with_one_edge(speed=1.0, traffic_weight=0.0)

    assert vehicle_effective_speed(graph, 0, 1, vehicle_speed=10.0) == pytest.approx(1.0)


def test_distance_covered_scales_with_time_step():
    assert distance_covered(effective_speed=2.0, time_step=1.0) == pytest.approx(2.0)
    assert distance_covered(effective_speed=2.0, time_step=0.5) == pytest.approx(1.0)
