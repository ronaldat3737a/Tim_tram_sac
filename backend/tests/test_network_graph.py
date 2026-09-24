"""Tests for network_graph.build_network (real OpenStreetMap street graph).

Never hits the real Overpass API: the autouse `mock_osm_fetch` fixture in
conftest.py transparently substitutes a small, hand-built, osmnx-shaped
MultiDiGraph (mirrors the exact shape osmnx's `graph_from_point` returns:
node `x`/`y` in (lng, lat), directed edges with a `length` in meters and
(sometimes) a shapely `geometry` LineString, some parallel/duplicate edges,
and two nodes disconnected from the rest) -- see conftest.py for why.
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest

from shapely.geometry import LineString, Point

from backend.config import DEFAULT_CONFIG
from backend.simulation import network_graph
from backend.simulation.network_graph import (
    build_network,
    get_depot_access_nodes,
    get_depot_nodes,
    get_station_access_nodes,
    get_station_nodes,
    shortest_path,
)
from backend.tests.conftest import fake_osm_multidigraph

_REAL_NODE_COUNT = 121  # conftest's fake extract, after dropping 2 isolated nodes
_TOTAL_POI_COUNT = DEFAULT_CONFIG.num_stations + DEFAULT_CONFIG.num_depots

# Captured at module-import time, before conftest's autouse fixture ever
# monkeypatches network_graph._load_or_fetch_osm_graph for a given test --
# lets test_load_or_fetch_osm_graph_caches_to_disk_on_first_call restore the
# real implementation for itself.
_REAL_LOAD_OR_FETCH_OSM_GRAPH = network_graph._load_or_fetch_osm_graph


def test_build_network_is_connected_with_correct_station_and_depot_count():
    graph = build_network(DEFAULT_CONFIG)

    assert nx.is_connected(graph)
    # Real road nodes, plus one POI node per station/depot (each connected
    # by its own spur edge -- see network_graph._add_poi_nodes).
    assert graph.number_of_nodes() == _REAL_NODE_COUNT + _TOTAL_POI_COUNT
    assert len(get_station_nodes(graph)) == DEFAULT_CONFIG.num_stations
    assert len(get_depot_nodes(graph)) == DEFAULT_CONFIG.num_depots
    assert len(get_station_access_nodes(graph)) == DEFAULT_CONFIG.num_stations
    assert len(get_depot_access_nodes(graph)) == DEFAULT_CONFIG.num_depots


def test_build_network_node_coordinates_are_real_lng_lat():
    graph = build_network(DEFAULT_CONFIG)

    for _, data in graph.nodes(data=True):
        assert -180.0 <= data["x"] <= 180.0
        assert -90.0 <= data["y"] <= 90.0


def test_build_network_edge_attributes_present_and_consistent():
    graph = build_network(DEFAULT_CONFIG)

    for u, v, data in graph.edges(data=True):
        # A POI's spur edge is not a real road edge -- it deliberately runs
        # at the fixed, slow poi_approach_speed_mps, not the randomized
        # real-road speed range.
        is_spur_edge = graph.nodes[u]["is_station"] or graph.nodes[u]["is_depot"] or graph.nodes[v]["is_station"] or graph.nodes[v]["is_depot"]

        assert data["distance"] > 0
        if is_spur_edge:
            assert data["speed"] == pytest.approx(DEFAULT_CONFIG.poi_approach_speed_mps)
            assert data["traffic_weight"] == 0.0
        else:
            assert DEFAULT_CONFIG.min_speed <= data["speed"] <= DEFAULT_CONFIG.max_speed
            assert (
                DEFAULT_CONFIG.min_traffic_weight
                <= data["traffic_weight"]
                <= DEFAULT_CONFIG.max_traffic_weight
            )
        effective_speed = data["speed"] / (1.0 + data["traffic_weight"])
        expected_travel_time = data["distance"] / effective_speed
        assert data["travel_time"] == pytest.approx(expected_travel_time)
        # Task 1: every edge's geometry is a real shapely LineString.
        assert len(data["geometry"].coords) >= 2
        assert data["geometry_start"] in (u, v)


def test_build_network_station_and_depot_nodes_flagged_on_graph():
    graph = build_network(DEFAULT_CONFIG)
    station_nodes = get_station_nodes(graph)
    depot_nodes = get_depot_nodes(graph)

    for node_id in station_nodes:
        assert graph.nodes[node_id]["is_station"] is True
    for node_id in depot_nodes:
        assert graph.nodes[node_id]["is_depot"] is True
    non_station_nodes = set(graph.nodes()) - set(station_nodes)
    for node_id in non_station_nodes:
        assert graph.nodes[node_id]["is_station"] is False


def test_build_network_station_poi_is_connected_to_its_access_node_by_a_spur():
    # Task 2/3: a station is a POI node reached by exactly one short spur
    # edge from its real access node -- routing/movement need no special
    # casing, they just traverse this like any other edge.
    graph = build_network(DEFAULT_CONFIG)
    station_nodes = get_station_nodes(graph)
    access_nodes = get_station_access_nodes(graph)

    for poi_node, access_node in zip(station_nodes, access_nodes):
        assert graph.has_edge(access_node, poi_node)
        edge = graph.edges[access_node, poi_node]
        assert edge["distance"] == pytest.approx(DEFAULT_CONFIG.poi_offset_meters)
        assert graph.degree(poi_node) == 1  # a dead-end parking spur, nothing else attaches to it


def test_build_network_depot_poi_is_connected_to_its_access_node_by_a_spur():
    graph = build_network(DEFAULT_CONFIG)
    depot_nodes = get_depot_nodes(graph)
    access_nodes = get_depot_access_nodes(graph)

    for poi_node, access_node in zip(depot_nodes, access_nodes):
        assert graph.has_edge(access_node, poi_node)
        edge = graph.edges[access_node, poi_node]
        assert edge["distance"] == pytest.approx(DEFAULT_CONFIG.poi_offset_meters)
        assert graph.degree(poi_node) == 1


# --- Task 2's required proof: the orthogonal-offset algorithm must return a
# point that does NOT lie on the original road edge it was offset from.
# Unit-tested directly against a controlled LineString (rather than checking
# every edge incident to a real access node in the built graph): the fake
# test fixture's streets are perfectly axis-aligned, so a real access node
# can coincidentally have a SECOND, unrelated road edge exactly collinear
# with the offset direction -- a fixture quirk, not a property of the
# offset algorithm itself, which this isolates against. ---------------------


def test_perpendicular_offset_point_is_not_on_a_north_south_edge():
    line = LineString([(105.80, 21.00), (105.80, 21.01)])  # constant longitude
    offset_point = Point(network_graph._perpendicular_offset_point(line, (105.80, 21.00), 10.0))

    assert line.distance(offset_point) > 0


def test_perpendicular_offset_point_is_not_on_an_east_west_edge():
    line = LineString([(105.80, 21.00), (105.81, 21.00)])  # constant latitude
    offset_point = Point(network_graph._perpendicular_offset_point(line, (105.80, 21.00), 10.0))

    assert line.distance(offset_point) > 0


def test_station_poi_offsets_are_distinct_per_station():
    # Different stations must not collapse onto the same displayed point.
    graph = build_network(DEFAULT_CONFIG)
    station_nodes = get_station_nodes(graph)
    positions = [(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in station_nodes]

    assert len(set(positions)) == len(positions)


def test_build_network_reproducible_with_same_seed():
    graph_a = build_network(DEFAULT_CONFIG, seed=123)
    graph_b = build_network(DEFAULT_CONFIG, seed=123)

    assert sorted(graph_a.nodes()) == sorted(graph_b.nodes())
    assert sorted(graph_a.edges()) == sorted(graph_b.edges())
    assert get_station_nodes(graph_a) == get_station_nodes(graph_b)
    assert get_depot_nodes(graph_a) == get_depot_nodes(graph_b)
    assert get_station_access_nodes(graph_a) == get_station_access_nodes(graph_b)
    assert get_depot_access_nodes(graph_a) == get_depot_access_nodes(graph_b)
    for node_id in graph_a.nodes():
        assert graph_a.nodes[node_id]["x"] == graph_b.nodes[node_id]["x"]
        assert graph_a.nodes[node_id]["y"] == graph_b.nodes[node_id]["y"]
    for u, v in graph_a.edges():
        assert graph_a.edges[u, v]["travel_time"] == graph_b.edges[u, v]["travel_time"]


def test_build_network_different_seeds_produce_different_traffic():
    graph_a = build_network(DEFAULT_CONFIG, seed=1)
    graph_b = build_network(DEFAULT_CONFIG, seed=2)

    # Node positions come from the (fixed, mocked) OSM extract, so only the
    # randomized traffic/speed/station/depot assignment differs by seed.
    travel_times_a = [data["travel_time"] for _, _, data in graph_a.edges(data=True)]
    travel_times_b = [data["travel_time"] for _, _, data in graph_b.edges(data=True)]
    assert travel_times_a != travel_times_b


def test_build_network_at_max_configured_station_scale():
    max_config = replace(DEFAULT_CONFIG, num_stations=DEFAULT_CONFIG.max_num_stations)

    graph = build_network(max_config)

    assert len(get_station_nodes(graph)) == max_config.max_num_stations
    assert nx.is_connected(graph)


def test_build_network_rejects_more_stations_than_available_nodes():
    bad_config = replace(DEFAULT_CONFIG, num_stations=10_000)

    with pytest.raises(ValueError):
        build_network(bad_config)


def test_build_network_rejects_more_depots_than_available_nodes():
    bad_config = replace(DEFAULT_CONFIG, num_depots=10_000)

    with pytest.raises(ValueError):
        build_network(bad_config)


def test_build_network_rejects_invalid_routing_weight():
    bad_config = replace(DEFAULT_CONFIG, routing_weight="not_a_real_weight")

    with pytest.raises(ValueError):
        build_network(bad_config)


def test_build_network_collapses_parallel_edges_to_the_shortest():
    graph = build_network(DEFAULT_CONFIG)

    # conftest's fake extract has a 1.0m duplicate edge between two grid
    # neighbors; the real ~66m edge between them must have been discarded.
    shortest_edge_distance = min(data["distance"] for _, _, data in graph.edges(data=True))
    assert shortest_edge_distance == pytest.approx(1.0)


def test_build_network_drops_nodes_outside_the_largest_component():
    graph = build_network(DEFAULT_CONFIG)

    # conftest's fake extract has 2 nodes with no path to the main grid.
    assert graph.number_of_nodes() == _REAL_NODE_COUNT + _TOTAL_POI_COUNT


def _manual_diamond_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge(0, 1, distance=10.0, travel_time=5.0)
    graph.add_edge(1, 3, distance=10.0, travel_time=5.0)
    graph.add_edge(0, 2, distance=1.0, travel_time=20.0)
    graph.add_edge(2, 3, distance=1.0, travel_time=20.0)
    graph.graph["routing_weight"] = "travel_time"
    return graph


def test_shortest_path_uses_configured_weight_key():
    graph = _manual_diamond_graph()

    path, total_distance, total_travel_time = shortest_path(graph, 0, 3)

    assert path == [0, 1, 3]
    assert total_distance == pytest.approx(20.0)
    assert total_travel_time == pytest.approx(10.0)


def test_shortest_path_can_override_weight_key():
    graph = _manual_diamond_graph()

    path, total_distance, total_travel_time = shortest_path(
        graph, 0, 3, weight="distance"
    )

    assert path == [0, 2, 3]
    assert total_distance == pytest.approx(2.0)
    assert total_travel_time == pytest.approx(40.0)


def test_shortest_path_same_source_and_target_is_trivial():
    graph = _manual_diamond_graph()

    path, total_distance, total_travel_time = shortest_path(graph, 0, 0)

    assert path == [0]
    assert total_distance == 0
    assert total_travel_time == 0


def test_shortest_path_rejects_invalid_weight():
    graph = _manual_diamond_graph()

    with pytest.raises(ValueError):
        shortest_path(graph, 0, 3, weight="not_a_real_weight")


def test_shortest_path_reachable_between_all_station_pairs_in_built_network():
    graph = build_network(DEFAULT_CONFIG)
    station_nodes = get_station_nodes(graph)

    for source in station_nodes:
        for target in station_nodes:
            path, _, _ = shortest_path(graph, source, target)
            assert path[0] == source
            assert path[-1] == target


def test_load_or_fetch_osm_graph_caches_to_disk_on_first_call(tmp_path, monkeypatch):
    """Exercises the REAL `_load_or_fetch_osm_graph` (disk cache read/write,
    osmnx call signature) rather than the conftest-mocked version every
    other test in this module uses -- only the network call itself (`ox.
    graph_from_point`, unreachable Overpass API from this sandbox) is
    stubbed. Proves the caching path -- fetch once, save, then load from
    disk on every subsequent call -- actually works end to end."""
    import osmnx as ox

    # conftest's autouse fixture already replaced _load_or_fetch_osm_graph
    # for the duration of every test; restore the original function just for
    # this one test.
    monkeypatch.setattr(network_graph, "_load_or_fetch_osm_graph", _REAL_LOAD_OR_FETCH_OSM_GRAPH)

    fake_graph = fake_osm_multidigraph()
    cache_path = tmp_path / "osm_cache.graphml"
    config = replace(DEFAULT_CONFIG, osm_cache_path=str(cache_path))

    fetch_calls = []
    monkeypatch.setattr(
        ox, "graph_from_point", lambda point, dist, network_type: (fetch_calls.append(1), fake_graph.copy())[1]
    )

    assert not cache_path.exists()
    first = network_graph._load_or_fetch_osm_graph(config)
    assert cache_path.exists()
    assert len(fetch_calls) == 1
    assert first.number_of_nodes() == fake_graph.number_of_nodes()

    second = network_graph._load_or_fetch_osm_graph(config)
    assert len(fetch_calls) == 1  # not re-fetched: loaded from the on-disk cache
    assert second.number_of_nodes() == fake_graph.number_of_nodes()
