"""Shared pytest fixtures for the backend test suite.

The traffic graph is always fetched from OpenStreetMap now (section 11) --
there is no synthetic generator to fall back to. Every test that builds a
Simulator/EVEnv/SimulationManager (almost all of them) would otherwise try
to hit the real Overpass API, which is both unreachable from this sandbox
and undesirable in any CI run (section 26/33: tests must be reproducible and
have no network dependency on the hot path). This autouse, session-wide
fixture transparently substitutes a small, hand-built, osmnx-shaped
MultiDiGraph for every test, so existing test bodies need no per-test
mocking of their own.
"""

from __future__ import annotations

import networkx as nx
import pytest
from shapely.geometry import LineString

from backend.simulation import network_graph

# Bach Khoa/Nghia Do area, same order of magnitude as the real config.
_BASE_LAT = 21.0385
_BASE_LNG = 105.7973
_GRID_STEP_DEGREES = 0.0006  # roughly ~60-70m at this latitude
_GRID_ROWS = 11
_GRID_COLS = 11  # 121 nodes: enough headroom for num_vehicles up to 100
# and max_num_stations=10 alongside num_depots.


def fake_osm_multidigraph() -> nx.MultiDiGraph:
    """An 11x11 grid of "streets" (121 connected nodes) plus 2 isolated
    nodes, mirroring the structural quirks a real osmnx 'drive' extract has:
    directed edge pairs for two-way streets, a `geometry` LineString on some
    edges and not others, and a duplicate parallel edge on one pair."""
    graph = nx.MultiDiGraph()
    grid_ids = {}
    node_id = 1000
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            lng = _BASE_LNG + col * _GRID_STEP_DEGREES
            lat = _BASE_LAT + row * _GRID_STEP_DEGREES
            graph.add_node(node_id, x=lng, y=lat)
            grid_ids[(row, col)] = node_id
            node_id += 1

    def add_two_way(a: int, b: int, with_geometry: bool) -> None:
        xa, ya = graph.nodes[a]["x"], graph.nodes[a]["y"]
        xb, yb = graph.nodes[b]["x"], graph.nodes[b]["y"]
        length = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5 * 111_000  # ~degrees to meters
        kwargs = {"length": length}
        if with_geometry:
            mid = ((xa + xb) / 2 + 0.00005, (ya + yb) / 2 + 0.00005)
            kwargs["geometry"] = LineString([(xa, ya), mid, (xb, yb)])
        graph.add_edge(a, b, **kwargs)
        graph.add_edge(b, a, **kwargs)

    edge_toggle = True
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            here = grid_ids[(row, col)]
            if col + 1 < _GRID_COLS:
                add_two_way(here, grid_ids[(row, col + 1)], edge_toggle)
                edge_toggle = not edge_toggle
            if row + 1 < _GRID_ROWS:
                add_two_way(here, grid_ids[(row + 1, col)], edge_toggle)
                edge_toggle = not edge_toggle

    # A duplicate/parallel edge between two already-connected nodes, shorter
    # than the "official" one above -- exercises the "keep the shortest
    # parallel edge" collapsing rule in _convert_osm_graph.
    a, b = grid_ids[(0, 0)], grid_ids[(0, 1)]
    graph.add_edge(a, b, length=1.0, key=99)

    # Two nodes with no edges at all to the grid: must be dropped when only
    # the largest connected component is kept.
    graph.add_node(9001, x=_BASE_LNG - 0.01, y=_BASE_LAT - 0.01)
    graph.add_node(9002, x=_BASE_LNG - 0.011, y=_BASE_LAT - 0.01)
    graph.add_edge(9001, 9002, length=5.0)
    graph.add_edge(9002, 9001, length=5.0)

    return graph


@pytest.fixture(autouse=True)
def mock_osm_fetch(monkeypatch):
    """Every test gets this fake graph instead of hitting Overpass, and
    never touches a disk cache file. Individual tests that specifically
    need the REAL `_load_or_fetch_osm_graph` (disk caching, osmnx call
    shape) restore it explicitly via their own monkeypatch, overriding this
    one for just that test."""
    fake_graph = fake_osm_multidigraph()
    monkeypatch.setattr(
        network_graph, "_load_or_fetch_osm_graph", lambda config: fake_graph.copy()
    )
