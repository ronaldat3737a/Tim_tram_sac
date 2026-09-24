"""Traffic network graph: real street geometry, routing, station/depot POIs.

The graph is always a real OpenStreetMap street network (section 11), never
a synthetic/abstract one. Every real traffic node is an actual intersection/
point on an actual street, with accurate (x=lng, y=lat); every edge's
`geometry` is a real `shapely.geometry.LineString` following the street's
actual curve (or a straight 2-point LineString when OSM itself records that
segment as straight -- that is real data, not an invented shortcut).

A station/depot is NOT itself a traffic node (Task 2/3 of the spatial-
geometry refactor): it is a small POI node, connected to one real "access
node" by a short synthetic spur edge, offset perpendicular to the access
node's road by a real physical distance (config.poi_offset_meters) so it
renders beside the road, never on it. Because the spur is an ordinary graph
edge like any other, the existing routing/movement/geometry-interpolation
code handles an EV crawling into and back out of a station/depot with no
special-casing at all -- shortest_path simply routes through it.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from backend.config import SimulationConfig

VALID_ROUTING_WEIGHTS = ("distance", "travel_time")

# Local flat-earth approximation for converting between degrees and meters,
# accurate enough at the scale of this simulation (a ~1km-radius extract,
# offsets of ~10m) without needing a full projected CRS (e.g. via pyproj).
_METERS_PER_DEGREE_LAT = 111_320.0


def _meters_per_degree_lng(latitude_deg: float) -> float:
    return _METERS_PER_DEGREE_LAT * math.cos(math.radians(latitude_deg))


def build_network(config: SimulationConfig, seed: int | None = None) -> nx.Graph:
    """Build the traffic graph from a real OpenStreetMap street network
    around `config.osm_center_lat/lng`.

    Downloaded once via osmnx and cached to `config.osm_cache_path` as
    GraphML -- repeated calls (every test run, every simulator reset) reuse
    the cached file instead of re-querying the Overpass API, which keeps
    this reproducible (section 26) and avoids a network dependency on the
    hot path (section 33).

    The node/edge count is whatever OSM returns for the requested radius --
    there is no "num_nodes" to configure. config.num_stations/num_depots
    control how many POI nodes get added (see module docstring).
    """
    if config.routing_weight not in VALID_ROUTING_WEIGHTS:
        raise ValueError(f"routing_weight must be one of {VALID_ROUTING_WEIGHTS}")

    rng = np.random.default_rng(seed if seed is not None else config.random_seed)

    raw_graph = _load_or_fetch_osm_graph(config)
    graph = _convert_osm_graph(raw_graph)
    real_node_ids = list(graph.nodes())

    if config.num_stations > len(real_node_ids):
        raise ValueError(
            "num_stations cannot exceed the number of nodes in the OSM graph "
            f"({len(real_node_ids)} nodes fetched for the configured radius)"
        )
    if config.num_depots > len(real_node_ids):
        raise ValueError(
            "num_depots cannot exceed the number of nodes in the OSM graph "
            f"({len(real_node_ids)} nodes fetched for the configured radius)"
        )

    # Traffic/speed only ever apply to real road edges -- randomized before
    # any POI spur is added, so this loop never touches one.
    _randomize_traffic_and_speed(graph, rng, config)

    station_access_nodes = _pick_access_nodes(rng, real_node_ids, config.num_stations)
    station_nodes = _add_poi_nodes(
        graph, rng, station_access_nodes, config, is_station=True
    )
    depot_access_nodes = _pick_access_nodes(rng, real_node_ids, config.num_depots)
    depot_nodes = _add_poi_nodes(
        graph, rng, depot_access_nodes, config, is_station=False
    )

    graph.graph["station_nodes"] = station_nodes
    graph.graph["station_access_nodes"] = station_access_nodes
    graph.graph["depot_nodes"] = depot_nodes
    graph.graph["depot_access_nodes"] = depot_access_nodes

    xs = [data["x"] for _, data in graph.nodes(data=True)]
    ys = [data["y"] for _, data in graph.nodes(data=True)]
    graph.graph["routing_weight"] = config.routing_weight
    graph.graph["bounds"] = (min(xs), min(ys), max(xs), max(ys))
    return graph


def shortest_path(
    graph: nx.Graph, source: int, target: int, weight: str | None = None
) -> tuple[list[int], float, float]:
    """Return (node path, total distance, total travel_time) via NetworkX."""
    weight_key = weight or graph.graph.get("routing_weight", "travel_time")
    if weight_key not in VALID_ROUTING_WEIGHTS:
        raise ValueError(f"weight must be one of {VALID_ROUTING_WEIGHTS}")

    path = nx.shortest_path(graph, source, target, weight=weight_key)
    total_distance = sum(
        graph.edges[u, v]["distance"] for u, v in zip(path, path[1:])
    )
    total_travel_time = sum(
        graph.edges[u, v]["travel_time"] for u, v in zip(path, path[1:])
    )
    return path, total_distance, total_travel_time


def get_station_nodes(graph: nx.Graph) -> list[int]:
    """Return each station's POI node id (routing target and display
    position both -- see module docstring)."""
    return list(graph.graph["station_nodes"])


def get_depot_nodes(graph: nx.Graph) -> list[int]:
    """Return each depot's POI node id (section 13 extension: where an EV
    parks once it finishes charging)."""
    return list(graph.graph["depot_nodes"])


def get_station_access_nodes(graph: nx.Graph) -> list[int]:
    """Return each station's real traffic (access) node id, in the same
    order as get_station_nodes -- the point on the actual road its spur
    edge connects to."""
    return list(graph.graph["station_access_nodes"])


def get_depot_access_nodes(graph: nx.Graph) -> list[int]:
    """Same as get_station_access_nodes, for depots."""
    return list(graph.graph["depot_access_nodes"])


def _randomize_traffic_and_speed(
    graph: nx.Graph, rng: np.random.Generator, config: SimulationConfig
) -> None:
    """Given edges that already carry a `distance` attribute, randomize
    speed/traffic_weight and derive travel_time (Phase 0 D.2: traffic is
    static per episode, unaffected by this being a real street graph)."""
    for u, v in graph.edges():
        distance = graph.edges[u, v]["distance"]
        speed = float(rng.uniform(config.min_speed, config.max_speed))
        traffic_weight = float(
            rng.uniform(config.min_traffic_weight, config.max_traffic_weight)
        )
        effective_speed = speed / (1.0 + traffic_weight)
        travel_time = distance / effective_speed
        graph.edges[u, v].update(
            speed=speed,
            traffic_weight=traffic_weight,
            travel_time=travel_time,
        )


def _pick_access_nodes(
    rng: np.random.Generator, real_node_ids: list[int], count: int
) -> list[int]:
    return sorted(int(n) for n in rng.choice(real_node_ids, size=count, replace=False))


def _perpendicular_offset_point(
    edge_line: LineString, at_xy: tuple[float, float], offset_meters: float
) -> tuple[float, float]:
    """Task 2: a real orthogonal-offset computation. Takes one real edge
    incident to `at_xy` (an access node), computes that edge's direction
    vector at the end touching `at_xy`, rotates it 90 degrees to get the
    perpendicular (normal) direction, and returns a point exactly
    `offset_meters` away along that normal -- so it lands beside the road
    regardless of the road's own bearing, never on top of it. The direction
    vector and the offset are both computed in local meters (via a flat
    equirectangular approximation) rather than raw degrees, since a degree
    of longitude and a degree of latitude are not the same physical
    distance -- an offset expressed as a flat degree delta would not
    actually be `offset_meters` in reality, nor consistently perpendicular.
    """
    coords = list(edge_line.coords)
    at_point = Point(at_xy)
    if Point(coords[0]).distance(at_point) <= Point(coords[-1]).distance(at_point):
        p0, p1 = coords[0], coords[1] if len(coords) > 1 else coords[0]
    else:
        p0, p1 = coords[-1], coords[-2] if len(coords) > 1 else coords[-1]

    meters_per_degree_lng = _meters_per_degree_lng(at_xy[1])
    dx_m = (p1[0] - p0[0]) * meters_per_degree_lng
    dy_m = (p1[1] - p0[1]) * _METERS_PER_DEGREE_LAT
    norm = math.hypot(dx_m, dy_m)
    if norm < 1e-6:
        dx_m, dy_m, norm = 1.0, 0.0, 1.0  # degenerate zero-length segment

    perp_x_m = -dy_m / norm
    perp_y_m = dx_m / norm

    offset_x_deg = (perp_x_m * offset_meters) / meters_per_degree_lng
    offset_y_deg = (perp_y_m * offset_meters) / _METERS_PER_DEGREE_LAT
    return at_xy[0] + offset_x_deg, at_xy[1] + offset_y_deg


def _add_poi_nodes(
    graph: nx.Graph,
    rng: np.random.Generator,
    access_nodes: list[int],
    config: SimulationConfig,
    is_station: bool,
) -> list[int]:
    """Add one POI node per access node, connected by a short synthetic
    spur edge (Task 3: the "smooth transition trajectory" an EV crawls
    along to enter/exit -- an ordinary edge needs no special movement
    logic, unlike a one-off teleport would). Returns the new POI node ids,
    in the same order as access_nodes."""
    poi_nodes = []
    next_id = max(graph.nodes()) + 1
    for access_node in access_nodes:
        incident_edges = list(graph.edges(access_node, data=True))
        edge_index = int(rng.integers(0, len(incident_edges)))
        _, _, edge_data = incident_edges[edge_index]

        access_xy = (graph.nodes[access_node]["x"], graph.nodes[access_node]["y"])
        poi_xy = _perpendicular_offset_point(
            edge_data["geometry"], access_xy, config.poi_offset_meters
        )

        poi_node = next_id
        next_id += 1
        graph.add_node(poi_node, x=poi_xy[0], y=poi_xy[1], is_station=is_station, is_depot=not is_station)

        distance = config.poi_offset_meters
        travel_time = distance / config.poi_approach_speed_mps
        graph.add_edge(
            access_node,
            poi_node,
            distance=distance,
            geometry=LineString([access_xy, poi_xy]),
            geometry_start=access_node,
            speed=config.poi_approach_speed_mps,
            traffic_weight=0.0,
            travel_time=travel_time,
        )
        poi_nodes.append(poi_node)
    return poi_nodes


def _load_or_fetch_osm_graph(config: SimulationConfig):
    """Load the cached OSM graph from disk, or fetch it from the Overpass
    API (via osmnx) and cache it on first use. Kept as its own function so
    tests can monkeypatch it with a small handcrafted graph instead of
    requiring network access."""
    import osmnx as ox

    cache_path = Path(config.osm_cache_path)
    if cache_path.exists():
        return ox.load_graphml(cache_path)

    raw_graph = ox.graph_from_point(
        (config.osm_center_lat, config.osm_center_lng),
        dist=config.osm_radius_meters,
        network_type="drive",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(raw_graph, cache_path)
    return raw_graph


def _convert_osm_graph(raw_graph) -> nx.Graph:
    """Convert osmnx's directed MultiDiGraph into the simple, undirected
    nx.Graph this system's routing/simulation code expects: every street is
    treated as bidirectional (this system has no one-way-street concept),
    and parallel/duplicate edges collapse to the shortest one. Every edge's
    `geometry` is kept as a real `shapely.geometry.LineString` (Task 1: an
    EV's displayed position is always a `Point` that lies exactly on this
    LineString via `.interpolate()`, never a hand-rolled approximation) --
    the real curve when OSM recorded one, or a straight 2-point LineString
    when OSM itself records that segment as straight (real data, not an
    invented shortcut). Node ids are relabeled to a contiguous 0..N-1 range
    (the original OSM id is kept as `osm_id`) so POI-node ids added later
    have a simple next-integer to use. Only the largest connected component
    is kept, so `shortest_path` is always well-defined between any two
    nodes."""
    graph = nx.Graph()
    for node_id, data in raw_graph.nodes(data=True):
        graph.add_node(
            node_id, x=float(data["x"]), y=float(data["y"]), is_station=False, is_depot=False
        )

    for u, v, data in raw_graph.edges(data=True):
        if u == v:
            continue  # skip self-loop artifacts
        length = float(data.get("length", 0.0))
        if length <= 0.0:
            continue
        if graph.has_edge(u, v) and graph.edges[u, v]["distance"] <= length:
            continue  # keep the shortest of any parallel/duplicate OSM edges

        line = data.get("geometry")
        if line is None:
            node_u, node_v = raw_graph.nodes[u], raw_graph.nodes[v]
            line = LineString([(node_u["x"], node_u["y"]), (node_v["x"], node_v["y"])])

        graph.add_edge(u, v, distance=length, geometry=line, geometry_start=u)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError("OSM graph has no usable nodes/edges after conversion")

    largest_component = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_component).copy()
    graph = nx.convert_node_labels_to_integers(
        graph, ordering="sorted", label_attribute="osm_id"
    )
    # convert_node_labels_to_integers only relabels the graph's own node
    # keys/edge tuples -- it has no way to know that `geometry_start` (an
    # edge attribute *value*) is itself a node id, so it's left holding a
    # stale raw OSM id unless remapped explicitly here.
    old_to_new = {data["osm_id"]: new_id for new_id, data in graph.nodes(data=True)}
    for u, v in graph.edges():
        graph.edges[u, v]["geometry_start"] = old_to_new[graph.edges[u, v]["geometry_start"]]
    return graph
