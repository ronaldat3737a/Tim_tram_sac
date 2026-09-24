"""Traffic network graph: real street data, routing, station/depot nodes.

The graph is always a real OpenStreetMap street network (section 11), never
a synthetic/abstract one -- per explicit user decision, the earlier
"abstract graph, display-only geo-mapping" mode has been removed entirely
rather than kept as a fallback. Every node is a real intersection/point on
an actual street, with accurate (x=lng, y=lat); every edge carries a real
`geometry` polyline following the street's actual curve.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np

from backend.config import SimulationConfig

VALID_ROUTING_WEIGHTS = ("distance", "travel_time")


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
    control how many of those real nodes become charging stations/depots.
    Distance/speed/traffic_weight/travel_time and the reward/observation/
    action contracts are otherwise unaffected by any of this.
    """
    if config.routing_weight not in VALID_ROUTING_WEIGHTS:
        raise ValueError(f"routing_weight must be one of {VALID_ROUTING_WEIGHTS}")

    rng = np.random.default_rng(seed if seed is not None else config.random_seed)

    raw_graph = _load_or_fetch_osm_graph(config)
    graph = _convert_osm_graph(raw_graph)

    if config.num_stations > graph.number_of_nodes():
        raise ValueError(
            "num_stations cannot exceed the number of nodes in the OSM graph "
            f"({graph.number_of_nodes()} nodes fetched for the configured radius)"
        )
    if config.num_depots > graph.number_of_nodes():
        raise ValueError(
            "num_depots cannot exceed the number of nodes in the OSM graph "
            f"({graph.number_of_nodes()} nodes fetched for the configured radius)"
        )

    _randomize_traffic_and_speed(graph, rng, config)
    _assign_station_nodes(graph, rng, config.num_stations)
    _assign_depot_nodes(graph, rng, config.num_depots)

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
    """Return the list of node ids designated as charging stations."""
    return list(graph.graph["station_nodes"])


def get_depot_nodes(graph: nx.Graph) -> list[int]:
    """Return the list of node ids designated as vehicle depots (section 13
    extension: where an EV parks once it finishes charging)."""
    return list(graph.graph["depot_nodes"])


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


def _assign_station_nodes(
    graph: nx.Graph, rng: np.random.Generator, num_stations: int
) -> None:
    num_nodes = graph.number_of_nodes()
    station_nodes = sorted(
        int(n) for n in rng.choice(num_nodes, size=num_stations, replace=False)
    )
    for node_id in station_nodes:
        graph.nodes[node_id]["is_station"] = True
    graph.graph["station_nodes"] = station_nodes


def _assign_depot_nodes(
    graph: nx.Graph, rng: np.random.Generator, num_depots: int
) -> None:
    """Pick `num_depots` real nodes as vehicle depots (Task 2: "Bãi tập kết
    ... lấy ngẫu nhiên 4-5 điểm từ danh sách Node"). Independent draw from
    station node selection -- a node can be both, this is not disallowed."""
    num_nodes = graph.number_of_nodes()
    depot_nodes = sorted(
        int(n) for n in rng.choice(num_nodes, size=num_depots, replace=False)
    )
    for node_id in depot_nodes:
        graph.nodes[node_id]["is_depot"] = True
    graph.graph["depot_nodes"] = depot_nodes


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
    and parallel/duplicate edges collapse to the shortest one. Node ids are
    relabeled to a contiguous 0..N-1 range (the original OSM id is kept as
    `osm_id`) so `_assign_station_nodes`/`_assign_depot_nodes`'s
    `rng.choice` has a simple integer range to draw from. Only the largest
    connected component is kept, so `shortest_path` is always well-defined
    between any two nodes."""
    import shapely.geometry as geom

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
            line = geom.LineString(
                [(node_u["x"], node_u["y"]), (node_v["x"], node_v["y"])]
            )
        geometry = [[float(x), float(y)] for x, y in line.coords]

        graph.add_edge(u, v, distance=length, geometry=geometry, geometry_start=u)

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
