"""User-defined "sandbox" maps: any center and radius, with stations and rest
stops spread evenly by K-Means instead of picked at random.

The fixed scenarios (network_graph.build_network) are untouched; this module
reuses its graph conversion, traffic and POI helpers and only differs in:

- Fetching: osmnx's "drive" network around (lat, lng), fetched with
  truncate_by_edge=True and simplified with ox.simplify_graph(), then trimmed
  to the circle of the requested radius -- an edge stays if at least one end
  is inside, so streets crossing the boundary are kept whole rather than
  cut. Raw extracts are cached under data/sandbox/ by (center, radius).
- Placement (anti-clustering): the road nodes are split into
  K = num_stations + num_rest_stops clusters by K-Means on their positions in
  meters, and a POI is placed at each cluster's most central node (the node
  nearest its centroid). The num_stations of those K spots that are farthest
  apart (farthest-point order) become charging stations, the rest become
  rest stops (the simulator's depots), so each set is spread out and the two
  never pile up on each other. Clustering by node, not by area, puts more
  stations where the road network (and so traffic) is denser.
- Battery: the range is widened if needed so that an EV asking for a station
  can reach the nearest one from anywhere on the map (see calibrate_battery).

EVs are placed by the unchanged Simulator logic: uniformly at random over
the road nodes that are not a station or rest stop.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import networkx as nx
import numpy as np

from backend.config import OSM_LARGE_CONFIG, SimulationConfig
from backend.simulation.network_graph import (
    _add_poi_nodes,
    _convert_osm_graph,
    _randomize_traffic_and_speed,
    get_station_nodes,
    shortest_path,
)
from backend.simulation.simulator import _LATE, _NON_APP, _PEAK, Simulator
from backend.simulation.vehicle import Vehicle, VehicleState

SANDBOX_CACHE_DIR = Path("data/sandbox")
# Physics, cohorts and charging of the large scenario; counts, map and
# battery range are set per sandbox.
SANDBOX_BASE_CONFIG = OSM_LARGE_CONFIG
# Range left when an EV asks for a station must cover this many times the
# farthest any road node sits from its nearest station.
BATTERY_SAFETY_FACTOR = 1.5
KMEANS_ITERATIONS = 50
# Seconds to wait before retrying an Overpass request that failed: the public
# server answers 5xx when at capacity and asks clients to back off and retry.
FETCH_RETRY_PAUSES = (5.0, 15.0, 30.0)

_METERS_PER_DEGREE_LAT = 111_320.0


class MapFetchError(RuntimeError):
    """The map could not be downloaded (Overpass unavailable)."""


@dataclass(frozen=True)
class SandboxParams:
    lat: float
    lng: float
    radius_km: float
    num_stations: int
    num_evs: int
    num_rest_stops: int
    seed: int = 42

    @property
    def radius_m(self) -> float:
        return self.radius_km * 1000.0


@dataclass
class SandboxNetwork:
    """A generated sandbox: the graph template every episode copies, the
    config to simulate it with, and how it was built."""

    params: SandboxParams
    config: SimulationConfig
    graph: nx.Graph
    from_cache: bool
    fetch_seconds: float
    build_seconds: float
    # Farthest any road node sits from its nearest station, meters of road.
    farthest_from_station_m: float


# --- fetching ----------------------------------------------------------------


def cache_path(params: SandboxParams) -> Path:
    return SANDBOX_CACHE_DIR / f"osm_{params.lat:.5f}_{params.lng:.5f}_{int(round(params.radius_m))}m.graphml"


def load_or_fetch_road_graph(params: SandboxParams):
    """(raw osmnx MultiDiGraph inside the circle, from_cache). Kept as its
    own function so tests can replace it without network access."""
    import osmnx as ox

    path = cache_path(params)
    if path.exists():
        return ox.load_graphml(path), True
    for attempt, pause in enumerate((*FETCH_RETRY_PAUSES, None)):
        try:
            raw = ox.graph_from_point(
                (params.lat, params.lng),
                dist=params.radius_m,
                network_type="drive",
                simplify=False,
                truncate_by_edge=True,
            )
            break
        except ox._errors.InsufficientResponseError as exc:
            raise ValueError(f"no drivable road found within {params.radius_km} km of ({params.lat}, {params.lng})") from exc
        except Exception as exc:  # Overpass overloaded (5xx), timeout, connection reset
            if pause is None:
                raise MapFetchError(
                    f"OpenStreetMap (Overpass API) did not answer after {attempt + 1} attempts: {exc}"
                ) from exc
            time.sleep(pause)
    raw = ox.simplify_graph(raw)
    raw = trim_to_circle(raw, params.lat, params.lng, params.radius_m)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(raw, path)
    return raw, False


def trim_to_circle(raw, lat: float, lng: float, radius_m: float):
    """Keep every edge with at least one end inside the circle (and those
    ends), so no street crossing the boundary is cut short."""
    inside = {
        node for node, data in raw.nodes(data=True) if _distance_m(lat, lng, data["y"], data["x"]) <= radius_m
    }
    keep = set(inside)
    for u, v in raw.edges():
        if u in inside or v in inside:
            keep.update((u, v))
    return raw.subgraph(keep).copy()


def _distance_m(lat0: float, lng0: float, lat: float, lng: float) -> float:
    dy = (lat - lat0) * _METERS_PER_DEGREE_LAT
    dx = (lng - lng0) * _METERS_PER_DEGREE_LAT * math.cos(math.radians(lat0))
    return math.hypot(dx, dy)


# --- placement -----------------------------------------------------------------


def kmeans(points: np.ndarray, k: int, rng: np.random.Generator, iterations: int = KMEANS_ITERATIONS) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's K-Means with k-means++ seeding. Returns (centroids (k, d),
    label of every point (n,))."""
    n = len(points)
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    centroids = [points[rng.integers(n)]]
    nearest_sq = ((points - centroids[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = nearest_sq.sum()
        index = rng.integers(n) if total <= 0 else rng.choice(n, p=nearest_sq / total)
        centroids.append(points[index])
        nearest_sq = np.minimum(nearest_sq, ((points - points[index]) ** 2).sum(axis=1))
    centroids = np.array(centroids, dtype=float)

    labels = np.zeros(n, dtype=int)
    point_sq = (points**2).sum(axis=1)[:, None]
    for iteration in range(iterations):
        # (n, k) squared distances without an (n, k, d) intermediate.
        distances = point_sq - 2 * points @ centroids.T + (centroids**2).sum(axis=1)[None, :]
        new_labels = distances.argmin(axis=1)
        if iteration > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = points[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
            else:  # empty cluster: restart it at the point farthest from its centroid
                far = distances[np.arange(n), labels].argmax()
                centroids[j] = points[far]
    return centroids, labels


def cluster_center_nodes(node_ids: list[int], points: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """K-Means over the nodes; for each cluster, the member node nearest its
    centroid (each node used once)."""
    centroids, labels = kmeans(points, k, rng)
    chosen: list[int] = []
    taken: set[int] = set()
    for j in range(k):
        members = np.flatnonzero(labels == j)
        if len(members) == 0:
            members = np.arange(len(points))
        order = members[np.argsort(((points[members] - centroids[j]) ** 2).sum(axis=1), kind="stable")]
        node = next((int(node_ids[i]) for i in order if int(node_ids[i]) not in taken), None)
        if node is None:  # every member already used: nearest unused node anywhere
            order = np.argsort(((points - centroids[j]) ** 2).sum(axis=1), kind="stable")
            node = next(int(node_ids[i]) for i in order if int(node_ids[i]) not in taken)
        chosen.append(node)
        taken.add(node)
    return chosen


def farthest_point_order(points: np.ndarray, start: int) -> list[int]:
    """Indices of `points` in greedy farthest-point order from `start`: each
    next index is the one farthest from all chosen so far."""
    order = [start]
    nearest = ((points - points[start]) ** 2).sum(axis=1)
    for _ in range(len(points) - 1):
        nearest[order] = -1.0
        index = int(nearest.argmax())
        order.append(index)
        nearest = np.minimum(nearest, ((points - points[index]) ** 2).sum(axis=1))
    return order


def place_pois(graph: nx.Graph, params: SandboxParams, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """(station access nodes, rest-stop access nodes), see module docstring."""
    node_ids = list(graph.nodes())
    k = params.num_stations + params.num_rest_stops
    if k > len(node_ids):
        raise ValueError(
            f"num_stations + num_rest_stops = {k} exceeds the {len(node_ids)} road nodes of this map; "
            "pick a larger radius or fewer stations"
        )
    cos_lat = math.cos(math.radians(params.lat))
    points = np.array(
        [
            [
                (graph.nodes[n]["x"] - params.lng) * _METERS_PER_DEGREE_LAT * cos_lat,
                (graph.nodes[n]["y"] - params.lat) * _METERS_PER_DEGREE_LAT,
            ]
            for n in node_ids
        ]
    )
    centers = cluster_center_nodes(node_ids, points, k, rng)
    index_of = {node: i for i, node in enumerate(node_ids)}
    center_points = np.array([points[index_of[n]] for n in centers])
    # Start from the spot nearest the map center, then spread outwards.
    start = int((center_points**2).sum(axis=1).argmin())
    order = farthest_point_order(center_points, start)
    stations = [centers[i] for i in order[: params.num_stations]]
    rest_stops = [centers[i] for i in order[params.num_stations :]]
    return stations, rest_stops


# --- building ------------------------------------------------------------------


def sandbox_config(params: SandboxParams) -> SimulationConfig:
    return replace(
        SANDBOX_BASE_CONFIG,
        num_stations=params.num_stations,
        num_vehicles=params.num_evs,
        num_depots=params.num_rest_stops,
        max_num_stations=max(SANDBOX_BASE_CONFIG.max_num_stations, params.num_stations),
        max_num_vehicles=max(SANDBOX_BASE_CONFIG.max_num_vehicles, params.num_evs),
        osm_center_lat=params.lat,
        osm_center_lng=params.lng,
        osm_radius_meters=params.radius_m,
        osm_cache_path=str(cache_path(params)),
        osm_road_layer_cache_path=str(cache_path(params)),
        random_seed=params.seed,
    )


def calibrate_battery(config: SimulationConfig, graph: nx.Graph) -> tuple[SimulationConfig, float]:
    """Widen the battery range (never narrow it) so the range left at
    low_battery_threshold covers BATTERY_SAFETY_FACTOR x the farthest any
    road node sits from its nearest station -- otherwise a sparse, wide map
    strands EVs no policy could save. Returns (config, that distance)."""
    to_station = nx.multi_source_dijkstra_path_length(graph, get_station_nodes(graph), weight="distance")
    farthest = max(to_station.values())
    needed = config.low_battery_threshold / (BATTERY_SAFETY_FACTOR * max(farthest, 1.0))
    if needed < config.battery_consumption_per_distance:
        config = replace(config, battery_consumption_per_distance=needed)
    return config, farthest


def build_sandbox(params: SandboxParams) -> SandboxNetwork:
    """Fetch (or load) the map, convert it, randomize traffic, place stations
    and rest stops by K-Means and calibrate the battery."""
    if params.num_stations < 1 or params.num_evs < 1 or params.num_rest_stops < 1:
        raise ValueError("num_stations, num_evs and num_rest_stops must all be >= 1")
    started = time.perf_counter()
    raw, from_cache = load_or_fetch_road_graph(params)
    fetched = time.perf_counter()

    config = sandbox_config(params)
    rng = np.random.default_rng(params.seed)
    graph = _convert_osm_graph(raw)
    _randomize_traffic_and_speed(graph, rng, config)
    station_access, rest_stop_access = place_pois(graph, params, rng)
    station_nodes = _add_poi_nodes(graph, rng, station_access, config, is_station=True)
    rest_stop_nodes = _add_poi_nodes(graph, rng, rest_stop_access, config, is_station=False)
    graph.graph.update(
        station_nodes=station_nodes,
        station_access_nodes=station_access,
        depot_nodes=rest_stop_nodes,
        depot_access_nodes=rest_stop_access,
        routing_weight=config.routing_weight,
    )
    xs = [data["x"] for _, data in graph.nodes(data=True)]
    ys = [data["y"] for _, data in graph.nodes(data=True)]
    graph.graph["bounds"] = (min(xs), min(ys), max(xs), max(ys))
    # network_graph.shortest_path caches one shortest-path tree per
    # station/depot in graph.graph. Graph.copy() copies graph.graph
    # shallowly, so creating the cache on the template makes every episode's
    # copy share it: each tree is built once per sandbox, not per episode.
    # Safe because the template (structure and traffic) never changes.
    graph.graph["_shortest_path_trees"] = {}

    config, farthest = calibrate_battery(config, graph)
    return SandboxNetwork(
        params=params,
        config=config,
        graph=graph,
        from_cache=from_cache,
        fetch_seconds=fetched - started,
        build_seconds=time.perf_counter() - fetched,
        farthest_from_station_m=farthest,
    )


class SandboxSimulator(Simulator):
    """The unchanged Simulator, on a sandbox graph instead of build_network,
    with lazy routing. The template is copied, so every episode starts from
    the same map; stations, rest stops (depots), movement, charging and
    dispatch are all the base Simulator's.

    Lazy routing: the base Simulator computes every EV's route to its
    personal destination when it creates the EV -- one shortest-path search
    each, ~14 ms on a 10 km map, so ~30 s for 2,000 EVs before anything can
    run. Here creating an EV only draws its origin, destination, battery,
    speed and departure tick (same rules: uniform over the non-POI road
    nodes, cohorts from the config); its route is computed by tick() right
    before its departure tick, with the same shortest_path the base
    Simulator uses. An EV dispatched to a station before that (possible only
    on its very departure tick) gets its route from assign_station as usual.
    route_departing() lets a caller compute the routes of the EVs departing
    at the start (the peak-hour cohort) ahead of the first tick, e.g. with a
    progress display, instead of inside it."""

    def __init__(self, config: SimulationConfig, graph_template: nx.Graph, seed: int | None = None):
        self.config = config
        self.seed = seed if seed is not None else config.random_seed
        self.rng = np.random.default_rng(self.seed)

        self.graph = graph_template.copy()
        self.stations = self._build_stations()
        self.depot_nodes = list(self.graph.graph["depot_nodes"])
        self.vehicles = self._build_vehicles()
        self.simulation_time = 0.0
        # EVs still waiting for their route, in departure order.
        self._unrouted: list[int] = sorted(self.vehicles, key=lambda vid: self.vehicles[vid].activation_tick)
        self._next_unrouted = 0

    @property
    def num_unrouted(self) -> int:
        return len(self._unrouted) - self._next_unrouted

    def num_departing_by(self, tick: float) -> int:
        """Unrouted EVs departing at or before `tick`."""
        return sum(
            1 for vid in self._unrouted[self._next_unrouted :] if self.vehicles[vid].activation_tick <= tick
        )

    def route_departing(self, tick: float, progress=None) -> int:
        """Compute the routes of every unrouted EV departing at or before
        `tick`. progress(done, total) is called after each route (it may
        raise to abort). Returns how many routes were computed."""
        total = self.num_departing_by(tick)
        done = 0
        while self._next_unrouted < len(self._unrouted):
            vehicle = self.vehicles[self._unrouted[self._next_unrouted]]
            if vehicle.activation_tick > tick:
                break
            self._route_to_destination(vehicle)
            self._next_unrouted += 1
            done += 1
            if progress is not None:
                progress(done, total)
        return done

    def tick(self) -> None:
        # EVs departing this tick need a route before _move_vehicles runs:
        # it treats a 1-node route as "already arrived".
        self.route_departing(self.simulation_time)
        super().tick()

    def _route_to_destination(self, vehicle: Vehicle) -> None:
        if vehicle.state != VehicleState.TRAVELING or vehicle.target_station is not None:
            return  # already dispatched (assign_station set its route)
        try:
            vehicle.route, _, _ = shortest_path(
                self.graph, vehicle.current_node, vehicle.destination_node, weight=self.config.routing_weight
            )
        except nx.NetworkXNoPath:
            # Same fallback as the base Simulator: complete in place.
            vehicle.destination_node = vehicle.current_node
            vehicle.route = [vehicle.current_node]

    def _build_vehicles(self) -> dict[int, Vehicle]:
        """Base Simulator._build_vehicles without the route search (see the
        class docstring). Destinations are drawn by rejection instead of
        building an "every other node" list per EV (1.3 ms each on a 10 km
        map); still uniform over the non-POI road nodes."""
        vehicles = {}
        cohorts = self._draw_cohorts()
        node_ids = np.array(
            [n for n, data in self.graph.nodes(data=True) if not data["is_station"] and not data["is_depot"]]
        )
        max_tick = self.config.max_activation_tick
        for vehicle_id in range(self.config.num_vehicles):
            current_node = int(self.rng.choice(node_ids))
            destination_node = current_node
            while destination_node == current_node and len(node_ids) > 1:
                destination_node = int(self.rng.choice(node_ids))
            battery_level = float(self.rng.uniform(self.config.low_battery_threshold, self.config.battery_capacity))
            speed = float(self.rng.uniform(self.config.min_speed, self.config.max_speed))
            cohort = cohorts[vehicle_id]
            if cohort == _PEAK or max_tick <= 0:
                activation_tick = 0
            elif cohort == _LATE:
                activation_tick = int(self.rng.integers(1, max_tick + 1))
            else:
                activation_tick = int(self.rng.integers(0, max_tick + 1))
            vehicles[vehicle_id] = Vehicle(
                vehicle_id=vehicle_id,
                current_node=current_node,
                destination_node=destination_node,
                battery_level=battery_level,
                battery_capacity=self.config.battery_capacity,
                speed=speed,
                # Placeholder until route_departing computes the real route.
                route=[current_node],
                activation_tick=activation_tick,
                is_non_app=cohort == _NON_APP,
            )
        return vehicles
