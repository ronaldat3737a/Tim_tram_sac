from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.api.routes import sandbox as sandbox_routes
from backend.config import DEFAULT_CONFIG
from backend.simulation import sandbox_network
from backend.simulation.network_graph import get_depot_nodes, get_station_nodes
from backend.simulation.sandbox_network import (
    MapFetchError,
    SandboxParams,
    SandboxSimulator,
    build_sandbox,
    farthest_point_order,
    kmeans,
    trim_to_circle,
)
from backend.tests.conftest import fake_osm_multidigraph
from backend.tests.scenario_helpers import single_scenario

# conftest's fake extract: an 11 x 11 grid starting at this corner.
CENTER = (21.0385 + 0.0025, 105.7973 + 0.0025)
PARAMS = SandboxParams(lat=CENTER[0], lng=CENTER[1], radius_km=1.0, num_stations=6, num_evs=20, num_rest_stops=3)


@pytest.fixture(autouse=True)
def fake_map(monkeypatch):
    monkeypatch.setattr(sandbox_network, "load_or_fetch_road_graph", lambda params: (fake_osm_multidigraph(), True))


def test_kmeans_separates_obvious_clusters():
    rng = np.random.default_rng(0)
    blobs = [rng.normal(center, 1.0, size=(50, 2)) for center in ((0, 0), (100, 0), (0, 100))]
    centroids, labels = kmeans(np.vstack(blobs), 3, rng)

    assert sorted(np.round(centroids / 100).astype(int).tolist()) == [[0, 0], [0, 1], [1, 0]]
    assert all(len(set(labels[i * 50 : (i + 1) * 50])) == 1 for i in range(3))


def test_kmeans_rejects_more_clusters_than_points():
    with pytest.raises(ValueError):
        kmeans(np.zeros((3, 2)), 4, np.random.default_rng(0))


def test_farthest_point_order_spreads_out():
    points = np.array([[0, 0], [1, 0], [10, 0], [5, 0]], dtype=float)
    assert farthest_point_order(points, 0) == [0, 2, 3, 1]


def test_trim_to_circle_keeps_edges_crossing_the_boundary_whole():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=0.009, y=0.0)  # ~1 km east: outside a 500 m circle
    graph.add_node(3, x=0.0, y=0.05)  # far away, no edge to the inside
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)

    trimmed = trim_to_circle(graph, 0.0, 0.0, 500.0)

    assert set(trimmed.nodes()) == {1, 2}
    assert trimmed.has_edge(1, 2)


def test_build_sandbox_places_the_requested_pois_on_distinct_road_nodes():
    sandbox = build_sandbox(PARAMS)
    graph = sandbox.graph

    stations, depots = get_station_nodes(graph), get_depot_nodes(graph)
    assert len(stations) == PARAMS.num_stations and len(depots) == PARAMS.num_rest_stops
    access = graph.graph["station_access_nodes"] + graph.graph["depot_access_nodes"]
    assert len(set(access)) == len(access)  # one POI per cluster center
    assert nx.is_connected(graph)
    config = sandbox.config
    assert (config.num_stations, config.num_vehicles, config.num_depots) == (6, 20, 3)
    assert config.osm_center_lat == PARAMS.lat and config.osm_radius_meters == 1000.0


def test_kmeans_stations_are_spread_wider_than_random_ones():
    graph = build_sandbox(PARAMS).graph

    def min_spacing(nodes):
        xy = np.array([[graph.nodes[n]["x"], graph.nodes[n]["y"]] for n in nodes])
        d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1)) + np.eye(len(nodes)) * 1e9
        return d.min()

    real = [n for n, data in graph.nodes(data=True) if not data["is_station"] and not data["is_depot"]]
    random_spacings = [
        min_spacing(np.random.default_rng(seed).choice(real, PARAMS.num_stations, replace=False)) for seed in range(20)
    ]
    assert min_spacing(get_station_nodes(graph)) > np.median(random_spacings)


def test_battery_range_covers_the_farthest_node_with_margin():
    sandbox = build_sandbox(PARAMS)
    config = sandbox.config
    request_range = config.low_battery_threshold / config.battery_consumption_per_distance
    assert request_range >= sandbox_network.BATTERY_SAFETY_FACTOR * sandbox.farthest_from_station_m - 1e-6
    # Only ever widened, never narrowed below the base scenario's range.
    assert config.battery_consumption_per_distance <= sandbox_network.SANDBOX_BASE_CONFIG.battery_consumption_per_distance


def test_too_many_pois_for_the_map_is_a_value_error():
    with pytest.raises(ValueError, match="exceeds"):
        build_sandbox(replace(PARAMS, num_stations=500))


def test_sandbox_simulator_uses_the_template_and_places_evs_on_road_nodes():
    sandbox = build_sandbox(PARAMS)
    simulator = SandboxSimulator(sandbox.config, sandbox.graph, seed=1)

    assert simulator.graph is not sandbox.graph
    assert len(simulator.stations) == PARAMS.num_stations
    assert simulator.depot_nodes == get_depot_nodes(sandbox.graph)
    assert len(simulator.vehicles) == PARAMS.num_evs
    for vehicle in simulator.vehicles.values():
        data = simulator.graph.nodes[vehicle.current_node]
        assert not data["is_station"] and not data["is_depot"]


def test_fetch_retries_overloaded_overpass_then_gives_up(monkeypatch, tmp_path):
    import osmnx as ox

    monkeypatch.undo()  # the real loader, with the fetch itself stubbed
    monkeypatch.setattr(sandbox_network, "SANDBOX_CACHE_DIR", tmp_path)
    monkeypatch.setattr(sandbox_network, "FETCH_RETRY_PAUSES", (0.0, 0.0))
    calls = []

    def overloaded(*args, **kwargs):
        calls.append(1)
        raise ox._errors.ResponseStatusCodeError("'overpass-api.de' responded: 502 Bad Gateway")

    monkeypatch.setattr(ox, "graph_from_point", overloaded)
    with pytest.raises(MapFetchError, match="3 attempts"):
        sandbox_network.load_or_fetch_road_graph(PARAMS)
    assert len(calls) == 3


# --- API ---------------------------------------------------------------------


@pytest.fixture
def client():
    sandbox_routes._active = None
    with TestClient(create_app(single_scenario(DEFAULT_CONFIG))) as test_client:
        yield test_client
    sandbox_routes._active = None


def _payload(**overrides):
    return {"lat": CENTER[0], "lng": CENTER[1], "radius_km": 1, "num_stations": 6, "num_evs": 20,
            "num_rest_stops": 3, **overrides}


def test_generate_env_returns_network_and_snapshot_in_the_dashboard_shapes(client):
    response = client.post("/api/sandbox/generate-env", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert len(body["network"]["stations"]) == 6 and len(body["network"]["depots"]) == 3
    assert body["network"]["scenario"] == "sandbox"
    assert all(len(edge["geometry"]) >= 2 for edge in body["network"]["edges"])
    node_ids = {node["id"] for node in body["network"]["nodes"]}
    assert {s["node_id"] for s in body["network"]["stations"]} <= node_ids
    assert body["snapshot"]["type"] == "simulation_update"
    assert len(body["snapshot"]["stations"]) == 6
    assert body["sandbox"]["num_vehicles_departed"] == len(body["snapshot"]["vehicles"])
    assert body["sandbox"]["battery"]["request_range_km"] >= 1.5 * body["sandbox"]["battery"]["farthest_from_station_km"] - 1e-6
    south, west, north, east = body["sandbox"]["map"]["bounds"]
    assert south < north and west < east


def test_get_env_returns_the_active_sandbox(client):
    assert client.get("/api/sandbox/env").status_code == 404
    generated = client.post("/api/sandbox/generate-env", json=_payload()).json()

    again = client.get("/api/sandbox/env").json()

    assert again["sandbox"]["sandbox_id"] == generated["sandbox"]["sandbox_id"]


def test_generate_env_errors_map_to_clear_status_codes(client, monkeypatch):
    assert client.post("/api/sandbox/generate-env", json=_payload(num_stations=500, num_rest_stops=100)).status_code in (400, 422)
    assert client.post("/api/sandbox/generate-env", json=_payload(radius_km=50)).status_code == 422

    def unavailable(params):
        raise MapFetchError("Overpass did not answer")

    monkeypatch.setattr(sandbox_network, "load_or_fetch_road_graph", unavailable)
    response = client.post("/api/sandbox/generate-env", json=_payload())
    assert response.status_code == 503 and "Overpass" in response.json()["detail"]


def test_a_second_generation_while_one_runs_gets_409(client):
    sandbox_routes._generate_lock.acquire()
    try:
        assert client.post("/api/sandbox/generate-env", json=_payload()).status_code == 409
    finally:
        sandbox_routes._generate_lock.release()


# --- lazy routing --------------------------------------------------------------


STAGGERED = replace(PARAMS, num_evs=30)


def _staggered_sandbox():
    sandbox = build_sandbox(STAGGERED)
    # 60% peak (depart at 0), the rest over 300 ticks, as the real cohorts.
    sandbox.config = replace(sandbox.config, max_activation_tick=300)
    return sandbox


def test_creating_evs_computes_no_route(monkeypatch):
    sandbox = _staggered_sandbox()
    calls = []
    monkeypatch.setattr(sandbox_network, "shortest_path", lambda *a, **k: calls.append(a) or ([a[1], a[2]], 0.0, 0.0))

    simulator = SandboxSimulator(sandbox.config, sandbox.graph, seed=1)

    assert calls == []
    assert all(v.route == [v.current_node] for v in simulator.vehicles.values())
    assert simulator.num_unrouted == STAGGERED.num_evs
    for vehicle in simulator.vehicles.values():
        assert vehicle.destination_node != vehicle.current_node
        assert not simulator.graph.nodes[vehicle.destination_node]["is_station"]


def test_each_ev_is_routed_right_before_it_departs():
    sandbox = _staggered_sandbox()
    simulator = SandboxSimulator(sandbox.config, sandbox.graph, seed=1)
    late = [v for v in simulator.vehicles.values() if v.activation_tick > 5]
    assert late, "expected some late joiners"

    for _ in range(3):
        simulator.tick()

    for vehicle in simulator.vehicles.values():
        if vehicle.activation_tick > simulator.simulation_time:
            assert vehicle.route == [vehicle.current_node]  # not routed yet
    target = min(late, key=lambda v: v.activation_tick)
    while simulator.simulation_time < target.activation_tick:
        simulator.tick()
    simulator.tick()
    assert target.state != sandbox_network.VehicleState.COMPLETED or target.current_node == target.destination_node
    assert len(target.route) >= 1 and (target.route[-1] == target.destination_node or target.target_station is not None)


def test_route_departing_routes_the_start_cohort_with_progress():
    sandbox = _staggered_sandbox()
    simulator = SandboxSimulator(sandbox.config, sandbox.graph, seed=1)
    expected = simulator.num_departing_by(0)
    progress = []

    done = simulator.route_departing(0, progress=lambda d, t: progress.append((d, t)))

    assert done == expected > 0
    assert progress[-1] == (expected, expected)
    for vehicle in simulator.vehicles.values():
        if vehicle.activation_tick == 0:
            assert vehicle.route[0] == vehicle.current_node and vehicle.route[-1] == vehicle.destination_node


def test_lazily_routed_episode_runs_to_completion_like_the_base_simulator():
    sandbox = _staggered_sandbox()
    simulator = SandboxSimulator(sandbox.config, sandbox.graph, seed=1)
    for _ in range(4000):
        if simulator.all_vehicles_done():
            break
        for vehicle in simulator.get_vehicles_needing_decision():
            reachable = [sid for sid in simulator.stations if simulator.is_station_reachable(vehicle.vehicle_id, sid)]
            if reachable:
                simulator.assign_station(vehicle.vehicle_id, reachable[0])
        simulator.tick()
    assert simulator.num_unrouted == 0
    assert all(v.state.value in ("COMPLETED", "FAILED", "CHARGING", "WAITING", "RETURNING_TO_DEPOT", "TRAVELING")
               for v in simulator.vehicles.values())
    assert sum(v.state.value == "COMPLETED" for v in simulator.vehicles.values()) > 0


# --- live session ----------------------------------------------------------------


@pytest.fixture
def gnn_model(tmp_path):
    from backend.ai_core.mappo_gnn import MAPPOGNNAgent

    path = tmp_path / "gnn.pth"
    MAPPOGNNAgent(hidden_size=16, heads=2).save(path, decision_window=5)
    return path


def _wait(predicate, timeout=20.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_session_waits_at_t0_runs_on_start_and_closes_its_worker(gnn_model):
    from backend.api.sandbox_session import SandboxSession

    session = SandboxSession("s1", _staggered_sandbox(), seed=1, model_path=gnn_model)
    session.open()
    try:
        assert _wait(lambda: session.status_message()["status"] == "paused")
        status = session.status_message()
        assert status["simulation_time"] == 0
        assert status["preparing"]["routes_done"] == status["preparing"]["routes_total"] > 0
        assert session.latest_snapshot()[1]["type"] == "simulation_update"

        session.control({"action": "speed", "value": 200})
        session.control({"action": "start"})
        assert _wait(lambda: session.status_message()["num_decisions"] > 0)
        session.control({"action": "pause"})
        assert _wait(lambda: session.status_message()["status"] == "paused")
        with pytest.raises(ValueError):
            session.control({"action": "speed", "value": 0})
        with pytest.raises(ValueError):
            session.control({"action": "fly"})
    finally:
        session.close()
    assert not session.worker_alive
    assert session.status_message()["status"] == "stopped"


def test_session_reports_a_missing_model_as_error(tmp_path):
    from backend.api.sandbox_session import SandboxSession

    session = SandboxSession("s2", _staggered_sandbox(), seed=1, model_path=tmp_path / "missing.pth")
    session.open()
    try:
        assert _wait(lambda: session.status_message()["status"] == "error")
        assert "missing.pth" in session.status_message()["error"]
    finally:
        session.close()


def test_websocket_streams_status_and_snapshots_and_stops_on_disconnect(client, gnn_model, monkeypatch):
    from backend.api import sandbox_session

    monkeypatch.setattr(sandbox_session, "GNN_MODEL_PATH", gnn_model)
    sandbox_id = client.post("/api/sandbox/generate-env", json=_payload()).json()["sandbox"]["sandbox_id"]

    with client.websocket_connect(f"/api/sandbox/ws/{sandbox_id}") as ws:
        seen = set()
        while not {"simulation_update", "sandbox_status"} <= seen:
            seen.add(ws.receive_json()["type"])
        ws.send_json({"action": "speed", "value": 200})
        ws.send_json({"action": "start"})
        status = None
        for _ in range(500):
            message = ws.receive_json()
            if message["type"] == "sandbox_status" and message["simulation_time"] > 0:
                status = message
                break
        assert status is not None and status["status"] == "running"
        ws.send_json({"action": "fly"})
        assert any(ws.receive_json()["type"] == "sandbox_error" for _ in range(50))
        session = sandbox_session.current_session()
        assert session is not None and session.worker_alive

    assert _wait(lambda: not session.worker_alive)
    assert sandbox_session.current_session() is None


def test_websocket_rejects_an_unknown_sandbox(client):
    with client.websocket_connect("/api/sandbox/ws/nope") as ws:
        assert ws.receive_json()["type"] == "sandbox_error"
