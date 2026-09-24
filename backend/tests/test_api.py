import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api import simulation_manager as simulation_manager_module
from backend.api.main import create_app
from backend.config import DEFAULT_CONFIG

TEST_CONFIG = replace(
    DEFAULT_CONFIG, num_stations=2, num_vehicles=5, max_episode_steps=800
)

FAST_SPEED = 2000.0  # ticks/sec -- keeps background-thread pacing negligible in tests


def test_cors_allows_frontend_dev_origin():
    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.get(
            "/api/simulation/status", headers={"Origin": "http://localhost:3000"}
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_read_root():
    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_network_matches_config_scale():
    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.get("/api/network")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) > 0
    assert len(body["stations"]) == TEST_CONFIG.num_stations
    assert len(body["depots"]) == TEST_CONFIG.num_depots
    # x/y are the node's own real (lng, lat) -- the graph is always a real
    # OpenStreetMap street network (section 11), never an abstract one.
    for node in body["nodes"]:
        assert -180.0 <= node["x"] <= 180.0
        assert -90.0 <= node["y"] <= 90.0
    for edge in body["edges"]:
        assert len(edge["geometry"]) >= 2


def test_initial_status_is_paused_before_any_start_call():
    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.get("/api/simulation/status")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "paused"
    assert body["num_vehicles_traveling"] + body["num_vehicles_completed"] == TEST_CONFIG.num_vehicles


def test_start_pause_resume_reset_flow():
    with TestClient(create_app(TEST_CONFIG)) as client:
        start_response = client.post(
            "/api/simulation/start",
            json={"seed": 5, "algorithm": "nearest_station", "speed": FAST_SPEED},
        )
        assert start_response.status_code == 200
        assert start_response.json()["status"] == "running"
        assert start_response.json()["episode_seed"] == 5

        pause_response = client.post("/api/simulation/pause")
        assert pause_response.json()["status"] == "paused"

        resume_response = client.post("/api/simulation/resume")
        assert resume_response.json()["status"] == "running"

        reset_response = client.post("/api/simulation/reset", params={"seed": 9})
        assert reset_response.json()["status"] == "paused"
        assert reset_response.json()["episode_seed"] == 9


def test_episode_metrics_accumulate_then_reset_to_zero():
    with TestClient(create_app(TEST_CONFIG)) as client:
        client.post(
            "/api/simulation/start",
            json={"seed": 5, "algorithm": "nearest_station", "speed": FAST_SPEED},
        )

        deadline = time.monotonic() + 10
        num_decisions = 0
        while time.monotonic() < deadline:
            body = client.get("/api/simulation/status").json()
            num_decisions = body["num_decisions"]
            if num_decisions > 0:
                break
            time.sleep(0.05)

        assert num_decisions > 0
        assert body["average_travel_time"] >= 0.0
        assert body["average_waiting_time"] >= 0.0
        assert body["total_system_cost"] >= 0.0
        assert body["num_invalid_actions"] <= body["num_decisions"]

        reset_body = client.post("/api/simulation/reset", params={"seed": 5}).json()
        assert reset_body["num_decisions"] == 0
        assert reset_body["episode_reward"] == 0.0


def test_start_with_invalid_algorithm_returns_422():
    # "algorithm" is a Pydantic Literal, so an unknown value fails request
    # body validation (422) before ever reaching the manager's own 400 checks
    # (e.g. the DQN-model-missing case below, which IS a valid Literal).
    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.post(
            "/api/simulation/start", json={"algorithm": "not_a_real_algorithm", "speed": FAST_SPEED}
        )

    assert response.status_code == 422


def test_start_with_dqn_missing_model_returns_400(monkeypatch, tmp_path):
    monkeypatch.setattr(simulation_manager_module, "DQN_MODEL_PATH", tmp_path / "missing.zip")

    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.post(
            "/api/simulation/start", json={"algorithm": "dqn", "speed": FAST_SPEED}
        )

    assert response.status_code == 400


def test_speed_endpoint_updates_reported_speed():
    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.post("/api/simulation/speed", json={"speed": 10.0})

    assert response.status_code == 200
    assert response.json()["speed"] == pytest.approx(10.0)


def test_speed_endpoint_rejects_non_positive_speed():
    with TestClient(create_app(TEST_CONFIG)) as client:
        response = client.post("/api/simulation/speed", json={"speed": 0.0})

    assert response.status_code == 400


def test_websocket_streams_simulation_update_matching_schema():
    with TestClient(create_app(TEST_CONFIG)) as client:
        client.post(
            "/api/simulation/start",
            json={"seed": 3, "algorithm": "nearest_station", "speed": FAST_SPEED},
        )
        with client.websocket_connect("/ws/simulation") as websocket:
            message = websocket.receive_json()

    assert message["type"] == "simulation_update"
    assert isinstance(message["timestamp"], (int, float))
    assert isinstance(message["vehicles"], list)
    assert isinstance(message["stations"], list)
    if message["vehicles"]:
        vehicle = message["vehicles"][0]
        assert set(vehicle) == {
            "id", "node", "x", "y", "battery", "state", "station_id", "eta_seconds", "depot_id",
        }
    if message["stations"]:
        station = message["stations"][0]
        assert set(station) == {"id", "queue", "charging"}


def test_repeated_websocket_connect_disconnect_does_not_leak_threads():
    with TestClient(create_app(TEST_CONFIG)) as client:
        client.post(
            "/api/simulation/start",
            json={"seed": 3, "algorithm": "nearest_station", "speed": FAST_SPEED},
        )
        time.sleep(0.1)
        baseline_threads = threading.active_count()

        for _ in range(8):
            with client.websocket_connect("/ws/simulation") as websocket:
                websocket.receive_json()

        time.sleep(0.2)
        # WebSocket connections run as asyncio tasks on the existing event
        # loop, not one OS thread each; only the single SimulationManager
        # background thread should ever be running regardless of how many
        # clients connected and disconnected.
        assert threading.active_count() == baseline_threads


def test_rapid_pause_resume_reset_cycles_stay_consistent():
    # A larger vehicle count than TEST_CONFIG's makes it overwhelmingly
    # likely every seed in the loop has at least one EV needing a decision
    # (see test_reset_with_degenerate_seed_returns_400_without_corrupting_state
    # for the dedicated, deliberately-degenerate-seed case).
    config = replace(TEST_CONFIG, num_vehicles=20)
    with TestClient(create_app(config)) as client:
        client.post(
            "/api/simulation/start",
            json={"seed": 5, "algorithm": "nearest_station", "speed": FAST_SPEED},
        )

        for i in range(10):
            client.post("/api/simulation/pause")
            client.post("/api/simulation/resume")
            reset_body = client.post("/api/simulation/reset", params={"seed": 200 + i}).json()
            assert reset_body["status"] == "paused"
            assert reset_body["episode_seed"] == 200 + i
            assert reset_body["num_decisions"] == 0

        final_status = client.get("/api/simulation/status").json()
        assert final_status["episode_seed"] == 209
        assert final_status["status"] == "paused"


def test_reset_with_degenerate_seed_returns_400_without_corrupting_state(monkeypatch):
    # EVEnv.reset() deliberately raises RuntimeError for a scenario/seed
    # where no EV ever needs a charging decision (ai_core/ev_env.py). Rather
    # than depending on a specific seed happening to be degenerate against
    # the current mocked graph (fragile: that depends on graph scale/
    # station placement, not on this test's actual intent), seed=100 is
    # made degenerate directly. The manager must surface the RuntimeError as
    # a clean 400, not a 500, and must not corrupt its own state: a
    # subsequent valid request should work exactly as if the bad one had
    # never been sent.
    from backend.ai_core.ev_env import EVEnv

    original_reset = EVEnv.reset

    def flaky_reset(self, *, seed=None, options=None):
        if seed == 100:
            raise RuntimeError("simulated degenerate scenario for this test")
        return original_reset(self, seed=seed, options=options)

    monkeypatch.setattr(EVEnv, "reset", flaky_reset)

    with TestClient(create_app(TEST_CONFIG)) as client:
        good_status = client.post(
            "/api/simulation/start",
            json={"seed": 3, "algorithm": "nearest_station", "speed": FAST_SPEED},
        ).json()
        assert good_status["status"] == "running"

        bad_response = client.post("/api/simulation/reset", params={"seed": 100})
        assert bad_response.status_code == 400

        # The manager must still reflect the last *successful* episode (seed
        # 3), not a half-updated mix of old and new state.
        status_after_failed_reset = client.get("/api/simulation/status").json()
        assert status_after_failed_reset["episode_seed"] == 3

        recovered = client.post("/api/simulation/reset", params={"seed": 7}).json()
        assert recovered["status"] == "paused"
        assert recovered["episode_seed"] == 7


def test_status_reports_a_valid_my_vehicle_id():
    with TestClient(create_app(TEST_CONFIG)) as client:
        status = client.get("/api/simulation/status").json()

    assert status["my_vehicle_id"] is not None
    assert 0 <= status["my_vehicle_id"] < TEST_CONFIG.num_vehicles


def test_my_car_preview_returns_station_and_route():
    with TestClient(create_app(TEST_CONFIG)) as client:
        status = client.post(
            "/api/simulation/start",
            json={"seed": 5, "algorithm": "nearest_station", "speed": FAST_SPEED},
        ).json()
        my_vehicle_id = status["my_vehicle_id"]

        response = client.get("/api/simulation/my-car/preview")

    assert response.status_code == 200
    body = response.json()
    assert body["vehicle_id"] == my_vehicle_id
    assert isinstance(body["station_id"], int)
    assert isinstance(body["reachable"], bool)
    assert len(body["route"]) >= 1
    for point in body["route"]:
        assert len(point) == 2


def test_my_car_preview_does_not_dispatch_the_vehicle():
    # Calling preview repeatedly must never assign a station itself -- only
    # the automatic dispatch loop does that (section 20).
    with TestClient(create_app(TEST_CONFIG)) as client:
        client.post(
            "/api/simulation/start",
            json={"seed": 5, "algorithm": "nearest_station", "speed": FAST_SPEED},
        )

        client.get("/api/simulation/my-car/preview")
        client.get("/api/simulation/my-car/preview")

        manager = client.app.state.manager
        vehicle = manager.env.simulator.vehicles[manager.my_vehicle_id]
        assert vehicle.target_station is None
        assert vehicle.state.value == "TRAVELING"
