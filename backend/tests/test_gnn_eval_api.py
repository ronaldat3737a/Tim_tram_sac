from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from backend.ai_core import evaluate_gnn
from backend.ai_core.mappo_gnn import MAPPOGNNAgent
from backend.ai_core.train_mappo import build_agent
from backend.api.main import create_app
from backend.api.routes import gnn_eval
from backend.config import DEFAULT_CONFIG
from backend.tests.scenario_helpers import single_scenario

EVAL_CONFIG = replace(
    DEFAULT_CONFIG,
    num_stations=3,
    num_vehicles=12,
    max_episode_steps=3000,
    max_activation_tick=300,
    peak_hour_fraction=0.6,
    non_app_fraction=0.2,
)


@pytest.fixture
def client(monkeypatch, tmp_path):
    """The evaluation API on a small test scenario, with a tiny MLP trained
    for 3 stations and a tiny GNN."""
    monkeypatch.setattr(
        evaluate_gnn,
        "scenario_config",
        lambda num_stations=None, num_evs=None: replace(
            EVAL_CONFIG,
            num_stations=num_stations or EVAL_CONFIG.num_stations,
            num_vehicles=num_evs or EVAL_CONFIG.num_vehicles,
        ),
    )
    mlp_path, gnn_path = tmp_path / "mlp.pth", tmp_path / "gnn.pth"
    build_agent(EVAL_CONFIG, hidden_sizes=(8,), sequential_decisions=True).save(mlp_path, decision_window=5)
    MAPPOGNNAgent(hidden_size=16, heads=2).save(gnn_path, decision_window=5, scenario="large")
    monkeypatch.setattr(evaluate_gnn, "MLP_MODEL_PATH", mlp_path)
    monkeypatch.setattr(evaluate_gnn, "GNN_MODEL_PATH", gnn_path)
    gnn_eval._cache.clear()
    with TestClient(create_app(single_scenario(EVAL_CONFIG))) as test_client:
        yield test_client
    gnn_eval._cache.clear()


def test_compare_baseline_runs_all_four_on_one_seed(client):
    response = client.post("/api/gnn-eval/compare-baseline", json={"seed": 7, "num_stations": 3, "num_evs": 12})

    assert response.status_code == 200
    body = response.json()
    assert body["scenario"]["seed"] == 7 and body["scenario"]["num_stations"] == 3
    assert [r["algorithm"] for r in body["results"]] == ["nearest_station", "least_queue", "mappo_mlp", "mappo_gnn"]
    for result in body["results"]:
        assert result["status"] == "ok" and result["crashed"] is False
        metrics = result["metrics"]
        assert metrics["num_evs"] == 12 and metrics["num_decisions"] > 0
        assert metrics["average_decision_time_ms"] >= 0
    assert body["results"][3]["model"]["input_dim"] is None  # GNN input does not depend on N
    assert body["cached"] is False


def test_repeating_a_request_is_served_from_the_cache(client):
    payload = {"seed": 7, "num_stations": 3, "num_evs": 12}
    first = client.post("/api/gnn-eval/compare-baseline", json=payload).json()
    second = client.post("/api/gnn-eval/compare-baseline", json=payload).json()

    assert second["cached"] is True
    assert second["results"] == first["results"]


def test_zero_shot_scale_reports_the_mlp_dimension_mismatch_and_runs_the_gnn(client):
    response = client.post("/api/gnn-eval/zero-shot-scale", json={"num_stations": 6, "num_evs": 15, "seed": 7})

    assert response.status_code == 200
    body = response.json()
    mlp, gnn, reference = body["mlp"], body["gnn"], body["reference"]
    assert mlp["status"] == "crashed" and mlp["crashed"] is True
    assert mlp["error_type"] == "dimension_mismatch"
    assert mlp["error_message"].startswith("MLP Model Crashed: Dimension Mismatch (expected shape")
    assert mlp["metrics"] is None
    assert (mlp["model"]["input_dim"], mlp["model"]["scenario_input_dim"]) == (5 + 7 * 3, 5 + 7 * 6)
    assert gnn["status"] == "ok" and gnn["metrics"]["num_evs"] == 15
    assert reference["algorithm"] == "nearest_station" and reference["status"] == "ok"


def test_a_missing_gnn_model_is_reported_not_raised(client, monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_gnn, "GNN_MODEL_PATH", tmp_path / "missing.pth")

    body = client.post("/api/gnn-eval/compare-baseline", json={"seed": 7, "num_stations": 3, "num_evs": 12}).json()

    gnn = body["results"][3]
    assert gnn["status"] == "unavailable" and gnn["error_type"] == "model_unavailable"
    assert all(r["status"] == "ok" for r in body["results"][:3])


def test_a_second_evaluation_while_one_runs_gets_409(client):
    gnn_eval._evaluation_lock.acquire()
    try:
        response = client.post("/api/gnn-eval/zero-shot-scale", json={"num_stations": 6, "num_evs": 15})
        status = client.get("/api/gnn-eval/status").json()
    finally:
        gnn_eval._evaluation_lock.release()

    assert response.status_code == 409
    assert status["busy"] is True


def test_status_describes_both_models(client):
    body = client.get("/api/gnn-eval/status").json()
    assert body["busy"] is False
    assert body["mlp_model"]["input_dim"] == 5 + 7 * 3
    assert body["gnn_model"]["input_dim"] is None and body["gnn_model"]["parameters"] > 0


@pytest.mark.parametrize("payload", [{"num_stations": 1}, {"num_stations": 500}, {"num_evs": 0}])
def test_out_of_range_scales_are_rejected(client, payload):
    assert client.post("/api/gnn-eval/zero-shot-scale", json=payload).status_code == 422
