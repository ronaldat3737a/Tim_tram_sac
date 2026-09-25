from dataclasses import replace
from pathlib import Path

import pytest

from backend.ai_core.evaluate import evaluate_all_policies, summarize
from backend.baseline.runner import EpisodeMetrics
from backend.config import DEFAULT_CONFIG

TRAINED_MODEL_PATH = Path("models/dqn_osm_model.zip")


def _make_metrics(**overrides) -> EpisodeMetrics:
    defaults = dict(
        policy_name="test_policy",
        seed=0,
        num_decisions=10,
        num_valid_decisions=8,
        num_invalid_actions=2,
        num_failed_vehicles=0,
        num_overloaded_events=1,
        total_travel_time=80.0,
        total_waiting_time=16.0,
        episode_reward=-146.0,
        terminated=True,
        truncated=False,
        simulation_time=500.0,
        average_queue_length=1.5,
        maximum_queue_length=4.0,
        station_utilization=0.6,
    )
    defaults.update(overrides)
    return EpisodeMetrics(**defaults)


def test_summarize_computes_mean_and_stdev_across_seeds():
    metrics_list = [
        _make_metrics(seed=1, total_travel_time=80.0, total_waiting_time=16.0, episode_reward=-96.0),
        _make_metrics(seed=2, total_travel_time=120.0, total_waiting_time=24.0, episode_reward=-144.0),
    ]

    summary = summarize(metrics_list)

    assert summary["policy_name"] == "test_policy"
    assert summary["num_test_seeds"] == 2
    expected_avg_travel = [m.average_travel_time for m in metrics_list]
    assert summary["average_travel_time_mean"] == pytest.approx(sum(expected_avg_travel) / 2)
    assert summary["episode_reward_mean"] == pytest.approx((-96.0 + -144.0) / 2)
    assert summary["num_overloaded_events_total"] == 2
    assert summary["terminated_count"] == 2
    assert summary["truncated_count"] == 0
    assert len(summary["per_seed"]) == 2


def test_summarize_single_seed_has_zero_stdev():
    summary = summarize([_make_metrics(seed=1)])

    assert summary["average_travel_time_stdev"] == 0.0
    assert summary["episode_reward_stdev"] == 0.0


def test_evaluate_all_policies_raises_when_model_missing():
    with pytest.raises(FileNotFoundError):
        evaluate_all_policies(DEFAULT_CONFIG, [1000], Path("models/does_not_exist.zip"))


def test_evaluate_all_policies_raises_on_station_count_mismatch():
    if not TRAINED_MODEL_PATH.exists():
        pytest.skip("no trained model artifact present (run backend.ai_core.train first)")

    mismatched_config = replace(DEFAULT_CONFIG, num_stations=DEFAULT_CONFIG.num_stations + 1)

    with pytest.raises(ValueError):
        evaluate_all_policies(mismatched_config, [1000], TRAINED_MODEL_PATH)


def test_evaluate_all_policies_runs_end_to_end_with_trained_model():
    if not TRAINED_MODEL_PATH.exists():
        pytest.skip("no trained model artifact present (run backend.ai_core.train first)")

    light_config = replace(DEFAULT_CONFIG, num_vehicles=6, max_episode_steps=1500)

    summaries = evaluate_all_policies(light_config, [1000, 1001], TRAINED_MODEL_PATH)

    assert set(summaries) == {"nearest_station", "shortest_time", "least_queue", "dqn"}
    for name, summary in summaries.items():
        assert summary["policy_name"] == name
        assert summary["num_test_seeds"] == 2
        assert summary["terminated_count"] + summary["truncated_count"] == 2
