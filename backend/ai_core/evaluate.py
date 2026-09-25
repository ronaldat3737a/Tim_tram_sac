"""Evaluation pipeline (PROJECT_SPEC.md sections 28, 29, 46 Phase 6).

Runs DQN and all 3 baselines on the exact same scenarios -- same network,
same initial EV/battery/traffic state, same charging-station configuration,
same random seeds (section 29) -- and reports the section 28 metrics for
each. Every number here comes from an actual run of
backend.ai_core.ev_env.EVEnv; nothing is fabricated or hard-coded (section 54).

Run:
    python -m backend.ai_core.evaluate
    python -m backend.ai_core.evaluate --model-path models/dqn_osm_model.zip --seeds 1000 1001 1002
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import DQN

from backend.baseline import least_queue, nearest_station, shortest_time
from backend.baseline.runner import ActionFn, EpisodeMetrics, run_baseline_episode, run_episode
from backend.config import OSM_DEMO_CONFIG, SimulationConfig

# A fixed, documented, held-out scenario set for fair comparison (sections
# 29/30). Training (ai_core/train.py's RandomScenarioPerEpisode) draws
# episode seeds uniformly from [0, 2**31), so the chance any of these 5
# specific seeds was ever seen during a ~50-episode training run is
# negligible -- these are effectively unseen test scenarios.
TEST_SEEDS = [1000, 1001, 1002, 1003, 1004]

RESULTS_PATH = Path("results") / "evaluation_comparison.json"


def dqn_action_fn(model: DQN) -> ActionFn:
    def _action_fn(env, obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return _action_fn


def summarize(metrics_list: list[EpisodeMetrics]) -> dict:
    def mean(values: list[float]) -> float:
        return statistics.mean(values) if values else 0.0

    def stdev(values: list[float]) -> float:
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    return {
        "policy_name": metrics_list[0].policy_name,
        "num_test_seeds": len(metrics_list),
        "average_travel_time_mean": mean([m.average_travel_time for m in metrics_list]),
        "average_travel_time_stdev": stdev([m.average_travel_time for m in metrics_list]),
        "average_waiting_time_mean": mean([m.average_waiting_time for m in metrics_list]),
        "average_waiting_time_stdev": stdev([m.average_waiting_time for m in metrics_list]),
        "total_travel_time_mean": mean([m.total_travel_time for m in metrics_list]),
        "total_waiting_time_mean": mean([m.total_waiting_time for m in metrics_list]),
        "total_system_cost_mean": mean([m.total_system_cost for m in metrics_list]),
        "total_system_cost_stdev": stdev([m.total_system_cost for m in metrics_list]),
        "average_queue_length_mean": mean([m.average_queue_length for m in metrics_list]),
        "maximum_queue_length_overall": max((m.maximum_queue_length for m in metrics_list), default=0.0),
        "station_utilization_mean": mean([m.station_utilization for m in metrics_list]),
        "num_failed_vehicles_total": sum(m.num_failed_vehicles for m in metrics_list),
        "num_overloaded_events_total": sum(m.num_overloaded_events for m in metrics_list),
        "num_invalid_actions_total": sum(m.num_invalid_actions for m in metrics_list),
        "num_stranded_vehicles_total": sum(m.num_stranded_vehicles for m in metrics_list),
        "episode_reward_mean": mean([m.episode_reward for m in metrics_list]),
        "episode_reward_stdev": stdev([m.episode_reward for m in metrics_list]),
        "terminated_count": sum(1 for m in metrics_list if m.terminated),
        "truncated_count": sum(1 for m in metrics_list if m.truncated),
        "per_seed": [asdict(m) for m in metrics_list],
    }


def evaluate_all_policies(
    config: SimulationConfig, seeds: list[int], model_path: Path
) -> dict[str, dict]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"DQN model not found at {model_path}. Train one first with "
            "`python -m backend.ai_core.train`."
        )
    model = DQN.load(str(model_path))
    if model.action_space.n != config.num_stations:
        raise ValueError(
            f"model was trained with {model.action_space.n} stations but config "
            f"has num_stations={config.num_stations}; evaluation requires the "
            "same charging-station configuration (section 29)."
        )

    summaries: dict[str, dict] = {}

    baseline_policies = {
        "nearest_station": nearest_station.choose_station,
        "shortest_time": shortest_time.choose_station,
        "least_queue": least_queue.choose_station,
    }
    for name, policy in baseline_policies.items():
        metrics_list = [run_baseline_episode(config, seed, policy, name) for seed in seeds]
        summaries[name] = summarize(metrics_list)

    dqn_metrics_list = [run_episode(config, seed, dqn_action_fn(model), "dqn") for seed in seeds]
    summaries["dqn"] = summarize(dqn_metrics_list)

    return summaries


def print_comparison_table(summaries: dict[str, dict]) -> None:
    header = (
        f"{'policy':<16}{'avg_travel':>12}{'avg_wait':>10}{'sys_cost':>12}"
        f"{'avg_queue':>10}{'max_queue':>10}{'util':>8}{'failed':>8}"
        f"{'overload':>10}{'invalid':>9}{'reward_mean':>14}{'term/trunc':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        print(
            f"{name:<16}{s['average_travel_time_mean']:>12.2f}{s['average_waiting_time_mean']:>10.2f}"
            f"{s['total_system_cost_mean']:>12.2f}{s['average_queue_length_mean']:>10.2f}"
            f"{s['maximum_queue_length_overall']:>10.2f}{s['station_utilization_mean']:>8.2f}"
            f"{s['num_failed_vehicles_total']:>8d}{s['num_overloaded_events_total']:>10d}"
            f"{s['num_invalid_actions_total']:>9d}{s['episode_reward_mean']:>14.2f}"
            f"{s['terminated_count']:>6d}/{s['truncated_count']:<5d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DQN against baseline dispatch policies.")
    parser.add_argument("--model-path", type=str, default="models/dqn_osm_model.zip")
    parser.add_argument("--seeds", type=int, nargs="+", default=TEST_SEEDS)
    args = parser.parse_args()

    summaries = evaluate_all_policies(OSM_DEMO_CONFIG, args.seeds, Path(args.model_path))

    print(f"Evaluated on {len(args.seeds)} held-out test seeds: {args.seeds}\n")
    print_comparison_table(summaries)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"seeds": args.seeds, "policies": summaries}, indent=2))
    print(f"\nsaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
