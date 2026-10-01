"""Evaluation pipeline (PROJECT_SPEC.md sections 28, 29, 46 Phase 6).

Runs DQN, MAPPO and all 3 baselines on the exact same scenarios -- same
network, same initial EV/battery/traffic state, same charging-station
configuration, same random seeds (section 29) -- and reports the section 28
metrics for each. Every number here comes from an actual run of
backend.ai_core.ev_env.EVEnv (baselines, DQN) or
backend.ai_core.marl_env.MultiAgentEVEnv (MAPPO); nothing is fabricated or
hard-coded (section 54).

Fairness for MAPPO: its decision window keeps EVs standing on the road,
already low on battery, before they are dispatched -- time EVEnv never
charges, since it dispatches on the very tick an EV runs low. Each EV's
decision_delay is therefore added to that EV's waiting time when its trip
resolves, so MAPPO's average waiting time (and system cost) pays for every
tick it held an EV in the window.

Run:
    python -m backend.ai_core.evaluate
    python -m backend.ai_core.evaluate --model-path models/dqn_osm_model.zip --seeds 1000 1001 1002
    python -m backend.ai_core.evaluate --mappo-model-path models/mappo_osm_model.pth
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN

from backend.ai_core.mappo import MAPPOAgent
from backend.ai_core.marl_env import MultiAgentEVEnv
from backend.baseline import least_queue, nearest_station, shortest_time
from backend.baseline.runner import (
    ActionFn,
    EpisodeMetrics,
    StationMetricsCollector,
    run_baseline_episode,
    run_episode,
)
from backend.config import OSM_DEMO_CONFIG, SimulationConfig

# A fixed, documented, held-out scenario set for fair comparison (sections
# 29/30). Training (ai_core/train.py's RandomScenarioPerEpisode) draws
# episode seeds uniformly from [0, 2**31), so the chance any of these 5
# specific seeds was ever seen during a ~50-episode training run is
# negligible -- these are effectively unseen test scenarios.
TEST_SEEDS = [1000, 1001, 1002, 1003, 1004]

RESULTS_PATH = Path("results") / "evaluation_comparison.json"
DEFAULT_MAPPO_MODEL_PATH = Path("models/mappo_osm_model.pth")
# Used only if the checkpoint does not record the window it was trained with.
MAPPO_DEFAULT_DECISION_WINDOW = 15


def dqn_action_fn(model: DQN) -> ActionFn:
    def _action_fn(env, obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return _action_fn


def run_mappo_episode(
    config: SimulationConfig, seed: int, agent: MAPPOAgent, decision_window: int
) -> EpisodeMetrics:
    """One MultiAgentEVEnv episode with the shared actor's argmax actions,
    measured exactly like run_episode, plus each EV's decision-window delay
    added to its own waiting time once its trip resolves."""
    collector = StationMetricsCollector()
    env = MultiAgentEVEnv(config, tick_hook=collector.record, decision_window=decision_window)
    obs, _ = env.reset(seed=seed)

    metrics = EpisodeMetrics(
        policy_name="mappo",
        seed=seed,
        num_decisions=0,
        num_valid_decisions=0,
        num_invalid_actions=0,
        num_failed_vehicles=0,
        num_overloaded_events=0,
        total_travel_time=0.0,
        total_waiting_time=0.0,
        episode_reward=0.0,
        terminated=False,
        truncated=False,
        simulation_time=0.0,
    )
    # Dispatched EV -> how long it stood in the window before dispatch.
    pending_delay: dict[int, float] = {}
    while env.agents:
        agents = list(env.agents)
        actions, _ = agent.get_action(np.stack([obs[a] for a in agents]), deterministic=True)
        obs, rewards, _, _, infos = env.step(dict(zip(agents, actions.tolist())))

        for a in agents:
            info = infos[a]
            metrics.num_decisions += 1
            metrics.episode_reward += rewards[a]
            if info["invalid_action"]:
                metrics.num_invalid_actions += 1
            else:
                metrics.num_valid_decisions += 1
            if info["stranded"]:
                metrics.num_stranded_vehicles += 1
                metrics.num_failed_vehicles += 1
            else:
                pending_delay[a] = info["decision_delay"]
            if info["station_overloaded"]:
                metrics.num_overloaded_events += 1
        for trip in env.last_resolved:
            delay = pending_delay.pop(trip["vehicle_id"])
            metrics.num_resolved_dispatches += 1
            metrics.total_travel_time += trip["travel_time"]
            metrics.total_waiting_time += trip["waiting_time"] + delay
            metrics.total_decision_delay += delay
            if trip["failed"]:
                metrics.num_failed_vehicles += 1

    metrics.terminated = env.simulator.all_vehicles_done()
    metrics.truncated = not metrics.terminated
    metrics.simulation_time = env.simulator.simulation_time
    metrics.average_queue_length = collector.average_queue_length
    metrics.maximum_queue_length = collector.maximum_queue_length
    metrics.station_utilization = collector.station_utilization
    return metrics


def load_mappo_model(config: SimulationConfig, model_path: Path) -> MAPPOAgent:
    if not model_path.exists():
        raise FileNotFoundError(
            f"MAPPO model not found at {model_path}. Train one first with "
            "`python -m backend.ai_core.train_mappo`."
        )
    agent = MAPPOAgent.load(model_path)
    env = MultiAgentEVEnv(config)
    expected = (env.observation_space(0).shape[0], env.state_space.shape[0], int(env.action_space(0).n))
    if (agent.obs_dim, agent.state_dim, agent.num_actions) != expected:
        raise ValueError(
            f"MAPPO model expects (obs_dim, state_dim, actions) = "
            f"{(agent.obs_dim, agent.state_dim, agent.num_actions)} but config needs {expected}; "
            "evaluation requires the same observation layout and charging-station configuration "
            "(section 29)."
        )
    return agent


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
        "average_decision_delay_mean": mean(
            [m.total_decision_delay / m.num_resolved_dispatches for m in metrics_list if m.num_resolved_dispatches]
        ),
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
    config: SimulationConfig, seeds: list[int], model_path: Path, mappo_model_path: Path | None = None
) -> dict[str, dict]:
    """Baselines and DQN, plus MAPPO when mappo_model_path is given."""
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

    if mappo_model_path is not None:
        agent = load_mappo_model(config, mappo_model_path)
        window = agent.metadata.get("decision_window", MAPPO_DEFAULT_DECISION_WINDOW)
        summaries["mappo"] = summarize([run_mappo_episode(config, seed, agent, window) for seed in seeds])

    return summaries


def print_comparison_table(summaries: dict[str, dict]) -> None:
    header = (
        f"{'policy':<16}{'avg_travel':>12}{'avg_wait':>10}{'(delay)':>9}{'sys_cost':>12}"
        f"{'avg_queue':>10}{'max_queue':>10}{'util':>8}{'failed':>8}"
        f"{'overload':>10}{'invalid':>9}{'reward_mean':>14}{'term/trunc':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        print(
            f"{name:<16}{s['average_travel_time_mean']:>12.2f}{s['average_waiting_time_mean']:>10.2f}"
            f"{s['average_decision_delay_mean']:>9.2f}"
            f"{s['total_system_cost_mean']:>12.2f}{s['average_queue_length_mean']:>10.2f}"
            f"{s['maximum_queue_length_overall']:>10.2f}{s['station_utilization_mean']:>8.2f}"
            f"{s['num_failed_vehicles_total']:>8d}{s['num_overloaded_events_total']:>10d}"
            f"{s['num_invalid_actions_total']:>9d}{s['episode_reward_mean']:>14.2f}"
            f"{s['terminated_count']:>6d}/{s['truncated_count']:<5d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DQN and MAPPO against baseline dispatch policies.")
    parser.add_argument("--model-path", type=str, default="models/dqn_osm_model.zip")
    parser.add_argument("--mappo-model-path", type=Path, default=DEFAULT_MAPPO_MODEL_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=TEST_SEEDS)
    args = parser.parse_args()

    mappo_model_path = args.mappo_model_path
    if not mappo_model_path.exists():
        print(
            f"WARNING: no MAPPO model at {mappo_model_path} -- MAPPO is left out of this comparison. "
            "Train one with `python -m backend.ai_core.train_mappo`.\n"
        )
        mappo_model_path = None

    summaries = evaluate_all_policies(OSM_DEMO_CONFIG, args.seeds, Path(args.model_path), mappo_model_path)

    print(f"Evaluated on {len(args.seeds)} held-out test seeds: {args.seeds}\n")
    print_comparison_table(summaries)
    print("\navg_wait includes (delay): time EVs stood waiting for a decision (MAPPO's decision window).")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({"seeds": args.seeds, "policies": summaries}, indent=2))
    print(f"\nsaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
