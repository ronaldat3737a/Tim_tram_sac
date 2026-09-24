"""Evaluation runner: drive EVEnv with a policy and collect PROJECT_SPEC.md
section 28 metrics.

Baselines and RL share the exact same EVEnv mechanics (reward, invalid-action
handling, overload-at-decision-time check, termination/truncation) so results
stay comparable per section 29. `run_episode` is the shared core used both by
the basic baseline-only smoke test in this module (Phase 4, section 65) and
by the full DQN-vs-baselines comparison in ai_core/evaluate.py (Phase 6).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from backend.ai_core.ev_env import EVEnv
from backend.config import SimulationConfig
from backend.simulation.simulator import Simulator

BaselinePolicy = Callable[[Simulator, int], int]
# General action-selection hook: given the env, the current observation and
# info dict, return the chosen station id. Lets a baseline (which only needs
# `simulator` + `vehicle_id`, both reachable via env/info) and a trained SB3
# model (which needs `obs`) share the same run_episode loop.
ActionFn = Callable[[EVEnv, np.ndarray, dict], int]


class StationMetricsCollector:
    """Samples every station's queue length and charger occupancy on every
    simulated second (via EVEnv's tick_hook), for time-weighted Average/
    Maximum Queue Length and Station Utilization (section 28)."""

    def __init__(self) -> None:
        self.queue_length_samples: list[int] = []
        self.utilization_samples: list[float] = []

    def record(self, simulator: Simulator) -> None:
        for station in simulator.stations.values():
            self.queue_length_samples.append(len(station.queue))
            if station.num_chargers > 0:
                self.utilization_samples.append(len(station.charging_vehicle_ids) / station.num_chargers)

    @property
    def average_queue_length(self) -> float:
        return statistics.mean(self.queue_length_samples) if self.queue_length_samples else 0.0

    @property
    def maximum_queue_length(self) -> float:
        return max(self.queue_length_samples) if self.queue_length_samples else 0.0

    @property
    def station_utilization(self) -> float:
        return statistics.mean(self.utilization_samples) if self.utilization_samples else 0.0


@dataclass
class EpisodeMetrics:
    policy_name: str
    seed: int
    num_decisions: int
    num_valid_decisions: int
    num_invalid_actions: int
    num_failed_vehicles: int
    num_overloaded_events: int
    total_travel_time: float
    total_waiting_time: float
    episode_reward: float
    terminated: bool
    truncated: bool
    simulation_time: float
    average_queue_length: float = 0.0
    maximum_queue_length: float = 0.0
    station_utilization: float = 0.0

    @property
    def average_travel_time(self) -> float:
        return self.total_travel_time / self.num_valid_decisions if self.num_valid_decisions else 0.0

    @property
    def average_waiting_time(self) -> float:
        return self.total_waiting_time / self.num_valid_decisions if self.num_valid_decisions else 0.0

    @property
    def total_system_cost(self) -> float:
        return self.total_travel_time + self.total_waiting_time


def run_episode(
    config: SimulationConfig, seed: int, action_fn: ActionFn, policy_name: str
) -> EpisodeMetrics:
    collector = StationMetricsCollector()
    env = EVEnv(config, tick_hook=collector.record)
    obs, info = env.reset(seed=seed)

    num_decisions = 0
    num_valid_decisions = 0
    num_invalid_actions = 0
    num_failed_vehicles = 0
    num_overloaded_events = 0
    total_travel_time = 0.0
    total_waiting_time = 0.0
    episode_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action = action_fn(env, obs, info)
        obs, reward, terminated, truncated, info = env.step(action)

        num_decisions += 1
        episode_reward += reward
        if info["invalid_action"]:
            num_invalid_actions += 1
        else:
            num_valid_decisions += 1
            total_travel_time += info["travel_time"]
            total_waiting_time += info["waiting_time"]
            if info["station_overloaded"]:
                num_overloaded_events += 1
            if info["vehicle_failed"]:
                num_failed_vehicles += 1

    return EpisodeMetrics(
        policy_name=policy_name,
        seed=seed,
        num_decisions=num_decisions,
        num_valid_decisions=num_valid_decisions,
        num_invalid_actions=num_invalid_actions,
        num_failed_vehicles=num_failed_vehicles,
        num_overloaded_events=num_overloaded_events,
        total_travel_time=total_travel_time,
        total_waiting_time=total_waiting_time,
        episode_reward=episode_reward,
        terminated=terminated,
        truncated=truncated,
        simulation_time=env.simulator.simulation_time,
        average_queue_length=collector.average_queue_length,
        maximum_queue_length=collector.maximum_queue_length,
        station_utilization=collector.station_utilization,
    )


def run_baseline_episode(
    config: SimulationConfig, seed: int, policy: BaselinePolicy, policy_name: str
) -> EpisodeMetrics:
    def action_fn(env: EVEnv, obs: np.ndarray, info: dict) -> int:
        return policy(env.simulator, info["next_vehicle_id"])

    return run_episode(config, seed, action_fn, policy_name)


if __name__ == "__main__":
    from backend.baseline import least_queue, nearest_station, shortest_time
    from backend.config import DEFAULT_CONFIG

    policies = {
        "nearest_station": nearest_station.choose_station,
        "shortest_time": shortest_time.choose_station,
        "least_queue": least_queue.choose_station,
    }

    results = []
    for name, policy in policies.items():
        metrics = run_baseline_episode(DEFAULT_CONFIG, seed=DEFAULT_CONFIG.random_seed, policy=policy, policy_name=name)
        results.append(asdict(metrics))
        print(
            f"{name}: decisions={metrics.num_decisions} "
            f"valid={metrics.num_valid_decisions} invalid={metrics.num_invalid_actions} "
            f"avg_travel_time={metrics.average_travel_time:.2f} "
            f"avg_waiting_time={metrics.average_waiting_time:.2f} "
            f"total_system_cost={metrics.total_system_cost:.2f} "
            f"failed={metrics.num_failed_vehicles} overloaded={metrics.num_overloaded_events} "
            f"episode_reward={metrics.episode_reward:.2f} "
            f"terminated={metrics.terminated} truncated={metrics.truncated}"
        )

    output_path = Path("results") / "baseline_smoke_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved metrics to {output_path}")
