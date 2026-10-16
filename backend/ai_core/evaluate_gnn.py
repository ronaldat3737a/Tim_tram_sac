"""Head-to-head and zero-shot evaluation of MAPPO-GNN against MAPPO-MLP and
the heuristics, on one shared scenario (same seed: same stations, EVs,
batteries and traffic for every algorithm).

Every algorithm is measured exactly as backend/ai_core/evaluate.py measures it
(its run_episode / run_mappo_episode are reused unchanged; run_gnn_episode is
the graph-env counterpart of run_mappo_episode), plus the wall time of each
policy's own decision call per EV ("decision time": for MAPPO-MLP and
MAPPO-GNN it includes building the observation they decide from).

- compare_baseline(): Nearest Station, Least Queue, MAPPO-MLP and MAPPO-GNN on
  the scenario both learned models were trained on (large: 10 stations,
  200 EVs) by default.
- zero_shot_scale(): MAPPO-GNN and MAPPO-MLP, without retraining, on a
  scenario with a different number of stations/EVs. MAPPO-MLP's input layer
  is sized for its training scenario's station count, so it fails on the
  first decision; that failure is caught and reported, not raised.

Run (from the repo root):
    python -m backend.ai_core.evaluate_gnn
    python -m backend.ai_core.evaluate_gnn --zero-shot --num-stations 50 --num-evs 500
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from backend.ai_core.dispatch_core import observation_dim
from backend.ai_core.evaluate import MAPPO_DEFAULT_DECISION_WINDOW, run_mappo_episode
from backend.ai_core.graph_env import GraphMultiAgentEVEnv
from backend.ai_core.mappo import MAPPOAgent
from backend.ai_core.mappo_gnn import MAPPOGNNAgent
from backend.ai_core.train_gnn import DEFAULT_OUTPUT_PATH as GNN_MODEL_PATH
from backend.baseline import least_queue, nearest_station
from backend.baseline.runner import EpisodeMetrics, StationMetricsCollector, run_episode
from backend.config import SCENARIOS, SimulationConfig

BASE_SCENARIO = "large"
MLP_MODEL_PATH = Path(SCENARIOS[BASE_SCENARIO].mappo_model_path)
DEFAULT_SEED = 1000

HEURISTICS: dict[str, tuple[str, Callable]] = {
    "nearest_station": ("Nearest Station", nearest_station.choose_station),
    "least_queue": ("Least Queue", least_queue.choose_station),
}
LABELS = {
    **{name: label for name, (label, _) in HEURISTICS.items()},
    "mappo_mlp": "MAPPO-MLP",
    "mappo_gnn": "MAPPO-GNN",
}


def scenario_config(num_stations: int | None = None, num_evs: int | None = None) -> SimulationConfig:
    """The base (large) scenario's map and physics, with the station/EV
    counts overridden."""
    base = SCENARIOS[BASE_SCENARIO].config
    return replace(
        base,
        num_stations=num_stations or base.num_stations,
        num_vehicles=num_evs or base.num_vehicles,
    )


def scenario_info(config: SimulationConfig, seed: int) -> dict[str, Any]:
    return {
        "base_scenario": BASE_SCENARIO,
        "num_stations": config.num_stations,
        "num_evs": config.num_vehicles,
        "num_depots": config.num_depots,
        "radius_meters": config.osm_radius_meters,
        "seed": seed,
    }


# --- per-algorithm runs ----------------------------------------------------


class _DecisionTimer:
    def __init__(self):
        self.seconds = 0.0
        self.decisions = 0

    def add(self, seconds: float, decisions: int) -> None:
        self.seconds += seconds
        self.decisions += decisions

    @property
    def average_ms(self) -> float:
        return 1000 * self.seconds / self.decisions if self.decisions else 0.0


class _TimedMLPAgent:
    """Times MAPPOAgent.act, so run_mappo_episode can be reused unchanged."""

    def __init__(self, agent: MAPPOAgent, timer: _DecisionTimer):
        self._agent = agent
        self._timer = timer

    def act(self, env, observations, deterministic: bool = False):
        started = time.perf_counter()
        result = self._agent.act(env, observations, deterministic=deterministic)
        self._timer.add(time.perf_counter() - started, len(result[0]))
        return result


def run_gnn_episode(
    config: SimulationConfig, seed: int, agent: MAPPOGNNAgent, decision_window: int, timer: _DecisionTimer | None = None
) -> EpisodeMetrics:
    """GraphMultiAgentEVEnv counterpart of evaluate.run_mappo_episode,
    measured identically (each EV's decision-window delay is added to its
    waiting time once its trip resolves)."""
    collector = StationMetricsCollector()
    env = GraphMultiAgentEVEnv(config, tick_hook=collector.record, decision_window=decision_window)
    env.reset(seed=seed)
    metrics = EpisodeMetrics(
        policy_name="mappo_gnn",
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
    pending_delay: dict[int, float] = {}
    while env.agents:
        started = time.perf_counter()
        agents, _, actions, _ = agent.act(env, deterministic=True)
        if timer is not None:
            timer.add(time.perf_counter() - started, len(agents))
        _, rewards, _, _, infos = env.step(dict(zip(agents, actions.tolist())))

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


def _metrics_dict(metrics: EpisodeMetrics, config: SimulationConfig, timer: _DecisionTimer) -> dict[str, Any]:
    failed_after_dispatch = metrics.num_failed_vehicles - metrics.num_stranded_vehicles
    return {
        "num_evs": config.num_vehicles,
        "num_decisions": metrics.num_decisions,
        # Dispatched EVs that reached their station and started charging.
        "num_charged": metrics.num_resolved_dispatches - failed_after_dispatch,
        "num_failed": metrics.num_failed_vehicles,
        "num_overloaded": metrics.num_overloaded_events,
        "total_waiting_time": metrics.total_waiting_time,
        "average_waiting_time": metrics.average_waiting_time,
        "total_travel_time": metrics.total_travel_time,
        "average_travel_time": metrics.average_travel_time,
        "total_system_cost": metrics.total_system_cost,
        "average_queue_length": metrics.average_queue_length,
        "max_queue_length": metrics.maximum_queue_length,
        "average_decision_time_ms": timer.average_ms,
        "simulation_time": metrics.simulation_time,
        "completed": metrics.terminated,
    }


def _result(algorithm: str, **fields) -> dict[str, Any]:
    result = {
        "algorithm": algorithm,
        "label": LABELS[algorithm],
        "status": "ok",
        "crashed": False,
        "error_type": None,
        "error_message": None,
        "metrics": None,
        "model": None,
        "wall_time_seconds": 0.0,
    }
    result.update(fields)
    return result


def evaluate_heuristic(name: str, config: SimulationConfig, seed: int) -> dict[str, Any]:
    _, policy = HEURISTICS[name]
    timer = _DecisionTimer()

    # Same action as baseline.runner.run_baseline_episode (station ids are
    # the action indices), only timed.
    def action_fn(env, obs, info):
        started = time.perf_counter()
        station_id = policy(env.simulator, info["next_vehicle_id"])
        timer.add(time.perf_counter() - started, 1)
        return station_id

    started = time.perf_counter()
    metrics = run_episode(config, seed, action_fn, name)
    return _result(
        name, metrics=_metrics_dict(metrics, config, timer), wall_time_seconds=time.perf_counter() - started
    )


def _model_info(path: Path, agent, input_dim: int | None, num_actions: int | None) -> dict[str, Any]:
    trained_on = agent.metadata.get("scenario", BASE_SCENARIO)
    trained_config = SCENARIOS[trained_on].config if trained_on in SCENARIOS else None
    return {
        "path": str(path),
        "parameters": sum(p.numel() for p in [*agent.actor.parameters(), *agent.critic.parameters()]),
        # None: the input size does not depend on the scenario (GNN).
        "input_dim": input_dim,
        "num_actions": num_actions,
        "trained_on_stations": trained_config.num_stations if trained_config else None,
        "trained_on_evs": trained_config.num_vehicles if trained_config else None,
        "trained_decisions": agent.metadata.get("decisions"),
        "validation_reward_per_decision": agent.metadata.get("validation_reward_per_decision"),
    }


def _unavailable(algorithm: str, path: Path, exc: Exception) -> dict[str, Any]:
    return _result(
        algorithm,
        status="unavailable",
        error_type="model_unavailable",
        error_message=f"{LABELS[algorithm]} model could not be loaded from {path}: {exc}",
    )


def evaluate_mlp(config: SimulationConfig, seed: int, model_path: Path | None = None) -> dict[str, Any]:
    """MAPPO-MLP on `config`. A model whose input layer does not fit the
    scenario fails on its first decision; that is caught and reported as
    crashed with error_type "dimension_mismatch"."""
    model_path = model_path or MLP_MODEL_PATH
    try:
        agent = MAPPOAgent.load(model_path)
    except Exception as exc:  # missing or unreadable file
        return _unavailable("mappo_mlp", model_path, exc)
    model = _model_info(model_path, agent, agent.obs_dim, agent.num_actions)
    model["scenario_input_dim"] = observation_dim(config)
    timer = _DecisionTimer()
    window = agent.metadata.get("decision_window", MAPPO_DEFAULT_DECISION_WINDOW)
    started = time.perf_counter()
    try:
        metrics = run_mappo_episode(config, seed, _TimedMLPAgent(agent, timer), window)
    except Exception as exc:
        mismatch = "expected shape" in str(exc)
        return _result(
            "mappo_mlp",
            status="crashed",
            crashed=True,
            error_type="dimension_mismatch" if mismatch else type(exc).__name__,
            error_message=(
                f"MLP Model Crashed: Dimension Mismatch ({exc})" if mismatch else f"MLP Model Crashed: {exc}"
            ),
            model=model,
            wall_time_seconds=time.perf_counter() - started,
        )
    return _result(
        "mappo_mlp",
        metrics=_metrics_dict(metrics, config, timer),
        model=model,
        wall_time_seconds=time.perf_counter() - started,
    )


def evaluate_gnn(config: SimulationConfig, seed: int, model_path: Path | None = None) -> dict[str, Any]:
    model_path = model_path or GNN_MODEL_PATH
    try:
        agent = MAPPOGNNAgent.load(model_path)
    except Exception as exc:
        return _unavailable("mappo_gnn", model_path, exc)
    agent.actor.eval()
    agent.critic.eval()
    model = _model_info(model_path, agent, None, None)
    timer = _DecisionTimer()
    window = agent.metadata.get("decision_window", MAPPO_DEFAULT_DECISION_WINDOW)
    started = time.perf_counter()
    try:
        metrics = run_gnn_episode(config, seed, agent, window, timer)
    except Exception as exc:
        return _result(
            "mappo_gnn",
            status="crashed",
            crashed=True,
            error_type=type(exc).__name__,
            error_message=f"GNN Model Crashed: {exc}",
            model=model,
            wall_time_seconds=time.perf_counter() - started,
        )
    return _result(
        "mappo_gnn",
        metrics=_metrics_dict(metrics, config, timer),
        model=model,
        wall_time_seconds=time.perf_counter() - started,
    )


def describe_models(
    mlp_model_path: Path | None = None, gnn_model_path: Path | None = None
) -> dict[str, dict[str, Any] | None]:
    """Model info of both learned models (None if a file cannot be loaded)."""
    mlp_model_path = mlp_model_path or MLP_MODEL_PATH
    gnn_model_path = gnn_model_path or GNN_MODEL_PATH
    described: dict[str, dict[str, Any] | None] = {"mlp_model": None, "gnn_model": None}
    try:
        mlp = MAPPOAgent.load(mlp_model_path)
        described["mlp_model"] = _model_info(mlp_model_path, mlp, mlp.obs_dim, mlp.num_actions)
    except Exception:
        pass
    try:
        described["gnn_model"] = _model_info(gnn_model_path, MAPPOGNNAgent.load(gnn_model_path), None, None)
    except Exception:
        pass
    return described


# --- the two experiments -----------------------------------------------------


def compare_baseline(
    seed: int = DEFAULT_SEED,
    num_stations: int | None = None,
    num_evs: int | None = None,
    mlp_model_path: Path | None = None,
    gnn_model_path: Path | None = None,
) -> dict[str, Any]:
    config = scenario_config(num_stations, num_evs)
    return {
        "scenario": scenario_info(config, seed),
        "results": [
            evaluate_heuristic("nearest_station", config, seed),
            evaluate_heuristic("least_queue", config, seed),
            evaluate_mlp(config, seed, mlp_model_path),
            evaluate_gnn(config, seed, gnn_model_path),
        ],
    }


def zero_shot_scale(
    num_stations: int,
    num_evs: int,
    seed: int = DEFAULT_SEED,
    include_baseline: bool = True,
    mlp_model_path: Path | None = None,
    gnn_model_path: Path | None = None,
) -> dict[str, Any]:
    config = scenario_config(num_stations, num_evs)
    base = SCENARIOS[BASE_SCENARIO].config
    return {
        "scenario": scenario_info(config, seed),
        "trained_on": {"num_stations": base.num_stations, "num_evs": base.num_vehicles},
        "gnn": evaluate_gnn(config, seed, gnn_model_path),
        "mlp": evaluate_mlp(config, seed, mlp_model_path),
        # Nearest Station on the same scaled scenario, as a yardstick for
        # the GNN's numbers.
        "reference": evaluate_heuristic("nearest_station", config, seed) if include_baseline else None,
    }


def _print_results(results: list[dict[str, Any]]) -> None:
    print(f"{'algorithm':<18}{'status':<10}{'avg_wait':>10}{'avg_travel':>12}{'charged':>9}{'failed':>8}"
          f"{'overload':>10}{'ms/decision':>13}")
    for r in results:
        m = r["metrics"]
        if m is None:
            print(f"{r['label']:<18}{r['status']:<10}  {r['error_message']}")
            continue
        print(f"{r['label']:<18}{r['status']:<10}{m['average_waiting_time']:>10.1f}{m['average_travel_time']:>12.1f}"
              f"{m['num_charged']:>9d}{m['num_failed']:>8d}{m['num_overloaded']:>10d}{m['average_decision_time_ms']:>13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MAPPO-GNN with MAPPO-MLP and heuristics.")
    parser.add_argument("--zero-shot", action="store_true", help="Run the zero-shot scale test instead.")
    parser.add_argument("--num-stations", type=int, default=None)
    parser.add_argument("--num-evs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json", type=Path, default=None, help="Also write the full result here.")
    args = parser.parse_args()

    if args.zero_shot:
        report = zero_shot_scale(args.num_stations or 50, args.num_evs or 500, args.seed)
        rows = [report["mlp"], report["gnn"]] + ([report["reference"]] if report["reference"] else [])
    else:
        report = compare_baseline(args.seed, args.num_stations, args.num_evs)
        rows = report["results"]
    print(f"scenario: {report['scenario']}")
    _print_results(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
