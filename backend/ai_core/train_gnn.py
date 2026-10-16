"""MAPPO-GNN training entry point.

Trains MAPPOGNNAgent (backend/ai_core/mappo_gnn.py) on GraphMultiAgentEVEnv
and saves the best model (by validation reward) to models/mappo_gnn_model.pth.
MAPPO-MLP's own trainer (train_mappo.py) is untouched; this one reuses its
PPO pieces as is -- PPOConfig, GAE along the chain of decision batches
(compute_gae), the clipped PPO update (ppo_update), the episode statistics
and the validation seeds -- so both models are trained by the very same
algorithm and differ only in their networks and observations.

Run (from the repo root):
    python backend/ai_core/train_gnn.py --smoke-test
    python backend/ai_core/train_gnn.py                        # large scenario, 1M decisions
    python backend/ai_core/train_gnn.py --scenario small --total-decisions 300000
    tensorboard --logdir ./logs/

Rollouts are played by n_workers CPU processes; the PPO update runs on
--device (CUDA when available). Workers return plain numpy arrays, never
tensors, so no shared-memory handles pile up between processes.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python backend/ai_core/train_gnn.py` as well as `python -m ...`.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from backend.ai_core.graph_env import DEFAULT_NEIGHBORS, GraphMultiAgentEVEnv
from backend.ai_core.mappo_gnn import DEFAULT_HEADS, DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, MAPPOGNNAgent
from backend.ai_core.train_mappo import (
    DECISION_WINDOW,
    MODELS_DIR,
    N_WORKERS,
    TENSORBOARD_LOG_DIR,
    VALIDATION_SEEDS,
    EpisodeStats,
    PPOConfig,
    compute_gae,
    ppo_update,
    summarize,
)
from backend.config import SCENARIOS, SimulationConfig, get_scenario

DEFAULT_SCENARIO_NAME = "large"
DEFAULT_OUTPUT_PATH = MODELS_DIR / "mappo_gnn_model.pth"


# --- rollout data --------------------------------------------------------


@dataclass
class GraphDecisionBatch:
    """One env.step(): n agents acting on one global state. Node features
    only; the episode's station edges are stored once on the Episode."""

    global_x: np.ndarray  # (N, GLOBAL_NODE_DIM)
    global_u: np.ndarray  # (GLOBAL_DIM,)
    local_x: np.ndarray  # (n, N, NODE_DIM)
    local_u: np.ndarray  # (n, EGO_DIM)
    actions: np.ndarray  # (n,)
    log_probs: np.ndarray  # (n,)
    rewards: np.ndarray  # (n,)


@dataclass
class GraphEpisode:
    seed: int
    edge_index: np.ndarray  # (2, E)
    edge_attr: np.ndarray  # (E, EDGE_DIM)
    batches: list[GraphDecisionBatch]
    # True if every EV finished; False if cut by max_episode_steps (GAE then
    # bootstraps from the final global state).
    terminated: bool
    final_global_x: np.ndarray
    final_global_u: np.ndarray
    stats: EpisodeStats = field(default_factory=EpisodeStats)


def run_episode(env: GraphMultiAgentEVEnv, agent: MAPPOGNNAgent, seed: int, deterministic: bool) -> GraphEpisode:
    env.reset(seed=seed)
    batches: list[GraphDecisionBatch] = []
    stats = EpisodeStats()
    first = env.get_global_graph()
    while env.agents:
        global_graph = env.get_global_graph()
        agents, graphs, actions, log_probs = agent.act(env, deterministic=deterministic)
        _, rewards, _, _, infos = env.step(dict(zip(agents, actions.tolist())))

        reward_batch = np.array([rewards[a] for a in agents], dtype=np.float32)
        batches.append(
            GraphDecisionBatch(
                global_graph.x.numpy(),
                global_graph.u[0].numpy(),
                np.stack([g.x.numpy() for g in graphs]),
                np.stack([g.u[0].numpy() for g in graphs]),
                actions,
                log_probs,
                reward_batch,
            )
        )
        stats.num_decisions += len(agents)
        stats.total_reward += float(reward_batch.sum())
        for a in agents:
            stats.num_invalid += infos[a]["invalid_action"]
            stats.num_overloaded += infos[a]["station_overloaded"]
            stats.num_stranded += infos[a]["stranded"]
            stats.total_decision_delay += infos[a]["decision_delay"]
        for trip in env.last_resolved:
            stats.travel_times.append(trip["travel_time"])
            stats.waiting_times.append(trip["waiting_time"])
    final = env.get_global_graph()
    return GraphEpisode(
        seed,
        first.edge_index.numpy(),
        first.edge_attr.numpy(),
        batches,
        env.simulator.all_vehicles_done(),
        final.x.numpy(),
        final.u[0].numpy(),
        stats,
    )


# --- parallel rollout workers -------------------------------------------

_worker_env: GraphMultiAgentEVEnv | None = None


def _worker_init(config: SimulationConfig, decision_window: int, num_neighbors: int) -> None:
    global _worker_env
    torch.set_num_threads(1)
    _worker_env = GraphMultiAgentEVEnv(config, decision_window=decision_window, num_neighbors=num_neighbors)


def _worker_run(job: tuple[dict, tuple[int, int, int], int, bool]) -> GraphEpisode:
    actor_state, (hidden_size, num_layers, heads), seed, deterministic = job
    agent = MAPPOGNNAgent(hidden_size, num_layers, heads)
    agent.actor.load_state_dict(actor_state)
    return run_episode(_worker_env, agent, seed, deterministic)


class RolloutRunner:
    """Plays episodes with the current actor on CPU, in n_workers processes
    (or inline when n_workers == 1)."""

    def __init__(self, config: SimulationConfig, decision_window: int, num_neighbors: int, n_workers: int):
        self._pool = None
        self._env = None
        if n_workers > 1:
            # spawn: never fork a process that already holds torch threads.
            self._pool = mp.get_context("spawn").Pool(
                n_workers, initializer=_worker_init, initargs=(config, decision_window, num_neighbors)
            )
        else:
            self._env = GraphMultiAgentEVEnv(config, decision_window=decision_window, num_neighbors=num_neighbors)

    def run(self, agent: MAPPOGNNAgent, seeds: list[int], deterministic: bool) -> list[GraphEpisode]:
        sizes = (agent.hidden_size, agent.num_layers, agent.heads)
        actor_state = {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}
        if self._pool is None:
            cpu_agent = MAPPOGNNAgent(*sizes)
            cpu_agent.actor.load_state_dict(actor_state)
            return [run_episode(self._env, cpu_agent, seed, deterministic) for seed in seeds]
        return self._pool.map(_worker_run, [(actor_state, sizes, seed, deterministic) for seed in seeds])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()


# --- rollout -> PPO rows -------------------------------------------------


class GraphRows:
    """A list of graphs that train_mappo.ppo_update can index like a tensor:
    rows[idx] (idx a tensor of row numbers) is the Batch of those graphs."""

    def __init__(self, graphs: list[Data]):
        self.graphs = graphs

    def __getitem__(self, idx: torch.Tensor) -> Batch:
        return Batch.from_data_list([self.graphs[i] for i in idx.tolist()])

    def __len__(self) -> int:
        return len(self.graphs)


@dataclass
class GraphRolloutTensors:
    """Same fields as train_mappo.RolloutTensors, with graphs for obs/state."""

    obs: GraphRows
    global_states: GraphRows
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return len(self.actions)


def _data(x: np.ndarray, u: np.ndarray, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Data:
    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
        u=torch.from_numpy(u).reshape(1, -1),
        num_nodes=x.shape[0],
    )


def build_rollout(episodes: list[GraphEpisode], agent: MAPPOGNNAgent, ppo: PPOConfig) -> GraphRolloutTensors:
    """One row per agent decision: its local graph, its batch's global
    graph, its GAE advantage and return (GAE exactly as for MAPPO-MLP)."""
    local, global_rows, actions, log_probs, advantages, returns = [], [], [], [], [], []
    for episode in episodes:
        if not episode.batches:
            continue
        edge_index = torch.from_numpy(episode.edge_index)
        edge_attr = torch.from_numpy(episode.edge_attr)
        global_graphs = [_data(b.global_x, b.global_u, edge_index, edge_attr) for b in episode.batches]
        values = agent.get_value(global_graphs)
        final_value = float(
            agent.get_value(_data(episode.final_global_x, episode.final_global_u, edge_index, edge_attr))[0]
        )
        gae = compute_gae(episode, values, final_value, ppo.gamma, ppo.gae_lambda)
        for t, (batch, adv) in enumerate(zip(episode.batches, gae)):
            for i in range(len(batch.actions)):
                local.append(_data(batch.local_x[i], batch.local_u[i], edge_index, edge_attr))
                global_rows.append(global_graphs[t])
            actions.append(batch.actions)
            log_probs.append(batch.log_probs)
            advantages.append(adv)
            returns.append(adv + values[t])

    def cat(parts, dtype=torch.float32):
        return torch.as_tensor(np.concatenate(parts), dtype=dtype, device=agent.device)

    return GraphRolloutTensors(
        GraphRows(local),
        GraphRows(global_rows),
        cat(actions, torch.long),
        cat(log_probs),
        cat(advantages),
        cat(returns),
    )


# --- training loop -------------------------------------------------------


def train(
    config: SimulationConfig,
    seed: int = 42,
    total_decisions: int = 1_000_000,
    episodes_per_update: int = 16,
    n_workers: int = N_WORKERS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    ppo: PPOConfig = PPOConfig(),
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    heads: int = DEFAULT_HEADS,
    num_neighbors: int = DEFAULT_NEIGHBORS,
    decision_window: int = DECISION_WINDOW,
    eval_every: int = 10,
    validation_seeds: list[int] = VALIDATION_SEEDS,
    device: str = "cpu",
    scenario_name: str = DEFAULT_SCENARIO_NAME,
    tensorboard: bool = True,
) -> Path:
    torch.manual_seed(seed)
    seed_rng = np.random.default_rng(seed)
    agent = MAPPOGNNAgent(hidden_size, num_layers, heads, device=device)
    optimizer = torch.optim.Adam(
        [*agent.actor.parameters(), *agent.critic.parameters()], lr=ppo.learning_rate, eps=1e-5
    )
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(TENSORBOARD_LOG_DIR / f"{output_path.stem}_{int(time.time())}"))

    runner = RolloutRunner(config, decision_window, num_neighbors, n_workers)
    best_score = -float("inf")
    decisions_done, update = 0, 0
    started = time.time()
    print(
        f"MAPPO-GNN: params={agent.num_parameters()} hidden={hidden_size} layers={num_layers} heads={heads} "
        f"k={num_neighbors} stations={config.num_stations} EVs={config.num_vehicles} window={decision_window} "
        f"workers={n_workers} episodes/update={episodes_per_update} device={device}"
    )
    try:
        while decisions_done < total_decisions:
            update += 1
            seeds = [int(s) for s in seed_rng.integers(0, 2**31 - 1, size=episodes_per_update)]
            episodes = runner.run(agent, seeds, deterministic=False)
            rollout = build_rollout(episodes, agent, ppo)
            losses = ppo_update(agent, optimizer, rollout, ppo)
            stats = summarize(episodes)
            decisions_done += len(rollout)

            print(
                f"[update {update:4d}] decisions={decisions_done:7d} "
                f"reward/decision={stats['reward_per_decision']:8.3f} episode_return={stats['episode_return']:8.2f} "
                f"invalid={stats['invalid_rate']:.3f} overload={stats['overload_rate']:.3f} "
                f"stranded={stats['stranded']} | actor_loss={losses['actor_loss']:.4f} "
                f"value_loss={losses['value_loss']:.3f} entropy={losses['entropy']:.3f} "
                f"kl={losses['approx_kl']:.4f} | {time.time() - started:6.0f}s",
                flush=True,
            )
            if writer is not None:
                for key, value in {**stats, **losses}.items():
                    writer.add_scalar(f"mappo_gnn/{key}", value, decisions_done)

            if update % eval_every == 0 or decisions_done >= total_decisions:
                validation = summarize(runner.run(agent, validation_seeds, deterministic=True))
                score = validation["reward_per_decision"]
                if writer is not None:
                    for key, value in validation.items():
                        writer.add_scalar(f"validation_gnn/{key}", value, decisions_done)
                improved = score > best_score
                print(
                    f"           validation reward/decision={score:8.3f} "
                    f"invalid={validation['invalid_rate']:.3f} overload={validation['overload_rate']:.3f} "
                    f"travel={validation['travel_time']:.1f}s waiting={validation['waiting_time']:.1f}s"
                    + ("  -> new best, saved" if improved else ""),
                    flush=True,
                )
                if improved:
                    best_score = score
                    agent.save(
                        output_path,
                        scenario=scenario_name,
                        update=update,
                        decisions=decisions_done,
                        validation_reward_per_decision=score,
                        decision_window=decision_window,
                        num_neighbors=num_neighbors,
                        ppo=asdict(ppo),
                    )
    finally:
        runner.close()
        if writer is not None:
            writer.close()

    print(f"best model (validation reward/decision={best_score:.3f}) saved to {output_path}")
    return output_path


def run_smoke_test(config: SimulationConfig, seed: int, n_workers: int, device: str) -> None:
    print("[1/4] checking graph observations...")
    env = GraphMultiAgentEVEnv(config)
    env.reset(seed=VALIDATION_SEEDS[0])
    graph = env.get_graph_observation(env.agents[0])
    assert graph.num_nodes == config.num_stations and torch.isfinite(graph.x).all()
    assert env.observation_space(env.agents[0]).shape[0] > 0  # flat space untouched
    print(f"      {graph}")

    smoke_path = MODELS_DIR / "checkpoints" / "mappo_gnn_smoke_test.pth"
    print(f"[2/4] training one short update ({n_workers} workers, update on {device})...")
    train(
        config,
        seed,
        total_decisions=1,
        episodes_per_update=max(n_workers, 2),
        n_workers=n_workers,
        output_path=smoke_path,
        eval_every=1,
        validation_seeds=VALIDATION_SEEDS[:2],
        device=device,
        tensorboard=False,
    )

    print("[3/4] reloading the saved model and checking its actions...")
    agent = MAPPOGNNAgent.load(smoke_path)
    episode = run_episode(GraphMultiAgentEVEnv(config), agent, VALIDATION_SEEDS[0], True)
    actions = np.concatenate([b.actions for b in episode.batches])
    assert ((0 <= actions) & (actions < config.num_stations)).all()
    print(f"      {len(actions)} deterministic actions, all within the action space -> OK")

    bigger = replace(config, num_stations=config.num_stations * 3, num_vehicles=config.num_vehicles * 2)
    print(f"[4/4] same weights, zero-shot on {bigger.num_stations} stations / {bigger.num_vehicles} EVs...")
    env = GraphMultiAgentEVEnv(bigger)
    env.reset(seed=VALIDATION_SEEDS[0])
    _, _, actions, _ = agent.act(env, deterministic=True)
    assert ((0 <= actions) & (actions < bigger.num_stations)).all()
    print(f"      first batch actions {actions.tolist()} -> OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO-GNN for multi-agent EV charging-station dispatch.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default=DEFAULT_SCENARIO_NAME,
                        help="Which config to train on (episodes/update and minibatch default to its own).")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick pre-training checks instead of full training.")
    parser.add_argument("--total-decisions", type=int, default=1_000_000)
    parser.add_argument("--episodes-per-update", type=int, default=None, help="Default: the scenario's.")
    parser.add_argument("--minibatch-size", type=int, default=None, help="Default: the scenario's.")
    parser.add_argument("--n-workers", type=int, default=N_WORKERS, help="Parallel rollout processes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--num-neighbors", type=int, default=DEFAULT_NEIGHBORS, help="k of the station kNN graph.")
    parser.add_argument("--decision-window", type=int, default=DECISION_WINDOW)
    parser.add_argument("--eval-every", type=int, default=10, help="Updates between validation runs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coef)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device of the PPO update (rollouts always run on CPU).")
    args = parser.parse_args()

    scenario = get_scenario(args.scenario)
    print(f"scenario={scenario.name} ({scenario.label})")
    if args.smoke_test:
        run_smoke_test(scenario.config, args.seed, args.n_workers, args.device)
        return

    train(
        scenario.config,
        args.seed,
        total_decisions=args.total_decisions,
        episodes_per_update=args.episodes_per_update or scenario.mappo_episodes_per_update,
        n_workers=args.n_workers,
        output_path=args.output,
        ppo=PPOConfig(
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            minibatch_size=args.minibatch_size or scenario.mappo_minibatch_size,
        ),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        heads=args.heads,
        num_neighbors=args.num_neighbors,
        decision_window=args.decision_window,
        eval_every=args.eval_every,
        device=args.device,
        scenario_name=scenario.name,
    )


if __name__ == "__main__":
    main()
