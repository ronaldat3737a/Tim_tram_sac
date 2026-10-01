"""MAPPO training entry point.

Trains a shared Actor + centralized Critic (backend/ai_core/mappo.py) on
MultiAgentEVEnv with OSM_DEMO_CONFIG and a 15-tick decision window, and
saves the best model (by validation reward) to models/mappo_osm_model.pth,
the path backend/api/simulation_manager.py loads.

Run (from the repo root):
    python backend/ai_core/train_mappo.py --smoke-test
    python backend/ai_core/train_mappo.py                       # 300k decisions (default)
    python backend/ai_core/train_mappo.py --n-workers 2         # fewer parallel workers
    tensorboard --logdir ./logs/

Rollouts: each update collects whole episodes, played by n_workers
processes in parallel. A decision batch holds 1..n agents, so the buffer
keeps one row per agent (obs, action, log_prob, reward) next to one entry
per batch (the global state at decision time).

Advantages: an app EV decides only once, so treating each agent as its own
one-step episode would make it purely selfish. Instead the decision batches
of an episode form one chain, and GAE runs along it with per-agent rewards:
    delta_i = r_i + gamma * V(s_{t+1}) - V(s_t)
    A_i     = delta_i + gamma * lambda * mean_j(A_j of batch t+1)
where V is the centralized critic on the global state. A dispatch that
congests stations for later EVs lowers V(s_{t+1}), so it is penalized even
though its own reward only prices its own trip.

Training never runs inside FastAPI -- this is a standalone CLI.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python backend/ai_core/train_mappo.py` as well as
    # `python -m backend.ai_core.train_mappo` (see train.py).
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch import nn

from backend.ai_core.mappo import DEFAULT_HIDDEN_SIZES, MAPPOAgent
from backend.ai_core.marl_env import MultiAgentEVEnv
from backend.config import OSM_DEMO_CONFIG, SimulationConfig

DECISION_WINDOW = 15
# Parallel rollout processes; each loads its own copy of the OSM graph, as
# in train.py.
N_WORKERS = 4
MODELS_DIR = Path("models")
TENSORBOARD_LOG_DIR = Path("logs")
DEFAULT_OUTPUT_PATH = MODELS_DIR / "mappo_osm_model.pth"
# Fixed scenarios the best model is picked on. Disjoint from
# evaluate.TEST_SEEDS (1000-1004), which stay unseen for the final A/B test.
VALIDATION_SEEDS = [2000, 2001, 2002, 2003, 2004]


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_epochs: int = 10
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


# --- rollout data --------------------------------------------------------


@dataclass
class DecisionBatch:
    """One MultiAgentEVEnv.step(): n agents acting on one global state."""

    global_state: np.ndarray  # (state_dim,)
    obs: np.ndarray  # (n, obs_dim)
    actions: np.ndarray  # (n,)
    log_probs: np.ndarray  # (n,)
    rewards: np.ndarray  # (n,)


@dataclass
class EpisodeStats:
    num_decisions: int = 0
    num_invalid: int = 0
    num_overloaded: int = 0
    num_stranded: int = 0
    total_decision_delay: float = 0.0
    total_reward: float = 0.0
    travel_times: list[float] = field(default_factory=list)
    waiting_times: list[float] = field(default_factory=list)


@dataclass
class Episode:
    seed: int
    batches: list[DecisionBatch]
    # True if every EV finished; False if cut by max_episode_steps, in which
    # case GAE bootstraps from the final global state.
    terminated: bool
    final_state: np.ndarray
    stats: EpisodeStats


def run_episode(env: MultiAgentEVEnv, agent: MAPPOAgent, seed: int, deterministic: bool) -> Episode:
    obs, _ = env.reset(seed=seed)
    batches: list[DecisionBatch] = []
    stats = EpisodeStats()
    while env.agents:
        agents = list(env.agents)
        global_state = env.state()
        obs_batch = np.stack([obs[a] for a in agents])
        actions, log_probs = agent.get_action(obs_batch, deterministic=deterministic)
        obs, rewards, _, _, infos = env.step(dict(zip(agents, actions.tolist())))

        reward_batch = np.array([rewards[a] for a in agents], dtype=np.float32)
        batches.append(DecisionBatch(global_state, obs_batch, actions, log_probs, reward_batch))
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
    return Episode(seed, batches, env.simulator.all_vehicles_done(), env.state(), stats)


# --- parallel rollout workers -------------------------------------------

_worker_env: MultiAgentEVEnv | None = None


def _worker_init(config: SimulationConfig, decision_window: int) -> None:
    global _worker_env
    torch.set_num_threads(1)
    _worker_env = MultiAgentEVEnv(config, decision_window=decision_window)


def _worker_run(job: tuple[dict, int, int, int, list[int], int, bool]) -> Episode:
    actor_state, obs_dim, state_dim, num_actions, hidden_sizes, seed, deterministic = job
    agent = MAPPOAgent(obs_dim, state_dim, num_actions, hidden_sizes)
    agent.actor.load_state_dict(actor_state)
    return run_episode(_worker_env, agent, seed, deterministic)


class RolloutRunner:
    """Plays episodes with the current actor, in n_workers processes (or
    inline when n_workers == 1)."""

    def __init__(self, config: SimulationConfig, decision_window: int, n_workers: int):
        self._pool = None
        self._env = None
        if n_workers > 1:
            # spawn: never fork a process that already holds torch threads.
            self._pool = mp.get_context("spawn").Pool(
                n_workers, initializer=_worker_init, initargs=(config, decision_window)
            )
        else:
            self._env = MultiAgentEVEnv(config, decision_window=decision_window)

    def run(self, agent: MAPPOAgent, seeds: list[int], deterministic: bool) -> list[Episode]:
        if self._pool is None:
            return [run_episode(self._env, agent, seed, deterministic) for seed in seeds]
        actor_state = {k: v.cpu() for k, v in agent.actor.state_dict().items()}
        dims = (agent.obs_dim, agent.state_dim, agent.num_actions, list(agent.hidden_sizes))
        return self._pool.map(_worker_run, [(actor_state, *dims, seed, deterministic) for seed in seeds])

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()


# --- GAE + PPO -----------------------------------------------------------


def compute_gae(
    episode: Episode, values: np.ndarray, final_value: float, gamma: float, gae_lambda: float
) -> list[np.ndarray]:
    """Per-agent advantages for every batch of one episode (see the module
    docstring). values[t] = V(global state of batch t)."""
    advantages: list[np.ndarray] = [np.empty(0, dtype=np.float32)] * len(episode.batches)
    next_value = 0.0 if episode.terminated else final_value
    next_mean_advantage = 0.0
    for t in reversed(range(len(episode.batches))):
        delta = episode.batches[t].rewards + gamma * next_value - values[t]
        advantages[t] = (delta + gamma * gae_lambda * next_mean_advantage).astype(np.float32)
        next_value = float(values[t])
        next_mean_advantage = float(advantages[t].mean())
    return advantages


@dataclass
class RolloutTensors:
    obs: torch.Tensor
    global_states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __len__(self) -> int:
        return len(self.actions)


def build_rollout(episodes: list[Episode], agent: MAPPOAgent, ppo: PPOConfig) -> RolloutTensors:
    """Flatten episodes into one row per agent decision, each row carrying
    the global state of its batch, its GAE advantage and its return."""
    obs, states, actions, log_probs, advantages, returns = [], [], [], [], [], []
    for episode in episodes:
        if not episode.batches:
            continue
        values = agent.get_value(np.stack([b.global_state for b in episode.batches]))
        final_value = float(agent.get_value(episode.final_state)[0])
        for t, (batch, adv) in enumerate(
            zip(episode.batches, compute_gae(episode, values, final_value, ppo.gamma, ppo.gae_lambda))
        ):
            n = len(batch.actions)
            obs.append(batch.obs)
            states.append(np.repeat(batch.global_state[None, :], n, axis=0))
            actions.append(batch.actions)
            log_probs.append(batch.log_probs)
            advantages.append(adv)
            returns.append(adv + values[t])

    def cat(parts, dtype=torch.float32):
        return torch.as_tensor(np.concatenate(parts), dtype=dtype, device=agent.device)

    return RolloutTensors(
        cat(obs), cat(states), cat(actions, torch.long), cat(log_probs), cat(advantages), cat(returns)
    )


def ppo_update(
    agent: MAPPOAgent, optimizer: torch.optim.Optimizer, rollout: RolloutTensors, ppo: PPOConfig
) -> dict[str, float]:
    advantages = rollout.advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    logs: dict[str, list[float]] = {k: [] for k in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
    for _ in range(ppo.n_epochs):
        for idx in torch.randperm(len(rollout), device=agent.device).split(ppo.minibatch_size):
            log_prob, entropy, value = agent.evaluate_actions(
                rollout.obs[idx], rollout.global_states[idx], rollout.actions[idx]
            )
            log_ratio = log_prob - rollout.old_log_probs[idx]
            ratio = log_ratio.exp()
            surrogate = torch.min(
                ratio * advantages[idx],
                ratio.clamp(1 - ppo.clip_range, 1 + ppo.clip_range) * advantages[idx],
            )
            actor_loss = -surrogate.mean() - ppo.entropy_coef * entropy.mean()
            value_loss = nn.functional.mse_loss(value, rollout.returns[idx])

            optimizer.zero_grad()
            (actor_loss + ppo.value_coef * value_loss).backward()
            # Clipped per network: early value errors are large (returns run
            # to tens of units), and a joint norm would let the critic's
            # gradient shrink the actor's to nothing.
            nn.utils.clip_grad_norm_(agent.actor.parameters(), ppo.max_grad_norm)
            nn.utils.clip_grad_norm_(agent.critic.parameters(), ppo.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                logs["actor_loss"].append(actor_loss.item())
                logs["value_loss"].append(value_loss.item())
                logs["entropy"].append(entropy.mean().item())
                logs["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                logs["clip_fraction"].append(((ratio - 1).abs() > ppo.clip_range).float().mean().item())
    return {k: statistics.mean(v) for k, v in logs.items()}


# --- training loop -------------------------------------------------------


def summarize(episodes: list[Episode]) -> dict[str, float]:
    decisions = sum(e.stats.num_decisions for e in episodes) or 1
    travel = [t for e in episodes for t in e.stats.travel_times]
    waiting = [w for e in episodes for w in e.stats.waiting_times]
    return {
        "decisions": decisions,
        "reward_per_decision": sum(e.stats.total_reward for e in episodes) / decisions,
        "episode_return": statistics.mean(e.stats.total_reward for e in episodes),
        "invalid_rate": sum(e.stats.num_invalid for e in episodes) / decisions,
        "overload_rate": sum(e.stats.num_overloaded for e in episodes) / decisions,
        "stranded": sum(e.stats.num_stranded for e in episodes),
        "decision_delay": sum(e.stats.total_decision_delay for e in episodes) / decisions,
        "travel_time": statistics.mean(travel) if travel else 0.0,
        "waiting_time": statistics.mean(waiting) if waiting else 0.0,
    }


def build_agent(config: SimulationConfig, hidden_sizes) -> MAPPOAgent:
    env = MultiAgentEVEnv(config)
    return MAPPOAgent(
        env.observation_space(0).shape[0], env.state_space.shape[0], env.action_space(0).n, hidden_sizes
    )


def train(
    config: SimulationConfig = OSM_DEMO_CONFIG,
    seed: int = OSM_DEMO_CONFIG.random_seed,
    total_decisions: int = 300_000,
    episodes_per_update: int = 32,
    n_workers: int = N_WORKERS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    ppo: PPOConfig = PPOConfig(),
    hidden_sizes=DEFAULT_HIDDEN_SIZES,
    decision_window: int = DECISION_WINDOW,
    eval_every: int = 10,
    validation_seeds: list[int] = VALIDATION_SEEDS,
    tensorboard: bool = True,
) -> Path:
    torch.manual_seed(seed)
    seed_rng = np.random.default_rng(seed)
    agent = build_agent(config, hidden_sizes)
    optimizer = torch.optim.Adam(
        [*agent.actor.parameters(), *agent.critic.parameters()], lr=ppo.learning_rate, eps=1e-5
    )
    writer = None
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(TENSORBOARD_LOG_DIR / f"{output_path.stem}_{int(time.time())}"))

    runner = RolloutRunner(config, decision_window, n_workers)
    best_score = -float("inf")
    decisions_done, update = 0, 0
    started = time.time()
    print(
        f"MAPPO: obs_dim={agent.obs_dim} state_dim={agent.state_dim} actions={agent.num_actions} "
        f"window={decision_window} workers={n_workers} episodes/update={episodes_per_update}"
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
                    writer.add_scalar(f"mappo/{key}", value, decisions_done)

            if update % eval_every == 0 or decisions_done >= total_decisions:
                validation = summarize(runner.run(agent, validation_seeds, deterministic=True))
                score = validation["reward_per_decision"]
                if writer is not None:
                    for key, value in validation.items():
                        writer.add_scalar(f"validation/{key}", value, decisions_done)
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
                        update=update,
                        decisions=decisions_done,
                        validation_reward_per_decision=score,
                        decision_window=decision_window,
                        ppo=asdict(ppo),
                    )
    finally:
        runner.close()
        if writer is not None:
            writer.close()

    print(f"best model (validation reward/decision={best_score:.3f}) saved to {output_path}")
    return output_path


def run_smoke_test(config: SimulationConfig, seed: int, n_workers: int) -> None:
    from pettingzoo.test import parallel_api_test

    print("[1/3] checking environment (PettingZoo parallel API)...")
    parallel_api_test(MultiAgentEVEnv(config, decision_window=DECISION_WINDOW), num_cycles=1000)

    smoke_path = MODELS_DIR / "checkpoints" / "mappo_smoke_test.pth"
    print(f"[2/3] training one short update ({n_workers} workers)...")
    train(
        config,
        seed,
        total_decisions=1,
        episodes_per_update=max(n_workers, 2),
        n_workers=n_workers,
        output_path=smoke_path,
        eval_every=1,
        validation_seeds=VALIDATION_SEEDS[:2],
        tensorboard=False,
    )

    print("[3/3] reloading the saved model and checking its actions...")
    agent = MAPPOAgent.load(smoke_path)
    episode = run_episode(MultiAgentEVEnv(config, decision_window=DECISION_WINDOW), agent, VALIDATION_SEEDS[0], True)
    actions = np.concatenate([b.actions for b in episode.batches])
    assert ((0 <= actions) & (actions < config.num_stations)).all()
    print(f"      {len(actions)} deterministic actions, all within the action space -> OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPPO for multi-agent EV charging-station dispatch.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick pre-training checks instead of full training.")
    parser.add_argument("--total-decisions", type=int, default=300_000)
    parser.add_argument("--episodes-per-update", type=int, default=32)
    parser.add_argument("--n-workers", type=int, default=N_WORKERS, help="Parallel rollout processes.")
    parser.add_argument("--seed", type=int, default=OSM_DEMO_CONFIG.random_seed)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES))
    parser.add_argument("--decision-window", type=int, default=DECISION_WINDOW)
    parser.add_argument("--eval-every", type=int, default=10, help="Updates between validation runs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--learning-rate", type=float, default=PPOConfig.learning_rate)
    parser.add_argument("--entropy-coef", type=float, default=PPOConfig.entropy_coef)
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(OSM_DEMO_CONFIG, args.seed, args.n_workers)
        return

    train(
        OSM_DEMO_CONFIG,
        args.seed,
        total_decisions=args.total_decisions,
        episodes_per_update=args.episodes_per_update,
        n_workers=args.n_workers,
        output_path=args.output,
        ppo=PPOConfig(learning_rate=args.learning_rate, entropy_coef=args.entropy_coef),
        hidden_sizes=args.hidden_sizes,
        decision_window=args.decision_window,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
