"""DQN training entry point (PROJECT_SPEC.md sections 31, 46 Phase 5).

Trains on OSM_DEMO_CONFIG (the real Nghia Do street map the API serves) and
saves to models/dqn_osm_model.zip, the path backend/api/simulation_manager.py
loads.

Run (from the repo root):
    python backend/ai_core/train.py --smoke-test
    python backend/ai_core/train.py                      # 500k steps (default)
    tensorboard --logdir ./logs/

Training never runs inside FastAPI (section 36) -- this is a standalone CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow `python backend/ai_core/train.py` as well as
    # `python -m backend.ai_core.train`: the former puts backend/ai_core/ on
    # sys.path instead of the repo root, so `backend.*` imports would fail.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from stable_baselines3.common.monitor import Monitor

from backend.ai_core.ev_env import EVEnv
from backend.config import OSM_DEMO_CONFIG, SimulationConfig

DEFAULT_NET_ARCH = [64, 64]
MODELS_DIR = Path("models")
TENSORBOARD_LOG_DIR = Path("logs")
DEFAULT_OUTPUT_NAME = "dqn_osm_model"


class RandomScenarioPerEpisode(gym.Wrapper):
    """Draws a fresh scenario seed for every episode reset during training.

    EVEnv.reset(seed=None) deterministically falls back to config.random_seed
    (needed for reproducible tests/evaluation), but Stable-Baselines3 calls
    env.reset() without a seed at every episode boundary during training. Left
    unwrapped, the agent would train on one single fixed scenario for the
    entire run. This wrapper draws a new seed per episode from its own RNG,
    which is itself seeded, so the overall training run stays reproducible
    for a given --seed even though individual episodes vary.
    """

    def __init__(self, env: gym.Env, seed_rng: np.random.Generator):
        super().__init__(env)
        self._seed_rng = seed_rng

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        episode_seed = seed if seed is not None else int(self._seed_rng.integers(0, 2**31 - 1))
        return self.env.reset(seed=episode_seed, options=options)


class TensorboardCallback(BaseCallback):
    """Logs per-decision dispatch outcomes from EVEnv's step() info dict to
    TensorBoard (under dispatch/), alongside SB3's own rollout/ and train/
    scalars. Averaged over each SB3 logging window (record_mean), so the
    curves show how often the policy picks unreachable stations, overloads
    a station or strands an EV, and the travel/waiting time it causes."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "invalid_action" not in info:
                continue
            self.logger.record_mean("dispatch/invalid_action_rate", float(info["invalid_action"]))
            self.logger.record_mean("dispatch/stranded_rate", float(info["stranded"]))
            self.logger.record_mean("dispatch/vehicle_failed_rate", float(info["vehicle_failed"]))
            if info["stranded"]:
                continue
            self.logger.record_mean("dispatch/travel_time", info["travel_time"])
            self.logger.record_mean("dispatch/waiting_time", info["waiting_time"])
            self.logger.record_mean("dispatch/overload_rate", float(info["station_overloaded"]))
        return True


def build_model(
    env: gym.Env, seed: int, net_arch: list[int], tensorboard_log: str | None = None
) -> DQN:
    return DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=net_arch),
        seed=seed,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )


def make_training_env(config: SimulationConfig, seed: int) -> Monitor:
    env = RandomScenarioPerEpisode(EVEnv(config), seed_rng=np.random.default_rng(seed))
    return Monitor(env)


def run_smoke_test(
    config: SimulationConfig, seed: int, net_arch: list[int], smoke_timesteps: int
) -> None:
    print("[1/3] checking environment (Gymnasium + Stable-Baselines3 compatibility)...")
    sb3_check_env(EVEnv(config), warn=True)
    print("      environment check passed")

    print(f"[2/3] running short training smoke test ({smoke_timesteps} timesteps)...")
    env = make_training_env(config, seed)
    model = build_model(env, seed, net_arch)
    model.learn(total_timesteps=smoke_timesteps)
    print(f"      trained for {smoke_timesteps} timesteps without error")

    print("[3/3] verifying predicted action is valid...")
    obs, _ = env.reset(seed=seed)
    action, _ = model.predict(obs, deterministic=True)
    action_int = int(action)
    if not env.action_space.contains(action_int):
        raise RuntimeError(f"model predicted invalid action {action_int}")
    print(f"      predicted action={action_int} is within action_space -> OK")


def train(
    config: SimulationConfig,
    seed: int,
    total_timesteps: int,
    net_arch: list[int],
    checkpoint_freq: int,
    output_name: str,
) -> Path:
    env = make_training_env(config, seed)
    model = build_model(env, seed, net_arch, tensorboard_log=str(TENSORBOARD_LOG_DIR))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = MODELS_DIR / "checkpoints"
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix=output_name,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList([checkpoint_callback, TensorboardCallback()]),
        tb_log_name=output_name,
    )

    final_path = MODELS_DIR / output_name
    model.save(str(final_path))
    print(f"model saved to {final_path}.zip")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for EV charging-station dispatch.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the Phase 5 pre-training checks instead of full training.")
    parser.add_argument("--smoke-timesteps", type=int, default=1000)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=OSM_DEMO_CONFIG.random_seed)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--net-arch", type=int, nargs="+", default=DEFAULT_NET_ARCH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_NAME)
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(OSM_DEMO_CONFIG, args.seed, args.net_arch, args.smoke_timesteps)
        return

    train(
        OSM_DEMO_CONFIG,
        args.seed,
        args.total_timesteps,
        args.net_arch,
        args.checkpoint_freq,
        args.output,
    )


if __name__ == "__main__":
    main()
