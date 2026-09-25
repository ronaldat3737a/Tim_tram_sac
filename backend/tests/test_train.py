from dataclasses import replace

import numpy as np
import pytest

from backend.ai_core.ev_env import EVEnv
from backend.ai_core.train import RandomScenarioPerEpisode, build_model, make_training_env, run_smoke_test
from backend.config import DEFAULT_CONFIG

SMALL_CONFIG = replace(DEFAULT_CONFIG, num_stations=2, num_vehicles=10, max_episode_steps=1500)


def test_random_scenario_wrapper_varies_seed_across_episodes():
    env = RandomScenarioPerEpisode(EVEnv(SMALL_CONFIG), seed_rng=np.random.default_rng(123))

    obs_first, _ = env.reset()
    obs_second, _ = env.reset()

    assert not np.array_equal(obs_first, obs_second)


def test_random_scenario_wrapper_reproducible_given_same_rng_seed():
    env_a = RandomScenarioPerEpisode(EVEnv(SMALL_CONFIG), seed_rng=np.random.default_rng(999))
    env_b = RandomScenarioPerEpisode(EVEnv(SMALL_CONFIG), seed_rng=np.random.default_rng(999))

    obs_a1, _ = env_a.reset()
    obs_a2, _ = env_a.reset()
    obs_b1, _ = env_b.reset()
    obs_b2, _ = env_b.reset()

    assert np.array_equal(obs_a1, obs_b1)
    assert np.array_equal(obs_a2, obs_b2)


def test_random_scenario_wrapper_respects_explicit_seed_override():
    env = RandomScenarioPerEpisode(EVEnv(SMALL_CONFIG), seed_rng=np.random.default_rng(7))

    obs_explicit_a, _ = env.reset(seed=55)
    obs_explicit_b, _ = env.reset(seed=55)

    assert np.array_equal(obs_explicit_a, obs_explicit_b)


def test_training_vec_env_workers_run_distinct_scenarios():
    env = make_training_env(SMALL_CONFIG, seed=11, n_envs=2)
    try:
        first = env.reset()
        assert first.shape == (2, *env.observation_space.shape)
        assert not np.array_equal(first[0], first[1])
    finally:
        env.close()


def test_build_model_uses_mlp_policy_and_configured_net_arch():
    env = EVEnv(SMALL_CONFIG)
    model = build_model(env, seed=1, net_arch=[32, 32])

    assert model.policy_class.__name__ == "DQNPolicy"
    assert model.policy_kwargs["net_arch"] == [32, 32]
    assert model.action_space.n == SMALL_CONFIG.num_stations


@pytest.mark.parametrize(("n_envs", "expected_train_freq"), [(1, 4), (2, 2), (4, 1)])
def test_build_model_keeps_one_update_per_four_transitions(n_envs, expected_train_freq):
    env = make_training_env(SMALL_CONFIG, seed=5, n_envs=n_envs)
    try:
        model = build_model(env, seed=1, net_arch=[16, 16])
        assert model.train_freq.frequency == expected_train_freq
        assert model.gradient_steps == 1
    finally:
        env.close()


def test_build_model_on_plain_env_uses_train_freq_four():
    model = build_model(EVEnv(SMALL_CONFIG), seed=1, net_arch=[16, 16])

    assert model.train_freq.frequency == 4


def test_run_smoke_test_executes_end_to_end_without_error():
    run_smoke_test(SMALL_CONFIG, seed=3, net_arch=[16, 16], smoke_timesteps=100, n_envs=2)
