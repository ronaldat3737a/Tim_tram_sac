from dataclasses import replace

import numpy as np
import pytest
import torch

from backend.ai_core.mappo import MAPPOAgent
from backend.ai_core.marl_env import MultiAgentEVEnv
from backend.ai_core.train_mappo import (
    DecisionBatch,
    Episode,
    EpisodeStats,
    PPOConfig,
    build_agent,
    build_rollout,
    compute_gae,
    ppo_update,
    run_episode,
    train,
)
from backend.config import DEFAULT_CONFIG

MARL_CONFIG = replace(
    DEFAULT_CONFIG,
    num_stations=2,
    num_vehicles=15,
    max_episode_steps=3000,
    max_activation_tick=500,
    peak_hour_fraction=0.6,
    non_app_fraction=0.2,
)
OBS_DIM, STATE_DIM, NUM_ACTIONS = 5 + 7 * 2, 3 * 2 + 1, 2


@pytest.fixture
def agent():
    torch.manual_seed(0)
    return MAPPOAgent(OBS_DIM, STATE_DIM, NUM_ACTIONS, hidden_sizes=(16,))


def test_get_action_accepts_a_single_observation_or_a_batch(agent):
    single_actions, single_log_probs = agent.get_action(np.zeros(OBS_DIM, dtype=np.float32))
    batch_actions, batch_log_probs = agent.get_action(np.zeros((3, OBS_DIM), dtype=np.float32))

    assert single_actions.shape == single_log_probs.shape == (1,)
    assert batch_actions.shape == batch_log_probs.shape == (3,)
    assert ((0 <= batch_actions) & (batch_actions < NUM_ACTIONS)).all()
    assert (batch_log_probs <= 0).all()


def test_deterministic_action_is_the_actor_argmax(agent):
    obs = np.random.default_rng(0).random((4, OBS_DIM), dtype=np.float32)

    actions, _ = agent.get_action(obs, deterministic=True)

    with torch.no_grad():
        expected = agent.actor(torch.as_tensor(obs)).argmax(dim=-1).numpy()
    assert (actions == expected).all()


def test_wrong_input_width_is_rejected(agent):
    with pytest.raises(ValueError):
        agent.get_action(np.zeros(OBS_DIM + 1, dtype=np.float32))
    with pytest.raises(ValueError):
        agent.get_value(np.zeros(OBS_DIM, dtype=np.float32))


def test_evaluate_actions_matches_get_action_and_has_gradients(agent):
    obs = np.random.default_rng(1).random((5, OBS_DIM), dtype=np.float32)
    states = np.random.default_rng(2).random((5, STATE_DIM), dtype=np.float32)
    actions, log_probs = agent.get_action(obs)

    new_log_probs, entropy, values = agent.evaluate_actions(
        torch.as_tensor(obs), torch.as_tensor(states), torch.as_tensor(actions)
    )

    assert new_log_probs.shape == entropy.shape == values.shape == (5,)
    assert np.allclose(new_log_probs.detach().numpy(), log_probs, atol=1e-6)
    assert np.allclose(values.detach().numpy(), agent.get_value(states), atol=1e-6)
    (new_log_probs.sum() + values.sum()).backward()
    assert all(p.grad is not None for p in [*agent.actor.parameters(), *agent.critic.parameters()])


def test_evaluate_actions_handles_a_single_row(agent):
    log_prob, entropy, value = agent.evaluate_actions(
        torch.zeros(OBS_DIM), torch.zeros(STATE_DIM), torch.tensor(1)
    )

    assert log_prob.shape == entropy.shape == value.shape == (1,)


def test_save_and_load_round_trip(agent, tmp_path):
    path = tmp_path / "model.pth"
    agent.save(path, decision_window=15)

    loaded = MAPPOAgent.load(path)

    obs = np.random.default_rng(3).random((4, OBS_DIM), dtype=np.float32)
    states = np.random.default_rng(4).random((4, STATE_DIM), dtype=np.float32)
    assert (loaded.get_action(obs, deterministic=True)[0] == agent.get_action(obs, deterministic=True)[0]).all()
    assert np.allclose(loaded.get_value(states), agent.get_value(states))
    assert loaded.hidden_sizes == (16,) and loaded.metadata == {"decision_window": 15}


def _episode(rewards_per_batch, terminated=True):
    batches = [
        DecisionBatch(
            global_state=np.zeros(STATE_DIM, dtype=np.float32),
            obs=np.zeros((len(r), OBS_DIM), dtype=np.float32),
            actions=np.zeros(len(r), dtype=np.int64),
            log_probs=np.zeros(len(r), dtype=np.float32),
            rewards=np.array(r, dtype=np.float32),
        )
        for r in rewards_per_batch
    ]
    return Episode(0, batches, terminated, np.zeros(STATE_DIM, dtype=np.float32), EpisodeStats())


def test_gae_chains_per_agent_rewards_across_decision_batches():
    episode = _episode([[1.0, 3.0], [2.0]])
    values = np.array([0.5, 1.0])
    gamma, lam = 0.9, 0.8

    advantages = compute_gae(episode, values, final_value=99.0, gamma=gamma, gae_lambda=lam)

    # Last batch ends the (terminated) episode: no bootstrap.
    last = 2.0 - 1.0
    assert advantages[1] == pytest.approx([last])
    # First batch bootstraps from V(s_1) and the next batch's mean advantage.
    expected_first = [r + gamma * 1.0 - 0.5 + gamma * lam * last for r in (1.0, 3.0)]
    assert advantages[0] == pytest.approx(expected_first)


def test_gae_bootstraps_from_the_final_state_when_truncated():
    episode = _episode([[1.0]], terminated=False)

    (advantage,) = compute_gae(episode, np.array([0.5]), final_value=2.0, gamma=0.9, gae_lambda=0.8)

    assert advantage == pytest.approx([1.0 + 0.9 * 2.0 - 0.5])


def test_rollout_has_one_row_per_agent_with_its_batch_global_state():
    env = MultiAgentEVEnv(MARL_CONFIG, decision_window=15)
    agent = build_agent(MARL_CONFIG, hidden_sizes=(16,))
    episode = run_episode(env, agent, seed=3, deterministic=False)

    rollout = build_rollout([episode], agent, PPOConfig())

    rows = sum(len(b.actions) for b in episode.batches)
    assert rows == episode.stats.num_decisions == len(rollout)
    assert rollout.obs.shape == (rows, agent.obs_dim)
    assert rollout.global_states.shape == (rows, agent.state_dim)
    first = len(episode.batches[0].actions)
    assert torch.equal(rollout.global_states[0], torch.as_tensor(episode.batches[0].global_state))
    assert torch.equal(rollout.global_states[first - 1], torch.as_tensor(episode.batches[0].global_state))
    assert max(len(b.actions) for b in episode.batches) > 1


def test_ppo_update_changes_both_networks():
    env = MultiAgentEVEnv(MARL_CONFIG, decision_window=15)
    agent = build_agent(MARL_CONFIG, hidden_sizes=(16,))
    ppo = PPOConfig(n_epochs=2, minibatch_size=7)
    rollout = build_rollout([run_episode(env, agent, seed=s, deterministic=False) for s in (3, 4)], agent, ppo)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    optimizer = torch.optim.Adam([*agent.actor.parameters(), *agent.critic.parameters()], lr=1e-3)

    logs = ppo_update(agent, optimizer, rollout, ppo)

    assert set(logs) == {"actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"}
    assert all(np.isfinite(v) for v in logs.values())
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters()))


def test_train_saves_a_loadable_best_model(tmp_path):
    path = tmp_path / "mappo.pth"

    train(
        MARL_CONFIG,
        seed=0,
        total_decisions=30,
        episodes_per_update=2,
        n_workers=1,
        output_path=path,
        ppo=PPOConfig(n_epochs=1, minibatch_size=16),
        hidden_sizes=(16,),
        eval_every=1,
        validation_seeds=[2000],
        tensorboard=False,
    )

    loaded = MAPPOAgent.load(path)
    assert (loaded.obs_dim, loaded.state_dim, loaded.num_actions) == (OBS_DIM, STATE_DIM, NUM_ACTIONS)
    assert loaded.metadata["decision_window"] == 15
    assert "validation_reward_per_decision" in loaded.metadata
