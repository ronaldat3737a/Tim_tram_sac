from dataclasses import replace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from backend.ai_core.graph_env import GraphMultiAgentEVEnv
from backend.ai_core.mappo_gnn import MAPPOGNNAgent
from backend.ai_core.train_gnn import GraphRows, build_rollout, run_episode, train
from backend.ai_core.train_mappo import PPOConfig, ppo_update
from backend.config import DEFAULT_CONFIG

GNN_CONFIG = replace(
    DEFAULT_CONFIG,
    num_stations=3,
    num_vehicles=15,
    max_episode_steps=3000,
    max_activation_tick=500,
    peak_hour_fraction=0.6,
    non_app_fraction=0.2,
)


@pytest.fixture
def agent():
    torch.manual_seed(0)
    return MAPPOGNNAgent(hidden_size=16, num_layers=2, heads=2)


def _graphs(num_stations: int, seed: int = 3):
    env = GraphMultiAgentEVEnv(replace(GNN_CONFIG, num_stations=num_stations), decision_window=200)
    env.reset(seed=seed)
    return env, env.get_graph_observation(env.agents[0]), env.get_global_graph()


def test_weights_do_not_depend_on_the_number_of_stations(agent):
    shapes = {name: tuple(p.shape) for name, p in agent.actor.named_parameters()}
    for n in (2, 3, 8):
        _, graph, global_graph = _graphs(n)
        actions, _ = agent.get_action(graph)
        assert 0 <= actions[0] < n
        assert agent.get_value(global_graph).shape == (1,)
    assert {name: tuple(p.shape) for name, p in agent.actor.named_parameters()} == shapes


def test_one_batch_can_mix_graphs_of_different_sizes(agent):
    graphs = [_graphs(n)[1] for n in (2, 5, 3)]
    actions, log_probs = agent.get_action(graphs)
    assert all(0 <= a < n for a, n in zip(actions, (2, 5, 3)))
    assert np.isfinite(log_probs).all()


def test_deterministic_action_is_the_argmax_and_log_probs_match_evaluate(agent):
    _, graph, global_graph = _graphs(4)
    logits = agent.actor(Batch.from_data_list([graph]))[0]

    action, log_prob = agent.get_action(graph, deterministic=True)
    new_log_prob, entropy, value = agent.evaluate_actions([graph], [global_graph], torch.as_tensor(action))

    assert action[0] == int(logits.argmax())
    assert np.allclose(new_log_prob.detach().numpy(), log_prob, atol=1e-5)
    assert entropy.item() > 0 and value.shape == (1,)


def test_save_and_load_round_trip(agent, tmp_path):
    _, graph, global_graph = _graphs(4)
    agent.save(tmp_path / "gnn.pth", decision_window=15)

    loaded = MAPPOGNNAgent.load(tmp_path / "gnn.pth")

    assert loaded.get_action(graph, True)[0] == agent.get_action(graph, True)[0]
    assert np.allclose(loaded.get_value(global_graph), agent.get_value(global_graph))
    assert loaded.metadata == {"decision_window": 15}


def test_act_returns_one_aligned_row_per_agent(agent):
    env, _, _ = _graphs(4)
    agents, graphs, actions, log_probs = agent.act(env)
    assert agents == env.agents
    assert len(graphs) == len(agents) and actions.shape == log_probs.shape == (len(agents),)


def test_rollout_and_ppo_update_change_the_networks(agent):
    env = GraphMultiAgentEVEnv(GNN_CONFIG)
    episodes = [run_episode(env, agent, seed=s, deterministic=False) for s in (3, 4)]
    ppo = PPOConfig(n_epochs=2, minibatch_size=32)
    rollout = build_rollout(episodes, agent, ppo)
    assert len(rollout) == sum(e.stats.num_decisions for e in episodes)
    assert isinstance(rollout.obs[torch.arange(3)], Batch)

    before = [p.detach().clone() for p in agent.actor.parameters()]
    optimizer = torch.optim.Adam([*agent.actor.parameters(), *agent.critic.parameters()], lr=1e-3)
    losses = ppo_update(agent, optimizer, rollout, ppo)

    assert all(np.isfinite(v) for v in losses.values())
    assert any(not torch.equal(b, a) for b, a in zip(before, agent.actor.parameters()))


def test_graph_rows_index_like_a_tensor():
    _, graph, _ = _graphs(3)
    rows = GraphRows([graph] * 5)
    assert len(rows) == 5
    assert rows[torch.tensor([0, 2])].num_graphs == 2


def test_train_saves_a_model_that_acts_zero_shot_on_more_stations(tmp_path):
    path = train(
        GNN_CONFIG,
        total_decisions=1,
        episodes_per_update=2,
        n_workers=1,
        output_path=tmp_path / "gnn.pth",
        ppo=PPOConfig(n_epochs=1, minibatch_size=64),
        hidden_size=16,
        heads=2,
        eval_every=1,
        validation_seeds=[3],
        tensorboard=False,
    )
    agent = MAPPOGNNAgent.load(path)
    env, _, _ = _graphs(7)
    _, _, actions, _ = agent.act(env, deterministic=True)
    assert ((0 <= actions) & (actions < 7)).all()
