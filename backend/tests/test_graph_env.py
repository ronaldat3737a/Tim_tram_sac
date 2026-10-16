from dataclasses import replace

import pytest
import torch

from backend.ai_core.graph_env import (
    EDGE_DIM,
    EGO_DIM,
    GLOBAL_DIM,
    GLOBAL_NODE_DIM,
    NODE_DIM,
    NODE_FEATURES,
    GraphMultiAgentEVEnv,
)
from backend.ai_core.marl_env import MultiAgentEVEnv
from backend.config import DEFAULT_CONFIG

GRAPH_CONFIG = replace(
    DEFAULT_CONFIG,
    num_stations=6,
    num_vehicles=15,
    max_episode_steps=3000,
    max_activation_tick=500,
    peak_hour_fraction=0.6,
    non_app_fraction=0.2,
)


def _env(**kwargs) -> GraphMultiAgentEVEnv:
    env = GraphMultiAgentEVEnv(GRAPH_CONFIG, decision_window=200, **kwargs)
    env.reset(seed=3)
    return env


def test_flat_observation_space_is_the_unchanged_mlp_one():
    env = _env()
    assert env.observation_space(0) == MultiAgentEVEnv(GRAPH_CONFIG).observation_space(0)


def test_graph_observation_has_one_node_per_station_in_action_order():
    env = _env()
    agent = env.agents[0]
    station = env.simulator.stations[env.station_id_for_action(4)]
    station.queue.extend([900, 901])  # make action 4's station visibly busy

    graph = env.get_graph_observation(agent)

    assert graph.x.shape == (GRAPH_CONFIG.num_stations, NODE_DIM)
    assert graph.u.shape == (1, EGO_DIM)
    assert graph.edge_attr.shape[1] == EDGE_DIM
    assert torch.isfinite(graph.x).all()
    queue = NODE_FEATURES.index("queue")
    assert graph.x[4, queue] == pytest.approx(2 / station.capacity)
    assert (graph.x[[i for i in range(6) if i != 4], queue] == 0).all()


def test_reachable_feature_matches_the_simulator():
    env = _env()
    agent = env.agents[0]
    graph = env.get_graph_observation(agent)
    reachable = graph.x[:, NODE_FEATURES.index("reachable")].tolist()
    expected = [
        float(env.simulator.is_station_reachable(agent, env.station_id_for_action(i)))
        for i in range(GRAPH_CONFIG.num_stations)
    ]
    assert reachable == expected


@pytest.mark.parametrize("k", [1, 3, 10])
def test_every_station_receives_its_k_nearest_neighbours(k):
    env = _env(num_neighbors=k)
    graph = env.get_global_graph()
    expected = min(k, GRAPH_CONFIG.num_stations - 1)

    incoming = torch.bincount(graph.edge_index[1], minlength=GRAPH_CONFIG.num_stations)
    assert incoming.tolist() == [expected] * GRAPH_CONFIG.num_stations
    assert (graph.edge_index[0] != graph.edge_index[1]).all()  # no self loops
    assert graph.x.shape == (GRAPH_CONFIG.num_stations, GLOBAL_NODE_DIM)
    assert graph.u.shape == (1, GLOBAL_DIM)


def test_decide_sequentially_graph_shows_earlier_picks_then_restores_the_env():
    env = GraphMultiAgentEVEnv(GRAPH_CONFIG, decision_window=200)
    for seed in range(50):
        env.reset(seed=seed)
        if len(env.agents) >= 2:
            break
    else:
        pytest.fail("no decision batch with two or more agents found")
    incoming_before = {sid: s.incoming_count for sid, s in env.simulator.stations.items()}
    incoming = NODE_FEATURES.index("incoming")
    capacity = env.simulator.stations[env.station_id_for_action(0)].capacity

    actions, graphs = env.decide_sequentially_graph(lambda graph: 0)

    assert list(actions) == env.agents and set(actions.values()) == {0}
    for i, graph in enumerate(graphs):
        before = incoming_before[env.station_id_for_action(0)]
        assert graph.x[0, incoming].item() == pytest.approx(min((before + i) / capacity, 2.0))
    assert {sid: s.incoming_count for sid, s in env.simulator.stations.items()} == incoming_before


def test_decide_sequentially_graph_rejects_an_out_of_range_action_without_side_effects():
    env = _env()
    incoming_before = {sid: s.incoming_count for sid, s in env.simulator.stations.items()}
    with pytest.raises(ValueError):
        env.decide_sequentially_graph(lambda graph: 99)
    assert {sid: s.incoming_count for sid, s in env.simulator.stations.items()} == incoming_before


def test_a_single_station_graph_has_no_edges():
    env = GraphMultiAgentEVEnv(replace(GRAPH_CONFIG, num_stations=1), decision_window=200)
    env.reset(seed=3)
    assert env.get_global_graph().edge_index.shape == (2, 0)
