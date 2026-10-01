import warnings
from dataclasses import replace

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from backend.ai_core.marl_env import MultiAgentEVEnv
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


def _run_episode(env, seed, policy=None):
    """Play one episode, returning every step's (actions, results)."""
    policy = policy or (lambda agent: env.action_space(agent).sample())
    env.reset(seed=seed)
    steps = []
    while env.agents:
        actions = {agent: policy(agent) for agent in env.agents}
        steps.append((actions, env.step(actions)))
    return steps


def test_passes_pettingzoo_parallel_api_test():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Expected: an app EV that reaches its destination without ever
        # running low never becomes an agent, so it is never terminated.
        warnings.filterwarnings(
            "ignore", message="No agents present but not all possible_agents are terminated or truncated"
        )
        parallel_api_test(MultiAgentEVEnv(MARL_CONFIG), num_cycles=1000)


def test_spaces_are_shared_and_match_single_agent_layout():
    env = MultiAgentEVEnv(MARL_CONFIG)
    env.reset(seed=3)

    assert env.observation_space(env.possible_agents[0]) is env.observation_space(env.possible_agents[-1])
    assert env.observation_space(0).shape == (5 + 7 * MARL_CONFIG.num_stations,)
    assert env.action_space(0).n == MARL_CONFIG.num_stations
    assert env.state_space.shape == (3 * MARL_CONFIG.num_stations + 1,)
    assert set(env.observation_spaces) == set(env.action_spaces) == set(env.possible_agents)


def test_possible_agents_are_exactly_the_app_vehicles():
    env = MultiAgentEVEnv(MARL_CONFIG)
    env.reset(seed=3)

    app_ids = sorted(vid for vid, v in env.simulator.vehicles.items() if not v.is_non_app)
    assert env.possible_agents == app_ids
    assert len(app_ids) == MARL_CONFIG.num_vehicles - round(MARL_CONFIG.num_vehicles * MARL_CONFIG.non_app_fraction)


def test_non_app_vehicles_never_become_agents():
    env = MultiAgentEVEnv(MARL_CONFIG)
    env.reset(seed=3)
    non_app = {vid for vid, v in env.simulator.vehicles.items() if v.is_non_app}
    seen = set(env.agents)
    while env.agents:
        env.step({agent: 0 for agent in env.agents})
        seen |= set(env.agents)

    assert seen and not seen & non_app


def test_each_agent_acts_once_and_is_then_terminated():
    env = MultiAgentEVEnv(MARL_CONFIG)
    acted = []
    for actions, (obs, rewards, terminations, truncations, infos) in _run_episode(env, seed=5):
        for agent in actions:
            assert terminations[agent] and not truncations[agent]
            assert "stranded" in infos[agent]
        for agent in env.agents:
            assert not terminations[agent] and rewards[agent] == 0.0
            assert obs[agent] in env.observation_space(agent)
        acted += list(actions)

    assert len(acted) == len(set(acted))


def test_single_agent_reward_is_the_semi_mdp_proxy_cost():
    env = MultiAgentEVEnv(MARL_CONFIG)
    env.reset(seed=5)
    while len(env.agents) != 1:
        env.step({agent: 0 for agent in env.agents})
        assert env.agents, "no single-agent batch in this episode"
    (agent,) = env.agents
    station_id = env.station_id_for_action(1)
    assert env.simulator.is_station_reachable(agent, station_id)
    station = env.simulator.stations[station_id]
    expected = -env._context.expected_dispatch_cost(agent, station_id)
    if station.occupancy + 1 > station.capacity:
        expected += MARL_CONFIG.station_overload_penalty

    _, rewards, _, _, infos = env.step({agent: 1})

    assert infos[agent]["station_id"] == station_id
    assert rewards[agent] == pytest.approx(expected * MARL_CONFIG.reward_scale)


def test_every_dispatched_trip_resolves_and_incoming_counts_drain():
    env = MultiAgentEVEnv(MARL_CONFIG)
    dispatched, resolved = [], []
    env.reset(seed=8)
    while env.agents:
        _, _, _, _, infos = env.step({agent: env.action_space(agent).sample() for agent in env.agents})
        dispatched += [a for a, info in infos.items() if info.get("station_id") is not None]
        resolved += [t["vehicle_id"] for t in env.last_resolved]

    app_dispatched = sorted(dispatched)
    assert sorted(v for v in resolved if v in env.possible_agents) == app_dispatched
    assert all(station.incoming_count == 0 for station in env.simulator.stations.values())


def test_global_state_reports_every_station_load():
    env = MultiAgentEVEnv(MARL_CONFIG)
    env.reset(seed=3)
    agent = env.agents[0]
    incoming_before = {sid: s.incoming_count for sid, s in env.simulator.stations.items()}

    _, info = env._context.dispatch(agent, env.station_id_for_action(0))
    station_id = info["station_id"]
    incoming_before = incoming_before[station_id]
    state = env.state()

    assert state.shape == env.state_space.shape and state.dtype == np.float32
    assert state in env.state_space
    station = env.simulator.stations[station_id]
    index = env._context.station_ids.index(station_id) * 3
    assert state[index] == pytest.approx(min(len(station.queue) / station.capacity, 1.0))
    assert state[index + 2] == pytest.approx(min((incoming_before + 1) / station.capacity, 1.0))


def test_step_rejects_missing_or_extra_actions():
    env = MultiAgentEVEnv(MARL_CONFIG)
    env.reset(seed=3)

    with pytest.raises(ValueError):
        env.step({})
    with pytest.raises(ValueError):
        env.step({**{agent: 0 for agent in env.agents}, -1: 0})


def test_reset_is_reproducible_with_same_seed():
    first = _run_episode(MultiAgentEVEnv(MARL_CONFIG), seed=11, policy=lambda agent: agent % 2)
    second = _run_episode(MultiAgentEVEnv(MARL_CONFIG), seed=11, policy=lambda agent: agent % 2)

    assert [a for a, _ in first] == [a for a, _ in second]
    assert [r[1] for _, r in first] == [r[1] for _, r in second]


def _batches(env, seed):
    """(agents, infos) of every step of one episode, always picking action 0."""
    env.reset(seed=seed)
    batches = []
    while env.agents:
        agents = list(env.agents)
        _, _, _, _, infos = env.step({agent: 0 for agent in agents})
        batches.append((agents, {agent: infos[agent] for agent in agents}))
    return batches


def test_decision_window_batches_evs_running_low_within_it():
    batches = _batches(MultiAgentEVEnv(MARL_CONFIG, decision_window=15), seed=3)
    tick_exact = _batches(MultiAgentEVEnv(MARL_CONFIG, decision_window=0), seed=3)

    delays = [info["decision_delay"] for _, infos in batches for info in infos.values()]
    assert all(0 <= delay <= 15 for delay in delays)
    assert max(len(agents) for agents, _ in batches) > 1
    assert len(batches) < len(tick_exact)


def test_zero_decision_window_dispatches_on_the_tick_evs_run_low():
    batches = _batches(MultiAgentEVEnv(MARL_CONFIG, decision_window=0), seed=3)

    assert all(info["decision_delay"] == 0 for _, infos in batches for info in infos.values())


def test_evs_waiting_in_the_window_keep_draining_idle_battery():
    env = MultiAgentEVEnv(MARL_CONFIG, decision_window=15)
    env.reset(seed=3)
    now = env.simulator.simulation_time
    waited = [agent for agent in env.agents if now - env._pending_since[agent] > 0]
    assert waited

    for agent in waited:
        delay = now - env._pending_since[agent]
        battery = env.simulator.vehicles[agent].battery_level
        drain = MARL_CONFIG.idle_battery_drain_per_tick * MARL_CONFIG.time_step * delay
        assert battery <= MARL_CONFIG.low_battery_threshold - drain + 1e-9


def test_decision_window_closes_early_when_the_episode_ends():
    env = MultiAgentEVEnv(MARL_CONFIG, decision_window=100_000)

    # The first EV runs low long before max_episode_steps, so the window
    # would outlast the episode: it closes at truncation, with no batch.
    with pytest.raises(RuntimeError):
        env.reset(seed=3)
    assert env.simulator.simulation_time == MARL_CONFIG.max_episode_steps


def test_negative_decision_window_is_rejected():
    with pytest.raises(ValueError):
        MultiAgentEVEnv(MARL_CONFIG, decision_window=-1)


def test_global_state_ends_with_episode_time_progress():
    env = MultiAgentEVEnv(MARL_CONFIG)
    env.reset(seed=3)
    progress = [env.state()[-1]]
    while env.agents:
        env.step({agent: 0 for agent in env.agents})
        progress.append(env.state()[-1])

    assert all(0.0 <= p <= 1.0 for p in progress)
    assert progress == sorted(progress) and progress[-1] > progress[0]
    assert progress[-1] == pytest.approx(
        min(env.simulator.simulation_time / MARL_CONFIG.max_episode_steps, 1.0)
    )
