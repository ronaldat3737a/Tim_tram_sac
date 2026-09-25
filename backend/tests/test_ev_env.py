from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from backend.ai_core.ev_env import EVEnv
from backend.config import DEFAULT_CONFIG
from backend.simulation.network_graph import get_station_nodes
from backend.simulation.vehicle import Vehicle, VehicleState

SMALL_CONFIG = replace(
    DEFAULT_CONFIG,
    num_stations=2,
    # A generous vehicle count makes it overwhelmingly likely that at least
    # one EV needs a charging decision during reset(), regardless of seed.
    num_vehicles=15,
    max_episode_steps=3000,
)


def test_observation_and_action_space_shapes():
    env = EVEnv(SMALL_CONFIG)

    assert env.action_space.n == SMALL_CONFIG.num_stations
    assert env.observation_space.shape == (5 + 7 * SMALL_CONFIG.num_stations,)
    assert env.observation_space.dtype == np.float32


def test_reset_returns_observation_within_space():
    env = EVEnv(SMALL_CONFIG)

    observation, info = env.reset(seed=1)

    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)
    assert "next_vehicle_id" in info


def test_reset_reproducible_with_same_seed():
    env_a = EVEnv(SMALL_CONFIG)
    env_b = EVEnv(SMALL_CONFIG)

    obs_a, info_a = env_a.reset(seed=99)
    obs_b, info_b = env_b.reset(seed=99)

    assert np.array_equal(obs_a, obs_b)
    assert info_a["next_vehicle_id"] == info_b["next_vehicle_id"]


def test_step_with_reachable_station_returns_valid_types():
    # Low consumption relative to the graph's extent guarantees the EV that
    # triggers a decision still has enough battery left to reach at least
    # one station, regardless of where stations happen to land.
    config = replace(SMALL_CONFIG, battery_consumption_per_distance=0.0001)
    env = EVEnv(config)
    obs, info = env.reset(seed=2)
    vehicle_id = info["next_vehicle_id"]

    reachable = [
        s for s in range(config.num_stations)
        if env.simulator.is_station_reachable(vehicle_id, s)
    ]
    assert reachable, "test scenario expects at least one reachable station"

    observation, reward, terminated, truncated, info2 = env.step(reachable[0])

    assert observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info2["invalid_action"] is False


def _reset_with_one_reachable_and_one_unreachable_station(env):
    """Returns (vehicle_id, reachable_station_id, unreachable_station_id)
    after lowering the current EV's battery so exactly its nearest station
    stays reachable."""
    for seed in range(50):
        _, info = env.reset(seed=seed)
        vehicle_id = info["next_vehicle_id"]
        energy = {sid: env.simulator.energy_required(vehicle_id, sid) for sid in env.simulator.stations}
        near, far = sorted(energy, key=energy.get)[0], sorted(energy, key=energy.get)[-1]
        if energy[far] - energy[near] > 1e-3:
            vehicle = env.simulator.vehicles[vehicle_id]
            vehicle.battery_level = (energy[near] + energy[far]) / 2
            return vehicle_id, near, far
    raise AssertionError("no seed with distinct station distances")


def test_invalid_action_is_penalized_and_falls_back_to_nearest_reachable_station():
    env = EVEnv(SMALL_CONFIG)
    vehicle_id, reachable_id, unreachable_id = _reset_with_one_reachable_and_one_unreachable_station(env)

    observation, reward, terminated, truncated, info = env.step(unreachable_id)

    assert info["invalid_action"] is True
    assert info["stranded"] is False
    assert info["requested_station_id"] == unreachable_id
    assert info["station_id"] == reachable_id
    # The EV was dispatched to the fallback at once, not left pending.
    vehicle = env.simulator.vehicles[vehicle_id]
    assert vehicle.target_station == reachable_id
    assert vehicle not in env.simulator.get_vehicles_needing_decision()
    assert reward <= SMALL_CONFIG.invalid_action_penalty * SMALL_CONFIG.reward_scale


def test_invalid_action_always_scores_worse_than_choosing_the_fallback_directly():
    env_invalid = EVEnv(SMALL_CONFIG)
    _, reachable_id, unreachable_id = _reset_with_one_reachable_and_one_unreachable_station(env_invalid)
    _, reward_invalid, *_ = env_invalid.step(unreachable_id)

    env_direct = EVEnv(SMALL_CONFIG)
    _reset_with_one_reachable_and_one_unreachable_station(env_direct)
    _, reward_direct, *_ = env_direct.step(reachable_id)

    assert reward_invalid == pytest.approx(reward_direct + SMALL_CONFIG.invalid_action_penalty * SMALL_CONFIG.reward_scale)


def test_vehicle_with_no_reachable_station_is_stranded_not_counted_invalid():
    env = EVEnv(SMALL_CONFIG)
    obs, info = env.reset(seed=5)
    vehicle_id = info["next_vehicle_id"]
    vehicle = env.simulator.vehicles[vehicle_id]
    vehicle.battery_level = 0.0

    observation, reward, terminated, truncated, info2 = env.step(0)

    assert info2["invalid_action"] is False
    assert info2["stranded"] is True
    assert vehicle.state == VehicleState.FAILED
    assert reward == pytest.approx(SMALL_CONFIG.battery_failure_penalty * SMALL_CONFIG.reward_scale)


def test_worst_case_policy_resolves_each_ev_at_most_once():
    # Regression: an unresolved invalid action used to leave the same EV
    # frozen and pending, so a bad policy could rack up thousands of
    # decisions in one episode. Every step now resolves one EV for good.
    env = EVEnv(SMALL_CONFIG)
    env.reset(seed=8)
    num_decisions = 0
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.n - 1)
        num_decisions += 1
        done = terminated or truncated

    assert num_decisions <= SMALL_CONFIG.num_vehicles


def test_observation_is_finite_and_within_unit_box_across_an_episode():
    env = EVEnv(SMALL_CONFIG)
    obs, _ = env.reset(seed=9)
    done = False
    while not done:
        assert np.all(np.isfinite(obs)) and obs.min() >= 0.0 and obs.max() <= 1.0
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated


def test_reward_matches_travel_and_waiting_time_formula():
    config = replace(DEFAULT_CONFIG, num_stations=2, num_vehicles=1)
    env = EVEnv(config)
    obs, info = env.reset(seed=3)
    vehicle_id = info["next_vehicle_id"]
    vehicle = env.simulator.vehicles[vehicle_id]
    station_id = next(iter(env.simulator.stations))
    station = env.simulator.stations[station_id]

    vehicle.current_node = station.node_id
    vehicle.battery_level = 1.0

    observation, reward, terminated, truncated, info2 = env.step(station_id)

    # The only EV, so this one step runs until its trip resolves.
    assert info2["invalid_action"] is False
    [trip] = info2["resolved"]
    assert trip["vehicle_id"] == vehicle_id and trip["failed"] is False
    assert trip["travel_time"] == pytest.approx(0.0)
    assert trip["waiting_time"] == pytest.approx(config.time_step)
    expected_reward = -(
        config.reward_travel_weight * trip["travel_time"]
        + config.reward_waiting_weight * trip["waiting_time"]
    )
    if info2["station_overloaded"]:
        expected_reward += config.station_overload_penalty
    assert reward == pytest.approx(expected_reward * config.reward_scale)


def test_station_overload_penalty_applied_without_terminating_episode():
    # num_vehicles > 1 so the episode has other still-traveling EVs left
    # once the ego vehicle's own decision resolves this step.
    config = replace(
        DEFAULT_CONFIG,
        num_stations=1,
        num_vehicles=4,
        station_capacity=1,
        num_chargers_per_station=1,
    )
    env = EVEnv(config)
    obs, info = env.reset(seed=13)
    vehicle_id = info["next_vehicle_id"]
    vehicle = env.simulator.vehicles[vehicle_id]
    station_id = next(iter(env.simulator.stations))
    station = env.simulator.stations[station_id]

    vehicle.current_node = station.node_id
    vehicle.battery_level = 1.0

    filler_kwargs = dict(
        current_node=station.node_id,
        destination_node=station.node_id,
        battery_level=1.0,
        battery_capacity=config.battery_capacity,
        speed=config.max_speed,
        target_station=station_id,
        state=VehicleState.WAITING,
        route=[station.node_id],
    )
    filler_a = Vehicle(vehicle_id=901, **filler_kwargs)
    filler_b = Vehicle(vehicle_id=902, **filler_kwargs)
    env.simulator.vehicles[901] = filler_a
    env.simulator.vehicles[902] = filler_b
    station.queue.append(901)
    station.queue.append(902)

    observation, reward, terminated, truncated, info2 = env.step(station_id)

    assert info2["station_overloaded"] is True
    assert terminated is False


def test_full_episode_with_simple_greedy_policy_terminates():
    config = replace(DEFAULT_CONFIG, num_stations=3, num_vehicles=6, max_episode_steps=500)
    env = EVEnv(config)
    obs, info = env.reset(seed=11)

    terminated = truncated = False
    steps = 0
    max_rl_steps = 600
    while not (terminated or truncated) and steps < max_rl_steps:
        vehicle_id = info["next_vehicle_id"]
        reachable = [
            s for s in range(config.num_stations)
            if env.simulator.is_station_reachable(vehicle_id, s)
        ]
        action = (
            min(reachable, key=lambda s: env.simulator.energy_required(vehicle_id, s))
            if reachable
            else 0
        )
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert terminated or truncated
    assert steps < max_rl_steps


def test_tick_hook_is_called_once_per_simulator_tick():
    config = replace(DEFAULT_CONFIG, num_stations=2, num_vehicles=3)
    tick_count = 0

    def hook(simulator):
        nonlocal tick_count
        tick_count += 1

    env = EVEnv(config, tick_hook=hook)
    obs, info = env.reset(seed=17)
    ticks_after_reset = tick_count
    assert ticks_after_reset == env.simulator.simulation_time

    vehicle_id = info["next_vehicle_id"]
    reachable = [s for s in range(config.num_stations) if env.simulator.is_station_reachable(vehicle_id, s)]
    action = reachable[0] if reachable else 0
    env.step(action)

    assert tick_count > ticks_after_reset
    assert tick_count == env.simulator.simulation_time


def test_check_env_gymnasium_compliance():
    config = replace(
        DEFAULT_CONFIG, num_stations=2, num_vehicles=15, max_episode_steps=3000
    )
    env = EVEnv(config)

    check_env(env, skip_render_check=True)


def test_build_observation_matches_internal_observation_for_current_vehicle():
    config = replace(DEFAULT_CONFIG, num_stations=2, num_vehicles=10)
    env = EVEnv(config)
    obs, info = env.reset(seed=11)

    public_obs = env.build_observation(info["next_vehicle_id"])

    assert np.array_equal(obs, public_obs)


def test_build_observation_works_for_an_arbitrary_non_current_vehicle():
    config = replace(DEFAULT_CONFIG, num_stations=2, num_vehicles=10)
    env = EVEnv(config)
    obs, info = env.reset(seed=11)

    other_vehicle_id = next(
        vid for vid in env.simulator.vehicles if vid != info["next_vehicle_id"]
    )
    other_obs = env.build_observation(other_vehicle_id)

    assert other_obs.shape == env.observation_space.shape
    assert env.observation_space.contains(other_obs)


def test_observation_coordinates_are_scaled_to_map_bounding_box():
    env = EVEnv(SMALL_CONFIG)
    obs, info = env.reset(seed=4)
    graph = env.simulator.graph
    x_min, y_min, x_max, y_max = graph.graph["bounds"]
    vehicle = env.simulator.vehicles[info["next_vehicle_id"]]
    current = graph.nodes[vehicle.current_node]

    assert obs[0] == pytest.approx((current["x"] - x_min) / (x_max - x_min), abs=1e-6)
    assert obs[1] == pytest.approx((current["y"] - y_min) / (y_max - y_min), abs=1e-6)
    for index, station_id in enumerate(sorted(env.simulator.stations)):
        station_node = graph.nodes[env.simulator.stations[station_id].node_id]
        offset = 5 + 7 * index
        assert obs[offset] == pytest.approx((station_node["x"] - x_min) / (x_max - x_min), abs=1e-6)
        assert obs[offset + 1] == pytest.approx((station_node["y"] - y_min) / (y_max - y_min), abs=1e-6)


def test_action_index_maps_to_real_station_node():
    env = EVEnv(SMALL_CONFIG)
    env.reset(seed=4)
    station_nodes = set(get_station_nodes(env.simulator.graph))

    for action in range(env.action_space.n):
        station_id = env.station_id_for_action(action)
        assert env.simulator.stations[station_id].node_id in station_nodes
    assert len({env.station_id_for_action(a) for a in range(env.action_space.n)}) == env.action_space.n


def test_no_ev_is_ever_held_on_the_road_awaiting_a_decision():
    # Regression: step() used to fast-forward until the dispatched EV
    # started charging, while every other EV that hit the threshold in the
    # meantime stood still on the road (thousands of ticks on the OSM map).
    # An EV can cross the threshold during a tick; it is only "held" if it
    # is still undecided after a further tick.
    held_ticks = 0
    pending_after_last_tick: set[int] = set()

    def count_held(simulator):
        nonlocal held_ticks, pending_after_last_tick
        pending = {v.vehicle_id for v in simulator.get_vehicles_needing_decision()}
        held_ticks += len(pending & pending_after_last_tick)
        pending_after_last_tick = pending

    env = EVEnv(SMALL_CONFIG, tick_hook=count_held)
    env.reset(seed=8)
    done = False
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated

    assert held_ticks == 0


def test_step_returns_without_ticking_when_another_ev_is_already_pending():
    env = EVEnv(SMALL_CONFIG)
    _, info = env.reset(seed=8)
    other = next(
        v for v in env.simulator.vehicles.values()
        if v.vehicle_id != info["next_vehicle_id"] and v.state == VehicleState.TRAVELING
    )
    other.battery_level = SMALL_CONFIG.low_battery_threshold
    time_before = env.simulator.simulation_time

    _, _, _, _, info2 = env.step(0)

    assert env.simulator.simulation_time == time_before
    assert info2["next_vehicle_id"] == other.vehicle_id


def test_episode_reward_totals_all_travel_and_waiting_time():
    config = replace(SMALL_CONFIG, station_overload_penalty=0.0)
    env = EVEnv(config)
    env.reset(seed=8)
    total_reward = 0.0
    trips = []
    penalties = 0.0
    done = False
    while not done:
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total_reward += reward
        trips += info["resolved"]
        penalties += config.invalid_action_penalty * info["invalid_action"]
        penalties += config.battery_failure_penalty * info["stranded"]
        done = terminated or truncated

    assert terminated and not env._in_flight  # every dispatched trip resolved
    expected = penalties + sum(
        -(config.reward_travel_weight * t["travel_time"] + config.reward_waiting_weight * t["waiting_time"])
        + (config.battery_failure_penalty if t["failed"] else 0.0)
        for t in trips
    )
    assert total_reward == pytest.approx(expected * config.reward_scale)
