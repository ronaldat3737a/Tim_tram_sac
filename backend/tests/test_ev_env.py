from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from backend.ai_core.ev_env import EVEnv
from backend.config import DEFAULT_CONFIG
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
    assert env.observation_space.shape == (5 + 5 * SMALL_CONFIG.num_stations,)
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


def test_step_with_unreachable_station_is_penalized_and_blocks_assignment():
    env = EVEnv(SMALL_CONFIG)
    obs, info = env.reset(seed=5)
    vehicle_id = info["next_vehicle_id"]
    vehicle = env.simulator.vehicles[vehicle_id]
    vehicle.battery_level = 0.0

    observation, reward, terminated, truncated, info2 = env.step(0)

    assert info2["invalid_action"] is True
    assert reward == SMALL_CONFIG.invalid_action_penalty
    assert vehicle.target_station is None
    assert terminated is False


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

    assert info2["invalid_action"] is False
    assert info2["travel_time"] == pytest.approx(0.0)
    assert info2["waiting_time"] == pytest.approx(config.time_step)
    expected_reward = -(
        config.reward_travel_weight * info2["travel_time"]
        + config.reward_waiting_weight * info2["waiting_time"]
    )
    if info2["station_overloaded"]:
        expected_reward += config.station_overload_penalty
    assert reward == pytest.approx(expected_reward)


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
