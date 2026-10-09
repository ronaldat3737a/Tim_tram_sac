from pathlib import Path

import pytest

from backend.ai_core.marl_env import MultiAgentEVEnv
from backend.ai_core.train_mappo import build_agent
from backend.config import DEFAULT_SCENARIO, OSM_DEMO_CONFIG, SCENARIOS, get_scenario


def test_small_scenario_is_the_unchanged_phase_1_setup():
    small = SCENARIOS["small"]

    assert DEFAULT_SCENARIO == "small"
    assert small.config is OSM_DEMO_CONFIG
    assert small.mappo_model_path == "models/mappo_osm_model.pth"
    assert small.dqn_model_path == "models/dqn_osm_model.zip"
    assert small.config.osm_cache_path == "data/osm_nghia_do.graphml"
    assert not small.config.osm_full_drive_network


def test_large_scenario_never_shares_files_with_small():
    small, large = SCENARIOS["small"], SCENARIOS["large"]

    small_files = {
        small.mappo_model_path, small.dqn_model_path,
        small.config.osm_cache_path, small.config.osm_road_layer_cache_path,
    }
    large_files = {
        large.mappo_model_path, large.dqn_model_path,
        large.config.osm_cache_path, large.config.osm_road_layer_cache_path,
    }
    assert small_files.isdisjoint(large_files)


def test_large_scenario_scale_and_cohort_fractions():
    small, large = SCENARIOS["small"].config, SCENARIOS["large"].config

    assert (large.num_stations, large.num_vehicles) == (10, 200)
    assert 10 <= large.num_depots <= 15
    assert large.osm_radius_meters == 2 * small.osm_radius_meters
    assert large.osm_full_drive_network
    assert large.low_battery_threshold == 0.8
    assert (large.peak_hour_fraction, large.non_app_fraction) == (small.peak_hour_fraction, small.non_app_fraction)


def test_large_scenario_shapes_follow_num_stations():
    scenario = SCENARIOS["large"]
    env = MultiAgentEVEnv(scenario.config)

    assert env.observation_space(0).shape == (5 + 7 * 10,)
    assert env.state_space.shape == (3 * 10 + 1,)
    assert env.action_space(0).n == 10

    agent = build_agent(scenario.config, scenario.mappo_hidden_sizes)
    assert (agent.obs_dim, agent.state_dim, agent.num_actions) == (75, 31, 10)
    assert agent.hidden_sizes == (256, 256)


def test_get_scenario_rejects_unknown_names():
    assert get_scenario("large") is SCENARIOS["large"]
    with pytest.raises(ValueError, match="unknown scenario"):
        get_scenario("huge")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_model_paths_live_in_models_dir(name):
    scenario = SCENARIOS[name]
    assert Path(scenario.mappo_model_path).parent == Path("models")
    assert Path(scenario.dqn_model_path).parent == Path("models")
