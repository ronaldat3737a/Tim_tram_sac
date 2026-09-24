import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from stable_baselines3 import DQN

from backend.api import simulation_manager as simulation_manager_module
from backend.api.simulation_manager import SimulationManager
from backend.config import DEFAULT_CONFIG

SMALL_CONFIG = replace(DEFAULT_CONFIG, num_vehicles=5, max_episode_steps=800)
TRAINED_MODEL_PATH = Path("models/dqn_ev_dispatch.zip")


@pytest.fixture
def manager():
    m = SimulationManager(SMALL_CONFIG)
    m.start_background_thread()
    yield m
    m.stop_background_thread()


def test_on_tick_ignores_snapshot_from_a_replaced_episode(manager):
    # Simulates a tick_hook call arriving from an orphaned env.step() that
    # was still mid-flight when a concurrent reset already replaced
    # manager.env (Phase 9 finding): such a stale tick must not overwrite
    # the current episode's published snapshot.
    stale_simulator = manager.env.simulator

    manager.request_reset(seed=123)
    snapshot_after_reset = manager.get_snapshot()

    manager._on_tick(stale_simulator)

    assert manager.get_snapshot() == snapshot_after_reset


def test_on_tick_does_not_sleep_for_a_stale_snapshot(manager):
    manager.request_speed(1.0)  # tick_delay = 1s for the *current* episode
    stale_simulator = manager.env.simulator
    manager.request_reset(seed=7)

    start = time.monotonic()
    manager._on_tick(stale_simulator)
    elapsed = time.monotonic() - start

    assert elapsed < 0.1  # stale ticks fast-forward instead of pacing


def test_on_tick_still_publishes_and_paces_current_episode_ticks(manager):
    manager.request_speed(1000.0)  # negligible tick_delay for this check
    current_simulator = manager.env.simulator

    manager._on_tick(current_simulator)

    assert manager.get_snapshot()["timestamp"] == current_simulator.simulation_time


def test_dqn_model_loading_does_not_block_status_requests(monkeypatch):
    if not TRAINED_MODEL_PATH.exists():
        pytest.skip("no trained model artifact present (run backend.ai_core.train first)")
    real_model = DQN.load(str(TRAINED_MODEL_PATH))

    def slow_load(path, *args, **kwargs):
        time.sleep(1.0)
        return real_model

    monkeypatch.setattr(simulation_manager_module.DQN, "load", staticmethod(slow_load))

    config = replace(DEFAULT_CONFIG, num_vehicles=5, max_episode_steps=800)
    manager = SimulationManager(config)
    manager.start_background_thread()
    try:
        start_thread = threading.Thread(
            target=lambda: manager.request_start(seed=1, algorithm="dqn", speed=5.0)
        )
        start_thread.start()
        time.sleep(0.2)  # request_start should now be inside the ~1s slow_load, unlocked

        started = time.monotonic()
        manager.get_status()
        elapsed = time.monotonic() - started

        start_thread.join(timeout=5)
        assert not start_thread.is_alive()
        assert elapsed < 0.5, "get_status() was blocked by the in-progress DQN.load"
        assert manager.algorithm == "dqn"
    finally:
        manager.stop_background_thread()


def test_network_info_nodes_are_real_geographic_coordinates(manager):
    # The graph is always a real OSM street network now (section 11) --
    # x/y served to the frontend are the node's own real (lng, lat), with
    # no separate abstract-to-geo mapping step anymore.
    info = manager.get_network_info()
    for node in info["nodes"]:
        assert -180.0 <= node["x"] <= 180.0
        assert -90.0 <= node["y"] <= 90.0


def test_network_info_edge_geometry_matches_graph_distance_semantics(manager):
    # Regression guard: displayed geometry must never be consulted by
    # anything that computes simulation physics -- distance/travel_time
    # stay the graph's own real values regardless of how geometry renders.
    graph = manager.env.simulator.graph
    for u, v, data in graph.edges(data=True):
        assert "distance" in data and data["distance"] > 0
        assert "travel_time" in data and data["travel_time"] > 0
        assert len(data["geometry"]) >= 2
