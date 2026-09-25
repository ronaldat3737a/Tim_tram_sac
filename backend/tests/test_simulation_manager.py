import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from stable_baselines3 import DQN

from backend.api import simulation_manager as simulation_manager_module
from backend.api.simulation_manager import SimulationManager, _interpolate_position, build_snapshot
from backend.config import DEFAULT_CONFIG
from backend.simulation.simulator import Simulator
from backend.simulation.vehicle import VehicleState

SMALL_CONFIG = replace(DEFAULT_CONFIG, num_vehicles=5, max_episode_steps=800)
TRAINED_MODEL_PATH = Path("models/dqn_osm_model.zip")


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
        assert len(data["geometry"].coords) >= 2


def test_network_info_stations_and_depots_report_access_node(manager):
    # Spatial-geometry refactor: a station/depot's own (x, y) -- its POI
    # node, in `nodes` -- is beside the road; access_node_id is the real
    # traffic node its spur edge connects to.
    info = manager.get_network_info()
    assert len(info["stations"]) > 0
    assert len(info["depots"]) > 0
    node_ids = {node["id"] for node in info["nodes"]}
    for station in info["stations"]:
        assert station["access_node_id"] in node_ids
        assert station["access_node_id"] != station["node_id"]
    for depot in info["depots"]:
        assert depot["access_node_id"] in node_ids
        assert depot["access_node_id"] != depot["node_id"]


def test_interpolate_position_reaches_the_station_poi_node_while_waiting_or_charging(manager):
    # The station's POI is a real graph node reached via a real spur edge
    # (network_graph._add_poi_nodes) -- once arrived, display position is
    # simply that node's own (x, y), the same as for any other node.
    simulator = manager.env.simulator
    vehicle = next(iter(simulator.vehicles.values()))
    station = next(iter(simulator.stations.values()))
    vehicle.current_node = station.node_id
    vehicle.target_station = station.station_id
    vehicle.route = [station.node_id]

    expected = (simulator.graph.nodes[station.node_id]["x"], simulator.graph.nodes[station.node_id]["y"])
    for state in (VehicleState.WAITING, VehicleState.CHARGING):
        vehicle.state = state
        assert _interpolate_position(simulator, vehicle) == expected


def test_interpolate_position_reaches_the_depot_poi_node_once_completed(manager):
    simulator = manager.env.simulator
    vehicle = next(iter(simulator.vehicles.values()))
    depot_node = simulator.depot_nodes[0]
    vehicle.current_node = depot_node
    vehicle.route = [depot_node]
    vehicle.state = VehicleState.COMPLETED

    expected = (simulator.graph.nodes[depot_node]["x"], simulator.graph.nodes[depot_node]["y"])
    assert _interpolate_position(simulator, vehicle) == expected


class _CrashingModel:
    def predict(self, obs, deterministic=True):
        raise RuntimeError("simulated model crash")


def test_dqn_predict_crash_falls_back_to_baseline_without_raising(manager):
    manager.algorithm = "dqn"
    manager._dqn_model = _CrashingModel()
    vehicle_id = manager._info["next_vehicle_id"]

    action = manager._choose_action()

    expected = simulation_manager_module._BASELINE_POLICIES[
        simulation_manager_module.DQN_FALLBACK_ALGORITHM
    ](manager.env.simulator, vehicle_id)
    assert action == expected


def test_incompatible_dqn_model_is_rejected_at_load(monkeypatch, tmp_path):
    from backend.ai_core.ev_env import EVEnv
    from backend.ai_core.train import build_model

    stale_config = replace(SMALL_CONFIG, num_stations=SMALL_CONFIG.num_stations + 1)
    stale_path = tmp_path / "stale.zip"
    build_model(EVEnv(stale_config), seed=0, net_arch=[8]).save(str(stale_path))
    monkeypatch.setattr(simulation_manager_module, "DQN_MODEL_PATH", stale_path)

    manager = SimulationManager(SMALL_CONFIG)
    with pytest.raises(ValueError, match="observation shape"):
        manager._ensure_dqn_model_loaded()
    assert manager._dqn_model is None


def test_snapshot_leaves_out_vehicles_that_have_not_departed_yet():
    simulator = Simulator(replace(SMALL_CONFIG, max_activation_tick=100), seed=1)
    vehicles = list(simulator.vehicles.values())
    for vehicle in vehicles:
        vehicle.activation_tick = 0
    vehicles[0].activation_tick = 50

    snapshot = build_snapshot(simulator, simulator.config, tick_delay=0.0)

    ids = {v["id"] for v in snapshot["vehicles"]}
    assert vehicles[0].vehicle_id not in ids
    assert ids == {v.vehicle_id for v in vehicles[1:]}
