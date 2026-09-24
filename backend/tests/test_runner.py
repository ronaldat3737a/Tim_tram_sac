from dataclasses import replace

import pytest

from backend.baseline import least_queue, nearest_station, shortest_time
from backend.baseline.runner import EpisodeMetrics, StationMetricsCollector, run_baseline_episode
from backend.config import DEFAULT_CONFIG
from backend.simulation.simulator import Simulator

SMALL_CONFIG = replace(
    DEFAULT_CONFIG, num_stations=3, num_vehicles=8, max_episode_steps=800
)

POLICIES = {
    "nearest_station": nearest_station.choose_station,
    "shortest_time": shortest_time.choose_station,
    "least_queue": least_queue.choose_station,
}


@pytest.mark.parametrize("name", POLICIES)
def test_run_baseline_episode_terminates_with_consistent_counts(name):
    metrics = run_baseline_episode(SMALL_CONFIG, seed=7, policy=POLICIES[name], policy_name=name)

    assert metrics.policy_name == name
    assert metrics.terminated or metrics.truncated
    assert metrics.num_decisions == metrics.num_valid_decisions + metrics.num_invalid_actions
    assert metrics.total_travel_time >= 0
    assert metrics.total_waiting_time >= 0
    assert metrics.simulation_time > 0


def test_run_baseline_episode_reproducible_with_same_seed():
    metrics_a = run_baseline_episode(
        SMALL_CONFIG, seed=11, policy=nearest_station.choose_station, policy_name="nearest_station"
    )
    metrics_b = run_baseline_episode(
        SMALL_CONFIG, seed=11, policy=nearest_station.choose_station, policy_name="nearest_station"
    )

    assert metrics_a.num_decisions == metrics_b.num_decisions
    assert metrics_a.total_travel_time == metrics_b.total_travel_time
    assert metrics_a.total_waiting_time == metrics_b.total_waiting_time
    assert metrics_a.episode_reward == metrics_b.episode_reward


@pytest.mark.parametrize("name", POLICIES)
def test_run_episode_populates_queue_and_utilization_metrics(name):
    metrics = run_baseline_episode(SMALL_CONFIG, seed=13, policy=POLICIES[name], policy_name=name)

    assert metrics.average_queue_length >= 0.0
    assert metrics.maximum_queue_length >= metrics.average_queue_length
    assert 0.0 <= metrics.station_utilization <= 1.0


def test_station_metrics_collector_records_queue_and_utilization_per_station():
    simulator = Simulator(SMALL_CONFIG, seed=1)
    station_ids = list(simulator.stations)
    simulator.stations[station_ids[0]].queue.extend([101, 102])
    simulator.stations[station_ids[0]].charging_vehicle_ids = {201}
    simulator.stations[station_ids[1]].queue.extend([301, 302, 303])

    collector = StationMetricsCollector()
    collector.record(simulator)

    queue_lengths = sorted(collector.queue_length_samples)
    assert queue_lengths == sorted(len(s.queue) for s in simulator.stations.values())
    expected_utilization = len(simulator.stations[station_ids[0]].charging_vehicle_ids) / simulator.stations[
        station_ids[0]
    ].num_chargers
    assert expected_utilization in collector.utilization_samples
    assert collector.maximum_queue_length == 3
    assert collector.average_queue_length == pytest.approx(sum(queue_lengths) / len(queue_lengths))


def test_station_metrics_collector_empty_before_any_record():
    collector = StationMetricsCollector()

    assert collector.average_queue_length == 0.0
    assert collector.maximum_queue_length == 0.0
    assert collector.station_utilization == 0.0


def test_episode_metrics_derived_properties():
    metrics = EpisodeMetrics(
        policy_name="test",
        seed=0,
        num_decisions=5,
        num_valid_decisions=4,
        num_invalid_actions=1,
        num_failed_vehicles=0,
        num_overloaded_events=0,
        total_travel_time=40.0,
        total_waiting_time=8.0,
        episode_reward=-48.0,
        terminated=True,
        truncated=False,
        simulation_time=100.0,
    )

    assert metrics.average_travel_time == pytest.approx(10.0)
    assert metrics.average_waiting_time == pytest.approx(2.0)
    assert metrics.total_system_cost == pytest.approx(48.0)


def test_episode_metrics_derived_properties_handle_zero_valid_decisions():
    metrics = EpisodeMetrics(
        policy_name="test",
        seed=0,
        num_decisions=1,
        num_valid_decisions=0,
        num_invalid_actions=1,
        num_failed_vehicles=0,
        num_overloaded_events=0,
        total_travel_time=0.0,
        total_waiting_time=0.0,
        episode_reward=-1000.0,
        terminated=False,
        truncated=True,
        simulation_time=5000.0,
    )

    assert metrics.average_travel_time == 0.0
    assert metrics.average_waiting_time == 0.0
