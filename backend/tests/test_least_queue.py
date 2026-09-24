from dataclasses import replace

from backend.baseline.least_queue import choose_station
from backend.config import DEFAULT_CONFIG
from backend.simulation.simulator import Simulator

SMALL_CONFIG = replace(DEFAULT_CONFIG, num_stations=3, num_vehicles=5)


def test_choose_station_picks_minimum_queue_among_reachable():
    simulator = Simulator(SMALL_CONFIG, seed=1)
    vehicle_id = next(iter(simulator.vehicles))
    simulator.vehicles[vehicle_id].battery_level = 1.0  # every station reachable

    station_ids = list(simulator.stations)
    simulator.stations[station_ids[0]].queue.extend([101, 102, 103])
    simulator.stations[station_ids[1]].queue.extend([201])
    # station_ids[2] left with an empty queue.

    chosen = choose_station(simulator, vehicle_id)

    assert chosen == station_ids[2]


def test_choose_station_filters_by_reachability_even_if_queue_is_shorter():
    # Exactly two stations so there is no third, untouched station that
    # could otherwise win on an empty default queue.
    two_station_config = replace(SMALL_CONFIG, num_stations=2)
    simulator = Simulator(two_station_config, seed=3)
    vehicle_id = next(iter(simulator.vehicles))
    vehicle = simulator.vehicles[vehicle_id]

    station_ids = list(simulator.stations)
    near_station, far_station = station_ids[0], station_ids[1]
    # Force a clear near/far split by energy_required, then starve the
    # near (reachable) station's battery margin and give the far
    # (unreachable) one the shortest queue.
    energy_near = simulator.energy_required(vehicle_id, near_station)
    energy_far = simulator.energy_required(vehicle_id, far_station)
    if energy_near > energy_far:
        near_station, far_station = far_station, near_station
        energy_near, energy_far = energy_far, energy_near
    vehicle.battery_level = energy_near + 0.01
    assert simulator.is_station_reachable(vehicle_id, near_station)
    assert not simulator.is_station_reachable(vehicle_id, far_station)

    simulator.stations[far_station].queue.clear()  # shortest queue, but unreachable
    simulator.stations[near_station].queue.extend([301, 302])  # longer queue, reachable

    chosen = choose_station(simulator, vehicle_id)

    assert chosen == near_station


def test_choose_station_falls_back_to_all_stations_when_none_reachable():
    simulator = Simulator(SMALL_CONFIG, seed=5)
    vehicle_id = next(iter(simulator.vehicles))
    simulator.vehicles[vehicle_id].battery_level = 0.0

    chosen = choose_station(simulator, vehicle_id)

    assert chosen in simulator.stations
