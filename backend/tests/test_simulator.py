from dataclasses import replace

import pytest

from backend.config import DEFAULT_CONFIG
from backend.simulation.simulator import Simulator
from backend.simulation.traffic_logic import vehicle_effective_speed
from backend.simulation.vehicle import Vehicle, VehicleState

SMALL_CONFIG = replace(
    DEFAULT_CONFIG,
    num_stations=2,
    num_vehicles=4,
    battery_consumption_per_distance=0.001,
)


def test_simulator_builds_correct_scale():
    simulator = Simulator(SMALL_CONFIG, seed=1)

    assert len(simulator.stations) == SMALL_CONFIG.num_stations
    assert len(simulator.vehicles) == SMALL_CONFIG.num_vehicles


def test_simulator_initial_vehicles_are_well_formed():
    simulator = Simulator(SMALL_CONFIG, seed=1)

    for vehicle in simulator.vehicles.values():
        assert vehicle.state == VehicleState.TRAVELING
        assert vehicle.current_node != vehicle.destination_node
        assert vehicle.route[0] == vehicle.current_node
        assert vehicle.route[-1] == vehicle.destination_node
        assert SMALL_CONFIG.low_battery_threshold <= vehicle.battery_level <= SMALL_CONFIG.battery_capacity


def test_simulator_reproducible_with_same_seed():
    sim_a = Simulator(SMALL_CONFIG, seed=42)
    sim_b = Simulator(SMALL_CONFIG, seed=42)

    for vehicle_id in sim_a.vehicles:
        va, vb = sim_a.vehicles[vehicle_id], sim_b.vehicles[vehicle_id]
        assert va.current_node == vb.current_node
        assert va.destination_node == vb.destination_node
        assert va.battery_level == vb.battery_level
        assert va.speed == vb.speed
        assert va.route == vb.route

    for station_id in sim_a.stations:
        assert sim_a.stations[station_id].node_id == sim_b.stations[station_id].node_id


def test_tick_advances_simulation_time():
    simulator = Simulator(SMALL_CONFIG, seed=1)

    simulator.tick()

    assert simulator.simulation_time == pytest.approx(SMALL_CONFIG.time_step)


def test_tick_moves_vehicle_and_consumes_battery_along_edge():
    simulator = Simulator(SMALL_CONFIG, seed=3)
    u, v = list(simulator.graph.edges())[0]
    edge = simulator.graph.edges[u, v]

    vehicle = Vehicle(
        vehicle_id=0,
        current_node=u,
        destination_node=v,
        battery_level=1.0,
        battery_capacity=SMALL_CONFIG.battery_capacity,
        speed=SMALL_CONFIG.max_speed,
        route=[u, v],
    )
    simulator.vehicles = {0: vehicle}

    simulator.tick()

    effective_speed = vehicle_effective_speed(simulator.graph, u, v, vehicle.speed)
    expected_move = min(effective_speed * SMALL_CONFIG.time_step, edge["distance"])
    expected_battery = 1.0 - SMALL_CONFIG.battery_consumption_per_distance * expected_move

    assert vehicle.battery_level == pytest.approx(expected_battery)
    if expected_move >= edge["distance"] - 1e-9:
        assert vehicle.current_node == v
    else:
        assert vehicle.edge_progress == pytest.approx(expected_move)


def test_vehicle_reaches_destination_and_completes_without_station():
    simulator = Simulator(SMALL_CONFIG, seed=5)
    u, v = min(simulator.graph.edges(), key=lambda e: simulator.graph.edges[e]["distance"])

    vehicle = Vehicle(
        vehicle_id=0,
        current_node=u,
        destination_node=v,
        battery_level=1.0,
        battery_capacity=SMALL_CONFIG.battery_capacity,
        speed=SMALL_CONFIG.max_speed,
        route=[u, v],
    )
    simulator.vehicles = {0: vehicle}

    for _ in range(2000):
        if vehicle.state != VehicleState.TRAVELING:
            break
        simulator.tick()

    assert vehicle.state == VehicleState.COMPLETED
    assert vehicle.current_node == v


def test_needs_decision_vehicle_is_blocked_until_station_assigned():
    simulator = Simulator(SMALL_CONFIG, seed=7)
    vehicle = next(iter(simulator.vehicles.values()))
    vehicle.battery_level = SMALL_CONFIG.low_battery_threshold
    vehicle.target_station = None
    route_before = list(vehicle.route)
    node_before = vehicle.current_node

    assert vehicle in simulator.get_vehicles_needing_decision()

    simulator.tick()

    assert vehicle.route == route_before
    assert vehicle.current_node == node_before

    station_id = next(iter(simulator.stations))
    simulator.assign_station(vehicle.vehicle_id, station_id)

    assert vehicle.target_station == station_id
    assert vehicle not in simulator.get_vehicles_needing_decision()
    assert vehicle.route[0] == vehicle.current_node
    assert vehicle.route[-1] == simulator.stations[station_id].node_id


def test_assigned_vehicle_accumulates_travel_time_since_assignment():
    simulator = Simulator(SMALL_CONFIG, seed=9)
    vehicle = next(iter(simulator.vehicles.values()))
    station_id = next(iter(simulator.stations))
    simulator.assign_station(vehicle.vehicle_id, station_id)

    simulator.tick()

    assert vehicle.time_since_station_assigned == pytest.approx(SMALL_CONFIG.time_step)


def test_vehicle_enters_charging_after_arriving_and_queue_admits_fifo():
    simulator = Simulator(SMALL_CONFIG, seed=11)
    station = simulator.stations[0]
    station.num_chargers = 1

    vehicle_a = Vehicle(
        vehicle_id=100,
        current_node=station.node_id,
        destination_node=station.node_id,
        battery_level=SMALL_CONFIG.battery_capacity - SMALL_CONFIG.charging_rate,
        battery_capacity=SMALL_CONFIG.battery_capacity,
        speed=SMALL_CONFIG.max_speed,
        target_station=0,
        state=VehicleState.CHARGING,
        route=[station.node_id],
    )
    vehicle_b = Vehicle(
        vehicle_id=101,
        current_node=station.node_id,
        destination_node=station.node_id,
        battery_level=1.0,
        battery_capacity=SMALL_CONFIG.battery_capacity,
        speed=SMALL_CONFIG.max_speed,
        target_station=0,
        state=VehicleState.WAITING,
        route=[station.node_id],
    )
    simulator.vehicles = {100: vehicle_a, 101: vehicle_b}
    station.charging_vehicle_ids = {100}
    station.queue.clear()
    station.queue.append(101)

    simulator.tick()

    # Finishing charging no longer completes the trip immediately -- the EV
    # is routed to its nearest depot first (Task 3 of the OSM/depot
    # upgrade) and only becomes COMPLETED once it physically arrives there.
    assert vehicle_a.state == VehicleState.RETURNING_TO_DEPOT
    assert vehicle_a.target_station is None
    assert vehicle_a.target_depot is not None
    assert vehicle_a.route[0] == station.node_id
    assert vehicle_a.route[-1] == simulator.depot_nodes[vehicle_a.target_depot]
    assert vehicle_b.state == VehicleState.CHARGING
    assert station.charging_vehicle_ids == {101}
    assert list(station.queue) == []

    for _ in range(2000):
        if vehicle_a.state != VehicleState.RETURNING_TO_DEPOT:
            break
        simulator.tick()

    assert vehicle_a.state == VehicleState.COMPLETED
    assert vehicle_a.target_depot is None


def test_waiting_vehicle_accumulates_waiting_time_while_queued():
    simulator = Simulator(SMALL_CONFIG, seed=13)
    station = simulator.stations[0]
    station.num_chargers = 0

    vehicle = Vehicle(
        vehicle_id=200,
        current_node=station.node_id,
        destination_node=station.node_id,
        battery_level=1.0,
        battery_capacity=SMALL_CONFIG.battery_capacity,
        speed=SMALL_CONFIG.max_speed,
        target_station=0,
        state=VehicleState.WAITING,
        route=[station.node_id],
    )
    simulator.vehicles = {200: vehicle}
    station.queue.append(200)

    simulator.tick()
    simulator.tick()

    assert vehicle.waiting_time == pytest.approx(2 * SMALL_CONFIG.time_step)
    assert vehicle.state == VehicleState.WAITING


def test_vehicle_fails_when_battery_depletes_mid_edge():
    simulator = Simulator(SMALL_CONFIG, seed=19)
    u, v = max(simulator.graph.edges(), key=lambda e: simulator.graph.edges[e]["distance"])

    vehicle = Vehicle(
        vehicle_id=300,
        current_node=u,
        destination_node=v,
        battery_level=0.05,
        battery_capacity=SMALL_CONFIG.battery_capacity,
        speed=SMALL_CONFIG.max_speed,
        route=[u, v],
        target_station=999,  # bypass needs_charging_decision blocking for this test
    )
    config = replace(SMALL_CONFIG, battery_consumption_per_distance=1.0)
    simulator.config = config
    simulator.vehicles = {300: vehicle}

    simulator.tick()

    assert vehicle.state == VehicleState.FAILED
    assert vehicle.battery_level == 0.0


def test_is_station_reachable_true_when_battery_exceeds_required_energy():
    simulator = Simulator(SMALL_CONFIG, seed=23)
    vehicle = next(iter(simulator.vehicles.values()))
    vehicle.battery_level = 1.0
    station_id = next(iter(simulator.stations))

    assert simulator.is_station_reachable(vehicle.vehicle_id, station_id) is True


def test_is_station_reachable_false_when_battery_insufficient():
    simulator = Simulator(SMALL_CONFIG, seed=23)
    vehicle = next(iter(simulator.vehicles.values()))
    vehicle.battery_level = 0.0
    station_id = next(iter(simulator.stations))
    station = simulator.stations[station_id]
    if vehicle.current_node == station.node_id:
        vehicle.current_node = next(n for n in simulator.graph.nodes() if n != station.node_id)

    assert simulator.is_station_reachable(vehicle.vehicle_id, station_id) is False


def test_assign_station_raises_when_unreachable():
    simulator = Simulator(SMALL_CONFIG, seed=23)
    vehicle = next(iter(simulator.vehicles.values()))
    vehicle.battery_level = 0.0
    station_id = next(iter(simulator.stations))
    station = simulator.stations[station_id]
    if vehicle.current_node == station.node_id:
        vehicle.current_node = next(n for n in simulator.graph.nodes() if n != station.node_id)

    with pytest.raises(ValueError):
        simulator.assign_station(vehicle.vehicle_id, station_id)


def test_vehicle_assigned_station_already_at_current_node_arrives_immediately():
    simulator = Simulator(SMALL_CONFIG, seed=29)
    station_id = next(iter(simulator.stations))
    station = simulator.stations[station_id]

    vehicle = Vehicle(
        vehicle_id=400,
        current_node=station.node_id,
        destination_node=station.node_id,
        battery_level=1.0,
        battery_capacity=SMALL_CONFIG.battery_capacity,
        speed=SMALL_CONFIG.max_speed,
        route=[station.node_id, station.node_id],  # arbitrary pre-assign route
        target_station=None,
    )
    simulator.vehicles = {400: vehicle}
    simulator.assign_station(400, station_id)
    assert vehicle.route == [station.node_id]

    simulator.tick()

    assert vehicle.state in (VehicleState.WAITING, VehicleState.CHARGING)
    assert 400 in station.queue or 400 in station.charging_vehicle_ids


def test_all_vehicles_done_reflects_completion_state():
    simulator = Simulator(SMALL_CONFIG, seed=17)

    assert simulator.all_vehicles_done() is False

    for vehicle in simulator.vehicles.values():
        vehicle.state = VehicleState.COMPLETED

    assert simulator.all_vehicles_done() is True
