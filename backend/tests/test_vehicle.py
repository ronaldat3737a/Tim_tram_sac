from dataclasses import replace

from backend.config import DEFAULT_CONFIG
from backend.simulation.vehicle import Vehicle, VehicleState, needs_charging_decision


def _make_vehicle(**overrides) -> Vehicle:
    defaults = dict(
        vehicle_id=0,
        current_node=0,
        destination_node=1,
        battery_level=1.0,
        battery_capacity=1.0,
        speed=1.0,
        route=[0, 1],
    )
    defaults.update(overrides)
    return Vehicle(**defaults)


def test_set_battery_level_clamps_upper_bound():
    vehicle = _make_vehicle(battery_capacity=1.0)

    vehicle.set_battery_level(1.5)

    assert vehicle.battery_level == 1.0


def test_set_battery_level_clamps_lower_bound():
    vehicle = _make_vehicle()

    vehicle.set_battery_level(-0.2)

    assert vehicle.battery_level == 0.0


def test_set_battery_level_accepts_value_within_range():
    vehicle = _make_vehicle()

    vehicle.set_battery_level(0.42)

    assert vehicle.battery_level == 0.42


def test_needs_charging_decision_true_when_low_battery_and_no_station():
    config = replace(DEFAULT_CONFIG, low_battery_threshold=0.3)
    vehicle = _make_vehicle(battery_level=0.3, state=VehicleState.TRAVELING, target_station=None)

    assert needs_charging_decision(vehicle, config) is True


def test_needs_charging_decision_false_when_battery_above_threshold():
    config = replace(DEFAULT_CONFIG, low_battery_threshold=0.3)
    vehicle = _make_vehicle(battery_level=0.31, state=VehicleState.TRAVELING, target_station=None)

    assert needs_charging_decision(vehicle, config) is False


def test_needs_charging_decision_false_when_station_already_assigned():
    config = replace(DEFAULT_CONFIG, low_battery_threshold=0.3)
    vehicle = _make_vehicle(battery_level=0.1, state=VehicleState.TRAVELING, target_station=2)

    assert needs_charging_decision(vehicle, config) is False


def test_needs_charging_decision_false_when_not_traveling():
    config = replace(DEFAULT_CONFIG, low_battery_threshold=0.3)
    vehicle = _make_vehicle(battery_level=0.1, state=VehicleState.WAITING, target_station=None)

    assert needs_charging_decision(vehicle, config) is False
