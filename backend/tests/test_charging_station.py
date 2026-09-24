from backend.simulation.charging_station import ChargingStation


def _make_station(**overrides) -> ChargingStation:
    defaults = dict(station_id=0, node_id=5, capacity=5, num_chargers=2)
    defaults.update(overrides)
    return ChargingStation(**defaults)


def test_enqueue_adds_to_queue_not_charging():
    station = _make_station()

    station.enqueue(1)

    assert list(station.queue) == [1]
    assert station.charging_vehicle_ids == set()


def test_admit_from_queue_fills_free_chargers_fifo():
    station = _make_station(num_chargers=2)
    for vehicle_id in (1, 2, 3):
        station.enqueue(vehicle_id)

    admitted = station.admit_from_queue()

    assert admitted == [1, 2]
    assert station.charging_vehicle_ids == {1, 2}
    assert list(station.queue) == [3]


def test_admit_from_queue_no_op_when_chargers_full():
    station = _make_station(num_chargers=1)
    station.enqueue(1)
    station.enqueue(2)
    station.admit_from_queue()

    admitted_second_call = station.admit_from_queue()

    assert admitted_second_call == []
    assert station.charging_vehicle_ids == {1}
    assert list(station.queue) == [2]


def test_release_frees_charger_slot():
    station = _make_station(num_chargers=1)
    station.enqueue(1)
    station.admit_from_queue()

    station.release(1)

    assert station.charging_vehicle_ids == set()


def test_occupancy_counts_queue_and_charging():
    station = _make_station(num_chargers=2)
    station.enqueue(1)
    station.enqueue(2)
    station.enqueue(3)
    station.admit_from_queue()

    assert station.occupancy == 3


def test_is_overloaded_false_when_within_capacity():
    station = _make_station(capacity=5, num_chargers=2)
    station.enqueue(1)
    station.enqueue(2)
    station.admit_from_queue()

    assert station.is_overloaded() is False


def test_is_overloaded_true_when_queue_plus_charging_exceeds_capacity():
    station = _make_station(capacity=2, num_chargers=1)
    station.enqueue(1)
    station.admit_from_queue()
    station.enqueue(2)
    station.enqueue(3)

    assert station.occupancy == 3
    assert station.is_overloaded() is True
