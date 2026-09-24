from dataclasses import replace

from backend.baseline.nearest_station import choose_station
from backend.config import DEFAULT_CONFIG
from backend.simulation.network_graph import shortest_path
from backend.simulation.simulator import Simulator

SMALL_CONFIG = replace(DEFAULT_CONFIG, num_stations=3, num_vehicles=5)


def test_choose_station_picks_minimum_distance():
    simulator = Simulator(SMALL_CONFIG, seed=1)
    vehicle_id = next(iter(simulator.vehicles))

    chosen = choose_station(simulator, vehicle_id)

    vehicle = simulator.vehicles[vehicle_id]
    distances = {
        station_id: shortest_path(
            simulator.graph, vehicle.current_node, station.node_id, weight=SMALL_CONFIG.routing_weight
        )[1]
        for station_id, station in simulator.stations.items()
    }
    assert distances[chosen] == min(distances.values())


def test_choose_station_ignores_reachability_by_design():
    simulator = Simulator(SMALL_CONFIG, seed=2)
    vehicle_id = next(iter(simulator.vehicles))
    vehicle = simulator.vehicles[vehicle_id]
    vehicle.battery_level = 0.0  # cannot safely reach any station

    chosen = choose_station(simulator, vehicle_id)

    vehicle_dist = {
        station_id: shortest_path(
            simulator.graph, vehicle.current_node, station.node_id, weight=SMALL_CONFIG.routing_weight
        )[1]
        for station_id, station in simulator.stations.items()
    }
    assert chosen == min(vehicle_dist, key=vehicle_dist.get)
    assert not simulator.is_station_reachable(vehicle_id, chosen)
