from dataclasses import replace

import networkx as nx

from backend.baseline.shortest_time import choose_station
from backend.config import DEFAULT_CONFIG
from backend.simulation.charging_station import ChargingStation
from backend.simulation.simulator import Simulator
from backend.simulation.vehicle import Vehicle

SMALL_CONFIG = replace(DEFAULT_CONFIG, num_stations=2, num_vehicles=1)


def _graph_where_distance_and_time_rankings_diverge() -> nx.Graph:
    graph = nx.Graph()
    # Node 1 is nearer by distance but slower (heavy traffic); node 2 is
    # farther by distance but faster overall -- a real ranking inversion.
    graph.add_edge(0, 1, distance=10.0, travel_time=50.0)
    graph.add_edge(0, 2, distance=20.0, travel_time=15.0)
    graph.graph["routing_weight"] = "travel_time"
    return graph


def test_choose_station_picks_minimum_travel_time_not_minimum_distance():
    simulator = Simulator(SMALL_CONFIG, seed=1)
    simulator.graph = _graph_where_distance_and_time_rankings_diverge()
    simulator.stations = {
        0: ChargingStation(station_id=0, node_id=1, capacity=5, num_chargers=2),
        1: ChargingStation(station_id=1, node_id=2, capacity=5, num_chargers=2),
    }
    vehicle = Vehicle(
        vehicle_id=0,
        current_node=0,
        destination_node=2,
        battery_level=1.0,
        battery_capacity=1.0,
        speed=1.0,
        route=[0, 2],
    )
    simulator.vehicles = {0: vehicle}

    chosen = choose_station(simulator, 0)

    assert chosen == 1  # station at node 2: slower distance, faster travel_time
