"""Simulation core: ties network, vehicles and stations into a tick loop."""

from __future__ import annotations

import networkx as nx
import numpy as np

from backend.config import SimulationConfig
from backend.simulation.charging_station import ChargingStation
from backend.simulation.network_graph import build_network, get_depot_nodes, get_station_nodes, shortest_path
from backend.simulation.traffic_logic import distance_covered, vehicle_effective_speed
from backend.simulation.vehicle import Vehicle, VehicleState, needs_charging_decision

_ARRIVAL_EPSILON = 1e-9


class Simulator:
    """Owns the traffic graph, vehicles and stations for one episode run."""

    def __init__(self, config: SimulationConfig, seed: int | None = None):
        self.config = config
        self.seed = seed if seed is not None else config.random_seed
        self.rng = np.random.default_rng(self.seed)

        self.graph: nx.Graph = build_network(config, seed=self.seed)
        self.stations: dict[int, ChargingStation] = self._build_stations()
        self.depot_nodes: list[int] = get_depot_nodes(self.graph)
        self.vehicles: dict[int, Vehicle] = self._build_vehicles()
        self.simulation_time: float = 0.0

    def tick(self) -> None:
        """Advance the simulation by one `config.time_step`."""
        self._move_vehicles()
        self._update_charging()
        self.simulation_time += self.config.time_step

    def energy_required(self, vehicle_id: int, station_id: int) -> float:
        """Battery needed to reach a station via the shortest path (section 20).
        Returns infinity if no path exists at all (defensive: the graph is
        built as a single connected component, so this should never trigger
        in practice, but a station must never be reported reachable, nor
        crash the simulation thread, if it somehow did)."""
        vehicle = self.vehicles[vehicle_id]
        station = self.stations[station_id]
        try:
            _, distance, _ = shortest_path(
                self.graph, vehicle.current_node, station.node_id, weight=self.config.routing_weight
            )
        except nx.NetworkXNoPath:
            return float("inf")
        return self.config.battery_consumption_per_distance * distance

    def is_station_reachable(self, vehicle_id: int, station_id: int) -> bool:
        """A station is reachable if current battery strictly exceeds the
        energy required to get there, leaving no exact-zero-on-arrival edge
        case that would otherwise collide with the battery-failure rule."""
        vehicle = self.vehicles[vehicle_id]
        return vehicle.battery_level > self.energy_required(vehicle_id, station_id)

    def assign_station(self, vehicle_id: int, station_id: int) -> None:
        """Assign a charging station to a vehicle and route it there."""
        if not self.is_station_reachable(vehicle_id, station_id):
            raise ValueError(
                f"station {station_id} is not reachable for vehicle {vehicle_id} "
                "with its current battery level"
            )
        vehicle = self.vehicles[vehicle_id]
        station = self.stations[station_id]
        try:
            path, _, _ = shortest_path(
                self.graph, vehicle.current_node, station.node_id, weight=self.config.routing_weight
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                f"station {station_id} is not reachable for vehicle {vehicle_id}: no path exists"
            ) from exc
        # Defensive: re-dispatching an EV already driving to another station
        # hands its incoming slot back there first.
        self._release_incoming(vehicle)
        vehicle.target_station = station_id
        vehicle.route = path
        vehicle.edge_progress = 0.0
        station.incoming_count += 1

    def fail_vehicle(self, vehicle: Vehicle) -> None:
        """Mark an EV FAILED, freeing its incoming slot if it was still
        driving to a station, so incoming_count never counts a dead EV."""
        self._release_incoming(vehicle)
        vehicle.state = VehicleState.FAILED

    def _release_incoming(self, vehicle: Vehicle) -> None:
        """Decrement incoming_count at the station this EV is driving to, if
        any. An EV counts as incoming exactly while it is TRAVELING with a
        target_station, so arrival and failure both end it."""
        if vehicle.state == VehicleState.TRAVELING and vehicle.target_station is not None:
            self.stations[vehicle.target_station].incoming_count -= 1

    def get_vehicles_needing_decision(self) -> list[Vehicle]:
        """EVs whose battery is low and have no charging station assigned yet."""
        return [
            vehicle
            for vehicle in self.vehicles.values()
            if self.is_active(vehicle) and needs_charging_decision(vehicle, self.config)
        ]

    def is_active(self, vehicle: Vehicle) -> bool:
        """False until the EV's staggered departure tick: it has not joined
        traffic yet."""
        return self.simulation_time >= vehicle.activation_tick

    def all_vehicles_done(self) -> bool:
        return all(
            vehicle.state in (VehicleState.COMPLETED, VehicleState.FAILED)
            for vehicle in self.vehicles.values()
        )

    def _build_stations(self) -> dict[int, ChargingStation]:
        stations = {}
        for station_id, node_id in enumerate(get_station_nodes(self.graph)):
            stations[station_id] = ChargingStation(
                station_id=station_id,
                node_id=node_id,
                capacity=self.config.station_capacity,
                num_chargers=self.config.num_chargers_per_station,
            )
        return stations

    def _build_vehicles(self) -> dict[int, Vehicle]:
        vehicles = {}
        # Station/depot POI nodes are not real traffic nodes (they only
        # exist to be routed *to*, via their spur edge) -- excluded here so
        # no EV ever spawns at, or is assigned a personal destination at,
        # a parking spot instead of a real intersection.
        node_ids = [
            n
            for n, data in self.graph.nodes(data=True)
            if not data["is_station"] and not data["is_depot"]
        ]
        for vehicle_id in range(self.config.num_vehicles):
            current_node = int(self.rng.choice(node_ids))
            destination_candidates = [n for n in node_ids if n != current_node]
            destination_node = int(self.rng.choice(destination_candidates))
            battery_level = float(
                self.rng.uniform(self.config.low_battery_threshold, self.config.battery_capacity)
            )
            speed = float(self.rng.uniform(self.config.min_speed, self.config.max_speed))
            # Drawn only when staggering is on, so scenarios with every EV
            # departing at tick 0 keep their exact random stream.
            activation_tick = (
                int(self.rng.integers(0, self.config.max_activation_tick + 1))
                if self.config.max_activation_tick > 0
                else 0
            )
            # Defensive: the graph is built as a single connected component,
            # so every node pair has a path in practice. If that ever fails
            # to hold, spawn this EV already at its destination (an
            # immediate, harmless COMPLETED) instead of crashing the
            # simulation thread.
            try:
                path, _, _ = shortest_path(
                    self.graph, current_node, destination_node, weight=self.config.routing_weight
                )
            except nx.NetworkXNoPath:
                destination_node = current_node
                path = [current_node]
            vehicles[vehicle_id] = Vehicle(
                vehicle_id=vehicle_id,
                current_node=current_node,
                destination_node=destination_node,
                battery_level=battery_level,
                battery_capacity=self.config.battery_capacity,
                speed=speed,
                route=path,
                activation_tick=activation_tick,
            )
        return vehicles

    def _move_vehicles(self) -> None:
        for vehicle in self.vehicles.values():
            if vehicle.state not in (VehicleState.TRAVELING, VehicleState.RETURNING_TO_DEPOT):
                continue
            if not self.is_active(vehicle):
                continue
            if needs_charging_decision(vehicle, self.config):
                # Held in place until it is dispatched. EVEnv never ticks
                # while a decision is pending, so in practice this lasts zero
                # ticks; if anything ever does tick past it, standing on the
                # road still costs battery (A/C, electronics) and can end in
                # FAILED, never in a free wait.
                self._drain_idle_battery(vehicle)
                continue
            if not vehicle.route:
                # Defensive: a vehicle on the road with no route at all could
                # never move again. Fail it instead of leaving it parked.
                self.fail_vehicle(vehicle)
                continue
            if len(vehicle.route) == 1:
                # Already standing at the route's target node (e.g. a
                # station assigned at the vehicle's current location):
                # resolve arrival immediately instead of never moving.
                self._handle_arrival(vehicle)
                continue

            if vehicle.target_station is not None:
                vehicle.time_since_station_assigned += self.config.time_step

            self._advance_vehicle_along_route(vehicle)

    def _drain_idle_battery(self, vehicle: Vehicle) -> None:
        vehicle.set_battery_level(vehicle.battery_level - self.config.idle_battery_drain_per_tick * self.config.time_step)
        if vehicle.battery_level <= 0.0:
            self.fail_vehicle(vehicle)

    def _advance_vehicle_along_route(self, vehicle: Vehicle) -> None:
        movement_budget = self._movement_budget(vehicle)

        while movement_budget > 0.0 and len(vehicle.route) >= 2:
            u, v = vehicle.route[0], vehicle.route[1]
            edge = self.graph.edges[u, v]
            remaining_edge_distance = edge["distance"] - vehicle.edge_progress
            move = min(movement_budget, remaining_edge_distance)

            vehicle.edge_progress += move
            vehicle.set_battery_level(
                vehicle.battery_level - self.config.battery_consumption_per_distance * move
            )
            movement_budget -= move

            if vehicle.battery_level <= 0.0:
                self.fail_vehicle(vehicle)
                return

            if vehicle.edge_progress >= edge["distance"] - _ARRIVAL_EPSILON:
                vehicle.current_node = v
                vehicle.edge_progress = 0.0
                vehicle.route.pop(0)

        if len(vehicle.route) == 1:
            self._handle_arrival(vehicle)

    def _movement_budget(self, vehicle: Vehicle) -> float:
        u, v = vehicle.route[0], vehicle.route[1]
        speed = vehicle_effective_speed(self.graph, u, v, vehicle.speed)
        return distance_covered(speed, self.config.time_step)

    def _handle_arrival(self, vehicle: Vehicle) -> None:
        if vehicle.target_station is not None:
            self._release_incoming(vehicle)
            vehicle.state = VehicleState.WAITING
            self.stations[vehicle.target_station].enqueue(vehicle.vehicle_id)
        else:
            # Covers both a trip that never needed charging at all, and a
            # RETURNING_TO_DEPOT vehicle reaching its depot node -- in
            # both cases target_station is already None and the vehicle is
            # simply done.
            vehicle.state = VehicleState.COMPLETED
            vehicle.target_depot = None

    def _route_to_depot(self, vehicle: Vehicle) -> None:
        """Once an EV finishes charging, send it to its nearest depot
        instead of marking it COMPLETED immediately -- it only actually
        completes once it physically arrives there, following the depot
        route's real road geometry exactly like any other trip (explicit
        user-requested extension: depot return is real simulated movement,
        not a frontend-only cosmetic)."""
        vehicle.target_station = None
        depot_index, path = self._nearest_depot(vehicle.current_node)
        if depot_index is None:
            # Defensive: no depot has a path from here at all (should never
            # happen -- the graph is one connected component -- but must
            # not crash the simulation thread if it somehow did). The EV
            # just completes its trip where it is instead of being stuck.
            vehicle.state = VehicleState.COMPLETED
            vehicle.target_depot = None
            return
        vehicle.target_depot = depot_index
        vehicle.state = VehicleState.RETURNING_TO_DEPOT
        vehicle.route = path
        vehicle.edge_progress = 0.0

    def _nearest_depot(self, node_id: int) -> tuple[int | None, list[int] | None]:
        """Returns (index into self.depot_nodes, path) for the closest
        depot -- an index, like target_station, rather than a raw node id,
        so the API/frontend can address depots the same way stations are.
        Returns (None, None) if no depot has a path from node_id at all."""
        best_index, best_path, best_cost = None, None, float("inf")
        for depot_index, depot_node in enumerate(self.depot_nodes):
            try:
                path, distance, travel_time = shortest_path(
                    self.graph, node_id, depot_node, weight=self.config.routing_weight
                )
            except nx.NetworkXNoPath:
                continue
            cost = distance if self.config.routing_weight == "distance" else travel_time
            if cost < best_cost:
                best_index, best_path, best_cost = depot_index, path, cost
        return best_index, best_path

    def _update_charging(self) -> None:
        for station in self.stations.values():
            for vehicle_id in list(station.charging_vehicle_ids):
                vehicle = self.vehicles[vehicle_id]
                vehicle.set_battery_level(
                    vehicle.battery_level + self.config.charging_rate * self.config.time_step
                )
                if vehicle.battery_level >= vehicle.battery_capacity - _ARRIVAL_EPSILON:
                    station.release(vehicle_id)
                    self._route_to_depot(vehicle)

            for vehicle_id in station.queue:
                self.vehicles[vehicle_id].waiting_time += self.config.time_step

            admitted = station.admit_from_queue()
            for vehicle_id in admitted:
                self.vehicles[vehicle_id].state = VehicleState.CHARGING


if __name__ == "__main__":
    from backend.config import DEFAULT_CONFIG

    simulator = Simulator(DEFAULT_CONFIG)
    for _ in range(200):
        simulator.tick()

    state_counts: dict[str, int] = {}
    for vehicle in simulator.vehicles.values():
        state_counts[vehicle.state.value] = state_counts.get(vehicle.state.value, 0) + 1

    print(f"simulation_time = {simulator.simulation_time}")
    print(f"vehicle states = {state_counts}")
    print(f"vehicles needing decision = {len(simulator.get_vehicles_needing_decision())}")
    for station in simulator.stations.values():
        print(
            f"station {station.station_id} (node {station.node_id}): "
            f"queue={len(station.queue)} charging={len(station.charging_vehicle_ids)} "
            f"overloaded={station.is_overloaded()}"
        )
