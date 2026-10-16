"""Graph observations for MAPPO-GNN, on top of the unchanged MultiAgentEVEnv.

GraphMultiAgentEVEnv is a subclass: the simulator, dispatch, reward, decision
window and flat observation space of MultiAgentEVEnv are inherited as is, so
MAPPO-MLP and the heuristics keep running on exactly the same environment.
It only adds graph views of the same state, as torch_geometric `Data`:

- get_graph_observation(ev_id): one EV's local view (actor input). Nodes are
  the charging stations, in action order (node i is action i), each with
  features relative to that EV; the EV's own features ride along as the
  graph-level attribute `u`.
- get_global_graph(): the station load graph (centralized critic input),
  independent of any EV, with the episode's time progress as `u`.

Edges connect every station to its k nearest stations by real road distance
(directed j -> i for j among i's k nearest), weighted by that distance. A
fixed k keeps every node's neighbourhood the same size at any scale, unlike a
metric radius (whose meaning changes with station density) or a complete
graph (N^2 edges, and over-smoothed features).

Every feature is expressed in physical units that do not depend on the map's
size or the number of stations -- distance as a fraction of a full charge,
times as a number of average charging sessions, loads as a fraction of a
station's capacity -- so a model trained on one scenario reads the same
numbers the same way on a larger one (zero-shot scale).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch_geometric.data import Data

from backend.ai_core.marl_env import MultiAgentEVEnv
from backend.config import DEFAULT_CONFIG, SimulationConfig
from backend.simulation.network_graph import shortest_path
from backend.simulation.traffic_logic import vehicle_effective_speed

# Local (EV-relative) features of every station node, in this order.
NODE_FEATURES = (
    "distance",  # road distance, as a fraction of a full charge's range
    "travel_time",  # this EV's expected drive time, in average charging sessions
    "path_traffic",  # mean traffic weight along the route, 0..1
    "queue",  # queued EVs / capacity
    "available_capacity",  # free slots (capacity - queued - charging - incoming) / capacity
    "incoming",  # EVs already driving there / capacity
    "expected_wait",  # EVs ahead per charger, in average charging sessions
    "reachable",  # 1 if this EV's battery can reach it, else 0
)
# The EV's own features (graph attribute `u` of the local graph).
EGO_FEATURES = ("battery", "time_progress")
# Station load features of the global graph (critic).
GLOBAL_NODE_FEATURES = ("queue", "available_capacity", "incoming", "charging", "expected_wait")
# Graph attribute `u` of the global graph.
GLOBAL_FEATURES = ("time_progress",)
EDGE_FEATURES = ("distance",)  # station-to-station road distance, fraction of a full charge

NODE_DIM = len(NODE_FEATURES)
EGO_DIM = len(EGO_FEATURES)
GLOBAL_NODE_DIM = len(GLOBAL_NODE_FEATURES)
GLOBAL_DIM = len(GLOBAL_FEATURES)
EDGE_DIM = len(EDGE_FEATURES)

DEFAULT_NEIGHBORS = 4
# Clip for the unbounded "in sessions" / "in charges" features.
_MAX_UNITS = 10.0
_MAX_LOAD = 2.0


class GraphMultiAgentEVEnv(MultiAgentEVEnv):
    """MultiAgentEVEnv plus graph observations (see module docstring)."""

    def __init__(
        self,
        config: SimulationConfig = DEFAULT_CONFIG,
        tick_hook: Callable | None = None,
        decision_window: int = 15,
        num_neighbors: int = DEFAULT_NEIGHBORS,
    ):
        super().__init__(config, tick_hook=tick_hook, decision_window=decision_window)
        if num_neighbors < 1:
            raise ValueError(f"num_neighbors must be >= 1, got {num_neighbors}")
        self.num_neighbors = num_neighbors
        # Station-to-station edges, fixed for an episode (stations never
        # move); built by reset().
        self._edge_index: torch.Tensor | None = None
        self._edge_attr: torch.Tensor | None = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        result = super().reset(seed=seed, options=options)
        self._edge_index, self._edge_attr = self._build_station_edges()
        return result

    # --- graph observations ------------------------------------------------

    def get_graph_observation(self, ev_id: int) -> Data:
        """One EV's local graph: x (num_stations, NODE_DIM), the shared
        station edges, and u (1, EGO_DIM) for the EV itself."""
        assert self._context is not None, "call reset() before get_graph_observation()"
        simulator, config = self.simulator, self.config
        graph = simulator.graph
        vehicle = simulator.vehicles[ev_id]
        charge_time = max(self._context.expected_charge_time, 1e-6)

        rows = []
        for station_id in self._context.station_ids:
            station = simulator.stations[station_id]
            path, distance, _ = shortest_path(graph, vehicle.current_node, station.node_id, weight=config.routing_weight)
            # The same drive-time estimate the reward charges this EV for.
            travel_time = sum(
                graph.edges[u, v]["distance"] / vehicle_effective_speed(graph, u, v, vehicle.speed)
                for u, v in zip(path, path[1:])
            )
            traffic = (
                sum(graph.edges[u, v]["traffic_weight"] for u, v in zip(path, path[1:])) / (len(path) - 1)
                if len(path) > 1
                else 0.0
            )
            energy = config.battery_consumption_per_distance * distance
            queue, available, incoming, _, expected_wait = self._load_features(station, charge_time)
            rows.append(
                [
                    min(energy / vehicle.battery_capacity, _MAX_LOAD),
                    min(travel_time / charge_time, _MAX_UNITS),
                    min(traffic / max(config.max_traffic_weight, 1e-6), 1.0),
                    queue,
                    available,
                    incoming,
                    expected_wait,
                    1.0 if vehicle.battery_level > energy else 0.0,
                ]
            )
        ego = [vehicle.battery_level / vehicle.battery_capacity, self._time_progress()]
        return self._graph(rows, ego)

    def get_global_graph(self) -> Data:
        """The station load graph for the centralized critic: x
        (num_stations, GLOBAL_NODE_DIM), the shared station edges, and u
        (1, GLOBAL_DIM)."""
        assert self._context is not None, "call reset() before get_global_graph()"
        charge_time = max(self._context.expected_charge_time, 1e-6)
        rows = [
            self._load_features(self.simulator.stations[station_id], charge_time)
            for station_id in self._context.station_ids
        ]
        return self._graph(rows, [self._time_progress()])

    def decide_sequentially_graph(self, choose: Callable[[Data], int]) -> tuple[dict[int, int], list[Data]]:
        """Graph counterpart of MultiAgentEVEnv.decide_sequentially: every
        agent of the current batch picks in dispatch order from its graph
        observation, which counts the earlier agents of the batch as already
        driving to the stations they picked. Reservations are undone before
        returning; pass the actions to step(). Returns (actions, the graph
        each agent chose from, in self.agents order)."""
        assert self._context is not None, "call reset() before decide_sequentially_graph()"
        actions: dict[int, int] = {}
        graphs: list[Data] = []
        reserved = []
        try:
            for agent in self.agents:
                graph = self.get_graph_observation(agent)
                action = int(choose(graph))
                if not self.action_space(agent).contains(action):
                    raise ValueError(f"action {action} for agent {agent} outside {self.action_space(agent)}")
                station = self.simulator.stations[self.station_id_for_action(action)]
                station.incoming_count += 1
                reserved.append(station)
                actions[agent] = action
                graphs.append(graph)
        finally:
            for station in reserved:
                station.incoming_count -= 1
        return actions, graphs

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _load_features(station, charge_time: float) -> list[float]:
        """queue, available capacity, incoming, charging (per charger) and
        expected wait (EVs ahead per charger, in charging sessions)."""
        capacity = max(station.capacity, 1)
        chargers = max(station.num_chargers, 1)
        charging = len(station.charging_vehicle_ids)
        evs_ahead = len(station.queue) + station.incoming_count + charging
        return [
            min(len(station.queue) / capacity, _MAX_LOAD),
            min(max(station.capacity - station.occupancy, 0) / capacity, 1.0),
            min(station.incoming_count / capacity, _MAX_LOAD),
            min(charging / chargers, 1.0),
            min(evs_ahead / chargers, _MAX_UNITS),
        ]

    def _time_progress(self) -> float:
        return min(self.simulator.simulation_time / max(self.config.max_episode_steps, 1), 1.0)

    def _graph(self, rows: list[list[float]], u: list[float]) -> Data:
        x = torch.nan_to_num(torch.tensor(rows, dtype=torch.float32))
        return Data(
            x=x,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
            u=torch.tensor([u], dtype=torch.float32),
            num_nodes=x.shape[0],
        )

    def _build_station_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Directed edges j -> i from each station i's k nearest stations
        by real road distance, with that distance (fraction of a full
        charge) as the edge feature."""
        graph = self.simulator.graph
        nodes = [self.simulator.stations[sid].node_id for sid in self._context.station_ids]
        n = len(nodes)
        k = min(self.num_neighbors, n - 1)
        if k <= 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, EDGE_DIM), dtype=torch.float32)
        distance = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance[i, j] = shortest_path(graph, nodes[j], nodes[i], weight="distance")[1]
        sources, targets, weights = [], [], []
        scale = self.config.battery_consumption_per_distance / self.config.battery_capacity
        for i in range(n):
            order = [j for j in np.argsort(distance[i], kind="stable") if j != i][:k]
            for j in order:
                sources.append(j)
                targets.append(i)
                weights.append([min(distance[i, j] * scale, _MAX_LOAD)])
        return (
            torch.tensor([sources, targets], dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )
