"""Pydantic schemas for REST responses and WebSocket messages (section 35, 42)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Algorithm = Literal["nearest_station", "shortest_time", "least_queue", "dqn", "mappo"]
SimulationStatus = Literal["stopped", "running", "paused"]
# A key of backend/config.SCENARIOS, validated by the SimulationManager.
ScenarioName = str


# --- Static network data (sent once via REST, not per tick -- section 34) ---


class NodeInfo(BaseModel):
    id: int
    x: float
    y: float
    is_station: bool


class EdgeInfo(BaseModel):
    source: int
    target: int
    distance: float
    travel_time: float
    speed: float
    traffic_weight: float
    # Detailed [[lat, lng], ...] polyline following this edge's real street
    # curve (section 11) -- what vehicles actually move along and what the
    # frontend renders, always populated (never a straight-line fallback).
    geometry: list[list[float]]
    # OSM road class (network_graph.HIGHWAY_CLASSES), or "spur" for a
    # station/depot connector. Display styling only.
    highway: str | None = None


class RoadInfo(BaseModel):
    """A street of the display-only road layer (vehicles never drive on
    these unless they are also in `edges`)."""

    highway: str
    geometry: list[list[float]]


class StationInfo(BaseModel):
    id: int
    # The station's own POI node id -- both the routing target and its
    # (x, y) display position (see NodeInfo for that node's coordinates).
    # Not a real traffic node itself: connects to one via a short spur edge.
    node_id: int
    # The real traffic node that spur edge connects to.
    access_node_id: int
    capacity: int
    num_chargers: int


class DepotInfo(BaseModel):
    id: int
    node_id: int
    access_node_id: int


class NetworkInfo(BaseModel):
    # The scenario this network belongs to (changes on a scenario switch).
    scenario: ScenarioName
    nodes: list[NodeInfo]
    edges: list[EdgeInfo]
    stations: list[StationInfo]
    depots: list[DepotInfo]
    roads: list[RoadInfo] = []


# --- Dynamic simulation data (sent every tick via WebSocket -- section 35) ---


class VehicleUpdate(BaseModel):
    id: int
    node: int
    x: float
    y: float
    battery: float
    state: str
    # station_id/eta_seconds are populated only while state == "CHARGING", so
    # the frontend side panel can show per-charger battery% + ETA without
    # ever computing charging physics itself (section 39).
    station_id: int | None = None
    eta_seconds: float | None = None
    # Populated only while state == "RETURNING_TO_DEPOT".
    depot_id: int | None = None
    # A non-app EV (exogenous noise) the agent never dispatches.
    is_non_app: bool = False


class StationUpdate(BaseModel):
    id: int
    queue: int
    charging: int


class SimulationUpdateMessage(BaseModel):
    type: Literal["simulation_update"] = "simulation_update"
    timestamp: float
    vehicles: list[VehicleUpdate]
    stations: list[StationUpdate]


# --- Simulation control (REST) ---


class StartSimulationRequest(BaseModel):
    seed: int | None = None
    algorithm: Algorithm = "nearest_station"
    speed: float = 5.0  # simulated ticks per real second
    # None keeps the active scenario; otherwise switch to it first.
    scenario: ScenarioName | None = None


class ScenarioRequest(BaseModel):
    scenario: ScenarioName


class ScenarioInfo(BaseModel):
    name: ScenarioName
    label: str
    num_stations: int
    num_vehicles: int
    num_depots: int
    radius_meters: float
    # Whether this scenario's trained model files exist on disk yet.
    has_mappo_model: bool
    has_dqn_model: bool


class SpeedRequest(BaseModel):
    speed: float


class SimulationStatusResponse(BaseModel):
    status: SimulationStatus
    scenario: ScenarioName
    simulation_time: float
    episode_seed: int
    algorithm: Algorithm
    speed: float
    num_vehicles_traveling: int
    num_vehicles_waiting: int
    num_vehicles_charging: int
    num_vehicles_returning_to_depot: int
    num_vehicles_completed: int
    num_vehicles_failed: int
    num_decisions: int
    num_invalid_actions: int
    num_overloaded_events: int
    average_travel_time: float
    average_waiting_time: float
    total_system_cost: float
    episode_reward: float
    my_vehicle_id: int | None = None


class MyCarPreviewResponse(BaseModel):
    vehicle_id: int
    station_id: int
    reachable: bool
    route: list[list[float]]  # [[lat, lng], ...] along the shortest path
