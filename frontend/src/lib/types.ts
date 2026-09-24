// Mirrors backend/api/schemas.py exactly (PROJECT_SPEC.md section 35, 42).
// The frontend never computes simulation/reward/routing values itself --
// every field here is populated verbatim from the backend API.

export type Algorithm =
  | "nearest_station"
  | "shortest_time"
  | "least_queue"
  | "dqn";

export type SimulationStatus = "stopped" | "running" | "paused";

export type VehicleState =
  | "TRAVELING"
  | "WAITING"
  | "CHARGING"
  | "RETURNING_TO_DEPOT"
  | "COMPLETED"
  | "FAILED";

// --- Static network data (REST, fetched once) ---

export interface NodeInfo {
  id: number;
  x: number;
  y: number;
  is_station: boolean;
}

export interface EdgeInfo {
  source: number;
  target: number;
  distance: number;
  travel_time: number;
  speed: number;
  traffic_weight: number;
  // Detailed [lat, lng] polyline following this edge's real street curve --
  // what vehicles actually move along and what the map renders.
  geometry: [number, number][];
}

export interface StationInfo {
  id: number;
  node_id: number;
  capacity: number;
  num_chargers: number;
}

export interface DepotInfo {
  id: number;
  node_id: number;
}

export interface NetworkInfo {
  nodes: NodeInfo[];
  edges: EdgeInfo[];
  stations: StationInfo[];
  depots: DepotInfo[];
}

// --- Dynamic simulation data (WebSocket, every tick) ---

export interface VehicleUpdate {
  id: number;
  node: number;
  x: number;
  y: number;
  battery: number;
  state: VehicleState;
  // Populated only while state is WAITING/CHARGING; eta_seconds only while
  // CHARGING. Computed backend-side (section 39: frontend never computes
  // simulation/charging physics itself).
  station_id: number | null;
  eta_seconds: number | null;
  // Populated only while state is RETURNING_TO_DEPOT.
  depot_id: number | null;
}

export interface StationUpdate {
  id: number;
  queue: number;
  charging: number;
}

export interface SimulationUpdateMessage {
  type: "simulation_update";
  timestamp: number;
  vehicles: VehicleUpdate[];
  stations: StationUpdate[];
}

// --- Simulation control (REST) ---

export interface StartSimulationRequest {
  seed?: number;
  algorithm: Algorithm;
  speed: number;
}

export interface SimulationStatusResponse {
  status: SimulationStatus;
  simulation_time: number;
  episode_seed: number;
  algorithm: Algorithm;
  speed: number;
  num_vehicles_traveling: number;
  num_vehicles_waiting: number;
  num_vehicles_charging: number;
  num_vehicles_returning_to_depot: number;
  num_vehicles_completed: number;
  num_vehicles_failed: number;
  num_decisions: number;
  num_invalid_actions: number;
  num_overloaded_events: number;
  average_travel_time: number;
  average_waiting_time: number;
  total_system_cost: number;
  episode_reward: number;
  my_vehicle_id: number | null;
}

export interface MyCarPreviewResponse {
  vehicle_id: number;
  station_id: number;
  reachable: boolean;
  route: [number, number][]; // [lat, lng] pairs along the shortest path
}
