// Client + types for the GNN Sandbox API (backend/api/routes/sandbox.py).
// `network` and `snapshot` reuse the simulation dashboard's own types.

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "@/lib/api";
import type { NetworkInfo, SimulationUpdateMessage } from "@/lib/types";

export interface GenerateEnvRequest {
  lat: number;
  lng: number;
  radius_km: number;
  num_stations: number;
  num_evs: number;
  num_rest_stops: number;
  seed?: number;
}

export interface SandboxInfo {
  sandbox_id: string;
  created_at: number;
  params: Required<GenerateEnvRequest>;
  map: {
    num_road_nodes: number;
    num_road_edges: number;
    bounds: [number, number, number, number]; // south, west, north, east
    from_cache: boolean;
    fetch_seconds: number;
    build_seconds: number;
  };
  battery: { full_range_km: number; request_range_km: number; farthest_from_station_km: number };
  num_vehicles_departed: number;
}

export interface GenerateEnvResponse {
  sandbox: SandboxInfo;
  network: NetworkInfo;
  snapshot: SimulationUpdateMessage;
}

export type SandboxRunStatus = "preparing" | "paused" | "running" | "finished" | "error" | "stopped";

export interface SandboxStatusMessage {
  type: "sandbox_status";
  sandbox_id: string;
  status: SandboxRunStatus;
  error: string | null;
  simulation_time: number;
  speed: number;
  preparing: { routes_done: number; routes_total: number };
  num_stations: number;
  num_vehicles_total: number;
  num_vehicles_departed: number;
  num_vehicles_traveling: number;
  num_vehicles_waiting: number;
  num_vehicles_charging: number;
  num_vehicles_returning_to_depot: number;
  num_vehicles_completed: number;
  num_vehicles_failed: number;
  num_decisions: number;
  num_invalid_actions: number;
  num_overloaded_events: number;
  num_stranded_vehicles: number;
  average_travel_time: number;
  average_waiting_time: number;
  total_system_cost: number;
  episode_reward: number;
  average_decision_time_ms: number;
}

export type SandboxControl =
  | { action: "start" | "pause" | "resume" | "reset" }
  | { action: "speed"; value: number };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = await response.text();
    try {
      const parsed = JSON.parse(detail).detail;
      detail = typeof parsed === "string" ? parsed : JSON.stringify(parsed);
    } catch {
      // not JSON: keep the raw text
    }
    throw new Error(`(${response.status}) ${detail}`);
  }
  return (await response.json()) as T;
}

export function generateSandbox(params: GenerateEnvRequest): Promise<GenerateEnvResponse> {
  return request<GenerateEnvResponse>("/api/sandbox/generate-env", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/** The active sandbox, or null if none has been generated yet. */
export async function fetchActiveSandbox(): Promise<GenerateEnvResponse | null> {
  const response = await fetch(`${API_BASE_URL}/api/sandbox/env`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`GET /api/sandbox/env failed (${response.status})`);
  return (await response.json()) as GenerateEnvResponse;
}

export type SandboxConnection = "connecting" | "open" | "closed";

/** Live MAPPO-GNN run of a sandbox over its WebSocket. Every connection is a
 *  fresh run from t=0 on the server, so this never reconnects on its own:
 *  `reconnect()` starts a new run deliberately. */
export function useSandboxSocket(sandboxId: string | null) {
  const [snapshot, setSnapshot] = useState<SimulationUpdateMessage | null>(null);
  const [status, setStatus] = useState<SandboxStatusMessage | null>(null);
  const [connection, setConnection] = useState<SandboxConnection>("connecting");
  const [lastError, setLastError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sandboxId) return;
    const url = `${API_BASE_URL.replace(/^http/, "ws")}/api/sandbox/ws/${sandboxId}`;
    const socket = new WebSocket(url);
    socketRef.current = socket;
    socket.onopen = () => setConnection("open");
    socket.onclose = () => setConnection("closed");
    socket.onerror = () => setLastError("Mất kết nối WebSocket tới backend.");
    socket.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === "simulation_update") setSnapshot(message as SimulationUpdateMessage);
      else if (message.type === "sandbox_status") setStatus(message as SandboxStatusMessage);
      else if (message.type === "sandbox_error") setLastError(message.message);
    };
    return () => {
      // Detach first: closing a socket that is still connecting fires
      // "error"/"close" for a connection this component no longer uses
      // (e.g. React's dev-mode double effect run).
      socket.onopen = socket.onclose = socket.onerror = socket.onmessage = null;
      socket.close();
      socketRef.current = null;
    };
  }, [sandboxId, attempt]);

  const send = useCallback((control: SandboxControl) => {
    const socket = socketRef.current;
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(control));
  }, []);

  const reconnect = useCallback(() => {
    setConnection("connecting");
    setStatus(null);
    setLastError(null);
    setAttempt((n) => n + 1);
  }, []);

  return { snapshot, status, connection, lastError, send, reconnect };
}
