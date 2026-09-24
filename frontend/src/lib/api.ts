// REST client for the FastAPI backend (backend/api/routes/simulation.py).
// Thin fetch wrappers only -- no simulation/reward logic here (section 39).

import type {
  Algorithm,
  MyCarPreviewResponse,
  NetworkInfo,
  SimulationStatusResponse,
} from "@/lib/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${init?.method ?? "GET"} ${path} failed (${response.status}): ${detail}`);
  }
  return (await response.json()) as T;
}

export function fetchNetworkInfo(): Promise<NetworkInfo> {
  return request<NetworkInfo>("/api/network");
}

export function fetchStatus(): Promise<SimulationStatusResponse> {
  return request<SimulationStatusResponse>("/api/simulation/status");
}

export function startSimulation(params: {
  seed?: number;
  algorithm: Algorithm;
  speed: number;
}): Promise<SimulationStatusResponse> {
  return request<SimulationStatusResponse>("/api/simulation/start", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export function pauseSimulation(): Promise<SimulationStatusResponse> {
  return request<SimulationStatusResponse>("/api/simulation/pause", { method: "POST" });
}

export function resumeSimulation(): Promise<SimulationStatusResponse> {
  return request<SimulationStatusResponse>("/api/simulation/resume", { method: "POST" });
}

export function resetSimulation(seed?: number): Promise<SimulationStatusResponse> {
  const query = seed !== undefined ? `?seed=${seed}` : "";
  return request<SimulationStatusResponse>(`/api/simulation/reset${query}`, { method: "POST" });
}

export function setSimulationSpeed(speed: number): Promise<SimulationStatusResponse> {
  return request<SimulationStatusResponse>("/api/simulation/speed", {
    method: "POST",
    body: JSON.stringify({ speed }),
  });
}

export function fetchMyCarPreview(): Promise<MyCarPreviewResponse> {
  return request<MyCarPreviewResponse>("/api/simulation/my-car/preview");
}
