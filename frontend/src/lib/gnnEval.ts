// Client + types for the MAPPO-GNN evaluation API (backend/api/routes/gnn_eval.py).
// Mirrors backend/api/gnn_schemas.py exactly; no evaluation logic here.

import { API_BASE_URL } from "@/lib/api";

export type AlgorithmName = "nearest_station" | "least_queue" | "mappo_mlp" | "mappo_gnn";
export type ResultStatus = "ok" | "crashed" | "unavailable";

export interface EvalScenario {
  base_scenario: string;
  num_stations: number;
  num_evs: number;
  num_depots: number;
  radius_meters: number;
  seed: number;
}

export interface EvalMetrics {
  num_evs: number;
  num_decisions: number;
  // Dispatched EVs that reached their station and started charging.
  num_charged: number;
  num_failed: number;
  num_overloaded: number;
  // Simulated seconds; waiting includes MAPPO's decision-window delay.
  total_waiting_time: number;
  average_waiting_time: number;
  total_travel_time: number;
  average_travel_time: number;
  total_system_cost: number;
  average_queue_length: number;
  max_queue_length: number;
  // Wall time of the policy's own decision call per EV.
  average_decision_time_ms: number;
  simulation_time: number;
  completed: boolean;
}

export interface ModelInfo {
  path: string;
  parameters: number;
  // Null when the input size does not depend on the scenario (GNN).
  input_dim: number | null;
  num_actions: number | null;
  scenario_input_dim: number | null;
  trained_on_stations: number | null;
  trained_on_evs: number | null;
  trained_decisions: number | null;
  validation_reward_per_decision: number | null;
}

export interface AlgorithmResult {
  algorithm: AlgorithmName;
  label: string;
  status: ResultStatus;
  crashed: boolean;
  error_type: string | null;
  error_message: string | null;
  metrics: EvalMetrics | null;
  model: ModelInfo | null;
  wall_time_seconds: number;
}

export interface CompareBaselineResponse {
  scenario: EvalScenario;
  // Always nearest_station, least_queue, mappo_mlp, mappo_gnn.
  results: AlgorithmResult[];
  cached: boolean;
}

export interface ZeroShotScaleResponse {
  scenario: EvalScenario;
  trained_on: { num_stations: number; num_evs: number };
  mlp: AlgorithmResult;
  gnn: AlgorithmResult;
  reference: AlgorithmResult | null;
  cached: boolean;
}

export interface GNNEvalStatus {
  busy: boolean;
  mlp_model: ModelInfo | null;
  gnn_model: ModelInfo | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = JSON.parse(detail).detail ?? detail;
    } catch {
      // not JSON: keep the raw text
    }
    throw new Error(response.status === 409 ? `Đang có một bài đánh giá khác chạy: ${detail}` : String(detail));
  }
  return (await response.json()) as T;
}

export function fetchGnnEvalStatus(): Promise<GNNEvalStatus> {
  return request<GNNEvalStatus>("/api/gnn-eval/status");
}

export function runCompareBaseline(params: {
  seed?: number;
  num_stations?: number;
  num_evs?: number;
}): Promise<CompareBaselineResponse> {
  return request<CompareBaselineResponse>("/api/gnn-eval/compare-baseline", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export function runZeroShotScale(params: {
  num_stations: number;
  num_evs: number;
  seed?: number;
}): Promise<ZeroShotScaleResponse> {
  return request<ZeroShotScaleResponse>("/api/gnn-eval/zero-shot-scale", {
    method: "POST",
    body: JSON.stringify(params),
  });
}
