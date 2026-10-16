"""Pydantic schemas of the MAPPO-GNN evaluation API (routes/gnn_eval.py).

Every algorithm's outcome is one AlgorithmResult with the same shape, ok or
not: `status` says which, `metrics` is filled only when it ran, and
`error_type` / `error_message` only when it did not.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AlgorithmName = Literal["nearest_station", "least_queue", "mappo_mlp", "mappo_gnn"]
ResultStatus = Literal["ok", "crashed", "unavailable"]


class CompareBaselineRequest(BaseModel):
    seed: int = 1000
    # Defaults: the large scenario both learned models were trained on.
    num_stations: int = Field(10, ge=2, le=150)
    num_evs: int = Field(200, ge=1, le=1000)


class ZeroShotScaleRequest(BaseModel):
    num_stations: int = Field(50, ge=2, le=150)
    num_evs: int = Field(500, ge=1, le=1000)
    seed: int = 1000
    # Also run Nearest Station on the scaled scenario as a yardstick.
    include_baseline: bool = True


class EvalScenario(BaseModel):
    base_scenario: str
    num_stations: int
    num_evs: int
    num_depots: int
    radius_meters: float
    seed: int


class EvalMetrics(BaseModel):
    num_evs: int
    num_decisions: int
    # Dispatched EVs that reached their station and started charging.
    num_charged: int
    num_failed: int
    num_overloaded: int
    # Seconds of simulated time. Waiting includes MAPPO's decision-window delay.
    total_waiting_time: float
    average_waiting_time: float
    total_travel_time: float
    average_travel_time: float
    total_system_cost: float
    average_queue_length: float
    max_queue_length: float
    # Wall time of the policy's own decision call per EV (for MAPPO-MLP/GNN
    # including building the observation it decides from).
    average_decision_time_ms: float
    simulation_time: float
    # Every EV finished within the episode's time cap.
    completed: bool


class ModelInfo(BaseModel):
    path: str
    parameters: int
    # Input width the network was built for; None when it does not depend
    # on the scenario (GNN).
    input_dim: int | None
    num_actions: int | None
    # Input width this scenario produces (MLP only).
    scenario_input_dim: int | None = None
    trained_on_stations: int | None
    trained_on_evs: int | None
    trained_decisions: int | None
    validation_reward_per_decision: float | None


class AlgorithmResult(BaseModel):
    algorithm: AlgorithmName
    label: str
    status: ResultStatus
    crashed: bool
    # "dimension_mismatch", "model_unavailable", or an exception class name.
    error_type: str | None
    error_message: str | None
    metrics: EvalMetrics | None
    model: ModelInfo | None
    wall_time_seconds: float


class CompareBaselineResponse(BaseModel):
    scenario: EvalScenario
    # Always 4 entries, in this order: nearest_station, least_queue,
    # mappo_mlp, mappo_gnn.
    results: list[AlgorithmResult]
    # True when served from the in-memory cache (same request, same models).
    cached: bool = False


class TrainedOn(BaseModel):
    num_stations: int
    num_evs: int


class ZeroShotScaleResponse(BaseModel):
    scenario: EvalScenario
    trained_on: TrainedOn
    mlp: AlgorithmResult
    gnn: AlgorithmResult
    reference: AlgorithmResult | None
    cached: bool = False


class GNNEvalStatus(BaseModel):
    # An evaluation is running right now (a new request would get 409).
    busy: bool
    mlp_model: ModelInfo | None
    gnn_model: ModelInfo | None
