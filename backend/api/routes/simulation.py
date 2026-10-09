"""REST endpoints: network info + simulation control (section 36, 38)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.api.schemas import (
    MyCarPreviewResponse,
    NetworkInfo,
    ScenarioInfo,
    ScenarioRequest,
    SimulationStatusResponse,
    SpeedRequest,
    StartSimulationRequest,
)
from backend.api.simulation_manager import SimulationManager

router = APIRouter(prefix="/api", tags=["simulation"])


def _manager(request: Request) -> SimulationManager:
    return request.app.state.manager


@router.get("/network", response_model=NetworkInfo)
def get_network(request: Request) -> NetworkInfo:
    return NetworkInfo(**_manager(request).get_network_info())


@router.get("/scenarios", response_model=list[ScenarioInfo])
def list_scenarios(request: Request) -> list[ScenarioInfo]:
    return [ScenarioInfo(**info) for info in _manager(request).list_scenarios()]


@router.post("/simulation/scenario", response_model=SimulationStatusResponse)
def switch_scenario(payload: ScenarioRequest, request: Request) -> SimulationStatusResponse:
    """Switch the map/config/models to another scenario: a fresh, paused
    episode. Clients then re-fetch /api/network."""
    manager = _manager(request)
    try:
        manager.request_scenario(payload.scenario)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SimulationStatusResponse(**manager.get_status())


@router.get("/simulation/status", response_model=SimulationStatusResponse)
def get_status(request: Request) -> SimulationStatusResponse:
    return SimulationStatusResponse(**_manager(request).get_status())


@router.post("/simulation/start", response_model=SimulationStatusResponse)
def start_simulation(payload: StartSimulationRequest, request: Request) -> SimulationStatusResponse:
    manager = _manager(request)
    try:
        manager.request_start(
            seed=payload.seed, algorithm=payload.algorithm, speed=payload.speed, scenario=payload.scenario
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        # RuntimeError covers EVEnv.reset() refusing a degenerate
        # scenario/seed where no EV in it ever needs a charging decision
        # (ai_core/ev_env.py) -- a bad request, not a server bug.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SimulationStatusResponse(**manager.get_status())


@router.post("/simulation/pause", response_model=SimulationStatusResponse)
def pause_simulation(request: Request) -> SimulationStatusResponse:
    manager = _manager(request)
    manager.request_pause()
    return SimulationStatusResponse(**manager.get_status())


@router.post("/simulation/resume", response_model=SimulationStatusResponse)
def resume_simulation(request: Request) -> SimulationStatusResponse:
    manager = _manager(request)
    manager.request_resume()
    return SimulationStatusResponse(**manager.get_status())


@router.post("/simulation/reset", response_model=SimulationStatusResponse)
def reset_simulation(request: Request, seed: int | None = None) -> SimulationStatusResponse:
    manager = _manager(request)
    try:
        manager.request_reset(seed=seed)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SimulationStatusResponse(**manager.get_status())


@router.post("/simulation/speed", response_model=SimulationStatusResponse)
def set_speed(payload: SpeedRequest, request: Request) -> SimulationStatusResponse:
    manager = _manager(request)
    try:
        manager.request_speed(payload.speed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SimulationStatusResponse(**manager.get_status())


@router.get("/simulation/my-car/preview", response_model=MyCarPreviewResponse)
def my_car_preview(request: Request) -> MyCarPreviewResponse:
    manager = _manager(request)
    try:
        preview = manager.get_my_car_preview()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MyCarPreviewResponse(**preview)
