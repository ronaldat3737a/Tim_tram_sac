"""FastAPI application entry point (section 36, 46 Phase 7).

Run:
    uvicorn backend.api.main:app --reload

FastAPI handles REST, WebSocket, live simulation control and DQN model
inference. It never trains a model (that runs separately via
`python -m backend.ai_core.train`, section 36).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from collections.abc import Mapping

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.simulation import router as simulation_router
from backend.api.simulation_manager import SimulationManager
from backend.api.websockets import router as websocket_router
from backend.config import DEFAULT_SCENARIO, SCENARIOS, Scenario

# The Next.js dev server (frontend/, section 7) runs on a different origin
# (typically localhost:3000) than this API (localhost:8000), so the browser
# needs an explicit CORS allowance for its REST calls to succeed. WebSocket
# connections aren't subject to CORS, but the REST control/status/network
# endpoints the dashboard also calls are.
DEV_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def create_app(
    scenarios: Mapping[str, Scenario] = SCENARIOS, default_scenario: str = DEFAULT_SCENARIO
) -> FastAPI:
    """App factory: each call gets its own SimulationManager and background
    thread, so tests can create isolated instances instead of sharing one
    long-lived singleton (a Thread can only be started once).

    `scenarios` (backend/config.SCENARIOS: "small" and "large") are defined
    in backend/config.py so ai_core/train*.py `--scenario <name>` trains on
    the exact config the live demo serves for it. The app starts on
    `default_scenario`; the UI switches via POST /api/simulation/scenario."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager.start_background_thread()
        yield
        app.state.manager.stop_background_thread()

    app = FastAPI(title="EV Charging Dispatch API", lifespan=lifespan)
    app.state.manager = SimulationManager(scenarios, default_scenario)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_FRONTEND_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(simulation_router)
    app.include_router(websocket_router)

    @app.get("/")
    def read_root() -> dict:
        return {"service": "ev-charging-dispatch-api", "status": "ok"}

    return app


app = create_app()
