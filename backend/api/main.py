"""FastAPI application entry point (section 36, 46 Phase 7).

Run:
    uvicorn backend.api.main:app --reload

FastAPI handles REST, WebSocket, live simulation control and DQN model
inference. It never trains a model (that runs separately via
`python -m backend.ai_core.train`, section 36).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.simulation import router as simulation_router
from backend.api.simulation_manager import SimulationManager
from backend.api.websockets import router as websocket_router
from backend.config import OSM_DEMO_CONFIG, SimulationConfig

# The Next.js dev server (frontend/, section 7) runs on a different origin
# (typically localhost:3000) than this API (localhost:8000), so the browser
# needs an explicit CORS allowance for its REST calls to succeed. WebSocket
# connections aren't subject to CORS, but the REST control/status/network
# endpoints the dashboard also calls are.
DEV_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Defined in backend/config.py so ai_core/train.py trains on the exact same
# scenario the live demo serves (see OSM_DEMO_CONFIG's own comment).
DEMO_CONFIG = OSM_DEMO_CONFIG


def create_app(config: SimulationConfig = DEMO_CONFIG) -> FastAPI:
    """App factory: each call gets its own SimulationManager and background
    thread, so tests can create isolated instances instead of sharing one
    long-lived singleton (a Thread can only be started once)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.manager.start_background_thread()
        yield
        app.state.manager.stop_background_thread()

    app = FastAPI(title="EV Charging Dispatch API", lifespan=lifespan)
    app.state.manager = SimulationManager(config)

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
