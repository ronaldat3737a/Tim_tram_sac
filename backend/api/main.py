"""FastAPI application entry point (section 36, 46 Phase 7).

Run:
    uvicorn backend.api.main:app --reload

FastAPI handles REST, WebSocket, live simulation control and DQN model
inference. It never trains a model (that runs separately via
`python -m backend.ai_core.train`, section 36).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.simulation import router as simulation_router
from backend.api.simulation_manager import SimulationManager
from backend.api.websockets import router as websocket_router
from backend.config import DEFAULT_CONFIG, SimulationConfig

# The Next.js dev server (frontend/, section 7) runs on a different origin
# (typically localhost:3000) than this API (localhost:8000), so the browser
# needs an explicit CORS allowance for its REST calls to succeed. WebSocket
# connections aren't subject to CORS, but the REST control/status/network
# endpoints the dashboard also calls are.
DEV_FRONTEND_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Live-demo-only override: a full 0.0 -> 1.0 charge takes ~900 simulated
# ticks, which is ~180 real seconds (~3 min) at the default UI speed of 5
# ticks/sec, so the battery bar visibly ticks up during a demo instead of
# taking the ~20 simulated ticks (config.charging_rate=0.05) calibrated for
# training/evaluation. This is a SEPARATE config from DEFAULT_CONFIG on
# purpose: DEFAULT_CONFIG stays exactly as-is so ai_core/train.py and
# ai_core/evaluate.py keep using the value the already-trained model and
# saved results/ were calibrated against (changing it there would silently
# invalidate that calibration). Real-time duration still scales with
# whatever speed the user picks in the UI.
DEMO_CONFIG = replace(DEFAULT_CONFIG, charging_rate=1.0 / 900)


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
