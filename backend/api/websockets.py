"""WebSocket endpoint streaming dynamic simulation state (section 34, 35).

Static data (network graph, station locations, node coordinates) is served
once via REST (`GET /api/network`), never repeated here. Only the dynamic
per-tick snapshot (vehicle positions/battery/state, station queue/charging)
is pushed, and only when it actually changed since the last send.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.simulation_manager import SimulationManager

router = APIRouter()

BROADCAST_POLL_INTERVAL = 0.1


@router.websocket("/ws/simulation")
async def simulation_updates(websocket: WebSocket) -> None:
    manager: SimulationManager = websocket.app.state.manager
    await websocket.accept()
    last_timestamp: float | None = None
    try:
        while True:
            snapshot = manager.get_snapshot()
            if snapshot is not None and snapshot["timestamp"] != last_timestamp:
                await websocket.send_json(snapshot)
                last_timestamp = snapshot["timestamp"]
            await asyncio.sleep(BROADCAST_POLL_INTERVAL)
    except WebSocketDisconnect:
        return
