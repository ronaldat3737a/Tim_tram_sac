"""GNN Sandbox: user-defined maps (new; the simulation dashboard is unchanged).

    POST /api/sandbox/generate-env   build a map around (lat, lng) and make it the active sandbox
    GET  /api/sandbox/env            the active sandbox again (e.g. after a page reload), or 404
    WS   /api/sandbox/ws/{id}        run the active sandbox live with MAPPO-GNN (see sandbox_ws)

The response reuses the simulation dashboard's own shapes, so its map
components can render it as is: `network` is a NetworkInfo and `snapshot` a
SimulationUpdateMessage (built by the same build_snapshot as its WebSocket).
`network.nodes` lists only station/rest-stop POI nodes and their access
nodes (what the map needs to place them); the full road graph can have tens
of thousands of nodes, and every road is already drawn from `network.edges`.
Edge geometry is simplified for display by at most SIMPLIFY_FRACTION of the
radius (a few meters), far below what the map can show at that zoom.

Generating fetches OpenStreetMap the first time a center/radius is used
(cached under data/sandbox/ afterwards), so it can take from a second to a
couple of minutes; one generation runs at a time (409 otherwise).
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import uuid
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.api import sandbox_session
from backend.api.schemas import NetworkInfo, SimulationUpdateMessage
from backend.api.simulation_manager import DEFAULT_TICK_DELAY, build_snapshot
from backend.simulation.network_graph import get_depot_access_nodes, get_station_access_nodes
from backend.simulation.sandbox_network import (
    MapFetchError,
    SandboxNetwork,
    SandboxParams,
    SandboxSimulator,
    build_sandbox,
)

router = APIRouter(prefix="/api/sandbox", tags=["gnn-sandbox"])

# How often the WebSocket pushes the latest snapshot/status (seconds).
SEND_INTERVAL = 0.1
# Status is resent at least this often even when unchanged (seconds).
STATUS_HEARTBEAT = 1.0

# Display simplification tolerance, as a fraction of the radius
# (1.5 m at 3 km, 5 m at 10 km).
SIMPLIFY_FRACTION = 1 / 2000
_METERS_PER_DEGREE = 111_320.0


class GenerateEnvRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Map center latitude")
    lng: float = Field(..., ge=-180, le=180, description="Map center longitude")
    radius_km: float = Field(3.0, ge=0.5, le=10)
    num_stations: int = Field(20, ge=1, le=200)
    num_evs: int = Field(500, ge=1, le=2000)
    num_rest_stops: int = Field(10, ge=1, le=100, description="Rest stops (depots) EVs park at after charging")
    seed: int = 42


class SandboxMapInfo(BaseModel):
    num_road_nodes: int
    num_road_edges: int
    # [south, west, north, east] of the whole extract.
    bounds: list[float]
    from_cache: bool
    fetch_seconds: float
    build_seconds: float


class SandboxBatteryInfo(BaseModel):
    full_range_km: float
    # Range left when an EV asks for a station (at low_battery_threshold).
    request_range_km: float
    farthest_from_station_km: float


class SandboxInfo(BaseModel):
    sandbox_id: str
    created_at: float
    params: GenerateEnvRequest
    map: SandboxMapInfo
    battery: SandboxBatteryInfo
    num_vehicles_departed: int


class GenerateEnvResponse(BaseModel):
    sandbox: SandboxInfo
    network: NetworkInfo
    snapshot: SimulationUpdateMessage


@dataclass
class ActiveSandbox:
    sandbox_id: str
    created_at: float
    network: SandboxNetwork
    # The episode the sandbox starts from (WebSocket stepping picks it up).
    simulator: SandboxSimulator
    response: dict


_generate_lock = threading.Lock()
_active: ActiveSandbox | None = None


def get_active_sandbox() -> ActiveSandbox | None:
    return _active


@router.post("/generate-env", response_model=GenerateEnvResponse)
def generate_env(payload: GenerateEnvRequest) -> dict:
    global _active
    if not _generate_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a sandbox is already being generated; try again when it finishes")
    try:
        try:
            sandbox = build_sandbox(SandboxParams(**payload.model_dump()))
        except MapFetchError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        simulator = SandboxSimulator(sandbox.config, sandbox.graph, seed=payload.seed)
        sandbox_id = uuid.uuid4().hex[:12]
        created_at = time.time()
        response = {
            "sandbox": _sandbox_info(sandbox_id, created_at, payload, sandbox, simulator),
            "network": network_info(sandbox),
            "snapshot": build_snapshot(simulator, sandbox.config, DEFAULT_TICK_DELAY),
        }
        _active = ActiveSandbox(sandbox_id, created_at, sandbox, simulator, response)
        return response
    finally:
        _generate_lock.release()


@router.get("/env", response_model=GenerateEnvResponse)
def get_env() -> dict:
    if _active is None:
        raise HTTPException(status_code=404, detail="no sandbox generated yet; POST /api/sandbox/generate-env first")
    return _active.response


def _sandbox_info(
    sandbox_id: str, created_at: float, payload: GenerateEnvRequest, sandbox: SandboxNetwork, simulator
) -> dict:
    graph, config = sandbox.graph, sandbox.config
    pois = len(graph.graph["station_nodes"]) + len(graph.graph["depot_nodes"])
    west, south, east, north = graph.graph["bounds"]
    full_range_km = 1 / config.battery_consumption_per_distance / 1000
    return {
        "sandbox_id": sandbox_id,
        "created_at": created_at,
        "params": payload.model_dump(),
        "map": {
            "num_road_nodes": graph.number_of_nodes() - pois,
            "num_road_edges": graph.number_of_edges() - pois,
            "bounds": [south, west, north, east],
            "from_cache": sandbox.from_cache,
            "fetch_seconds": sandbox.fetch_seconds,
            "build_seconds": sandbox.build_seconds,
        },
        "battery": {
            "full_range_km": full_range_km,
            "request_range_km": full_range_km * config.low_battery_threshold,
            "farthest_from_station_km": sandbox.farthest_from_station_m / 1000,
        },
        "num_vehicles_departed": sum(1 for v in simulator.vehicles.values() if simulator.is_active(v)),
    }


def network_info(sandbox: SandboxNetwork) -> dict:
    """NetworkInfo of a sandbox graph (see module docstring for `nodes` and
    the display simplification of `geometry`)."""
    graph, config = sandbox.graph, sandbox.config
    tolerance_deg = sandbox.params.radius_m * SIMPLIFY_FRACTION / _METERS_PER_DEGREE
    station_nodes = list(graph.graph["station_nodes"])
    depot_nodes = list(graph.graph["depot_nodes"])
    station_access = get_station_access_nodes(graph)
    depot_access = get_depot_access_nodes(graph)

    shown = {*station_nodes, *depot_nodes, *station_access, *depot_access}
    nodes = [
        {"id": n, "x": graph.nodes[n]["x"], "y": graph.nodes[n]["y"], "is_station": graph.nodes[n]["is_station"]}
        for n in sorted(shown)
    ]
    edges = []
    for u, v, data in graph.edges(data=True):
        line = data["geometry"]
        if tolerance_deg > 0 and len(line.coords) > 2:
            line = line.simplify(tolerance_deg, preserve_topology=False)
        coords = list(line.coords)
        if data["geometry_start"] != u:
            coords.reverse()
        edges.append(
            {
                "source": u,
                "target": v,
                "distance": data["distance"],
                "travel_time": data["travel_time"],
                "speed": data["speed"],
                "traffic_weight": data["traffic_weight"],
                "highway": data.get("highway"),
                "geometry": [[round(y, 6), round(x, 6)] for x, y in coords],
            }
        )
    stations = [
        {
            "id": i,
            "node_id": node,
            "access_node_id": station_access[i],
            "capacity": config.station_capacity,
            "num_chargers": config.num_chargers_per_station,
        }
        for i, node in enumerate(station_nodes)
    ]
    depots = [{"id": i, "node_id": node, "access_node_id": depot_access[i]} for i, node in enumerate(depot_nodes)]
    return {"scenario": "sandbox", "nodes": nodes, "edges": edges, "stations": stations, "depots": depots, "roads": []}


@router.websocket("/ws/{sandbox_id}")
async def sandbox_ws(websocket: WebSocket, sandbox_id: str) -> None:
    """Live MAPPO-GNN simulation of the active sandbox, one per connection.

    Server -> client, every SEND_INTERVAL while they change:
      {"type": "simulation_update", ...}  the dashboard's snapshot (EVs, station loads)
      {"type": "sandbox_status", ...}     status, progress, counts and metrics
      {"type": "sandbox_error", "message"} a rejected control message, or an unknown sandbox
    Client -> server:
      {"action": "start" | "pause" | "resume" | "reset"}, {"action": "speed", "value": ticks_per_second}

    The session starts paused at t=0 (after routing the EVs that depart at
    the start). Disconnecting stops and joins its worker thread."""
    await websocket.accept()
    active = _active
    if active is None or active.sandbox_id != sandbox_id:
        await websocket.send_json({"type": "sandbox_error", "message": f"unknown sandbox {sandbox_id!r}; generate one first"})
        await websocket.close(code=4404)
        return

    session = await asyncio.to_thread(
        sandbox_session.open_session, active.sandbox_id, active.network, active.network.params.seed
    )
    receiver = asyncio.create_task(_receive_controls(websocket, session))
    try:
        sent_version, last_status, last_status_at = -1, None, 0.0
        while not receiver.done():
            version, snapshot = session.latest_snapshot()
            if snapshot is not None and version != sent_version:
                await websocket.send_json(snapshot)
                sent_version = version
            status = session.status_message()
            now = time.monotonic()
            if status != last_status or now - last_status_at >= STATUS_HEARTBEAT:
                await websocket.send_json(status)
                last_status, last_status_at = status, now
            await asyncio.sleep(SEND_INTERVAL)
    except (WebSocketDisconnect, RuntimeError):
        pass  # client gone (RuntimeError: sending on a closed socket)
    finally:
        # Stop the worker first, synchronously: when the server cancels this
        # handler (client gone, shutdown), every await below is cancelled
        # too, and the worker must not outlive the connection.
        sandbox_session.release_session(session)
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
            await receiver
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.shield(asyncio.to_thread(session.close))


async def _receive_controls(websocket: WebSocket, session: sandbox_session.SandboxSession) -> None:
    """Apply control messages until the client disconnects (which ends this
    task, and with it the connection's send loop)."""
    while True:
        message = await websocket.receive_json()
        try:
            if not isinstance(message, dict):
                raise ValueError("control messages are JSON objects")
            # reset stops and joins the worker: keep that off the event loop.
            await asyncio.to_thread(session.control, message)
        except (ValueError, TypeError) as exc:
            await websocket.send_json({"type": "sandbox_error", "message": str(exc)})
