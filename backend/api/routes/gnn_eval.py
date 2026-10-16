"""MAPPO-GNN evaluation endpoints (new; the original API is unchanged).

    GET  /api/gnn-eval/status              both models' info, and whether an evaluation is running
    POST /api/gnn-eval/compare-baseline    Nearest Station, Least Queue, MAPPO-MLP, MAPPO-GNN on one shared seed
    POST /api/gnn-eval/zero-shot-scale     MAPPO-GNN vs MAPPO-MLP on a scenario of another size, no retraining

Each evaluation plays whole episodes (tens of seconds), so they run one at a
time: a request arriving while another runs gets 409 instead of piling up on
the CPU. Results are deterministic for a given request and model files, so
they are cached in memory until a model file changes (e.g. training saves a
better GNN checkpoint).
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.ai_core import evaluate_gnn
from backend.api.gnn_schemas import (
    CompareBaselineRequest,
    CompareBaselineResponse,
    GNNEvalStatus,
    ZeroShotScaleRequest,
    ZeroShotScaleResponse,
)

router = APIRouter(prefix="/api/gnn-eval", tags=["gnn-evaluation"])

_evaluation_lock = threading.Lock()
_cache: dict[tuple, dict] = {}
_CACHE_LIMIT = 32


def _model_versions() -> tuple:
    """Modification times of both model files, part of every cache key."""
    versions = []
    for path in (Path(evaluate_gnn.MLP_MODEL_PATH), Path(evaluate_gnn.GNN_MODEL_PATH)):
        versions.append(path.stat().st_mtime_ns if path.exists() else None)
    return tuple(versions)


def _run_exclusive(key: tuple, run) -> tuple[dict, bool]:
    """(report, cached): from the cache, or by running `run` alone."""
    key = (*key, _model_versions())
    if key in _cache:
        return _cache[key], True
    if not _evaluation_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="another evaluation is already running; try again when it finishes")
    try:
        report = run()
    finally:
        _evaluation_lock.release()
    if len(_cache) >= _CACHE_LIMIT:
        _cache.pop(next(iter(_cache)))
    _cache[key] = report
    return report, False


@router.get("/status", response_model=GNNEvalStatus)
def status() -> GNNEvalStatus:
    return GNNEvalStatus(busy=_evaluation_lock.locked(), **evaluate_gnn.describe_models())


@router.post("/compare-baseline", response_model=CompareBaselineResponse)
def compare_baseline(payload: CompareBaselineRequest) -> CompareBaselineResponse:
    report, cached = _run_exclusive(
        ("compare", payload.seed, payload.num_stations, payload.num_evs),
        lambda: evaluate_gnn.compare_baseline(payload.seed, payload.num_stations, payload.num_evs),
    )
    return CompareBaselineResponse(**report, cached=cached)


@router.post("/zero-shot-scale", response_model=ZeroShotScaleResponse)
def zero_shot_scale(payload: ZeroShotScaleRequest) -> ZeroShotScaleResponse:
    report, cached = _run_exclusive(
        ("zero-shot", payload.num_stations, payload.num_evs, payload.seed, payload.include_baseline),
        lambda: evaluate_gnn.zero_shot_scale(
            payload.num_stations, payload.num_evs, payload.seed, payload.include_baseline
        ),
    )
    return ZeroShotScaleResponse(**report, cached=cached)
