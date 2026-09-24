"""Baseline 3 — Least Queue (PROJECT_SPEC.md section 27): argmin(queue),
restricted to stations the EV can safely reach with its current battery.

If no station is reachable, falls back to argmin(queue) over all stations
(some action must still be returned; the environment will apply the usual
invalid-action penalty, consistent with how an unreachable RL action is
handled).
"""

from __future__ import annotations

from backend.simulation.simulator import Simulator


def choose_station(simulator: Simulator, vehicle_id: int) -> int:
    reachable = [
        station_id
        for station_id in simulator.stations
        if simulator.is_station_reachable(vehicle_id, station_id)
    ]
    candidates = reachable if reachable else list(simulator.stations)
    return min(candidates, key=lambda station_id: len(simulator.stations[station_id].queue))
