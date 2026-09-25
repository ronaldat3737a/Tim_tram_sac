"""Electric vehicle model (PROJECT_SPEC.md section 13)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from backend.config import SimulationConfig


class VehicleState(str, Enum):
    TRAVELING = "TRAVELING"
    WAITING = "WAITING"
    CHARGING = "CHARGING"
    # Charging finished; the EV is driving (along real road geometry, like
    # TRAVELING) to its nearest depot node. Explicit user-requested MDP/
    # vehicle-model extension -- distinct from the earlier frontend-only
    # cosmetic depot: this is now real simulated movement with real battery
    # consumption, only ending in COMPLETED once the depot node is reached.
    RETURNING_TO_DEPOT = "RETURNING_TO_DEPOT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Vehicle:
    vehicle_id: int
    current_node: int
    destination_node: int
    battery_level: float
    battery_capacity: float
    speed: float
    target_station: int | None = None
    # Set only while RETURNING_TO_DEPOT (the nearest depot node picked at
    # the moment charging finished); None otherwise.
    target_depot: int | None = None
    state: VehicleState = VehicleState.TRAVELING

    # Remaining nodes to visit; route[0] is always current_node.
    route: list[int] = field(default_factory=list)
    # Distance already covered along the edge (route[0] -> route[1]).
    edge_progress: float = 0.0

    # Simulation-time bookkeeping consumed by reward calculation in Phase 3.
    time_since_station_assigned: float = 0.0
    waiting_time: float = 0.0

    # Tick at which this EV joins traffic (staggered departures). Before it,
    # the Simulator leaves the EV untouched: no movement, no battery drain,
    # no charging decision.
    activation_tick: float = 0.0

    def set_battery_level(self, value: float) -> None:
        self.battery_level = min(max(value, 0.0), self.battery_capacity)


def needs_charging_decision(vehicle: Vehicle, config: SimulationConfig) -> bool:
    """True if this EV must be assigned a charging station right now."""
    return (
        vehicle.state == VehicleState.TRAVELING
        and vehicle.target_station is None
        and vehicle.battery_level <= config.low_battery_threshold
    )
