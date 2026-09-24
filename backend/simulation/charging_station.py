"""Charging station model (PROJECT_SPEC.md section 12)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ChargingStation:
    station_id: int
    # The station's own POI node id -- both the routing target and the
    # display position (its (x, y) on the graph is the offset point beside
    # the road; see network_graph.py's _add_poi_nodes). Not a real traffic
    # node itself: it connects to one via a short spur edge.
    node_id: int
    capacity: int
    num_chargers: int

    queue: deque[int] = field(default_factory=deque)
    charging_vehicle_ids: set[int] = field(default_factory=set)

    @property
    def occupancy(self) -> int:
        """Vehicles queued plus vehicles currently charging."""
        return len(self.queue) + len(self.charging_vehicle_ids)

    def is_overloaded(self) -> bool:
        """Overload = (queue + charging) > capacity (Phase 0 decision D.3)."""
        return self.occupancy > self.capacity

    def enqueue(self, vehicle_id: int) -> None:
        """Add a vehicle to the queue. Always accepted (PROJECT_SPEC.md section 24:
        a full station is penalized, not blocked)."""
        self.queue.append(vehicle_id)

    def admit_from_queue(self) -> list[int]:
        """Move queued vehicles into free chargers, FIFO. Returns admitted ids."""
        admitted: list[int] = []
        while self.queue and len(self.charging_vehicle_ids) < self.num_chargers:
            vehicle_id = self.queue.popleft()
            self.charging_vehicle_ids.add(vehicle_id)
            admitted.append(vehicle_id)
        return admitted

    def release(self, vehicle_id: int) -> None:
        """Remove a vehicle that finished charging, freeing its charger slot."""
        self.charging_vehicle_ids.discard(vehicle_id)
