"""Gymnasium environment: dispatch one low-battery EV to a charging station.

Step granularity: one env.step() = one charging-station decision for one
EV. The EV is dispatched immediately, then the simulator ticks only until
the NEXT EV needs a decision (possibly zero ticks, if several crossed the
low-battery threshold at once). An EV awaiting a decision is held in place
by the simulator, so the env must never keep ticking while one is pending:
an earlier version fast-forwarded until the dispatched EV started charging,
which parked every other low-battery EV on the road for thousands of ticks.

Reward is the per-decision semi-MDP proxy cost described in dispatch_core.
The EVs' real travel/waiting times are still reported in `info["resolved"]`
once each one starts charging or fails; evaluation metrics use those, never
the reward.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backend.ai_core.dispatch_core import DispatchContext, observation_dim
from backend.config import DEFAULT_CONFIG, SimulationConfig
from backend.simulation.simulator import Simulator


class EVEnv(gym.Env):
    """Single-agent charging-station dispatch environment."""

    def __init__(
        self,
        config: SimulationConfig = DEFAULT_CONFIG,
        tick_hook: Callable[[Simulator], None] | None = None,
    ):
        super().__init__()
        self.config = config
        # Optional observer called after every internal simulator.tick(),
        # e.g. for evaluation code (Phase 6) to sample per-second station
        # queue/occupancy history. None by default: no effect on behavior.
        self._tick_hook = tick_hook

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(observation_dim(config),), dtype=np.float32)
        self.action_space = spaces.Discrete(config.num_stations)

        self.simulator: Simulator | None = None
        # Observation features, reward pricing and dispatch for the current
        # episode. Set in reset().
        self._context: DispatchContext | None = None
        self._current_vehicle_id: int | None = None
        # Round-robin queue over EVs currently needing a decision. Without
        # this, always taking get_vehicles_needing_decision()[0] would let a
        # single vehicle whose best station is persistently unreachable (a
        # deterministic policy keeps re-picking the same invalid station for
        # it) monopolize every remaining decision slot in the episode,
        # starving every other pending EV until truncation.
        self._pending_queue: deque[int] = deque()

    @property
    def _station_ids(self) -> list[int]:
        """action index -> station_id, in the same order the observation
        lists stations."""
        return self._context.station_ids

    @property
    def _in_flight(self) -> dict[int, int]:
        """Dispatched EVs still driving to / queueing at their station."""
        return self._context.in_flight

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        resolved_seed = seed if seed is not None else self.config.random_seed

        self.simulator = Simulator(self.config, seed=resolved_seed)
        self._context = DispatchContext(self.simulator, self.config)
        self._pending_queue = deque()

        terminated, truncated = self._advance_until_next_event()
        if terminated or truncated:
            raise RuntimeError(
                "No EV ever needs a charging decision for this scenario/seed. "
                "Check low_battery_threshold and battery_consumption_per_distance."
            )

        self._current_vehicle_id = self._pick_next_vehicle()
        observation = self._build_observation(self._current_vehicle_id)
        info = {
            "next_vehicle_id": self._current_vehicle_id,
            "simulation_time": self.simulator.simulation_time,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.simulator is not None, "call reset() before step()"
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} outside action_space {self.action_space}")

        vehicle_id = self._current_vehicle_id
        info: dict[str, Any] = {
            "vehicle_id": vehicle_id,
            "requested_station_id": self.station_id_for_action(action),
        }
        reward, dispatch_info = self._context.dispatch(vehicle_id, info["requested_station_id"])
        info.update(dispatch_info)

        terminated, truncated = self._advance_until_next_event()
        info["resolved"] = self._context.collect_resolved()
        info["simulation_time"] = self.simulator.simulation_time

        if terminated or truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info["next_vehicle_id"] = None
        else:
            self._current_vehicle_id = self._pick_next_vehicle()
            observation = self._build_observation(self._current_vehicle_id)
            info["next_vehicle_id"] = self._current_vehicle_id

        reward *= self.config.reward_scale
        return observation, float(reward), terminated, truncated, info

    def _expected_dispatch_cost(self, vehicle_id: int, station_id: int) -> float:
        return self._context.expected_dispatch_cost(vehicle_id, station_id)

    def station_id_for_action(self, action: int) -> int:
        """Map a Discrete action index (0..num_stations-1) to the real
        station_id it dispatches to; that station's graph node (its POI
        node on the OSM map) is simulator.stations[station_id].node_id."""
        return self._station_ids[int(action)]

    def _advance_until_next_event(self) -> tuple[bool, bool]:
        """Tick until a decision is available or the episode ends."""
        while True:
            if self.simulator.all_vehicles_done():
                return True, False
            if self.simulator.simulation_time >= self.config.max_episode_steps:
                return False, True
            if self.simulator.get_vehicles_needing_decision():
                return False, False
            self._tick()

    def _tick(self) -> None:
        self.simulator.tick()
        if self._tick_hook is not None:
            self._tick_hook(self.simulator)

    def _pick_next_vehicle(self) -> int:
        """Pop the next pending EV in round-robin order (see _pending_queue)."""
        pending_ids = {v.vehicle_id for v in self.simulator.get_vehicles_needing_decision()}
        self._pending_queue = deque(vid for vid in self._pending_queue if vid in pending_ids)
        newly_eligible = sorted(pending_ids - set(self._pending_queue))
        self._pending_queue.extend(newly_eligible)
        return self._pending_queue.popleft()

    def build_observation(self, vehicle_id: int) -> np.ndarray:
        """Public accessor for the same normalized observation reset()/step()
        use internally, for an arbitrary vehicle_id -- not just whichever one
        is currently up for decision. Used for read-only previews (e.g. the
        API's "what would the policy suggest for this vehicle right now"
        endpoint); never called from reset()/step() itself, so it changes no
        existing behavior."""
        return self._build_observation(vehicle_id)

    def _build_observation(self, vehicle_id: int) -> np.ndarray:
        return self._context.build_observation(vehicle_id)
