"""GraphMultiAgentEVEnv on a sandbox map (backend/simulation/sandbox_network.py).

Same dispatch, reward, decision window and graph observations as the GNN's
training env; only the episode's Simulator differs (SandboxSimulator: the
user's map, K-Means stations, lazy routing). MultiAgentEVEnv.reset builds a
plain Simulator itself, so reset() is restated here around SandboxSimulator,
step for step otherwise.

The sandbox serves MAPPO-GNN only, which decides from graph observations, so
SandboxDispatchContext skips what only the flat MLP observation needs: the
map-wide normalization constants (one shortest-path tree per station per
episode) and the flat observation itself (one route lookup per station per
EV); step() and reset() still return a correctly shaped, all-zero vector in
its place.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from backend.ai_core.dispatch_core import DispatchContext, observation_dim
from backend.ai_core.graph_env import DEFAULT_NEIGHBORS, GraphMultiAgentEVEnv
from backend.simulation.sandbox_network import SandboxNetwork, SandboxSimulator


class SandboxDispatchContext(DispatchContext):
    """DispatchContext without the flat-observation-only parts (see module
    docstring); dispatch, reward and graph features are unchanged."""

    def _compute_normalization_constants(self) -> tuple[float, float]:
        return 1.0, 1.0

    def build_observation(self, vehicle_id: int) -> np.ndarray:
        return np.zeros(observation_dim(self.config), dtype=np.float32)


class SandboxGraphEnv(GraphMultiAgentEVEnv):
    def __init__(
        self,
        sandbox: SandboxNetwork,
        tick_hook: Callable | None = None,
        decision_window: int = 15,
        num_neighbors: int = DEFAULT_NEIGHBORS,
    ):
        super().__init__(sandbox.config, tick_hook=tick_hook, decision_window=decision_window, num_neighbors=num_neighbors)
        self.sandbox = sandbox
        # Called with (routes done, total) while the EVs departing at the
        # start are routed; it may raise to abort the reset.
        self.route_progress: Callable[[int, int], None] | None = None
        # Called once the episode is built and routed, before its first
        # tick (e.g. to publish the t=0 state and wait for "start").
        self.before_first_tick: Callable[[SandboxSimulator], None] | None = None

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        resolved_seed = seed if seed is not None else self.config.random_seed
        self.simulator = SandboxSimulator(self.config, self.sandbox.graph, seed=resolved_seed)
        self.simulator.route_departing(0, progress=self.route_progress)
        self._context = SandboxDispatchContext(self.simulator, self.config)
        self._edge_index, self._edge_attr = self._build_station_edges()

        self.possible_agents = sorted(
            vehicle_id for vehicle_id, vehicle in self.simulator.vehicles.items() if not vehicle.is_non_app
        )
        self.observation_spaces = dict.fromkeys(self.possible_agents, self._observation_space)
        self.action_spaces = dict.fromkeys(self.possible_agents, self._action_space)
        self.last_resolved = []
        self._pending_since = {}
        if self.before_first_tick is not None:
            self.before_first_tick(self.simulator)

        terminated, truncated = self._advance_until_next_event()
        if terminated or truncated:
            raise RuntimeError("No app EV ever needs a charging decision in this sandbox.")
        self.agents = self._pending_agents()
        observations = {agent: self._context.build_observation(agent) for agent in self.agents}
        infos = {agent: {"simulation_time": self.simulator.simulation_time} for agent in self.agents}
        return observations, infos
