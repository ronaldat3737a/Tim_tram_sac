"""PettingZoo ParallelEnv: every app EV awaiting a decision is an agent.

Same Simulator, observations and per-decision semi-MDP proxy reward as the
single-agent EVEnv (all shared through dispatch_core), but each step
dispatches the whole batch of app EVs that need a station at the current
tick at once, then ticks until the next batch appears. Non-app EVs are
dispatched by the Simulator itself and never become agents.

Agent lifecycle: an app EV asks for a station exactly once (after charging
it drives to a depot and never needs another decision), so it is an agent
for exactly one step. That step's results report it terminated, alongside
the agents of the next batch (terminated False, reward 0), as PettingZoo's
parallel API expects for agents joining mid-episode. The episode is over
once env.agents is empty.

Decision window: like a ride-hailing dispatcher, the env does not dispatch
the first EV that runs low on its own. It opens a window there and keeps the
simulation running for up to decision_window more ticks, collecting every
app EV that runs low meanwhile into the same batch. EVs waiting inside the
window stand still on the road, keep draining idle battery (and can fail)
exactly as the Simulator always treats an undispatched EV; how long each one
waited is reported as infos[agent]["decision_delay"], since no evaluation
metric (travel time counts from dispatch, waiting time from joining a
station queue) would see it otherwise. The window closes early if the
episode ends. decision_window=0 batches only EVs running low on the very
same tick. All of this lives in this env: the Simulator and EVEnv keep
their own timing.

Within a batch, EVs are dispatched in vehicle_id order, so a later EV's
expected waiting time already counts the EVs of the same batch sent to that
station before it.

For MAPPO's centralized critic (CTDE), state() returns every station's load
plus the episode's time progress (see DispatchContext.global_state); each
actor still only sees its own observation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from backend.ai_core.dispatch_core import DispatchContext, observation_dim, state_dim
from backend.config import DEFAULT_CONFIG, SimulationConfig
from backend.simulation.simulator import Simulator


class MultiAgentEVEnv(ParallelEnv):
    """Multi-agent charging-station dispatch environment."""

    metadata = {"name": "multi_agent_ev_v0", "render_modes": []}

    def __init__(
        self,
        config: SimulationConfig = DEFAULT_CONFIG,
        tick_hook: Callable[[Simulator], None] | None = None,
        decision_window: int = 15,
    ):
        if decision_window < 0:
            raise ValueError(f"decision_window must be >= 0, got {decision_window}")
        self.config = config
        self.decision_window = decision_window
        # Optional observer called after every internal simulator.tick(),
        # as in EVEnv. None by default: no effect on behavior.
        self._tick_hook = tick_hook

        # Every agent shares these exact space objects (parameter sharing).
        self._observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(observation_dim(config),), dtype=np.float32
        )
        self._action_space = spaces.Discrete(config.num_stations)
        self.state_space = spaces.Box(low=0.0, high=1.0, shape=(state_dim(config),), dtype=np.float32)

        self.simulator: Simulator | None = None
        self._context: DispatchContext | None = None
        # Which EVs use the app is drawn per scenario seed, so both are
        # filled in by reset(): possible_agents is every app EV of the
        # scenario, agents the ones awaiting a decision right now.
        self.possible_agents: list[int] = []
        self.agents: list[int] = []
        self.observation_spaces: dict[int, spaces.Box] = {}
        self.action_spaces: dict[int, spaces.Discrete] = {}
        # Dispatched trips that started charging (or failed) during the last
        # reset()/step(), with their real travel/waiting times, for
        # evaluation. Never feeds the reward.
        self.last_resolved: list[dict[str, Any]] = []
        # vehicle_id -> simulation_time it first ran low, for every app EV
        # still awaiting dispatch.
        self._pending_since: dict[int, float] = {}

    def observation_space(self, agent: int) -> spaces.Box:
        return self._observation_space

    def action_space(self, agent: int) -> spaces.Discrete:
        return self._action_space

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
        resolved_seed = seed if seed is not None else self.config.random_seed
        self.simulator = Simulator(self.config, seed=resolved_seed)
        self._context = DispatchContext(self.simulator, self.config)

        self.possible_agents = sorted(
            vehicle_id for vehicle_id, vehicle in self.simulator.vehicles.items() if not vehicle.is_non_app
        )
        self.observation_spaces = dict.fromkeys(self.possible_agents, self._observation_space)
        self.action_spaces = dict.fromkeys(self.possible_agents, self._action_space)
        self.last_resolved = []
        self._pending_since = {}

        terminated, truncated = self._advance_until_next_event()
        if terminated or truncated:
            raise RuntimeError(
                "No app EV ever needs a charging decision for this scenario/seed. "
                "Check low_battery_threshold and battery_consumption_per_distance."
            )
        self.agents = self._pending_agents()

        observations = {agent: self._context.build_observation(agent) for agent in self.agents}
        infos = {agent: {"simulation_time": self.simulator.simulation_time} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[int, int]
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[int, dict[str, Any]],
    ]:
        assert self.simulator is not None, "call reset() before step()"
        if set(actions) != set(self.agents):
            raise ValueError(
                f"step() needs exactly one action per agent awaiting a decision {self.agents}, "
                f"got {sorted(actions)}"
            )
        # Checked up front so a bad action never leaves a batch half dispatched.
        for agent, action in actions.items():
            if not self._action_space.contains(int(action)):
                raise ValueError(f"action {action} for agent {agent} outside {self._action_space}")

        acting = self.agents
        rewards: dict[int, float] = {}
        infos: dict[int, dict[str, Any]] = {}
        for agent in acting:
            requested_station_id = self.station_id_for_action(actions[agent])
            reward, dispatch_info = self._context.dispatch(agent, requested_station_id)
            rewards[agent] = float(reward * self.config.reward_scale)
            infos[agent] = {
                "requested_station_id": requested_station_id,
                "decision_delay": self.simulator.simulation_time - self._pending_since.pop(agent),
                **dispatch_info,
            }

        terminated, truncated = self._advance_until_next_event()
        self.last_resolved = self._context.collect_resolved()
        self.agents = [] if terminated or truncated else self._pending_agents()
        if not self.agents:
            self._pending_since = {}

        observations = {agent: np.zeros(self._observation_space.shape, dtype=np.float32) for agent in acting}
        terminations = dict.fromkeys(acting, True)
        truncations = dict.fromkeys(acting, False)
        for agent in self.agents:
            observations[agent] = self._context.build_observation(agent)
            rewards[agent] = 0.0
            terminations[agent] = False
            truncations[agent] = False
            infos[agent] = {}
        for info in infos.values():
            info["simulation_time"] = self.simulator.simulation_time
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        """Global state for the centralized critic: queue, available
        capacity and incoming EVs of every station, then time progress."""
        assert self._context is not None, "call reset() before state()"
        return self._context.global_state()

    def station_id_for_action(self, action: int) -> int:
        """Map a Discrete action index (0..num_stations-1) to the real
        station_id it dispatches to."""
        return self._context.station_ids[int(action)]

    def action_for_station_id(self, station_id: int) -> int:
        """Inverse of station_id_for_action."""
        return self._context.station_ids.index(station_id)

    def build_observation(self, vehicle_id: int) -> np.ndarray:
        """The local observation of any vehicle, not just a current agent
        (e.g. for the API's read-only "my car" preview)."""
        return self._context.build_observation(vehicle_id)

    def close(self) -> None:
        pass

    def _pending_agents(self) -> list[int]:
        agents = sorted(vehicle.vehicle_id for vehicle in self.simulator.get_vehicles_needing_decision())
        # Drops EVs that failed while waiting in the window.
        self._pending_since = {agent: self._pending_since[agent] for agent in agents}
        return agents

    def _advance_until_next_event(self) -> tuple[bool, bool]:
        """Tick until the next batch is ready or the episode ends: run until
        some app EV needs a decision, then keep the decision window open for
        up to decision_window more ticks, unless the episode ends first."""
        while True:
            episode_end = self._tick_until_pending()
            if episode_end is not None:
                return episode_end
            for _ in range(self.decision_window):
                if self._episode_end() is not None:
                    break
                self._tick()
                self._note_pending()
            episode_end = self._episode_end()
            if episode_end is not None:
                return episode_end
            if self.simulator.get_vehicles_needing_decision():
                return False, False
            # Every EV in the window failed while waiting: find the next one.

    def _tick_until_pending(self) -> tuple[bool, bool] | None:
        """Tick until some app EV needs a decision (returns None) or the
        episode ends (returns its (terminated, truncated))."""
        while True:
            episode_end = self._episode_end()
            if episode_end is not None:
                return episode_end
            if self._note_pending():
                return None
            self._tick()

    def _episode_end(self) -> tuple[bool, bool] | None:
        if self.simulator.all_vehicles_done():
            return True, False
        if self.simulator.simulation_time >= self.config.max_episode_steps:
            return False, True
        return None

    def _note_pending(self) -> bool:
        """Record when each app EV awaiting a decision first ran low.
        Returns whether any is awaiting one."""
        pending = self.simulator.get_vehicles_needing_decision()
        for vehicle in pending:
            self._pending_since.setdefault(vehicle.vehicle_id, self.simulator.simulation_time)
        return bool(pending)

    def _tick(self) -> None:
        self.simulator.tick()
        if self._tick_hook is not None:
            self._tick_hook(self.simulator)
