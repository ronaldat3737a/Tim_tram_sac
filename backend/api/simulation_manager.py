"""Live simulation controller for the FastAPI backend (section 36, Phase 7).

Runs EVEnv in a background thread (never inside a request handler, and never
training -- section 36), stepping through decisions with the selected policy
(a baseline or a loaded, already-trained DQN model). MAPPO runs
MultiAgentEVEnv instead, dispatching a whole decision batch per step with
the trained shared actor. The env's tick_hook
(added in Phase 6 for evaluation metrics) is reused here to publish a
dynamic-state snapshot after every simulated second, so WebSocket clients see
smooth per-second updates even though env.step() itself resolves one whole
decision (many ticks) per call.

Pause/reset take effect at the next decision boundary, not instantly
mid-tick: aborting env.step() partway through would leave it in an
inconsistent state. Decisions resolve in on the order of tens of ticks in
practice, so this is a small, bounded, documented latency.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString
from stable_baselines3 import DQN

from backend.ai_core.ev_env import EVEnv
from backend.ai_core.mappo import MAPPOAgent
from backend.ai_core.marl_env import MultiAgentEVEnv
from backend.baseline import least_queue, nearest_station, shortest_time
from backend.config import SimulationConfig
from backend.simulation.network_graph import get_depot_access_nodes, get_station_access_nodes, shortest_path
from backend.simulation.simulator import Simulator
from backend.simulation.vehicle import VehicleState

logger = logging.getLogger(__name__)

IDLE_POLL_INTERVAL = 0.1
DEFAULT_TICK_DELAY = 0.2
# Written by backend/ai_core/train.py (trained on OSM_DEMO_CONFIG).
DQN_MODEL_PATH = Path("models/dqn_osm_model.zip")

# Baseline used whenever DQN can't be: no trained model on disk yet, a model
# whose observation/action shape doesn't match this config, or predict()
# raising mid-episode. The failure is logged and the simulation keeps going
# rather than crashing the request or the background thread.
DQN_FALLBACK_ALGORITHM = "nearest_station"

# Written by backend/ai_core/train_mappo.py (trained on OSM_DEMO_CONFIG).
# Unlike DQN, a missing or incompatible MAPPO model is reported to the UI
# (400) rather than silently replaced by a baseline.
MAPPO_MODEL_PATH = Path("models/mappo_osm_model.pth")
# Used only if the checkpoint does not record the window it was trained
# with (train_mappo.DECISION_WINDOW).
MAPPO_DEFAULT_DECISION_WINDOW = 15

_BASELINE_POLICIES = {
    "nearest_station": nearest_station.choose_station,
    "shortest_time": shortest_time.choose_station,
    "least_queue": least_queue.choose_station,
}

def _point_along_geometry(line: LineString, fraction: float) -> tuple[float, float]:
    """A point at `fraction` (0..1) of the way along a real edge's
    LineString, guaranteed to lie exactly on it (Task 1 of the spatial-
    geometry refactor: an EV's displayed position is always a shapely
    `Point` produced by `LineString.interpolate()`, never a hand-rolled
    approximation), so movement along a curvy real road stays at a visually
    uniform speed."""
    fraction = min(max(fraction, 0.0), 1.0)
    point = line.interpolate(fraction, normalized=True)
    return point.x, point.y


def _interpolate_position(simulator: Simulator, vehicle) -> tuple[float, float]:
    """Returns this vehicle's current (lng, lat) for display. The
    authoritative physics (edge_progress, distance, battery consumption,
    reward -- all in simulator.py) stay untouched scalars; this only decides
    where that scalar progress maps to visually. A station/depot's POI is a
    real graph node reached via a real spur edge (network_graph.py's
    _add_poi_nodes), so the normal edge-geometry interpolation below already
    shows an EV smoothly crawling in and back out -- no special-casing by
    vehicle state is needed at all."""
    graph = simulator.graph
    if len(vehicle.route) >= 2:
        u, v = vehicle.route[0], vehicle.route[1]
        edge = graph.edges[u, v]
        fraction = 0.0 if edge["distance"] <= 0 else vehicle.edge_progress / edge["distance"]
        # `geometry` is stored in one fixed point order (graph edges are
        # undirected); if this vehicle is traversing it start-to-end in the
        # opposite direction from how it was stored, walk the polyline
        # backwards so position still advances the way the vehicle is
        # actually moving.
        line = edge["geometry"] if edge["geometry_start"] == u else edge["geometry"].reverse()
        return _point_along_geometry(line, fraction)
    node = graph.nodes[vehicle.current_node]
    return node["x"], node["y"]


@dataclass
class _EpisodeAccumulator:
    """Same accumulation pattern as baseline/runner.py's EpisodeMetrics, kept
    live for the current episode so the Dashboard (section 38) can show
    Average Travel Time / Average Waiting Time / Total System Cost without
    the frontend ever computing simulation or reward values itself
    (section 39)."""

    num_decisions: int = 0
    num_valid_decisions: int = 0
    num_invalid_actions: int = 0
    num_overloaded_events: int = 0
    num_stranded_vehicles: int = 0
    num_resolved_dispatches: int = 0
    total_travel_time: float = 0.0
    total_waiting_time: float = 0.0
    episode_reward: float = 0.0

    def record(self, reward: float, info: dict) -> None:
        """One EVEnv step: its decision plus the trips it resolved."""
        self.record_decision(reward, info)
        self.record_resolved(info["resolved"])

    def record_decision(self, reward: float, info: dict) -> None:
        self.num_decisions += 1
        self.episode_reward += reward
        if info["invalid_action"]:
            self.num_invalid_actions += 1
        else:
            self.num_valid_decisions += 1
        if info["stranded"]:
            self.num_stranded_vehicles += 1
        if info["station_overloaded"]:
            self.num_overloaded_events += 1

    def record_resolved(self, trips: list[dict]) -> None:
        # Trips (including invalid picks sent to the env's fallback station)
        # that resolved -- started charging or failed -- during a step.
        for trip in trips:
            self.num_resolved_dispatches += 1
            self.total_travel_time += trip["travel_time"]
            self.total_waiting_time += trip["waiting_time"]

    @property
    def average_travel_time(self) -> float:
        n = self.num_resolved_dispatches
        return self.total_travel_time / n if n else 0.0

    @property
    def average_waiting_time(self) -> float:
        n = self.num_resolved_dispatches
        return self.total_waiting_time / n if n else 0.0

    @property
    def total_system_cost(self) -> float:
        return self.total_travel_time + self.total_waiting_time


def build_snapshot(simulator: Simulator, config: SimulationConfig, tick_delay: float) -> dict:
    vehicles = []
    for vehicle in simulator.vehicles.values():
        if not simulator.is_active(vehicle):
            continue  # not departed yet: not on the map
        x, y = _interpolate_position(simulator, vehicle)
        station_id = None
        eta_seconds = None
        depot_id = None
        if vehicle.state in (VehicleState.WAITING, VehicleState.CHARGING):
            station_id = vehicle.target_station
        if vehicle.state == VehicleState.CHARGING and config.charging_rate > 0:
            remaining_battery = max(vehicle.battery_capacity - vehicle.battery_level, 0.0)
            ticks_needed = remaining_battery / config.charging_rate
            eta_seconds = ticks_needed * tick_delay
        if vehicle.state == VehicleState.RETURNING_TO_DEPOT:
            depot_id = vehicle.target_depot
        vehicles.append(
            {
                "id": vehicle.vehicle_id,
                "node": vehicle.current_node,
                "x": x,
                "y": y,
                "battery": vehicle.battery_level,
                "state": vehicle.state.value,
                "station_id": station_id,
                "eta_seconds": eta_seconds,
                "depot_id": depot_id,
                "is_non_app": vehicle.is_non_app,
            }
        )
    stations = [
        {
            "id": station.station_id,
            "queue": len(station.queue),
            "charging": len(station.charging_vehicle_ids),
        }
        for station in simulator.stations.values()
    ]
    return {
        "type": "simulation_update",
        "timestamp": simulator.simulation_time,
        "vehicles": vehicles,
        "stations": stations,
    }


class SimulationManager:
    """Owns the live EVEnv instance and the background stepping thread."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._lock = threading.RLock()

        self.status = "stopped"
        self.algorithm = "nearest_station"
        self.tick_delay = DEFAULT_TICK_DELAY
        self.episode_seed = config.random_seed

        self.env: EVEnv | MultiAgentEVEnv | None = None
        # EVEnv: the pending EV's observation. MultiAgentEVEnv: a dict of
        # observations, one per agent of the current decision batch.
        self._obs = None
        self._info: dict = {}
        self._latest_snapshot: dict | None = None
        self._dqn_model: DQN | None = None
        self._mappo_model: MAPPOAgent | None = None
        self._episode_metrics = _EpisodeAccumulator()
        # A single EV highlighted as "my car" in the UI (picked fresh, from
        # the episode's own seeded RNG, on every reset/start). Only ever read
        # for display and for the read-only preview endpoint below -- it is
        # dispatched through the exact same automatic policy loop as every
        # other EV, never force-assigned a station outside the normal
        # needs_charging_decision() rule (section 20).
        self.my_vehicle_id: int | None = None

        self._shutdown_requested = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    # --- lifecycle -----------------------------------------------------

    def start_background_thread(self) -> None:
        with self._lock:
            self._reset_episode_locked(self.episode_seed, self.algorithm)
            self.status = "paused"
        self._thread.start()

    def stop_background_thread(self) -> None:
        self._shutdown_requested = True

    # --- control (called from REST request handlers) -------------------

    def request_start(self, seed: int | None, algorithm: str, speed: float) -> None:
        if algorithm == "mappo":
            # Loaded before taking the lock, like DQN below. Raises
            # (-> 400) if there is no usable trained model.
            self._ensure_mappo_model_loaded()
        elif algorithm == "dqn":
            # Deserializing the model from disk can take a noticeable
            # fraction of a second; doing it while holding self._lock would
            # block status/pause/resume requests from other clients for
            # that whole time. Load it (once, cached) before taking the lock.
            try:
                self._ensure_dqn_model_loaded()
            except Exception:
                # Run the fallback baseline instead, and report it as the
                # active algorithm (get_status) so the dashboard never
                # labels baseline results as DQN.
                logger.exception(
                    "could not load DQN model from %s; running %s instead",
                    DQN_MODEL_PATH,
                    DQN_FALLBACK_ALGORITHM,
                )
                algorithm = DQN_FALLBACK_ALGORITHM
        with self._lock:
            if algorithm is not None:
                self._validate_algorithm_locked(algorithm)
            resolved_seed = seed if seed is not None else self.episode_seed
            # _reset_episode_locked can raise (degenerate scenario/seed);
            # only commit seed/algorithm/speed/status once it succeeds, so a
            # rejected request leaves every field exactly as it was rather
            # than partially applied (section 43).
            self._reset_episode_locked(resolved_seed, algorithm if algorithm is not None else self.algorithm)
            self.episode_seed = resolved_seed
            if algorithm is not None:
                self.algorithm = algorithm
            if speed is not None:
                self._set_speed_locked(speed)
            self.status = "running"

    def request_pause(self) -> None:
        with self._lock:
            if self.status == "running":
                self.status = "paused"

    def request_resume(self) -> None:
        with self._lock:
            if self.status == "paused":
                self.status = "running"

    def request_reset(self, seed: int | None) -> None:
        with self._lock:
            resolved_seed = seed if seed is not None else self.episode_seed
            self._reset_episode_locked(resolved_seed, self.algorithm)
            self.episode_seed = resolved_seed
            self.status = "paused"

    def request_speed(self, speed: float) -> None:
        with self._lock:
            self._set_speed_locked(speed)

    def get_status(self) -> dict:
        with self._lock:
            counts = {
                "TRAVELING": 0,
                "WAITING": 0,
                "CHARGING": 0,
                "RETURNING_TO_DEPOT": 0,
                "COMPLETED": 0,
                "FAILED": 0,
            }
            if self.env is not None:
                for vehicle in self.env.simulator.vehicles.values():
                    if self.env.simulator.is_active(vehicle):
                        counts[vehicle.state.value] += 1
            return {
                "status": self.status,
                "simulation_time": self.env.simulator.simulation_time if self.env else 0.0,
                "episode_seed": self.episode_seed,
                "algorithm": self.algorithm,
                "speed": 1.0 / self.tick_delay if self.tick_delay > 0 else 0.0,
                "num_vehicles_traveling": counts["TRAVELING"],
                "num_vehicles_waiting": counts["WAITING"],
                "num_vehicles_charging": counts["CHARGING"],
                "num_vehicles_returning_to_depot": counts["RETURNING_TO_DEPOT"],
                "num_vehicles_completed": counts["COMPLETED"],
                "num_vehicles_failed": counts["FAILED"],
                "num_decisions": self._episode_metrics.num_decisions,
                "num_invalid_actions": self._episode_metrics.num_invalid_actions,
                "num_overloaded_events": self._episode_metrics.num_overloaded_events,
                "average_travel_time": self._episode_metrics.average_travel_time,
                "average_waiting_time": self._episode_metrics.average_waiting_time,
                "total_system_cost": self._episode_metrics.total_system_cost,
                "episode_reward": self._episode_metrics.episode_reward,
                "my_vehicle_id": self.my_vehicle_id,
            }

    def get_network_info(self) -> dict:
        with self._lock:
            graph = self.env.simulator.graph
            nodes = [
                {
                    "id": node_id,
                    "x": data["x"],
                    "y": data["y"],
                    "is_station": data["is_station"],
                }
                for node_id, data in graph.nodes(data=True)
            ]
            edges = [
                {
                    "source": u,
                    "target": v,
                    "distance": data["distance"],
                    "travel_time": data["travel_time"],
                    "speed": data["speed"],
                    "traffic_weight": data["traffic_weight"],
                    "geometry": [
                        [y, x]
                        for x, y in (
                            list(data["geometry"].coords)
                            if data["geometry_start"] == u
                            else reversed(list(data["geometry"].coords))
                        )
                    ],
                }
                for u, v, data in graph.edges(data=True)
            ]
            station_access_nodes = get_station_access_nodes(graph)
            stations = [
                {
                    "id": station.station_id,
                    "node_id": station.node_id,
                    "access_node_id": station_access_nodes[station.station_id],
                    "capacity": station.capacity,
                    "num_chargers": station.num_chargers,
                }
                for station in self.env.simulator.stations.values()
            ]
            depot_access_nodes = get_depot_access_nodes(graph)
            depots = [
                {"id": depot_index, "node_id": node_id, "access_node_id": depot_access_nodes[depot_index]}
                for depot_index, node_id in enumerate(self.env.simulator.depot_nodes)
            ]
            return {
                "nodes": nodes,
                "edges": edges,
                "stations": stations,
                "depots": depots,
            }

    def get_snapshot(self) -> dict | None:
        with self._lock:
            return self._latest_snapshot

    def get_my_car_preview(self) -> dict:
        """Read-only "what would the current policy suggest for my car right
        now" query. Never calls assign_station -- the highlighted vehicle is
        only ever actually dispatched through the normal automatic loop, once
        it naturally needs a decision (section 20). Raises RuntimeError (->
        400) if there is no active episode or the car isn't in a state a
        preview makes sense for (already WAITING/CHARGING/COMPLETED/FAILED)."""
        with self._lock:
            if self.env is None or self.my_vehicle_id is None:
                raise RuntimeError("no active episode")
            simulator = self.env.simulator
            vehicle = simulator.vehicles.get(self.my_vehicle_id)
            if vehicle is None or vehicle.state != VehicleState.TRAVELING or not simulator.is_active(vehicle):
                state_name = vehicle.state.value if vehicle else "unknown"
                raise RuntimeError(
                    f"my car (vehicle {self.my_vehicle_id}) has no preview available "
                    f"right now (state={state_name})"
                )
            candidate_station_id = self._choose_station_locked(
                self.my_vehicle_id, lambda: self.env.build_observation(self.my_vehicle_id)
            )

            station = simulator.stations[candidate_station_id]
            path, _, _ = shortest_path(
                simulator.graph,
                vehicle.current_node,
                station.node_id,
                weight=self.config.routing_weight,
            )
            # path already ends at the station's own POI node (a real graph
            # node, reached via its spur edge) -- no manual nudge needed.
            route = [
                [simulator.graph.nodes[node_id]["y"], simulator.graph.nodes[node_id]["x"]]
                for node_id in path
            ]
            reachable = simulator.is_station_reachable(self.my_vehicle_id, candidate_station_id)
            return {
                "vehicle_id": self.my_vehicle_id,
                "station_id": candidate_station_id,
                "reachable": reachable,
                "route": route,
            }

    # --- internal --------------------------------------------------------

    def _validate_algorithm_locked(self, algorithm: str) -> None:
        if algorithm not in _BASELINE_POLICIES and algorithm not in ("dqn", "mappo"):
            raise ValueError(f"unknown algorithm {algorithm!r}")
        if algorithm == "mappo" and self._mappo_model is None:
            raise FileNotFoundError(
                f"no trained MAPPO model at {MAPPO_MODEL_PATH}; run "
                "`python backend/ai_core/train_mappo.py` first."
            )
        if algorithm == "dqn" and self._dqn_model is None:
            raise FileNotFoundError(
                f"no trained DQN model at {DQN_MODEL_PATH}; run "
                "`python backend/ai_core/train.py` first."
            )

    def _ensure_dqn_model_loaded(self) -> None:
        with self._lock:
            if self._dqn_model is not None:
                return
        if not DQN_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"no trained DQN model at {DQN_MODEL_PATH}; run "
                "`python backend/ai_core/train.py` first."
            )
        model = DQN.load(str(DQN_MODEL_PATH))
        # Catch a stale model (e.g. trained before the observation layout
        # changed, or for a different num_stations) once here, instead of
        # letting every single predict() call fail later.
        expected_env = EVEnv(self.config)
        if (
            model.observation_space.shape != expected_env.observation_space.shape
            or model.action_space.n != expected_env.action_space.n
        ):
            raise ValueError(
                f"DQN model at {DQN_MODEL_PATH} expects observation shape "
                f"{model.observation_space.shape} / {model.action_space.n} actions, but "
                f"this config needs {expected_env.observation_space.shape} / "
                f"{expected_env.action_space.n}; retrain with `python backend/ai_core/train.py`."
            )
        with self._lock:
            if self._dqn_model is None:
                self._dqn_model = model

    def _ensure_mappo_model_loaded(self) -> None:
        with self._lock:
            if self._mappo_model is not None:
                return
        if not MAPPO_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"no trained MAPPO model at {MAPPO_MODEL_PATH}; run "
                "`python backend/ai_core/train_mappo.py` first."
            )
        model = MAPPOAgent.load(MAPPO_MODEL_PATH)
        # Catch a stale model (different observation layout or station
        # count) once here, instead of failing every inference later.
        expected_env = MultiAgentEVEnv(self.config)
        expected = (
            expected_env.observation_space(0).shape[0],
            expected_env.state_space.shape[0],
            int(expected_env.action_space(0).n),
        )
        if (model.obs_dim, model.state_dim, model.num_actions) != expected:
            raise ValueError(
                f"MAPPO model at {MAPPO_MODEL_PATH} expects (obs_dim, state_dim, actions) = "
                f"{(model.obs_dim, model.state_dim, model.num_actions)}, but this config needs "
                f"{expected}; retrain with `python backend/ai_core/train_mappo.py`."
            )
        model.actor.eval()
        model.critic.eval()
        with self._lock:
            if self._mappo_model is None:
                self._mappo_model = model

    def _set_speed_locked(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.tick_delay = 1.0 / speed

    def _reset_episode_locked(self, seed: int, algorithm: str) -> None:
        new_env = self._make_env(algorithm)
        # env.reset() can raise (e.g. a scenario/seed where no EV in it ever
        # needs a decision -- ai_core/ev_env.py's own documented guard). Only
        # commit to self.env/_obs/_info/_latest_snapshot once it actually
        # succeeds, so a failed reset leaves the manager exactly as it was
        # rather than in a mismatched, half-updated state (section 43) that
        # could later crash the background thread. While new_env.reset() is
        # still searching for that first decision, self.env still points at
        # the outgoing episode, so _on_tick treats those search ticks as not
        # current: no pacing sleep and no snapshot flicker for a reset that
        # hasn't taken effect yet (ticks published only after this commit).
        # The caller commits self.episode_seed itself, for the same reason.
        obs, info = new_env.reset(seed=seed)

        self.env = new_env
        self._obs, self._info = obs, info
        self._latest_snapshot = build_snapshot(self.env.simulator, self.config, self.tick_delay)
        self._episode_metrics = _EpisodeAccumulator()
        # "My car" is an app user, never a non-app EV the policy cannot dispatch.
        app_vehicle_ids = [vid for vid, v in self.env.simulator.vehicles.items() if not v.is_non_app]
        self.my_vehicle_id = int(self.env.simulator.rng.choice(app_vehicle_ids))

    def _make_env(self, algorithm: str) -> EVEnv | MultiAgentEVEnv:
        if algorithm == "mappo":
            window = self._mappo_model.metadata.get("decision_window", MAPPO_DEFAULT_DECISION_WINDOW)
            return MultiAgentEVEnv(self.config, tick_hook=self._on_tick, decision_window=window)
        return EVEnv(self.config, tick_hook=self._on_tick)

    def _on_tick(self, simulator: Simulator) -> None:
        # A concurrent reset/start can replace self.env while this exact
        # simulator's env.step() is still mid-flight in the background
        # thread (it was already running before the reset happened, and
        # nothing can abort it mid-call -- see the module docstring). Ticks
        # from that now-orphaned run must not overwrite the new episode's
        # snapshot, and are fast-forwarded (no pacing sleep) so the stale
        # step() finishes quickly and the thread can pick up the new episode.
        with self._lock:
            is_current = self.env is not None and simulator is self.env.simulator
            if is_current:
                self._latest_snapshot = build_snapshot(simulator, self.config, self.tick_delay)
            delay = self.tick_delay if is_current else 0.0
        if delay:
            time.sleep(delay)

    def _choose_action(self) -> int:
        return self._choose_station_locked(self._info["next_vehicle_id"], lambda: self._obs)

    def _choose_station_locked(self, vehicle_id: int, get_obs) -> int:
        """Pick a station for vehicle_id with the active algorithm. For DQN,
        any failure (model missing, predict() raising, an out-of-range
        action) is logged and answered by DQN_FALLBACK_ALGORITHM instead, so
        neither the background loop nor a preview request ever crashes.
        get_obs is only called on the DQN/MAPPO paths."""
        if self.algorithm == "mappo":
            try:
                actions, _ = self._mappo_model.get_action(get_obs(), deterministic=True)
                return self.env.station_id_for_action(int(actions[0]))
            except Exception:
                logger.exception(
                    "MAPPO inference failed for vehicle %s; falling back to %s",
                    vehicle_id,
                    DQN_FALLBACK_ALGORITHM,
                )
                return _BASELINE_POLICIES[DQN_FALLBACK_ALGORITHM](self.env.simulator, vehicle_id)
        if self.algorithm == "dqn":
            try:
                if self._dqn_model is None:
                    raise RuntimeError("DQN model is not loaded")
                action, _ = self._dqn_model.predict(get_obs(), deterministic=True)
                return self.env.station_id_for_action(int(action))
            except Exception:
                logger.exception(
                    "DQN predict failed for vehicle %s; falling back to %s",
                    vehicle_id,
                    DQN_FALLBACK_ALGORITHM,
                )
                return _BASELINE_POLICIES[DQN_FALLBACK_ALGORITHM](self.env.simulator, vehicle_id)
        return _BASELINE_POLICIES[self.algorithm](self.env.simulator, vehicle_id)

    def _choose_batch_actions_locked(self) -> dict[int, int]:
        """MAPPO: one forward pass of the shared actor over every agent of
        the current decision batch (a batch of one is just a 1-row tensor),
        argmax per agent. If inference fails the batch is dispatched by
        DQN_FALLBACK_ALGORITHM instead, so the background loop never
        crashes."""
        agents = list(self.env.agents)
        try:
            actions, _ = self._mappo_model.get_action(
                np.stack([self._obs[agent] for agent in agents]), deterministic=True
            )
            return {agent: int(action) for agent, action in zip(agents, actions)}
        except Exception:
            logger.exception("MAPPO inference failed; falling back to %s", DQN_FALLBACK_ALGORITHM)
            policy = _BASELINE_POLICIES[DQN_FALLBACK_ALGORITHM]
            return {
                agent: self.env.action_for_station_id(policy(self.env.simulator, agent)) for agent in agents
            }

    def _run_loop(self) -> None:
        while not self._shutdown_requested:
            with self._lock:
                status = self.status
                env = self.env
            if status != "running" or env is None:
                time.sleep(IDLE_POLL_INTERVAL)
                continue

            if isinstance(env, MultiAgentEVEnv):
                self._step_multi_agent(env)
            else:
                self._step_single_agent(env)

    def _step_single_agent(self, env: EVEnv) -> None:
        with self._lock:
            action = self._choose_action()

        obs, reward, terminated, truncated, info = env.step(action)

        with self._lock:
            if env is not self.env:
                # A reset happened concurrently; discard this stale result.
                return
            self._obs, self._info = obs, info
            self._episode_metrics.record(reward, info)
            if terminated or truncated:
                self.status = "paused"

    def _step_multi_agent(self, env: MultiAgentEVEnv) -> None:
        with self._lock:
            actions = self._choose_batch_actions_locked()

        obs, rewards, _, _, infos = env.step(actions)

        with self._lock:
            if env is not self.env:
                # A reset happened concurrently; discard this stale result.
                return
            self._obs = obs
            for agent in actions:
                self._episode_metrics.record_decision(rewards[agent], infos[agent])
            self._episode_metrics.record_resolved(env.last_resolved)
            if not env.agents:
                self.status = "paused"
