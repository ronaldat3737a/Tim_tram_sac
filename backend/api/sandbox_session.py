"""One live MAPPO-GNN simulation of the active sandbox, owned by one WebSocket
connection (routes/sandbox.py).

The episode runs in its own worker thread (stepping it is CPU-bound and must
never block the event loop), dispatching every decision batch with the
trained MAPPO-GNN (argmax, sequential decisions; zero-shot on this map).
As in the simulation dashboard (simulation_manager.py), the env's tick hook
paces the simulation (1/speed seconds per tick) and publishes the state;
the snapshot uses that dashboard's own build_snapshot, so its WebSocket
clients can render it unchanged. Snapshots are rebuilt at most every
SNAPSHOT_INTERVAL seconds (what a client is sent anyway), not every tick:
with thousands of EVs that alone would cost more than the tick.

Lifecycle: "preparing" (routing the EVs that depart at the start, with
progress) -> "paused" at t=0 -> "running" <-> "paused" -> "finished" (every EV
done) or "error". close() stops the worker within one tick (or one route /
one 0.1 s pause poll) and joins it, so a closed connection leaves no thread,
env or graph copy behind.

Only one session exists at a time: opening one closes the previous (e.g. a
reloaded page whose old connection has not dropped yet).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from backend.ai_core.graph_env import DEFAULT_NEIGHBORS
from backend.ai_core.mappo_gnn import MAPPOGNNAgent
from backend.ai_core.sandbox_env import SandboxGraphEnv
from backend.ai_core.train_gnn import DEFAULT_OUTPUT_PATH as GNN_MODEL_PATH
from backend.api.simulation_manager import _EpisodeAccumulator, build_snapshot
from backend.simulation.sandbox_network import SandboxNetwork

logger = logging.getLogger(__name__)

DEFAULT_SPEED = 5.0  # simulated ticks per real second, as the dashboard
MAX_SPEED = 200.0
SNAPSHOT_INTERVAL = 0.1
PAUSE_POLL_INTERVAL = 0.1
DEFAULT_DECISION_WINDOW = 15
_STATES = ("TRAVELING", "WAITING", "CHARGING", "RETURNING_TO_DEPOT", "COMPLETED", "FAILED")


class _Stopped(Exception):
    """Raised inside the worker to unwind it when the session closes."""


class SandboxSession:
    def __init__(self, sandbox_id: str, sandbox: SandboxNetwork, seed: int, model_path: Path | None = None):
        self.sandbox_id = sandbox_id
        self.sandbox = sandbox
        self.seed = seed
        self.model_path = Path(model_path or GNN_MODEL_PATH)
        self.speed = DEFAULT_SPEED
        self._lock = threading.Lock()
        self._running = threading.Event()  # set: run; clear: pause
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._reset_state()

    # --- lifecycle -----------------------------------------------------

    def open(self) -> None:
        self._stop.clear()
        self._running.clear()
        self._reset_state()
        self._worker = threading.Thread(target=self._run, name=f"sandbox-{self.sandbox_id}", daemon=True)
        self._worker.start()

    def request_stop(self) -> None:
        """Tell the worker to stop, without waiting: it unwinds within one
        tick. Safe to call from anywhere, including a cancelled coroutine."""
        self._stop.set()

    def close(self, timeout: float = 10.0) -> None:
        """Stop the worker and wait for it to exit."""
        self._stop.set()
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout)
            if worker.is_alive():
                logger.warning("sandbox worker %s did not stop within %.0fs", worker.name, timeout)
        with self._lock:
            if self._phase not in ("finished", "error"):
                self._phase = "stopped"

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    # --- control (from the WebSocket) ------------------------------------

    def control(self, message: dict) -> None:
        """{"action": "start" | "resume" | "pause" | "reset"} or
        {"action": "speed", "value": ticks_per_second}. "start" before the
        session is ready makes it run as soon as it is. Raises ValueError
        for anything else."""
        action = message.get("action")
        if action in ("start", "resume"):
            self._running.set()
        elif action == "pause":
            self._running.clear()
        elif action == "speed":
            value = float(message.get("value", 0))
            if not 0 < value <= MAX_SPEED:
                raise ValueError(f"speed must be in (0, {MAX_SPEED:g}] ticks/s")
            self.speed = value
        elif action == "reset":
            self.close()
            self.open()
        else:
            raise ValueError(f"unknown action {action!r}")

    # --- state (to the WebSocket) ----------------------------------------

    def latest_snapshot(self) -> tuple[int, dict | None]:
        with self._lock:
            return self._snapshot_version, self._snapshot

    def status_message(self) -> dict:
        with self._lock:
            metrics = self._metrics
            status = self._phase
            if status == "ready":
                status = "running" if self._running.is_set() else "paused"
            return {
                "type": "sandbox_status",
                "sandbox_id": self.sandbox_id,
                "status": status,
                "error": self._error,
                "simulation_time": self._simulation_time,
                "speed": self.speed,
                "preparing": {"routes_done": self._routes_done, "routes_total": self._routes_total},
                "num_stations": self.sandbox.config.num_stations,
                "num_vehicles_total": self.sandbox.config.num_vehicles,
                "num_vehicles_departed": self._departed,
                **{f"num_vehicles_{state.lower()}": self._counts[state] for state in _STATES},
                "num_decisions": metrics.num_decisions,
                "num_invalid_actions": metrics.num_invalid_actions,
                "num_overloaded_events": metrics.num_overloaded_events,
                "num_stranded_vehicles": metrics.num_stranded_vehicles,
                "average_travel_time": metrics.average_travel_time,
                "average_waiting_time": metrics.average_waiting_time,
                "total_system_cost": metrics.total_system_cost,
                "episode_reward": metrics.episode_reward,
                "average_decision_time_ms": (
                    1000 * self._decision_seconds / metrics.num_decisions if metrics.num_decisions else 0.0
                ),
            }

    # --- worker ----------------------------------------------------------

    def _reset_state(self) -> None:
        self._phase = "preparing"  # preparing | ready | finished | error | stopped
        self._error: str | None = None
        self._routes_done = 0
        self._routes_total = 0
        self._metrics = _EpisodeAccumulator()
        self._decision_seconds = 0.0
        self._snapshot: dict | None = None
        self._snapshot_version = 0
        self._snapshot_at = 0.0
        self._simulation_time = 0.0
        self._departed = 0
        self._counts = dict.fromkeys(_STATES, 0)

    def _run(self) -> None:
        try:
            agent = MAPPOGNNAgent.load(self.model_path)
            agent.actor.eval()
            agent.critic.eval()
            env = SandboxGraphEnv(
                self.sandbox,
                tick_hook=self._on_tick,
                decision_window=agent.metadata.get("decision_window", DEFAULT_DECISION_WINDOW),
                num_neighbors=agent.metadata.get("num_neighbors", DEFAULT_NEIGHBORS),
            )
            env.route_progress = self._on_route_progress
            env.before_first_tick = self._on_ready
            env.reset(seed=self.seed)
            while env.agents:
                self._check_stop()
                started = time.perf_counter()
                agents, _, actions, _ = agent.act(env, deterministic=True)
                decided = time.perf_counter() - started
                _, rewards, _, _, infos = env.step(dict(zip(agents, actions.tolist())))
                with self._lock:
                    self._decision_seconds += decided
                    for a in agents:
                        self._metrics.record_decision(rewards[a], infos[a])
                    self._metrics.record_resolved(env.last_resolved)
            self._publish(env.simulator, force=True)
            with self._lock:
                self._phase = "finished"
        except _Stopped:
            pass
        except Exception as exc:
            logger.exception("sandbox session %s failed", self.sandbox_id)
            with self._lock:
                self._phase = "error"
                self._error = f"{type(exc).__name__}: {exc}"

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise _Stopped

    def _on_route_progress(self, done: int, total: int) -> None:
        self._check_stop()
        with self._lock:
            self._routes_done, self._routes_total = done, total

    def _on_ready(self, simulator) -> None:
        """The episode is built and routed, at t=0: publish it, then hold
        until "start" (or close)."""
        self._publish(simulator, force=True)
        with self._lock:
            self._phase = "ready"
        self._wait_while_paused()

    def _on_tick(self, simulator) -> None:
        self._check_stop()
        self._wait_while_paused()
        self._publish(simulator)
        if self._stop.wait(1.0 / self.speed):
            raise _Stopped

    def _wait_while_paused(self) -> None:
        while not self._running.wait(PAUSE_POLL_INTERVAL):
            self._check_stop()
        self._check_stop()

    def _publish(self, simulator, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._snapshot_at < SNAPSHOT_INTERVAL:
            with self._lock:
                self._simulation_time = simulator.simulation_time
            return
        snapshot = build_snapshot(simulator, self.sandbox.config, 1.0 / self.speed)
        counts = dict.fromkeys(_STATES, 0)
        departed = 0
        for vehicle in simulator.vehicles.values():
            if simulator.is_active(vehicle):
                departed += 1
                counts[vehicle.state.value] += 1
        with self._lock:
            self._snapshot = snapshot
            self._snapshot_version += 1
            self._snapshot_at = now
            self._simulation_time = simulator.simulation_time
            self._counts = counts
            self._departed = departed


_current: SandboxSession | None = None
_current_lock = threading.Lock()


def open_session(sandbox_id: str, sandbox: SandboxNetwork, seed: int, model_path: Path | None = None) -> SandboxSession:
    """Close the current session, if any, and open a new one."""
    global _current
    with _current_lock:
        previous, _current = _current, None
    if previous is not None:
        previous.close()
    session = SandboxSession(sandbox_id, sandbox, seed, model_path)
    session.open()
    with _current_lock:
        _current = session
    return session


def release_session(session: SandboxSession) -> None:
    """Stop `session` without waiting and forget it if it is the current one.
    Non-blocking, so it can run even in a cancelled WebSocket handler."""
    global _current
    session.request_stop()
    with _current_lock:
        if _current is session:
            _current = None


def close_session(session: SandboxSession) -> None:
    """release_session, then wait for the worker to exit."""
    release_session(session)
    session.close()


def current_session() -> SandboxSession | None:
    return _current
