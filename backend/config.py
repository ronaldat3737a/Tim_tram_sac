"""Centralized configuration for the EV charging RL platform.

All tunable parameters referenced by simulation, environment, training and
evaluation code must be defined here instead of being hard-coded across
multiple files (per PROJECT_SPEC.md section 44).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable configuration for one simulation/training scenario."""

    # --- Scale (PROJECT_SPEC.md section 5.1) ---
    # There is no `num_nodes`: the traffic graph is always a real
    # OpenStreetMap street network (see osm_* below), so node count is
    # whatever OSM returns for the requested radius, not a configured value.
    num_stations: int = 5
    num_vehicles: int = 50
    # Vehicle depots (section 13 extension): where an EV parks once it
    # finishes charging and its RETURNING_TO_DEPOT trip completes.
    num_depots: int = 4

    # Maximum configurable scale (PROJECT_SPEC.md section 5.1).
    max_num_stations: int = 10
    max_num_vehicles: int = 100

    # --- Simulation time (PROJECT_SPEC.md section 14) ---
    time_step: float = 1.0
    # Cap on simulation_time (simulated seconds), used for termination
    # condition #2 in PROJECT_SPEC.md section 25.
    max_episode_steps: int = 5000

    # --- Edge generation (PROJECT_SPEC.md section 10) ---
    # speed here is in real meters/second (edges carry real OSM distances in
    # meters -- see network_graph.py). Chosen as a plausible congested-urban
    # driving range (~11-43 km/h); this is a first-pass recalibration
    # alongside the move to real street distances, not an empirically swept
    # constant the way the old abstract-map value was.
    min_speed: float = 3.0
    max_speed: float = 12.0
    # Traffic congestion is static per episode (confirmed decision, Phase 0
    # D.2): sampled once per edge at graph build time and held fixed for the
    # whole episode. effective_speed = speed / (1 + traffic_weight).
    min_traffic_weight: float = 0.0
    max_traffic_weight: float = 1.0

    # Edge weight key used by NetworkX shortest-path routing
    # (PROJECT_SPEC.md section 16). Must be "distance" or "travel_time".
    routing_weight: str = "travel_time"

    # --- Battery (PROJECT_SPEC.md section 15) ---
    battery_capacity: float = 1.0
    # `distance` is now real meters (a real OSM extract's edges run from a
    # few meters up to a few hundred meters, trips up to a few km). A full
    # battery affording ~1.8 km -- roughly a ~1km-radius extract's diameter,
    # not several times it -- is chosen so a meaningful fraction of trips
    # actually need a mid-trip charge (an EV that can drive several times
    # the whole map on one charge would rarely, if ever, need to charge at
    # all, which would defeat the point of the demo). This is a first-pass,
    # honestly-unvalidated estimate -- unlike the old abstract-map value
    # (tuned from a 200-seed stranded-EV study, see low_battery_threshold
    # below), it has not been empirically swept against the real graph yet.
    battery_consumption_per_distance: float = 1.0 / 1800.0
    # Battery level below which an EV requests a charging-station decision
    # (Phase 0 addition, needed for the event-driven step model, D.1/D.4).
    # The 0.5 default carries over unchanged from the abstract-map study
    # (see git history) as a starting point; that study's specific
    # stranded-EV coverage numbers have not been re-measured against the
    # real street graph's station placement, and are not claimed to still
    # hold exactly.
    low_battery_threshold: float = 0.5
    # Battery drained per simulated second by an EV standing still on the
    # road (A/C, electronics) -- roughly a tenth of what driving at a
    # typical urban speed costs. Never applies inside a station (queued or
    # charging).
    idle_battery_drain_per_tick: float = 0.0001
    # Staggered departures: each EV joins traffic at a random tick in
    # [0, max_activation_tick]; before that it does not move, drain battery
    # or ask for a decision. 0 = every EV departs at tick 0.
    max_activation_tick: int = 0

    # --- Charging station (PROJECT_SPEC.md section 12) ---
    # capacity is the total number of slots at a station, i.e.
    # capacity = num_chargers + queue capacity (confirmed decision, Phase 0
    # D.3: overload = (queue + charging_vehicles) > capacity).
    station_capacity: int = 5
    num_chargers_per_station: int = 2
    charging_rate: float = 0.05

    # --- Reward (PROJECT_SPEC.md section 21) ---
    reward_travel_weight: float = 1.0
    reward_waiting_weight: float = 1.0
    battery_failure_penalty: float = -1000.0
    # Confirmed decision, Phase 0 D.3/D.4: only the failing EV is affected,
    # a station overload never terminates the whole episode; it is
    # discouraged through this configurable penalty instead.
    station_overload_penalty: float = -50.0
    # Picking a station the EV cannot safely reach with its current battery
    # (PROJECT_SPEC.md section 20). EVEnv overrides such a pick with the
    # nearest reachable station (so the decision still resolves and time
    # still advances) and adds this penalty on top of that trip's own cost.
    # 10x battery_failure_penalty, so an invalid pick is never a cheaper
    # "escape" than any real outcome, including stranding an EV.
    invalid_action_penalty: float = -10000.0
    # EVEnv.step() multiplies every reward by this before returning it
    # (penalties above stay in their natural units here). Brings the
    # -10000 invalid penalty down to -100 and a typical ~-200 trip cost to
    # ~-2, a range where DQN's Huber loss can still tell apart the small
    # differences between two valid stations instead of being dominated by
    # the rare huge penalties.
    reward_scale: float = 0.01

    # --- Reproducibility (PROJECT_SPEC.md section 26) ---
    random_seed: int = 42

    # --- Real street network (section 11). The traffic graph is always
    # fetched from OpenStreetMap via osmnx and disk-cached to
    # osm_cache_path as GraphML, so repeated calls (every test run, every
    # simulator reset) reuse the cached file instead of re-querying the
    # Overpass API (reproducible per section 26, no network dependency on
    # the hot path per section 33). Default center: Công viên Nghĩa Đô,
    # Cầu Giấy, Hà Nội (approximate park-center coordinates).
    osm_center_lat: float = 21.0385
    osm_center_lng: float = 105.7973
    osm_radius_meters: float = 1000.0
    osm_cache_path: str = "data/osm_nghia_do.graphml"

    # --- Station/depot POI (spatial-geometry refactor). A station/depot is
    # not itself a traffic node: it is a small POI node connected to one
    # real "access node" by a short spur edge, offset a real physical
    # distance perpendicular to the access node's road (Task 2 -- an actual
    # orthogonal-offset computation, not an arbitrary degree delta) so it
    # renders beside the road, never on top of it.
    poi_offset_meters: float = 10.0
    # Speed an EV crawls the spur at while entering/exiting a station or
    # depot (Task 3: a real, if brief, "smooth transition trajectory" --
    # not a teleport) -- a deliberately slow, fixed "parking maneuver" pace,
    # independent of the EV's own top speed on the real road network.
    poi_approach_speed_mps: float = 1.5


DEFAULT_CONFIG = SimulationConfig()

# The real-OSM-map scenario the live demo runs AND the DQN is trained on
# (Nghia Do extract, data/osm_nghia_do.graphml). Its one difference from
# DEFAULT_CONFIG is a slow charging_rate: a full 0.0 -> 1.0 charge takes ~900
# simulated ticks (~3 real minutes at the default UI speed of 5 ticks/sec),
# so the battery bar visibly ticks up during a demo. ai_core/train.py uses
# this exact config so the served model is trained on the same station-
# occupancy dynamics it faces in the API -- a model trained on the fast
# DEFAULT_CONFIG charging would badly underestimate how long stations stay
# occupied here.
#
# Physics recalibrated for the real map (road distances run far longer than
# straight-line distance across it), so that under the nearest_station
# baseline no EV fails or is stranded purely because of the environment:
# - battery_consumption_per_distance 1/5500: a full battery covers ~5.5 km
#   (battery_capacity stays the normalized 1.0 -- low_battery_threshold and
#   charging_rate are absolute battery units, so range is set here).
# - low_battery_threshold 0.75: an EV asks for a station with ~4.1 km of
#   range left. Measured over 200 seeds, the farthest any road node sits
#   from its nearest (randomly placed) station is up to ~3.45 km of real
#   road, so a lower threshold strands EVs no policy could save (0.2 left
#   ~2 stranded per episode). 0.75 gave 0 stranded / 0 failed under
#   nearest_station on 30 seeds with ~39 decisions per 50-EV episode; the
#   alternative of 0.2 with a ~20 km range also reached 0 stranded but
#   left only ~4 decisions per episode, too few to learn congestion from.
# - max_episode_steps 20000: each charge takes ~900 ticks, so a 5000-tick
#   cap truncated every episode before its EVs could finish.
# - max_activation_tick 8000: a full battery only drives ~180 ticks before
#   reaching the 0.75 threshold, so with every EV departing at tick 0 all
#   decisions landed in the first ~300 ticks. Staggering departures spreads
#   them over the episode.
OSM_DEMO_CONFIG = replace(
    DEFAULT_CONFIG,
    charging_rate=1.0 / 900,
    battery_consumption_per_distance=1.0 / 5500.0,
    low_battery_threshold=0.75,
    max_episode_steps=20_000,
    max_activation_tick=8_000,
)
