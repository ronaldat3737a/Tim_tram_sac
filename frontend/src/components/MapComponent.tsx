"use client";

import "leaflet/dist/leaflet.css";
import L, { type LatLngTuple } from "leaflet";
import { Car, ParkingSquare, Zap } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import type { MyCarPreviewResponse, NetworkInfo, StationUpdate, VehicleUpdate } from "@/lib/types";

// Initial view only: FitToNetwork fits the camera to the real network's
// bounds as soon as it is shown, and again on every scenario switch
// (section 11: the graph is always a real OpenStreetMap street network).
const FALLBACK_CENTER: LatLngTuple = [21.0385, 105.7973];
const DEFAULT_ZOOM = 15;
const FIT_PADDING_PX: [number, number] = [16, 16];

// Free, no-API-key tile source (section 7: "Không yêu cầu API trả phí").
// CartoDB's basemaps.cartocdn.com now requires a registered API key even
// for the free "light_all" style (confirmed by an "API KEY REQUIRED"
// watermark when tried, also for "rastertiles/voyager"). The standard
// tile.openstreetmap.org server is unreachable from some networks (their
// DNS answers 127.0.0.1 for it), which left the map with no basemap at all,
// so the default is OpenStreetMap France's key-free rendering of the same
// OSM data. NEXT_PUBLIC_TILE_URL overrides it.
const TILE_URL = process.env.NEXT_PUBLIC_TILE_URL ?? "https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, tiles by OpenStreetMap France';

// TRAVELING stays the plain default green; WAITING/CHARGING/
// RETURNING_TO_DEPOT each get a distinct color *and* a small badge (see
// buildVehicleIcon) so none of them ever reads as "frozen/stuck" -- color
// plus badge makes the state legible at a glance.
const VEHICLE_COLORS: Record<VehicleUpdate["state"], string> = {
  TRAVELING: "#22c55e",
  WAITING: "#f59e0b",
  CHARGING: "#3b82f6",
  RETURNING_TO_DEPOT: "#8b5cf6",
  COMPLETED: "#9ca3af",
  FAILED: "#ef4444",
};

// Neon-green halo/ping on every real EV marker signals "this is a vehicle
// the system is actively dispatching", independent of the body color (which
// still encodes vehicle.state) -- and visually separates real EVs from the
// plain gray decorative background traffic dots.
const EV_HALO_COLOR = "#39ff14";
const MY_CAR_RING_COLOR = "#facc15"; // gold ring: the one designated "my car"
// Non-app EVs (exogenous noise the policy never dispatches) swap the neon
// halo for a static, dashed orange ring with no ping, so they stand apart
// from AI-dispatched EVs while the body color still encodes vehicle.state.
const NON_APP_RING_COLOR = "#f97316";

// Once an EV has arrived (WAITING/CHARGING at a station, COMPLETED at a
// depot or its destination), vehicle.x/y IS that exact node position, so
// every car there shares one coordinate with the station/depot icon. The
// marker stays at that exact coordinate; only its icon is shifted, in
// screen pixels, onto a ring around the node: chargers on the inner ring,
// the queue on outer rings. A pixel layout stays clear of the 26px
// station icon and of the other cars at every zoom level, which a fixed
// geographic offset cannot (a few meters is ~1-4px at zoom 15).
const PARKED_ICON_SIZE = 20;
const MY_PARKED_ICON_SIZE = 26;
const INNER_RING_RADIUS_PX = 26;
const QUEUE_RING_RADIUS_PX = 46;
const QUEUE_RING_SPACING_PX = 20;
const QUEUE_RING_CAPACITY = 10;
// Leaflet stacks markers by screen y, not by JSX order, so a car just north
// of its station would sit under the station icon. Vehicles always win.
const VEHICLE_Z_INDEX_OFFSET = 10_000;
const MY_CAR_Z_INDEX_OFFSET = 20_000;
// Above this many EVs on the map, every EV but "my car" uses the lite icon.
const LITE_ICON_VEHICLE_THRESHOLD = 100;

type PixelOffset = [number, number];

function ringOffset(index: number, count: number, radius: number): PixelOffset {
  const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(count, 1);
  return [Math.round(Math.cos(angle) * radius), Math.round(Math.sin(angle) * radius)];
}

// Queue slots fill consecutive outer rings of QUEUE_RING_CAPACITY cars.
function queueOffset(index: number, count: number): PixelOffset {
  const ring = Math.floor(index / QUEUE_RING_CAPACITY);
  const onThisRing = Math.min(count - ring * QUEUE_RING_CAPACITY, QUEUE_RING_CAPACITY);
  return ringOffset(index % QUEUE_RING_CAPACITY, onThisRing, QUEUE_RING_RADIUS_PX + ring * QUEUE_RING_SPACING_PX);
}

// Icon offset for every vehicle parked at a station/depot/destination node,
// keyed by vehicle id. Slots are assigned in vehicle-id order within each
// group, so a car keeps its slot while the group is unchanged.
function parkedOffsets(vehicles: VehicleUpdate[]): Map<number, PixelOffset> {
  const groups = new Map<string, { charging: number[]; waiting: number[]; parked: number[] }>();
  const group = (key: string) => {
    let entry = groups.get(key);
    if (!entry) {
      entry = { charging: [], waiting: [], parked: [] };
      groups.set(key, entry);
    }
    return entry;
  };
  for (const vehicle of vehicles) {
    if ((vehicle.state === "CHARGING" || vehicle.state === "WAITING") && vehicle.station_id !== null) {
      const entry = group(`station-${vehicle.station_id}`);
      (vehicle.state === "CHARGING" ? entry.charging : entry.waiting).push(vehicle.id);
    } else if (vehicle.state === "COMPLETED") {
      group(`node-${vehicle.node}`).parked.push(vehicle.id);
    }
  }
  const offsets = new Map<number, PixelOffset>();
  for (const { charging, waiting, parked } of groups.values()) {
    for (const ids of [charging, waiting, parked]) ids.sort((a, b) => a - b);
    charging.forEach((id, i) => offsets.set(id, ringOffset(i, charging.length, INNER_RING_RADIUS_PX)));
    waiting.forEach((id, i) => offsets.set(id, queueOffset(i, waiting.length)));
    parked.forEach((id, i) => offsets.set(id, ringOffset(i, parked.length, INNER_RING_RADIUS_PX)));
  }
  return offsets;
}

// Backend node/vehicle (x, y) are always real (lng, lat) -- section 11: the
// graph is always a real OpenStreetMap street network -- so this is a
// direct [lat, lng] reorder, no scaling.
function toLatLng(x: number, y: number): LatLngTuple {
  return [y, x];
}

function edgeKey(source: number, target: number): string {
  return `${source}-${target}`;
}

// --- Icons -------------------------------------------------------------

const EV_ICON_HTML = renderToStaticMarkup(<Car color="#052e16" strokeWidth={2.5} size={15} />);
const EV_ICON_SMALL_HTML = renderToStaticMarkup(<Car color="#052e16" strokeWidth={2.5} size={11} />);
const STATION_ICON_HTML = renderToStaticMarkup(
  <Zap color="#ffffff" fill="#ffffff" strokeWidth={2} size={14} />,
);
const DEPOT_ICON_HTML = renderToStaticMarkup(
  <ParkingSquare color="#ffffff" fill="#ffffff" strokeWidth={2} size={16} />,
);
const BADGE_ZAP_HTML = renderToStaticMarkup(<Zap color="#ffffff" fill="#ffffff" strokeWidth={2} size={9} />);
const BADGE_PARKING_HTML = renderToStaticMarkup(
  <ParkingSquare color="#ffffff" fill="#ffffff" strokeWidth={2} size={9} />,
);

// A small state badge in the corner of the icon, distinct from body color,
// so WAITING ("W", pulsing), CHARGING (lightning bolt) and
// RETURNING_TO_DEPOT (parking icon) never read as a frozen/disconnected
// vehicle -- TRAVELING/COMPLETED/FAILED get no badge, only the states that
// could otherwise look "stuck".
function stateBadgeHtml(state: VehicleUpdate["state"], lite = false): string {
  const badgeBase =
    "position:absolute;top:-4px;right:-4px;width:14px;height:14px;border-radius:9999px;" +
    "display:flex;align-items:center;justify-content:center;border:1.5px solid #ffffff;" +
    "color:#ffffff;font-size:9px;font-weight:700;line-height:1;";
  if (state === "WAITING") {
    return `<span class="${lite ? "" : "animate-pulse"}" style="${badgeBase}background:#d97706;">W</span>`;
  }
  if (state === "CHARGING") {
    return `<span style="${badgeBase}background:#1d4ed8;">${BADGE_ZAP_HTML}</span>`;
  }
  if (state === "RETURNING_TO_DEPOT") {
    return `<span style="${badgeBase}background:#6d28d9;">${BADGE_PARKING_HTML}</span>`;
  }
  return "";
}

interface VehicleIconOptions {
  mine?: boolean;
  nonApp?: boolean;
  // Set for a car parked at a node: smaller icon shifted by this many pixels.
  offset?: PixelOffset;
  // Many EVs on screen (the large scenario): no ping animation, glow or
  // pulsing badge -- hundreds of continuously animated, shadowed DOM
  // elements are what make the browser stutter. Body color, ring and badge
  // still encode the same state.
  lite?: boolean;
}

function buildVehicleIcon(
  color: string,
  state: VehicleUpdate["state"],
  options: VehicleIconOptions = {},
): L.DivIcon {
  const ringColor = options.mine ? MY_CAR_RING_COLOR : options.nonApp ? NON_APP_RING_COLOR : EV_HALO_COLOR;
  const parked = options.offset !== undefined;
  const size = parked ? (options.mine ? MY_PARKED_ICON_SIZE : PARKED_ICON_SIZE) : options.mine ? 32 : 26;
  const [dx, dy] = options.offset ?? [0, 0];
  // The ping halo means "on the road, being dispatched"; a parked car has
  // none (it would also pulse over its neighbors on the ring).
  const ping = options.nonApp || parked || options.lite
    ? ""
    : `<span class="animate-ping" style="position:absolute;inset:0;border-radius:9999px;background:${ringColor};opacity:0.55;"></span>`;
  const borderWidth = options.mine || options.nonApp ? 3 : 2;
  const borderStyle = options.nonApp ? "dashed" : "solid";
  return L.divIcon({
    className: parked ? "ev-marker-icon ev-marker-parked" : "ev-marker-icon",
    html: `
      <div style="position:relative;width:${size}px;height:${size}px;">
        ${ping}
        <div style="position:relative;width:${size}px;height:${size}px;border-radius:9999px;background:${color};border:${borderWidth}px ${borderStyle} ${ringColor};display:flex;align-items:center;justify-content:center;${options.lite ? "" : `box-shadow:0 0 8px ${ringColor};`}">
          ${parked ? EV_ICON_SMALL_HTML : EV_ICON_HTML}
        </div>
        ${stateBadgeHtml(state, options.lite)}
      </div>
    `,
    iconSize: [size, size],
    // Shifting the anchor moves the icon by (dx, dy) pixels while the
    // marker itself stays at the vehicle's exact coordinate.
    iconAnchor: [size / 2 - dx, size / 2 - dy],
    tooltipAnchor: [dx, dy - size / 2],
  });
}

// Icons are rebuilt only for a new (state, kind, slot) combination.
const vehicleIconCache = new Map<string, L.DivIcon>();

function vehicleIcon(state: VehicleUpdate["state"], options: VehicleIconOptions): L.DivIcon {
  const key = `${state}|${options.mine ? 1 : 0}|${options.nonApp ? 1 : 0}|${options.lite ? 1 : 0}|${options.offset?.join(",") ?? "-"}`;
  let icon = vehicleIconCache.get(key);
  if (!icon) {
    icon = buildVehicleIcon(VEHICLE_COLORS[state], state, options);
    vehicleIconCache.set(key, icon);
  }
  return icon;
}

const STATION_ICON = L.divIcon({
  className: "station-marker-icon",
  html: `
    <div style="width:26px;height:26px;border-radius:9999px;background:#ea580c;border:2px solid #ffffff;display:flex;align-items:center;justify-content:center;box-shadow:0 1px 4px rgba(0,0,0,0.45);">
      ${STATION_ICON_HTML}
    </div>
  `,
  iconSize: [26, 26],
  iconAnchor: [13, 13],
});

const DEPOT_ICON = L.divIcon({
  className: "",
  html: `
    <div style="width:30px;height:30px;border-radius:8px;background:#334155;border:2px solid #ffffff;display:flex;align-items:center;justify-content:center;box-shadow:0 1px 4px rgba(0,0,0,0.45);">
      ${DEPOT_ICON_HTML}
    </div>
  `,
  iconSize: [30, 30],
  iconAnchor: [15, 15],
});

// --- Edge styling --------------------------------------------------------
// Task 4: three visually distinct layers prove, by eye, that vehicles/
// stations/depots stick exactly to the graph's real geometry:
//   1. MUTED_ROAD_* -- every real road edge, drawn faint underneath
//      everything else -- "the ground truth network".
//   2. Traffic-colored road edges on top of that -- the same edges,
//      colored by congestion.
//   3. Spur edges (a station/depot's short connector to its access node)
//      drawn in a third, distinct, fixed color -- visibly branching off
//      the muted road layer rather than being part of the road itself.

const MUTED_ROAD_COLOR = "#9ca3af";
const MUTED_ROAD_OPACITY = 0.4;
const STATION_SPUR_COLOR = "#ea580c";
const DEPOT_SPUR_COLOR = "#334155";

// Line weight per OSM road class, shared by the background road layer and
// the traffic-colored routing edges, so a primary road reads thicker than a
// residential street or an alley.
const ROAD_WEIGHTS: Record<string, number> = {
  motorway: 7, motorway_link: 4, trunk: 7, trunk_link: 4,
  primary: 6, primary_link: 4, secondary: 5, secondary_link: 3.5,
  tertiary: 4, tertiary_link: 3, unclassified: 3, residential: 3,
  living_street: 2.5, road: 2.5, service: 1.6,
};
const DEFAULT_ROAD_WEIGHT = 3;
// Background layer: every drivable street, in neutral tones, darker for
// bigger roads.
const BACKGROUND_ROAD_COLORS: Record<string, string> = {
  motorway: "#475569", trunk: "#475569", primary: "#475569",
  secondary: "#64748b", tertiary: "#64748b",
};
const BACKGROUND_ROAD_DEFAULT_COLOR = "#94a3b8";

function roadWeight(highway: string | null | undefined): number {
  return (highway && ROAD_WEIGHTS[highway]) || DEFAULT_ROAD_WEIGHT;
}

function backgroundRoadColor(highway: string): string {
  return BACKGROUND_ROAD_COLORS[highway.replace(/_link$/, "")] ?? BACKGROUND_ROAD_DEFAULT_COLOR;
}

// Streets are drawn in merged per-style groups (StaticRoadLayers), so the
// continuous green -> red congestion scale is quantized into this many steps.
const TRAFFIC_COLOR_BUCKETS = 10;
// The dashed underlay is drawn only up to this many routing edges (the small
// scenario has ~500; the large one's full network ~9.5k).
const DASHED_UNDERLAY_MAX_EDGES = 2000;
// Own pane for the street canvas, below Leaflet's overlayPane (400), so
// spurs, decorative traffic and the preview route stay on top of it.
const ROADS_PANE = "static-roads";
const ROADS_PANE_Z_INDEX = "350";

function trafficBucket(weight: number, maxWeight: number): number {
  const ratio = maxWeight > 0 ? Math.min(Math.max(weight / maxWeight, 0), 1) : 0;
  return Math.min(Math.floor(ratio * TRAFFIC_COLOR_BUCKETS), TRAFFIC_COLOR_BUCKETS - 1);
}

function trafficBucketColor(bucket: number): string {
  const ratio = (bucket + 0.5) / TRAFFIC_COLOR_BUCKETS;
  const hue = 120 - 120 * ratio; // 120=green (thông thoáng) -> 0=red (tắc)
  return `hsl(${hue.toFixed(0)}, 75%, 45%)`;
}

// --- Decorative-only background traffic: a handful of gray dots drifting
// along the real road polylines purely for visual atmosphere. Never sent to
// or derived from the backend, never affects simulation/reward/routing
// (section 39) -- pure client-side cosmetic state. ----------------------

const FAKE_TRAFFIC_COUNT = 8;
const FAKE_TRAFFIC_TICK_MS = 200;

interface FakeCar {
  id: number;
  position: LatLngTuple;
}

function useFakeTraffic(edgePositions: LatLngTuple[][]): FakeCar[] {
  const [cars, setCars] = useState<FakeCar[]>([]);
  const stateRef = useRef<{ edgeIndex: number; t: number; speed: number }[]>([]);

  useEffect(() => {
    if (edgePositions.length === 0) {
      return;
    }
    stateRef.current = Array.from({ length: FAKE_TRAFFIC_COUNT }, () => ({
      edgeIndex: Math.floor(Math.random() * edgePositions.length),
      t: Math.random(),
      speed: 0.01 + Math.random() * 0.02,
    }));

    const interval = setInterval(() => {
      stateRef.current = stateRef.current.map((car) => {
        let { edgeIndex, t } = car;
        t += car.speed;
        if (t >= 1) {
          t = 0;
          edgeIndex = Math.floor(Math.random() * edgePositions.length);
        }
        return { edgeIndex, t, speed: car.speed };
      });
      setCars(
        stateRef.current.map((car, id) => {
          const points = edgePositions[car.edgeIndex];
          const segment = pointAlongPolyline(points, car.t);
          return { id, position: segment };
        }),
      );
    }, FAKE_TRAFFIC_TICK_MS);

    return () => clearInterval(interval);
  }, [edgePositions]);

  return cars;
}

function pointAlongPolyline(points: LatLngTuple[], fraction: number): LatLngTuple {
  if (points.length === 1) return points[0];
  const index = Math.min(Math.floor(fraction * (points.length - 1)), points.length - 2);
  const localT = fraction * (points.length - 1) - index;
  const [lat1, lng1] = points[index];
  const [lat2, lng2] = points[index + 1];
  return [lat1 + (lat2 - lat1) * localT, lng1 + (lng2 - lng1) * localT];
}

// --- Depot "parked, then fades away" lifecycle (frontend-only cosmetic). -
// Backend now drives a vehicle to a real depot node via real simulated
// movement (RETURNING_TO_DEPOT), and only reports COMPLETED once it truly
// arrives there -- this hook is purely about how long the *frontend* keeps
// rendering that already-parked marker before decluttering the map: fully
// visible for DEPOT_FADE_START_DELAY_MS, then fades to transparent over
// DEPOT_FADE_DURATION_MS, then is removed from the rendered list entirely.
// None of this is sent to or read from the backend.

const DEPOT_FADE_START_DELAY_MS = 1500;
const DEPOT_FADE_DURATION_MS = 2000;

function useCompletedVehicleFade(vehicles: VehicleUpdate[]): {
  isFading: (vehicle: VehicleUpdate) => boolean;
  isHidden: (vehicle: VehicleUpdate) => boolean;
} {
  const [fadingIds, setFadingIds] = useState<Set<number>>(new Set());
  const [hiddenIds, setHiddenIds] = useState<Set<number>>(new Set());
  const timersRef = useRef<Map<number, ReturnType<typeof setTimeout>[]>>(new Map());

  useEffect(() => {
    for (const vehicle of vehicles) {
      if (vehicle.state === "COMPLETED" && !timersRef.current.has(vehicle.id)) {
        const fadeTimer = setTimeout(() => {
          setFadingIds((prev) => new Set(prev).add(vehicle.id));
        }, DEPOT_FADE_START_DELAY_MS);
        const hideTimer = setTimeout(() => {
          setHiddenIds((prev) => new Set(prev).add(vehicle.id));
        }, DEPOT_FADE_START_DELAY_MS + DEPOT_FADE_DURATION_MS);
        timersRef.current.set(vehicle.id, [fadeTimer, hideTimer]);
      }
    }
  }, [vehicles]);

  useEffect(() => {
    const timersMap = timersRef.current;
    return () => {
      for (const timers of timersMap.values()) {
        for (const timer of timers) clearTimeout(timer);
      }
    };
  }, []);

  // A vehicle id is reused across episodes. Rather than needing an explicit
  // "episode changed, wipe every timer/set" reset (which either fights the
  // lint rules against touching refs/state synchronously during render, or
  // needs an effect that races the very state it's resetting), the fade/
  // hidden membership is only ever honored while the SAME vehicle is
  // currently COMPLETED. The moment a reused id is TRAVELING again (a new
  // episode, or this same episode's next one), any stale membership from a
  // previous life is simply ignored -- self-correcting by construction.
  return {
    isFading: (vehicle) => vehicle.state === "COMPLETED" && fadingIds.has(vehicle.id),
    isHidden: (vehicle) => vehicle.state === "COMPLETED" && hiddenIds.has(vehicle.id),
  };
}

// -------------------------------------------------------------------------

interface MapComponentProps {
  network: NetworkInfo | null;
  vehicles: VehicleUpdate[];
  stations: StationUpdate[];
  myVehicleId: number | null;
  myCarPreview: MyCarPreviewResponse | null;
  onStationSelect: (stationId: number | null) => void;
}

export default function MapComponent({
  network,
  vehicles,
  stations,
  myVehicleId,
  myCarPreview,
  onStationSelect,
}: MapComponentProps) {
  const stationById = useMemo(() => {
    const map = new Map<number, StationUpdate>();
    for (const station of stations) map.set(station.id, station);
    return map;
  }, [stations]);

  const nodeById = useMemo(() => {
    if (!network) return new Map<number, NetworkInfo["nodes"][number]>();
    return new Map(network.nodes.map((node) => [node.id, node]));
  }, [network]);

  // A station/depot's POI node id -- used to tell a real road edge apart
  // from a spur edge (one endpoint is a POI node) with no backend change
  // needed, since network.stations/depots already give us node_id.
  const stationPoiNodeIds = useMemo(
    () => new Set(network?.stations.map((s) => s.node_id) ?? []),
    [network],
  );
  const depotPoiNodeIds = useMemo(() => new Set(network?.depots.map((d) => d.node_id) ?? []), [network]);

  // Backend-provided geometry per edge (network_graph.py's `geometry`, a
  // real shapely LineString under the hood) -- the exact path vehicles
  // move along, split into "real road" vs "spur" so each can get its own
  // layer/styling (Task 4). Keyed the same way network.edges is, so
  // lookups stay consistent even if some edge is skipped -- a plain
  // filtered array would let its indices drift out of sync with
  // network.edges once that happens. Already [lat, lng] pairs from the API.
  const { roadEdgePositions, stationSpurPositions, depotSpurPositions } = useMemo(() => {
    const road = new Map<string, LatLngTuple[]>();
    const stationSpur = new Map<string, LatLngTuple[]>();
    const depotSpur = new Map<string, LatLngTuple[]>();
    if (!network) return { roadEdgePositions: road, stationSpurPositions: stationSpur, depotSpurPositions: depotSpur };

    for (const edge of network.edges) {
      if (edge.geometry.length < 2) continue;
      const key = edgeKey(edge.source, edge.target);
      const positions = edge.geometry.map(([lat, lng]) => [lat, lng] as LatLngTuple);
      const isStationSpur = stationPoiNodeIds.has(edge.source) || stationPoiNodeIds.has(edge.target);
      const isDepotSpur = depotPoiNodeIds.has(edge.source) || depotPoiNodeIds.has(edge.target);
      if (isStationSpur) stationSpur.set(key, positions);
      else if (isDepotSpur) depotSpur.set(key, positions);
      else road.set(key, positions);
    }
    return { roadEdgePositions: road, stationSpurPositions: stationSpur, depotSpurPositions: depotSpur };
  }, [network, stationPoiNodeIds, depotPoiNodeIds]);

  const fakeTraffic = useFakeTraffic(useMemo(() => [...roadEdgePositions.values()], [roadEdgePositions]));
  const { isFading, isHidden } = useCompletedVehicleFade(vehicles);

  const maxTrafficWeight = useMemo(() => {
    if (!network || network.edges.length === 0) return 1;
    return Math.max(...network.edges.map((e) => e.traffic_weight), 0.0001);
  }, [network]);

  if (!network) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-gray-500">
        Loading network…
      </div>
    );
  }

  // Every marker is placed at the exact coordinate the backend reports;
  // cars parked at a node only get their icon shifted (see parkedOffsets).
  // Cars already faded out free their slot.
  const visibleVehicles = vehicles.filter((vehicle) => !isHidden(vehicle));
  const iconOffsets = parkedOffsets(visibleVehicles);
  const liteIcons = visibleVehicles.length > LITE_ICON_VEHICLE_THRESHOLD;

  return (
    // Canvas renderer: the full road layer is a few thousand polylines,
    // far cheaper to draw on one canvas than as SVG paths.
    <MapContainer center={FALLBACK_CENTER} zoom={DEFAULT_ZOOM} preferCanvas className="h-full w-full rounded-lg">
      <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
      <FitToNetwork network={network} />

      <StaticRoadLayers network={network} roadEdgePositions={roadEdgePositions} maxTrafficWeight={maxTrafficWeight} />

      {fakeTraffic.map((car) => (
        <CircleMarker
          key={`fake-traffic-${car.id}`}
          center={car.position}
          radius={2.5}
          pathOptions={{ color: "#9ca3af", fillColor: "#9ca3af", fillOpacity: 0.8, weight: 0 }}
        />
      ))}

      {/* Z-order 3: spur edges -- a station/depot's short, distinctly
          colored connector branching off the road, visibly separate from
          both the muted underlay and the traffic layer. */}
      {[...stationSpurPositions.entries()].map(([key, positions]) => (
        <Polyline
          key={`station-spur-${key}`}
          positions={positions}
          pathOptions={{ color: STATION_SPUR_COLOR, weight: 3, opacity: 0.9 }}
        />
      ))}
      {[...depotSpurPositions.entries()].map(([key, positions]) => (
        <Polyline
          key={`depot-spur-${key}`}
          positions={positions}
          pathOptions={{ color: DEPOT_SPUR_COLOR, weight: 3, opacity: 0.9 }}
        />
      ))}

      {myCarPreview && (
        <Polyline
          positions={myCarPreview.route as LatLngTuple[]}
          pathOptions={{ color: "#a3e635", weight: 5, opacity: 0.35, dashArray: "1 8" }}
        />
      )}

      {/* Z-order 4: station/depot icons (Leaflet's markerPane always
          renders above its overlayPane, i.e. above every Polyline/
          CircleMarker regardless of JSX order -- these still come before
          the vehicle markers below so a vehicle icon wins any direct
          overlap). */}
      {network.depots.map((depot) => {
        const node = nodeById.get(depot.node_id);
        if (!node) return null;
        return (
          <Marker key={`depot-${depot.id}`} position={toLatLng(node.x, node.y)} icon={DEPOT_ICON}>
            <Tooltip direction="top" offset={[0, -16]}>
              Depot {depot.id} — bãi đỗ xe sau khi sạc xong
            </Tooltip>
          </Marker>
        );
      })}

      {network.stations.map((station) => {
        const node = nodeById.get(station.node_id);
        if (!node) return null;
        const live = stationById.get(station.id);
        return (
          <Marker
            key={`station-${station.id}`}
            position={toLatLng(node.x, node.y)}
            icon={STATION_ICON}
            eventHandlers={{
              mouseover: () => onStationSelect(station.id),
              click: () => onStationSelect(station.id),
            }}
          >
            <Tooltip direction="top" offset={[0, -14]}>
              Station {station.id} — queue {live?.queue ?? 0}, charging{" "}
              {live?.charging ?? 0}/{station.num_chargers}
            </Tooltip>
          </Marker>
        );
      })}

      {/* Z-order 5 (top): vehicles, always stacked above station/depot
          icons (zIndexOffset). */}
      {visibleVehicles.map((vehicle) => {
        const isMine = vehicle.id === myVehicleId;
        const icon = vehicleIcon(vehicle.state, {
          mine: isMine,
          nonApp: vehicle.is_non_app,
          offset: iconOffsets.get(vehicle.id),
          lite: liteIcons && !isMine,
        });
        const opacity = isFading(vehicle) ? 0 : 1;
        return (
          <Marker
            key={`vehicle-${vehicle.id}`}
            position={toLatLng(vehicle.x, vehicle.y)}
            icon={icon}
            opacity={opacity}
            zIndexOffset={isMine ? MY_CAR_Z_INDEX_OFFSET : VEHICLE_Z_INDEX_OFFSET}
          >
            <Tooltip direction="top" offset={[0, -2]}>
              {isMine ? "★ Xe của tôi — " : ""}EV {vehicle.id}
              {vehicle.is_non_app ? " (Non-App)" : ""} — {vehicle.state} — battery{" "}
              {(vehicle.battery * 100).toFixed(0)}%
            </Tooltip>
          </Marker>
        );
      })}
      <MapLegend />
    </MapContainer>
  );
}

// Fits the camera to the network's bounds once per scenario: on first load
// and on every scenario switch (the large map covers ~4x the area), but not
// on a new episode of the same scenario, so a user's own pan/zoom survives
// resets.
function FitToNetwork({ network }: { network: NetworkInfo }) {
  const map = useMap();
  const fittedScenario = useRef<string | null>(null);
  useEffect(() => {
    if (fittedScenario.current === network.scenario || network.nodes.length === 0) return;
    fittedScenario.current = network.scenario;
    const bounds = L.latLngBounds(network.nodes.map((node) => toLatLng(node.x, node.y)));
    map.fitBounds(bounds, { padding: FIT_PADDING_PX });
  }, [map, network]);
  return null;
}

interface StaticRoadLayersProps {
  network: NetworkInfo;
  roadEdgePositions: Map<string, LatLngTuple[]>;
  maxTrafficWeight: number;
}

// Every street the backend sends, drawn in full: the display road layer and
// every routing edge (the large scenario: ~9.5k edges plus ~9.5k display
// roads). One Leaflet layer per street made the large map lag, so streets
// sharing a style are merged into one multi-polyline (a few dozen layers in
// total; traffic colors are quantized into TRAFFIC_COLOR_BUCKETS steps),
// drawn on a canvas of their own, in a pane below the overlay pane. Moving
// vehicles and decorative traffic therefore never trigger a redraw of the
// streets; only pan/zoom or a new network does. Built imperatively, once
// per network, outside React's per-tick re-render.
function StaticRoadLayers({ network, roadEdgePositions, maxTrafficWeight }: StaticRoadLayersProps) {
  const map = useMap();
  useEffect(() => {
    if (!map.getPane(ROADS_PANE)) {
      const pane = map.createPane(ROADS_PANE);
      pane.style.zIndex = ROADS_PANE_Z_INDEX;
      pane.style.pointerEvents = "none";
    }
    const renderer = L.canvas({ pane: ROADS_PANE });
    const groups = new Map<string, { options: L.PolylineOptions; lines: LatLngTuple[][] }>();
    const add = (key: string, options: L.PolylineOptions, line: LatLngTuple[]) => {
      let group = groups.get(key);
      if (!group) {
        group = { options, lines: [] };
        groups.set(key, group);
      }
      group.lines.push(line);
    };

    // Z-order 0 (bottom): every drivable street of the extract, alleys
    // included, weighted by road class.
    for (const road of network.roads ?? []) {
      const color = backgroundRoadColor(road.highway);
      const weight = roadWeight(road.highway);
      add(`road|${color}|${weight}`, { color, weight, opacity: 0.55 }, road.geometry);
    }
    const routingEdges = network.edges.flatMap((edge) => {
      const positions = roadEdgePositions.get(edgeKey(edge.source, edge.target));
      return positions ? [{ edge, positions }] : [];
    });
    // Z-order 1: muted dashed underlay over every routing edge -- the
    // "ground truth" geometry vehicles/stations/depots all visibly stick
    // to. Skipped on a full-city network, where it would only double the
    // drawing work under lines that already cover it.
    if (routingEdges.length <= DASHED_UNDERLAY_MAX_EDGES) {
      for (const { edge, positions } of routingEdges) {
        const weight = roadWeight(edge.highway) + 3;
        add(
          `muted|${weight}`,
          { color: MUTED_ROAD_COLOR, opacity: MUTED_ROAD_OPACITY, weight, dashArray: "2 8" },
          positions,
        );
      }
    }
    // Z-order 2: the same routing edges, colored by traffic congestion and
    // weighted by road class.
    for (const { edge, positions } of routingEdges) {
      const bucket = trafficBucket(edge.traffic_weight, maxTrafficWeight);
      const weight = roadWeight(edge.highway);
      add(`traffic|${bucket}|${weight}`, { color: trafficBucketColor(bucket), weight, opacity: 0.8 }, positions);
    }

    const layer = L.layerGroup(
      [...groups.values()].map(({ options, lines }) =>
        L.polyline(lines, { ...options, renderer, interactive: false }),
      ),
    ).addTo(map);
    return () => {
      layer.remove();
      renderer.remove();
    };
  }, [map, network, roadEdgePositions, maxTrafficWeight]);
  return null;
}

function MapLegend() {
  const swatch = (color: string, height: number) => (
    <span className="inline-block w-6 rounded-sm align-middle" style={{ height, background: color }} />
  );
  return (
    <div className="pointer-events-none absolute bottom-6 left-2 z-[1000] rounded-md bg-white/90 px-2.5 py-2 text-[11px] leading-5 text-gray-700 shadow dark:bg-gray-900/90 dark:text-gray-200">
      <div className="flex items-center gap-2">
        <span className="inline-block h-1.5 w-6 rounded-sm" style={{ background: "linear-gradient(90deg, hsl(120,75%,45%), hsl(60,75%,45%), hsl(0,75%,45%))" }} />
        Đường xe chạy trong mô phỏng (màu = mức tắc)
      </div>
      <div className="flex items-center gap-2">
        {swatch(BACKGROUND_ROAD_COLORS.primary, 4)}
        Đường lớn (nền bản đồ)
      </div>
      <div className="flex items-center gap-2">
        {swatch(BACKGROUND_ROAD_DEFAULT_COLOR, 2)}
        Ngõ / đường nhỏ (nền bản đồ)
      </div>
    </div>
  );
}
