"use client";

import "leaflet/dist/leaflet.css";
import L, { type LatLngTuple } from "leaflet";
import { Car, ParkingSquare, Zap } from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip } from "react-leaflet";
import type { MyCarPreviewResponse, NetworkInfo, StationUpdate, VehicleUpdate } from "@/lib/types";

// Fallback center used only until the real network graph loads; once it
// does, the map re-centers on the actual node average (section 11: the
// graph is always a real OpenStreetMap street network).
const FALLBACK_CENTER: LatLngTuple = [21.0385, 105.7973];
const DEFAULT_ZOOM = 15;

// Free, no-API-key tile source (section 7: "Không yêu cầu API trả phí").
// CartoDB's basemaps.cartocdn.com now requires a registered API key even
// for the free "light_all" style (confirmed by an "API KEY REQUIRED"
// watermark when tried), so this uses the standard OpenStreetMap tile
// server instead, which stays key-free.
const TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

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

// A station/depot is offset a small, fixed distance from its road node so
// it reads as a lot beside the street rather than sitting on top of the
// road. Vehicles queued/charging at a station are then clustered around
// that SAME offset point (not the raw road node) so they visually belong
// to the station lot instead of appearing to wait mid-road; a COMPLETED
// vehicle is clustered around its own real (already-at-the-depot-node)
// position for the same reason. All offsets are deterministic (derived
// from id via golden-angle spacing), so they are stable across renders
// instead of jittering.
const STATION_OFFSET_DEGREES = 0.00018;
const DEPOT_OFFSET_DEGREES = 0.00018;
const CHARGING_CLUSTER_RADIUS_DEGREES = 0.00006;
const QUEUE_CLUSTER_RADIUS_DEGREES = 0.00015;
const PARKED_CLUSTER_RADIUS_DEGREES = 0.00012;

function goldenAngleOffset(id: number, radiusDegrees: number): [number, number] {
  const angleRad = ((id * 137.508) % 360) * (Math.PI / 180);
  return [Math.cos(angleRad) * radiusDegrees, Math.sin(angleRad) * radiusDegrees];
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
function stateBadgeHtml(state: VehicleUpdate["state"]): string {
  const badgeBase =
    "position:absolute;top:-4px;right:-4px;width:14px;height:14px;border-radius:9999px;" +
    "display:flex;align-items:center;justify-content:center;border:1.5px solid #ffffff;" +
    "color:#ffffff;font-size:9px;font-weight:700;line-height:1;";
  if (state === "WAITING") {
    return `<span class="animate-pulse" style="${badgeBase}background:#d97706;">W</span>`;
  }
  if (state === "CHARGING") {
    return `<span style="${badgeBase}background:#1d4ed8;">${BADGE_ZAP_HTML}</span>`;
  }
  if (state === "RETURNING_TO_DEPOT") {
    return `<span style="${badgeBase}background:#6d28d9;">${BADGE_PARKING_HTML}</span>`;
  }
  return "";
}

function buildVehicleIcon(
  color: string,
  state: VehicleUpdate["state"],
  options: { mine?: boolean } = {},
): L.DivIcon {
  const ringColor = options.mine ? MY_CAR_RING_COLOR : EV_HALO_COLOR;
  const size = options.mine ? 32 : 26;
  return L.divIcon({
    className: "ev-marker-icon",
    html: `
      <div style="position:relative;width:${size}px;height:${size}px;">
        <span class="animate-ping" style="position:absolute;inset:0;border-radius:9999px;background:${ringColor};opacity:0.55;"></span>
        <div style="position:relative;width:${size}px;height:${size}px;border-radius:9999px;background:${color};border:${options.mine ? 3 : 2}px solid ${ringColor};display:flex;align-items:center;justify-content:center;box-shadow:0 0 8px ${ringColor};">
          ${EV_ICON_HTML}
        </div>
        ${stateBadgeHtml(state)}
      </div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

const VEHICLE_ICONS: Record<VehicleUpdate["state"], L.DivIcon> = {
  TRAVELING: buildVehicleIcon(VEHICLE_COLORS.TRAVELING, "TRAVELING"),
  WAITING: buildVehicleIcon(VEHICLE_COLORS.WAITING, "WAITING"),
  CHARGING: buildVehicleIcon(VEHICLE_COLORS.CHARGING, "CHARGING"),
  RETURNING_TO_DEPOT: buildVehicleIcon(VEHICLE_COLORS.RETURNING_TO_DEPOT, "RETURNING_TO_DEPOT"),
  COMPLETED: buildVehicleIcon(VEHICLE_COLORS.COMPLETED, "COMPLETED"),
  FAILED: buildVehicleIcon(VEHICLE_COLORS.FAILED, "FAILED"),
};

const MY_CAR_ICONS: Record<VehicleUpdate["state"], L.DivIcon> = {
  TRAVELING: buildVehicleIcon(VEHICLE_COLORS.TRAVELING, "TRAVELING", { mine: true }),
  WAITING: buildVehicleIcon(VEHICLE_COLORS.WAITING, "WAITING", { mine: true }),
  CHARGING: buildVehicleIcon(VEHICLE_COLORS.CHARGING, "CHARGING", { mine: true }),
  RETURNING_TO_DEPOT: buildVehicleIcon(VEHICLE_COLORS.RETURNING_TO_DEPOT, "RETURNING_TO_DEPOT", { mine: true }),
  COMPLETED: buildVehicleIcon(VEHICLE_COLORS.COMPLETED, "COMPLETED", { mine: true }),
  FAILED: buildVehicleIcon(VEHICLE_COLORS.FAILED, "FAILED", { mine: true }),
};

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

// --- Traffic color (static traffic_weight, section: Phase 0 D.2 -- traffic
// stays static per episode; only visualized here, physics unchanged) -------

function trafficColor(weight: number, maxWeight: number): string {
  const ratio = maxWeight > 0 ? Math.min(Math.max(weight / maxWeight, 0), 1) : 0;
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

  // Backend-provided road geometry per edge (network_graph.py's `geometry`)
  // -- the real street curve, and exactly what vehicles actually move
  // along, so this is the single source of truth for both the drawn road
  // and vehicle movement; the two can never visually diverge. Keyed the
  // same way network.edges is, so lookups stay consistent even if some
  // edge is skipped -- a plain filtered array would let its indices drift
  // out of sync with network.edges once that happens. Already [lat, lng]
  // pairs from the API.
  const edgeGeometryPositions = useMemo(() => {
    const map = new Map<string, LatLngTuple[]>();
    if (!network) return map;
    for (const edge of network.edges) {
      if (edge.geometry.length < 2) continue;
      map.set(
        edgeKey(edge.source, edge.target),
        edge.geometry.map(([lat, lng]) => [lat, lng] as LatLngTuple),
      );
    }
    return map;
  }, [network]);

  const fakeTraffic = useFakeTraffic(useMemo(() => [...edgeGeometryPositions.values()], [edgeGeometryPositions]));
  const { isFading, isHidden } = useCompletedVehicleFade(vehicles);

  const maxTrafficWeight = useMemo(() => {
    if (!network || network.edges.length === 0) return 1;
    return Math.max(...network.edges.map((e) => e.traffic_weight), 0.0001);
  }, [network]);

  const stationOffsetPositions = useMemo(() => {
    const map = new Map<number, LatLngTuple>();
    if (!network) return map;
    for (const station of network.stations) {
      const node = nodeById.get(station.node_id);
      if (!node) continue;
      const [dLat, dLng] = goldenAngleOffset(station.id, STATION_OFFSET_DEGREES);
      map.set(station.id, [node.y + dLat, node.x + dLng]);
    }
    return map;
  }, [network, nodeById]);

  const depotOffsetPositions = useMemo(() => {
    const map = new Map<number, LatLngTuple>();
    if (!network) return map;
    for (const depot of network.depots) {
      const node = nodeById.get(depot.node_id);
      if (!node) continue;
      // Offset in the opposite angular "family" from stations (id + a large
      // fixed jump) purely so a depot sharing a node with a station doesn't
      // land on the exact same offset point.
      const [dLat, dLng] = goldenAngleOffset(depot.id + 1000, DEPOT_OFFSET_DEGREES);
      map.set(depot.id, [node.y + dLat, node.x + dLng]);
    }
    return map;
  }, [network, nodeById]);

  const center = useMemo<LatLngTuple>(() => {
    if (!network || network.nodes.length === 0) return FALLBACK_CENTER;
    const avgLng = network.nodes.reduce((sum, n) => sum + n.x, 0) / network.nodes.length;
    const avgLat = network.nodes.reduce((sum, n) => sum + n.y, 0) / network.nodes.length;
    return [avgLat, avgLng];
  }, [network]);

  if (!network) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-gray-500">
        Loading network…
      </div>
    );
  }

  function displayPositionFor(vehicle: VehicleUpdate): LatLngTuple {
    if (vehicle.state === "COMPLETED") {
      // Once COMPLETED, vehicle.x/y IS already the real depot node's own
      // position (the simulator set current_node to it on arrival) -- only
      // a small per-vehicle offset is added so multiple cars parked at the
      // same depot don't render as one overlapping icon.
      const [dLat, dLng] = goldenAngleOffset(vehicle.id, PARKED_CLUSTER_RADIUS_DEGREES);
      const [lat, lng] = toLatLng(vehicle.x, vehicle.y);
      return [lat + dLat, lng + dLng];
    }
    if (vehicle.station_id !== null && (vehicle.state === "WAITING" || vehicle.state === "CHARGING")) {
      const stationPos = stationOffsetPositions.get(vehicle.station_id);
      if (stationPos) {
        const radius =
          vehicle.state === "CHARGING" ? CHARGING_CLUSTER_RADIUS_DEGREES : QUEUE_CLUSTER_RADIUS_DEGREES;
        const [dLat, dLng] = goldenAngleOffset(vehicle.id, radius);
        return [stationPos[0] + dLat, stationPos[1] + dLng];
      }
    }
    // TRAVELING and RETURNING_TO_DEPOT both render at the real, actively
    // moving position the backend reports (interpolated along real road
    // geometry) -- no offset, since they're genuinely in motion.
    return toLatLng(vehicle.x, vehicle.y);
  }

  return (
    <MapContainer center={center} zoom={DEFAULT_ZOOM} className="h-full w-full rounded-lg">
      <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />

      {network.edges.map((edge) => {
        const key = edgeKey(edge.source, edge.target);
        const positions = edgeGeometryPositions.get(key);
        if (!positions) return null;
        return (
          <Polyline
            key={`edge-${key}`}
            positions={positions}
            pathOptions={{
              color: trafficColor(edge.traffic_weight, maxTrafficWeight),
              weight: 3,
              opacity: 0.65,
            }}
          />
        );
      })}

      {network.nodes
        .filter((node) => !node.is_station)
        .map((node) => (
          <CircleMarker
            key={`node-${node.id}`}
            center={toLatLng(node.x, node.y)}
            radius={2.5}
            pathOptions={{ color: "#64748b", fillColor: "#64748b", fillOpacity: 1, weight: 0 }}
          />
        ))}

      {fakeTraffic.map((car) => (
        <CircleMarker
          key={`fake-traffic-${car.id}`}
          center={car.position}
          radius={2.5}
          pathOptions={{ color: "#9ca3af", fillColor: "#9ca3af", fillOpacity: 0.8, weight: 0 }}
        />
      ))}

      {network.depots.map((depot) => {
        const node = nodeById.get(depot.node_id);
        const offsetPosition = depotOffsetPositions.get(depot.id);
        if (!node || !offsetPosition) return null;
        return (
          <Fragment key={`depot-group-${depot.id}`}>
            <Polyline
              positions={[toLatLng(node.x, node.y), offsetPosition]}
              pathOptions={{ color: "#334155", weight: 1.5, dashArray: "3 4", opacity: 0.8 }}
            />
            <Marker position={offsetPosition} icon={DEPOT_ICON}>
              <Tooltip direction="top" offset={[0, -16]}>
                Depot {depot.id} — bãi đỗ xe sau khi sạc xong
              </Tooltip>
            </Marker>
          </Fragment>
        );
      })}

      {network.stations.map((station) => {
        const node = nodeById.get(station.node_id);
        const offsetPosition = stationOffsetPositions.get(station.id);
        if (!node || !offsetPosition) return null;
        const live = stationById.get(station.id);
        return (
          <Fragment key={`station-group-${station.id}`}>
            <Polyline
              positions={[toLatLng(node.x, node.y), offsetPosition]}
              pathOptions={{ color: "#ea580c", weight: 1.5, dashArray: "3 4", opacity: 0.8 }}
            />
            <Marker
              position={offsetPosition}
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
          </Fragment>
        );
      })}

      {myCarPreview && (
        <Polyline
          positions={myCarPreview.route as LatLngTuple[]}
          pathOptions={{ color: "#a3e635", weight: 5, opacity: 0.35, dashArray: "1 8" }}
        />
      )}

      {vehicles
        .filter((vehicle) => !isHidden(vehicle))
        .map((vehicle) => {
          const isMine = vehicle.id === myVehicleId;
          const icon = (isMine ? MY_CAR_ICONS : VEHICLE_ICONS)[vehicle.state];
          const opacity = isFading(vehicle) ? 0 : 1;
          return (
            <Marker
              key={`vehicle-${vehicle.id}`}
              position={displayPositionFor(vehicle)}
              icon={icon}
              opacity={opacity}
            >
              <Tooltip direction="top" offset={[0, -14]}>
                {isMine ? "★ Xe của tôi — " : ""}EV {vehicle.id} — {vehicle.state} — battery{" "}
                {(vehicle.battery * 100).toFixed(0)}%
              </Tooltip>
            </Marker>
          );
        })}
    </MapContainer>
  );
}
