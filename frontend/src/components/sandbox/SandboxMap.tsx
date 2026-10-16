"use client";

// Map of the GNN Sandbox page, in two modes:
// - setup: click to place the center, a translucent circle previews the radius;
// - simulation: the generated sandbox (roads, stations, rest stops) and its EVs.
// Built for thousands of EVs: roads are a few merged polylines on their own
// canvas, and EVs are canvas circles updated imperatively (setLatLng) from each
// snapshot, never React-rendered DOM markers.

import "leaflet/dist/leaflet.css";
import L, { type LatLngTuple } from "leaflet";
import { useEffect, useMemo, useRef } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ParkingSquare, Zap } from "lucide-react";
import { Circle, MapContainer, Marker, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import { VEHICLE_COLORS } from "@/components/sandbox/vehicleColors";
import type { GenerateEnvResponse } from "@/lib/sandbox";
import type { SimulationUpdateMessage, StationUpdate, VehicleUpdate } from "@/lib/types";

// Same key-free basemap as the simulation dashboard (see MapComponent.tsx).
const TILE_URL = process.env.NEXT_PUBLIC_TILE_URL ?? "https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, tiles by OpenStreetMap France';
const INITIAL_CENTER: LatLngTuple = [21.0385, 105.7973];
const INITIAL_ZOOM = 13;


const ROAD_WEIGHTS: Record<string, number> = {
  motorway: 5, motorway_link: 3, trunk: 5, trunk_link: 3, primary: 4, primary_link: 3,
  secondary: 3.5, secondary_link: 2.5, tertiary: 3, tertiary_link: 2, unclassified: 2,
  residential: 2, living_street: 1.5, road: 1.5, service: 1.2, spur: 2,
};
const TRAFFIC_BUCKETS = 10;

function trafficColor(bucket: number): string {
  const hue = 120 - 120 * ((bucket + 0.5) / TRAFFIC_BUCKETS);
  return `hsl(${hue.toFixed(0)}, 70%, 45%)`;
}

const STATION_ICON = L.divIcon({
  className: "",
  html: `<div style="width:20px;height:20px;border-radius:9999px;background:#ea580c;border:2px solid #fff;display:flex;align-items:center;justify-content:center;box-shadow:0 1px 3px rgba(0,0,0,.45)">${renderToStaticMarkup(<Zap color="#fff" fill="#fff" size={11} />)}</div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});
const REST_STOP_ICON = L.divIcon({
  className: "",
  html: `<div style="width:20px;height:20px;border-radius:6px;background:#334155;border:2px solid #fff;display:flex;align-items:center;justify-content:center;box-shadow:0 1px 3px rgba(0,0,0,.45)">${renderToStaticMarkup(<ParkingSquare color="#fff" size={12} />)}</div>`,
  iconSize: [20, 20],
  iconAnchor: [10, 10],
});
const CENTER_ICON = L.divIcon({
  className: "",
  html: `<div style="width:18px;height:18px;border-radius:9999px;background:#4f46e5;border:3px solid #fff;box-shadow:0 0 0 2px #4f46e5"></div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

interface SandboxMapProps {
  mode: "setup" | "simulation";
  center: LatLngTuple | null;
  radiusKm: number;
  onPick: (center: LatLngTuple) => void;
  env: GenerateEnvResponse | null;
  snapshot: SimulationUpdateMessage | null;
}

export default function SandboxMap({ mode, center, radiusKm, onPick, env, snapshot }: SandboxMapProps) {
  return (
    <MapContainer center={center ?? INITIAL_CENTER} zoom={INITIAL_ZOOM} preferCanvas className="h-full w-full">
      <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
      {mode === "setup" ? (
        <SetupLayers center={center} radiusKm={radiusKm} onPick={onPick} />
      ) : (
        env && <SimulationLayers env={env} snapshot={snapshot ?? env.snapshot} />
      )}
    </MapContainer>
  );
}

// --- setup mode --------------------------------------------------------------

function SetupLayers({
  center,
  radiusKm,
  onPick,
}: {
  center: LatLngTuple | null;
  radiusKm: number;
  onPick: (center: LatLngTuple) => void;
}) {
  const map = useMapEvents({
    click: (event) => onPick([event.latlng.lat, event.latlng.lng]),
  });
  // Keep the whole preview circle in view as the center or radius changes.
  useEffect(() => {
    if (!center) return;
    map.fitBounds(L.latLng(center).toBounds(radiusKm * 2000), { padding: [40, 40], animate: false });
  }, [map, center, radiusKm]);
  if (!center) return null;
  return (
    <>
      <Circle
        center={center}
        radius={radiusKm * 1000}
        pathOptions={{ color: "#4f46e5", weight: 2, fillColor: "#6366f1", fillOpacity: 0.12, interactive: false }}
      />
      <Marker position={center} icon={CENTER_ICON} interactive={false} />
    </>
  );
}

// --- simulation mode -----------------------------------------------------------

function SimulationLayers({ env, snapshot }: { env: GenerateEnvResponse; snapshot: SimulationUpdateMessage }) {
  const map = useMap();
  const fitted = useRef<string | null>(null);
  useEffect(() => {
    if (fitted.current === env.sandbox.sandbox_id) return;
    fitted.current = env.sandbox.sandbox_id;
    const [south, west, north, east] = env.sandbox.map.bounds;
    map.fitBounds([[south, west], [north, east]], { padding: [20, 20] });
  }, [map, env]);

  const nodeById = useMemo(() => new Map(env.network.nodes.map((n) => [n.id, n])), [env]);
  const liveStations = useMemo(() => {
    const byId = new Map<number, StationUpdate>();
    for (const station of snapshot.stations) byId.set(station.id, station);
    return byId;
  }, [snapshot]);

  return (
    <>
      <RoadCanvas env={env} />
      {env.network.depots.map((depot) => {
        const node = nodeById.get(depot.node_id);
        return node ? (
          <Marker key={`rest-${depot.id}`} position={[node.y, node.x]} icon={REST_STOP_ICON}>
            <Tooltip direction="top" offset={[0, -10]}>Trạm dừng {depot.id}</Tooltip>
          </Marker>
        ) : null;
      })}
      {env.network.stations.map((station) => {
        const node = nodeById.get(station.node_id);
        const live = liveStations.get(station.id);
        return node ? (
          <Marker key={`station-${station.id}`} position={[node.y, node.x]} icon={STATION_ICON}>
            <Tooltip direction="top" offset={[0, -10]}>
              Trạm {station.id}: đang sạc {live?.charging ?? 0}/{station.num_chargers}, hàng đợi {live?.queue ?? 0}
            </Tooltip>
          </Marker>
        ) : null;
      })}
      <VehicleCanvas vehicles={snapshot.vehicles} />
    </>
  );
}

const ROADS_PANE = "sandbox-roads";
const VEHICLES_PANE = "sandbox-vehicles";

function ensurePane(map: L.Map, name: string, zIndex: string) {
  if (!map.getPane(name)) {
    const pane = map.createPane(name);
    pane.style.zIndex = zIndex;
  }
}

/** Every road edge, merged per (traffic color, width) into one polyline each,
 *  on a canvas of its own below markers; rebuilt only for a new sandbox. */
function RoadCanvas({ env }: { env: GenerateEnvResponse }) {
  const map = useMap();
  useEffect(() => {
    ensurePane(map, ROADS_PANE, "350");
    map.getPane(ROADS_PANE)!.style.pointerEvents = "none";
    const renderer = L.canvas({ pane: ROADS_PANE });
    const edges = env.network.edges;
    const maxTraffic = edges.reduce((m, e) => Math.max(m, e.traffic_weight), 1e-6);
    const groups = new Map<string, { color: string; weight: number; lines: LatLngTuple[][] }>();
    for (const edge of edges) {
      if (edge.geometry.length < 2) continue;
      const bucket = Math.min(Math.floor((edge.traffic_weight / maxTraffic) * TRAFFIC_BUCKETS), TRAFFIC_BUCKETS - 1);
      const weight = (edge.highway && ROAD_WEIGHTS[edge.highway]) || 2;
      const key = `${bucket}|${weight}`;
      let group = groups.get(key);
      if (!group) {
        group = { color: trafficColor(bucket), weight, lines: [] };
        groups.set(key, group);
      }
      group.lines.push(edge.geometry as LatLngTuple[]);
    }
    const layer = L.layerGroup(
      [...groups.values()].map((g) =>
        L.polyline(g.lines, { renderer, color: g.color, weight: g.weight, opacity: 0.75, interactive: false }),
      ),
    ).addTo(map);
    return () => {
      layer.remove();
      renderer.remove();
    };
  }, [map, env]);
  return null;
}

/** EVs as canvas circles, updated in place from every snapshot. COMPLETED EVs
 *  (parked at a rest stop) are hidden to keep the map readable. */
function VehicleCanvas({ vehicles }: { vehicles: VehicleUpdate[] }) {
  const map = useMap();
  const rendererRef = useRef<L.Canvas | null>(null);
  const markersRef = useRef(new Map<number, L.CircleMarker>());
  const dataRef = useRef(new Map<number, VehicleUpdate>());

  useEffect(() => {
    ensurePane(map, VEHICLES_PANE, "450");
    const renderer = L.canvas({ pane: VEHICLES_PANE });
    rendererRef.current = renderer;
    const markers = markersRef.current;
    return () => {
      for (const marker of markers.values()) marker.remove();
      markers.clear();
      renderer.remove();
      rendererRef.current = null;
    };
  }, [map]);

  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    const markers = markersRef.current;
    const data = dataRef.current;
    const seen = new Set<number>();
    for (const vehicle of vehicles) {
      if (vehicle.state === "COMPLETED") continue;
      seen.add(vehicle.id);
      data.set(vehicle.id, vehicle);
      const color = VEHICLE_COLORS[vehicle.state];
      let marker = markers.get(vehicle.id);
      if (!marker) {
        marker = L.circleMarker([vehicle.y, vehicle.x], {
          renderer,
          radius: 4,
          weight: 1.5,
          color: vehicle.is_non_app ? "#f97316" : "#ffffff",
          dashArray: vehicle.is_non_app ? "2 2" : undefined,
          fillColor: color,
          fillOpacity: 0.95,
        });
        const id = vehicle.id;
        marker.bindTooltip(() => {
          const v = data.get(id);
          if (!v) return "";
          return `EV ${v.id}${v.is_non_app ? " (non-app)" : ""} — ${v.state} — pin ${(v.battery * 100).toFixed(0)}%`;
        });
        marker.addTo(map);
        markers.set(vehicle.id, marker);
      } else {
        marker.setLatLng([vehicle.y, vehicle.x]);
        if (marker.options.fillColor !== color) marker.setStyle({ fillColor: color });
      }
    }
    for (const [id, marker] of markers) {
      if (!seen.has(id)) {
        marker.remove();
        markers.delete(id);
        data.delete(id);
      }
    }
  }, [map, vehicles]);

  return null;
}
