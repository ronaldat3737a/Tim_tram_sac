"use client";

// Fixed side panel showing charging-station detail on hover/click (replaces
// an on-map popup so it never covers the map). Every number here is read
// directly from the WebSocket snapshot / network info -- no computation of
// simulation/charging physics happens in this component (section 39).

import { Battery, MapPin, Zap } from "lucide-react";
import type { StationInfo, StationUpdate, VehicleUpdate } from "@/lib/types";

interface StationDetailPanelProps {
  station: StationInfo | null;
  liveStation: StationUpdate | null;
  vehicles: VehicleUpdate[];
}

function formatEta(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function StationDetailPanel({ station, liveStation, vehicles }: StationDetailPanelProps) {
  if (!station) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-gray-300 p-4 text-center text-xs text-gray-400 dark:border-gray-700">
        <MapPin className="h-5 w-5" />
        Hover hoặc click vào một trạm sạc trên bản đồ để xem chi tiết
      </div>
    );
  }

  const chargingVehicles = vehicles.filter(
    (v) => v.station_id === station.id && v.state === "CHARGING",
  );
  const queuedVehicles = vehicles.filter(
    (v) => v.station_id === station.id && v.state === "WAITING",
  );
  const busy = liveStation?.charging ?? 0;
  const free = Math.max(station.num_chargers - busy, 0);

  return (
    <div className="flex h-full flex-col gap-3 rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-center gap-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-orange-500 text-white">
          <Zap className="h-4 w-4" />
        </span>
        <h3 className="text-sm font-semibold">Station {station.id}</h3>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-md bg-gray-50 p-2 dark:bg-gray-800">
          <div className="text-gray-400">Tổng trụ</div>
          <div className="text-base font-semibold">{station.num_chargers}</div>
        </div>
        <div className="rounded-md bg-gray-50 p-2 dark:bg-gray-800">
          <div className="text-gray-400">Đang bận</div>
          <div className="text-base font-semibold text-orange-500">{busy}</div>
        </div>
        <div className="rounded-md bg-gray-50 p-2 dark:bg-gray-800">
          <div className="text-gray-400">Còn trống</div>
          <div className="text-base font-semibold text-emerald-500">{free}</div>
        </div>
      </div>

      <div>
        <div className="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">
          Trụ đang sạc ({chargingVehicles.length})
        </div>
        {chargingVehicles.length === 0 ? (
          <p className="text-xs text-gray-400">Không có xe nào đang sạc.</p>
        ) : (
          <ul className="flex flex-col gap-1.5">
            {chargingVehicles.map((v) => (
              <li
                key={v.id}
                className="flex items-center justify-between gap-2 rounded-md border border-gray-100 px-2 py-1.5 text-xs dark:border-gray-800"
              >
                <span className="font-medium">EV {v.id}</span>
                <span className="flex items-center gap-1 text-emerald-600">
                  <Battery className="h-3.5 w-3.5" /> {(v.battery * 100).toFixed(0)}%
                </span>
                <span className="text-gray-400">ETA {formatEta(v.eta_seconds)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <div className="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">
          Khu vực chờ ({queuedVehicles.length})
        </div>
        {queuedVehicles.length === 0 ? (
          <p className="text-xs text-gray-400">Không có xe chờ.</p>
        ) : (
          <div className="flex flex-wrap gap-1">
            {queuedVehicles.map((v) => (
              <span
                key={v.id}
                className="rounded-full bg-amber-100 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-900 dark:text-amber-300"
              >
                EV {v.id}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
