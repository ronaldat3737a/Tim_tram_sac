"use client";

// Dashboard summary (PROJECT_SPEC.md section 38). Every number here is read
// directly from the backend's SimulationStatusResponse / WebSocket snapshot
// -- nothing is computed or simulated client-side (section 39).

import { Activity, Battery, Clock, Gauge, TimerReset, Zap } from "lucide-react";
import type { NetworkInfo, SimulationStatusResponse, StationUpdate } from "@/lib/types";

interface DashboardChartsProps {
  status: SimulationStatusResponse | null;
  stations: StationUpdate[];
  network: NetworkInfo | null;
}

function StatTile({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
      <Icon className="h-5 w-5 shrink-0 text-indigo-500" />
      <div className="min-w-0">
        <div className="truncate text-xs text-gray-500 dark:text-gray-400">{label}</div>
        <div className="text-lg font-semibold tabular-nums">{value}</div>
      </div>
    </div>
  );
}

export default function DashboardCharts({ status, stations, network }: DashboardChartsProps) {
  const totalEvs =
    (status?.num_vehicles_traveling ?? 0) +
    (status?.num_vehicles_waiting ?? 0) +
    (status?.num_vehicles_charging ?? 0) +
    (status?.num_vehicles_returning_to_depot ?? 0) +
    (status?.num_vehicles_completed ?? 0) +
    (status?.num_vehicles_failed ?? 0);
  const activeEvs =
    (status?.num_vehicles_traveling ?? 0) +
    (status?.num_vehicles_waiting ?? 0) +
    (status?.num_vehicles_charging ?? 0) +
    (status?.num_vehicles_returning_to_depot ?? 0);

  const stationsById = new Map(stations.map((s) => [s.id, s]));

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <StatTile icon={Activity} label="Total EVs" value={String(totalEvs)} />
        <StatTile icon={Zap} label="Active EVs" value={String(activeEvs)} />
        <StatTile
          icon={Battery}
          label="Completed / Failed"
          value={`${status?.num_vehicles_completed ?? 0} / ${status?.num_vehicles_failed ?? 0}`}
        />
        <StatTile
          icon={Clock}
          label="Avg Travel Time"
          value={`${(status?.average_travel_time ?? 0).toFixed(1)}s`}
        />
        <StatTile
          icon={TimerReset}
          label="Avg Waiting Time"
          value={`${(status?.average_waiting_time ?? 0).toFixed(1)}s`}
        />
        <StatTile
          icon={Gauge}
          label="Total System Cost"
          value={(status?.total_system_cost ?? 0).toFixed(1)}
        />
      </div>

      <div className="rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
        <div className="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">
          Station Utilization
        </div>
        <div className="flex flex-col gap-2">
          {(network?.stations ?? []).map((station) => {
            const live = stationsById.get(station.id);
            const charging = live?.charging ?? 0;
            const queue = live?.queue ?? 0;
            const utilization = station.num_chargers > 0 ? charging / station.num_chargers : 0;
            return (
              <div key={station.id} className="flex items-center gap-2 text-xs">
                <span className="w-16 shrink-0 text-gray-500 dark:text-gray-400">
                  Station {station.id}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-100 dark:bg-gray-800">
                  <div
                    className="h-full rounded-full bg-emerald-500 transition-all"
                    style={{ width: `${Math.min(utilization, 1) * 100}%` }}
                  />
                </div>
                <span className="w-24 shrink-0 tabular-nums text-gray-500 dark:text-gray-400">
                  {charging}/{station.num_chargers} chg, {queue} queue
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
