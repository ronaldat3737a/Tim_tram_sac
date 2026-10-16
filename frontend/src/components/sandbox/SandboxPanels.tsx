"use client";

// Overlay panels of the GNN Sandbox page: setup (choose the area and the
// counts, generate) and simulation (start/pause/speed/reset and live metrics).

import { ArrowLeft, Loader2, Pause, Play, RotateCcw, Wand2 } from "lucide-react";
import type { LatLngTuple } from "leaflet";
import type { GenerateEnvResponse, SandboxConnection, SandboxStatusMessage } from "@/lib/sandbox";
import { VEHICLE_COLORS } from "@/components/sandbox/vehicleColors";

const PANEL = "w-80 max-w-[calc(100vw-2rem)] rounded-lg border border-gray-200 bg-white/95 p-4 shadow-lg backdrop-blur dark:border-gray-800 dark:bg-gray-900/95";

function fmt(value: number, digits = 1): string {
  return value.toLocaleString("vi-VN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs font-medium text-gray-500 dark:text-gray-400">
      <span className="flex justify-between">
        {props.label}
        <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          {props.value}
          {props.unit ?? ""}
        </span>
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

export interface SetupValues {
  radiusKm: number;
  numStations: number;
  numEvs: number;
  numRestStops: number;
}

export function SetupPanel({
  center,
  values,
  onChange,
  onGenerate,
  generating,
  error,
  previous,
  onReopen,
}: {
  center: LatLngTuple | null;
  values: SetupValues;
  onChange: (values: SetupValues) => void;
  onGenerate: () => void;
  generating: boolean;
  error: string | null;
  previous: GenerateEnvResponse | null;
  onReopen: () => void;
}) {
  const set = (patch: Partial<SetupValues>) => onChange({ ...values, ...patch });
  return (
    <div className={PANEL}>
      <h2 className="text-base font-semibold">Tạo bản đồ tùy chỉnh</h2>
      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
        {center
          ? `Tâm: ${center[0].toFixed(5)}, ${center[1].toFixed(5)} (bấm chỗ khác để đổi)`
          : "Bấm vào bản đồ để chọn tâm vùng mô phỏng."}
      </p>
      <div className="mt-3 flex flex-col gap-3">
        <Slider label="Bán kính" value={values.radiusKm} min={0.5} max={10} step={0.5} unit=" km" onChange={(v) => set({ radiusKm: v })} />
        <Slider label="Số trạm sạc" value={values.numStations} min={1} max={200} step={1} onChange={(v) => set({ numStations: v })} />
        <Slider label="Số xe EV" value={values.numEvs} min={10} max={2000} step={10} onChange={(v) => set({ numEvs: v })} />
        <Slider label="Số trạm dừng (bãi đỗ)" value={values.numRestStops} min={1} max={100} step={1} onChange={(v) => set({ numRestStops: v })} />
      </div>
      <button
        onClick={onGenerate}
        disabled={!center || generating}
        className="mt-4 flex w-full items-center justify-center gap-1.5 rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
      >
        {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
        Generate Custom Environment
      </button>
      {generating && (
        <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
          Đang tải đường phố từ OpenStreetMap và rải trạm bằng K-Means… Lần đầu ở một vị trí mới có thể mất 1–2 phút.
        </p>
      )}
      {error && <p className="mt-2 text-xs text-red-600 dark:text-red-400">{error}</p>}
      {previous && !generating && (
        <button
          onClick={onReopen}
          className="mt-3 w-full rounded-md border border-gray-300 px-3 py-1.5 text-xs font-medium hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
        >
          Mở lại sandbox gần nhất ({previous.sandbox.params.num_stations} trạm, {previous.sandbox.params.num_evs} xe,{" "}
          {previous.sandbox.params.radius_km} km)
        </button>
      )}
    </div>
  );
}

const STATUS_LABELS: Record<SandboxStatusMessage["status"], string> = {
  preparing: "đang chuẩn bị",
  paused: "tạm dừng",
  running: "đang chạy",
  finished: "đã xong",
  error: "lỗi",
  stopped: "đã dừng",
};

export function SimulationPanel({
  env,
  status,
  connection,
  error,
  speed,
  onSpeed,
  onStart,
  onPause,
  onReset,
  onReconnect,
  onBack,
}: {
  env: GenerateEnvResponse;
  status: SandboxStatusMessage | null;
  connection: SandboxConnection;
  error: string | null;
  speed: number;
  onSpeed: (speed: number) => void;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onReconnect: () => void;
  onBack: () => void;
}) {
  const { params, battery } = env.sandbox;
  const run = status?.status;
  const prep = status?.preparing;
  const counts: [string, keyof SandboxStatusMessage, string][] = [
    ["Đang chạy", "num_vehicles_traveling", VEHICLE_COLORS.TRAVELING],
    ["Chờ sạc", "num_vehicles_waiting", VEHICLE_COLORS.WAITING],
    ["Đang sạc", "num_vehicles_charging", VEHICLE_COLORS.CHARGING],
    ["Về bãi đỗ", "num_vehicles_returning_to_depot", VEHICLE_COLORS.RETURNING_TO_DEPOT],
    ["Hoàn tất", "num_vehicles_completed", VEHICLE_COLORS.COMPLETED],
    ["Hỏng", "num_vehicles_failed", VEHICLE_COLORS.FAILED],
  ];
  return (
    <div className={`${PANEL} max-h-[calc(100vh-2rem)] overflow-y-auto`}>
      <div className="flex items-start justify-between gap-2">
        <div>
          <h2 className="text-base font-semibold">GNN Sandbox</h2>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {params.num_stations} trạm · {params.num_evs} xe · {params.num_rest_stops} trạm dừng · {params.radius_km} km ·{" "}
            {env.sandbox.map.num_road_nodes.toLocaleString("vi-VN")} nút đường
          </p>
        </div>
        <span
          className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
            run === "running"
              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
              : run === "error"
                ? "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300"
                : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300"
          }`}
        >
          {connection !== "open" ? (connection === "connecting" ? "đang kết nối" : "mất kết nối") : run ? STATUS_LABELS[run] : "…"}
        </span>
      </div>

      {run === "preparing" && prep && (
        <div className="mt-3">
          <div className="text-xs text-gray-500 dark:text-gray-400">
            Tính tuyến cho xe khởi hành đầu tiên: {prep.routes_done}/{prep.routes_total}
          </div>
          <div className="mt-1 h-1.5 rounded bg-gray-200 dark:bg-gray-800">
            <div
              className="h-1.5 rounded bg-indigo-500"
              style={{ width: `${prep.routes_total ? (100 * prep.routes_done) / prep.routes_total : 0}%` }}
            />
          </div>
        </div>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        {run === "running" ? (
          <button onClick={onPause} className="flex items-center gap-1 rounded-md bg-amber-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-400">
            <Pause className="h-4 w-4" /> Pause
          </button>
        ) : (
          <button
            onClick={onStart}
            disabled={connection !== "open" || run === "finished" || run === "error"}
            className="flex items-center gap-1 rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            <Play className="h-4 w-4" /> {run === "preparing" ? "Start (khi sẵn sàng)" : "Start"}
          </button>
        )}
        <button
          onClick={connection === "open" ? onReset : onReconnect}
          className="flex items-center gap-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
        >
          <RotateCcw className="h-4 w-4" /> {connection === "open" ? "Reset" : "Kết nối lại"}
        </button>
        <button
          onClick={onBack}
          className="flex items-center gap-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
        >
          <ArrowLeft className="h-4 w-4" /> Bản đồ khác
        </button>
      </div>

      <div className="mt-3">
        <Slider label="Tốc độ" value={speed} min={1} max={100} step={1} unit=" tick/s" onChange={onSpeed} />
      </div>

      {(error || status?.error) && <p className="mt-2 text-xs text-red-600 dark:text-red-400">{status?.error ?? error}</p>}

      {status && (
        <>
          <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
            t = {fmt(status.simulation_time, 0)} s · đã khởi hành {status.num_vehicles_departed}/{status.num_vehicles_total} xe
          </div>
          <div className="mt-2 grid grid-cols-3 gap-1.5">
            {counts.map(([label, key, color]) => (
              <div key={key} className="rounded-md bg-gray-50 p-1.5 dark:bg-gray-800">
                <div className="flex items-center gap-1 text-[10px] text-gray-500 dark:text-gray-400">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
                  {label}
                </div>
                <div className="text-sm font-semibold tabular-nums">{status[key] as number}</div>
              </div>
            ))}
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1 text-xs tabular-nums">
            <dt className="text-gray-500 dark:text-gray-400">Quyết định GNN</dt>
            <dd className="text-right font-medium">{status.num_decisions}</dd>
            <dt className="text-gray-500 dark:text-gray-400">Chờ TB</dt>
            <dd className="text-right font-medium">{fmt(status.average_waiting_time)} s</dd>
            <dt className="text-gray-500 dark:text-gray-400">Đi TB</dt>
            <dd className="text-right font-medium">{fmt(status.average_travel_time)} s</dd>
            <dt className="text-gray-500 dark:text-gray-400">Tổng chi phí</dt>
            <dd className="text-right font-medium">{fmt(status.total_system_cost, 0)} s</dd>
            <dt className="text-gray-500 dark:text-gray-400">Quá tải</dt>
            <dd className="text-right font-medium">{status.num_overloaded_events}</dd>
            <dt className="text-gray-500 dark:text-gray-400">GNN / quyết định</dt>
            <dd className="text-right font-medium">{fmt(status.average_decision_time_ms, 2)} ms</dd>
          </dl>
        </>
      )}

      <p className="mt-3 border-t border-gray-200 pt-2 text-[11px] text-gray-500 dark:border-gray-800 dark:text-gray-400">
        Pin: đầy {fmt(battery.full_range_km)} km, xin sạc khi còn {fmt(battery.request_range_km)} km; điểm xa trạm
        nhất {fmt(battery.farthest_from_station_km, 2)} km. Mô hình MAPPO-GNN train trên 10 trạm / 200 xe, chạy
        zero-shot trên bản đồ này. Viền cam đứt = xe không dùng app.
      </p>
    </div>
  );
}
