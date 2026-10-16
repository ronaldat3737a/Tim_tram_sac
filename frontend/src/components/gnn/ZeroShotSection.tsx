"use client";

// Section 2: the same trained MAPPO-MLP and MAPPO-GNN, without retraining, on
// a scenario with a different number of stations/EVs.

import { useState } from "react";
import { CheckCircle2, Loader2, OctagonAlert, Play } from "lucide-react";
import { type AlgorithmResult, type EvalMetrics, runZeroShotScale, type ZeroShotScaleResponse } from "@/lib/gnnEval";

const SEED = 1000;

function fmt(value: number, digits = 1): string {
  return value.toLocaleString("vi-VN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export default function ZeroShotSection() {
  const [numStations, setNumStations] = useState(50);
  const [numEvs, setNumEvs] = useState(500);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ZeroShotScaleResponse | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await runZeroShotScale({ num_stations: numStations, num_evs: numEvs, seed: SEED }));
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="flex flex-col gap-4">
      <header>
        <h2 className="text-lg font-semibold">2. Zero-Shot Scalability Test</h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Hai mô hình đã train trên 10 trạm / 200 xe được đưa thẳng vào một kịch bản quy mô khác, không train lại.
        </p>
      </header>

      <div className="flex flex-wrap items-end gap-6 rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
        <Slider label="Số trạm sạc" value={numStations} min={20} max={100} step={1} onChange={setNumStations} />
        <Slider label="Số xe EV" value={numEvs} min={100} max={1000} step={50} onChange={setNumEvs} />
        <button
          onClick={run}
          disabled={loading}
          className="flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-60"
        >
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          Run Zero-Shot Scale Test
        </button>
        <span className="text-xs text-gray-500 dark:text-gray-400">seed {SEED} · khoảng 30–90 giây tùy quy mô</span>
      </div>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      {loading && (
        <div className="grid gap-4 md:grid-cols-2" aria-busy="true">
          <div className="h-56 animate-pulse rounded-lg bg-gray-100 dark:bg-gray-800" />
          <div className="h-56 animate-pulse rounded-lg bg-gray-100 dark:bg-gray-800" />
        </div>
      )}

      {!loading && data && (
        <>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Kịch bản mới: {data.scenario.num_stations} trạm · {data.scenario.num_evs} xe · seed {data.scenario.seed} —
            cả hai mô hình được train trên {data.trained_on.num_stations} trạm / {data.trained_on.num_evs} xe
            {data.cached && " · kết quả từ cache"}
          </p>
          <div className="grid gap-4 md:grid-cols-2">
            <MlpPanel result={data.mlp} numStations={data.scenario.num_stations} />
            <GnnPanel result={data.gnn} reference={data.reference} data={data} />
          </div>
        </>
      )}
    </section>
  );
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="flex w-56 flex-col gap-1 text-xs font-medium text-gray-500 dark:text-gray-400">
      <span>
        {props.label}: <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">{props.value}</span>
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
      <span className="flex justify-between text-[10px]">
        <span>{props.min}</span>
        <span>{props.max}</span>
      </span>
    </label>
  );
}

function StatusBox({
  tone,
  icon,
  title,
  children,
}: {
  tone: "critical" | "good" | "neutral";
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}) {
  const color = tone === "critical" ? "var(--status-critical)" : tone === "good" ? "var(--status-good)" : "var(--axis)";
  return (
    <div
      role={tone === "critical" ? "alert" : "status"}
      className="flex h-full flex-col gap-3 rounded-lg border-2 p-4"
      style={{
        borderColor: color,
        background: `color-mix(in srgb, ${color} 9%, var(--surface-1))`,
        color: "var(--text-primary)",
      }}
    >
      <div className="flex items-center gap-2">
        <span style={{ color }}>{icon}</span>
        <h3 className="text-base font-bold tracking-tight">{title}</h3>
      </div>
      {children}
    </div>
  );
}

function MlpPanel({ result, numStations }: { result: AlgorithmResult; numStations: number }) {
  const model = result.model;
  if (result.status === "crashed") {
    return (
      <StatusBox tone="critical" icon={<OctagonAlert className="h-6 w-6" />} title="CRITICAL FAILURE: Dimension Mismatch">
        <p className="text-sm font-semibold">MAPPO-MLP (mô hình cũ) không chạy được trên kịch bản này.</p>
        {model && model.input_dim !== null && (
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Lớp đầu vào của MLP bị cố định theo không gian quan sát lúc train: <b>{model.input_dim} chiều</b> (5 đặc
            trưng của xe + 7 × {model.trained_on_stations} trạm) và <b>{model.num_actions} action</b>. Với{" "}
            {numStations} trạm, môi trường tạo vector <b>{model.scenario_input_dim} chiều</b> và cần {numStations}{" "}
            action, nên mạng dừng ngay ở quyết định đầu tiên. Muốn dùng phải thiết kế lại và train lại từ đầu.
          </p>
        )}
        <pre className="whitespace-pre-wrap break-words rounded-md bg-black/80 p-2 font-mono text-xs text-red-200">
          {result.error_message}
        </pre>
      </StatusBox>
    );
  }
  if (result.status === "unavailable") {
    return (
      <StatusBox tone="neutral" icon={<OctagonAlert className="h-6 w-6" />} title="MAPPO-MLP: chưa có model">
        <p className="text-sm">{result.error_message}</p>
      </StatusBox>
    );
  }
  return (
    <StatusBox tone="neutral" icon={<CheckCircle2 className="h-6 w-6" />} title="MAPPO-MLP chạy được">
      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
        Số trạm trùng với lúc train nên kích thước đầu vào khớp.
      </p>
      {result.metrics && <MetricsGrid metrics={result.metrics} />}
    </StatusBox>
  );
}

function GnnPanel({
  result,
  reference,
  data,
}: {
  result: AlgorithmResult;
  reference: AlgorithmResult | null;
  data: ZeroShotScaleResponse;
}) {
  if (result.status !== "ok" || !result.metrics) {
    return (
      <StatusBox
        tone={result.status === "crashed" ? "critical" : "neutral"}
        icon={<OctagonAlert className="h-6 w-6" />}
        title={result.status === "crashed" ? "MAPPO-GNN gặp lỗi" : "MAPPO-GNN: chưa có model"}
      >
        <p className="text-sm">{result.error_message}</p>
      </StatusBox>
    );
  }
  const m = result.metrics;
  const ref = reference?.metrics ?? null;
  return (
    <StatusBox tone="good" icon={<CheckCircle2 className="h-6 w-6" />} title="SUCCESS: MAPPO-GNN thích ứng thành công">
      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
        Cùng <b>{result.model?.parameters.toLocaleString("vi-VN")} tham số</b> train trên {data.trained_on.num_stations}{" "}
        trạm, mô hình đồ thị tự nhận {data.scenario.num_stations} nút trạm sạc và điều phối {data.scenario.num_evs} xe
        mà không cần train lại.
      </p>
      <MetricsGrid metrics={m} />
      {ref && (
        <p className="text-xs" style={{ color: "var(--text-secondary)" }}>
          Đối chiếu Nearest Station trên cùng kịch bản: chờ TB {fmt(ref.average_waiting_time)} s · đi TB{" "}
          {fmt(ref.average_travel_time)} s · sạc thành công {ref.num_charged} · quá tải {ref.num_overloaded}.
        </p>
      )}
    </StatusBox>
  );
}

function MetricsGrid({ metrics: m }: { metrics: EvalMetrics }) {
  const items: [string, string][] = [
    ["Chờ TB", `${fmt(m.average_waiting_time)} s`],
    ["Đi TB", `${fmt(m.average_travel_time)} s`],
    ["Sạc thành công", `${m.num_charged}`],
    ["Hỏng", `${m.num_failed}`],
    ["Quá tải", `${m.num_overloaded}`],
    ["Quyết định/xe", `${fmt(m.average_decision_time_ms, 2)} ms`],
  ];
  return (
    <dl className="grid grid-cols-3 gap-2">
      {items.map(([label, value]) => (
        <div key={label} className="rounded-md p-2" style={{ background: "var(--surface-1)" }}>
          <dt className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
            {label}
          </dt>
          <dd className="text-base font-semibold tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  );
}
