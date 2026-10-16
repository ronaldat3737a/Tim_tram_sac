"use client";

// Section 1: Nearest Station, Least Queue, MAPPO-MLP and MAPPO-GNN on one
// shared scenario (same seed: same stations, EVs, batteries, traffic).

import { useState } from "react";
import { Loader2, Play } from "lucide-react";
import AlgorithmBarChart, { ALGORITHM_COLORS, type BarDatum } from "@/components/gnn/AlgorithmBarChart";
import { type AlgorithmResult, type CompareBaselineResponse, runCompareBaseline } from "@/lib/gnnEval";

const DEFAULT_SEED = 1000;

function fmt(value: number, digits = 1): string {
  return value.toLocaleString("vi-VN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function bars(results: AlgorithmResult[], pick: (r: AlgorithmResult) => number): BarDatum[] {
  return results
    .filter((r) => r.metrics !== null)
    .map((r) => ({ algorithm: r.algorithm, label: r.label, value: pick(r) }));
}

export default function HeadToHeadSection() {
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CompareBaselineResponse | null>(null);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await runCompareBaseline({ seed }));
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    } finally {
      setLoading(false);
    }
  };

  const mlp = data?.results.find((r) => r.algorithm === "mappo_mlp");
  const gnn = data?.results.find((r) => r.algorithm === "mappo_gnn");
  const missing = data?.results.filter((r) => r.metrics === null) ?? [];

  return (
    <section className="flex flex-col gap-4">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">1. Head-to-Head Comparison</h2>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Cùng một kịch bản cho cả 4 thuật toán: cùng seed, cùng 10 trạm sạc, 200 xe, cùng pin và kẹt xe ban đầu.
          </p>
        </div>
        <div className="flex items-end gap-2">
          <label className="flex flex-col gap-1 text-xs font-medium text-gray-500 dark:text-gray-400">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-24 rounded-md border border-gray-300 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100"
            />
          </label>
          <button
            onClick={run}
            disabled={loading}
            className="flex items-center gap-1.5 rounded-md bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-60"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Benchmark (10 Stations / 200 EVs)
          </button>
        </div>
      </header>

      {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

      {loading && (
        <div className="flex flex-col gap-3" aria-busy="true">
          <p className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
            <Loader2 className="h-4 w-4 animate-spin" /> Đang chạy lần lượt 4 thuật toán trên cùng kịch bản (khoảng
            15–30 giây)…
          </p>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="h-72 animate-pulse rounded-lg bg-gray-100 dark:bg-gray-800" />
            <div className="h-72 animate-pulse rounded-lg bg-gray-100 dark:bg-gray-800" />
          </div>
        </div>
      )}

      {!loading && data && (
        <>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Kịch bản: {data.scenario.num_stations} trạm · {data.scenario.num_evs} xe · {data.scenario.num_depots} bãi đỗ ·
            bán kính {fmt(data.scenario.radius_meters / 1000)} km · seed {data.scenario.seed}
            {data.cached && " · kết quả từ cache (cùng yêu cầu, cùng model)"}
          </p>

          {mlp?.metrics && gnn?.metrics && (
            <div className="grid gap-3 sm:grid-cols-3">
              <DeltaTile
                title="Thời gian chờ TB: GNN so với MLP"
                gnn={gnn.metrics.average_waiting_time}
                mlp={mlp.metrics.average_waiting_time}
                unit="s"
              />
              <DeltaTile
                title="Thời gian đi TB: GNN so với MLP"
                gnn={gnn.metrics.average_travel_time}
                mlp={mlp.metrics.average_travel_time}
                unit="s"
              />
              <DeltaTile
                title="Thời gian quyết định/xe: GNN so với MLP"
                gnn={gnn.metrics.average_decision_time_ms}
                mlp={mlp.metrics.average_decision_time_ms}
                unit="ms"
                digits={2}
              />
            </div>
          )}

          {missing.map((r) => (
            <p key={r.algorithm} className="text-sm text-amber-700 dark:text-amber-400">
              {r.label}: {r.error_message}
            </p>
          ))}

          <div className="grid gap-4 md:grid-cols-2">
            <AlgorithmBarChart
              title="Thời gian chờ trung bình"
              unit="giây"
              data={bars(data.results, (r) => r.metrics!.average_waiting_time)}
            />
            <AlgorithmBarChart
              title="Thời gian quyết định trung bình mỗi xe"
              unit="ms"
              digits={2}
              data={bars(data.results, (r) => r.metrics!.average_decision_time_ms)}
            />
          </div>

          <MetricsTable results={data.results} />
        </>
      )}
    </section>
  );
}

function DeltaTile({
  title,
  gnn,
  mlp,
  unit,
  digits = 1,
}: {
  title: string;
  gnn: number;
  mlp: number;
  unit: string;
  digits?: number;
}) {
  const change = mlp > 0 ? (gnn - mlp) / mlp : 0;
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
      <div className="text-xs text-gray-500 dark:text-gray-400">{title}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">
        {change > 0 ? "+" : ""}
        {fmt(change * 100)}%
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400">
        GNN {fmt(gnn, digits)} {unit} · MLP {fmt(mlp, digits)} {unit} (thấp hơn là tốt hơn)
      </div>
    </div>
  );
}

function MetricsTable({ results }: { results: AlgorithmResult[] }) {
  const columns: { label: string; value: (r: AlgorithmResult) => string }[] = [
    { label: "Chờ TB (s)", value: (r) => fmt(r.metrics!.average_waiting_time) },
    { label: "Đi TB (s)", value: (r) => fmt(r.metrics!.average_travel_time) },
    { label: "Tổng chờ (s)", value: (r) => fmt(r.metrics!.total_waiting_time, 0) },
    { label: "Tổng đi (s)", value: (r) => fmt(r.metrics!.total_travel_time, 0) },
    { label: "Sạc thành công", value: (r) => `${r.metrics!.num_charged}` },
    { label: "Hỏng", value: (r) => `${r.metrics!.num_failed}` },
    { label: "Quá tải", value: (r) => `${r.metrics!.num_overloaded}` },
    { label: "Quyết định/xe (ms)", value: (r) => fmt(r.metrics!.average_decision_time_ms, 2) },
  ];
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900">
      <table className="w-full text-sm tabular-nums">
        <thead className="text-xs text-gray-500 dark:text-gray-400">
          <tr className="border-b border-gray-200 dark:border-gray-800">
            <th className="px-3 py-2 text-left font-medium">Thuật toán</th>
            {columns.map((c) => (
              <th key={c.label} className="px-3 py-2 text-right font-medium">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.algorithm} className="border-b border-gray-100 last:border-0 dark:border-gray-800">
              <td className="px-3 py-2">
                <span className="flex items-center gap-2">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: ALGORITHM_COLORS[r.algorithm] }} />
                  {r.label}
                </span>
              </td>
              {r.metrics ? (
                columns.map((c) => (
                  <td key={c.label} className="px-3 py-2 text-right">
                    {c.value(r)}
                  </td>
                ))
              ) : (
                <td colSpan={columns.length} className="px-3 py-2 text-right text-gray-500 dark:text-gray-400">
                  {r.status}: {r.error_message}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
