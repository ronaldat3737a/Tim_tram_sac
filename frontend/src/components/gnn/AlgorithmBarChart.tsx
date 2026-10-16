"use client";

// One measure across the four algorithms, as a bar chart (Recharts). Bars
// are coloured by algorithm (gnnViz.module.css roles), labelled with their
// value in text ink, and hoverable for the exact number.

import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AlgorithmName } from "@/lib/gnnEval";

export const ALGORITHM_COLORS: Record<AlgorithmName, string> = {
  nearest_station: "var(--series-heuristic)",
  least_queue: "var(--series-heuristic)",
  mappo_mlp: "var(--series-mlp)",
  mappo_gnn: "var(--series-gnn)",
};

export interface BarDatum {
  algorithm: AlgorithmName;
  label: string;
  value: number;
}

interface AlgorithmBarChartProps {
  title: string;
  unit: string;
  data: BarDatum[];
  digits?: number;
}

function format(value: number, digits: number): string {
  return value.toLocaleString("vi-VN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export default function AlgorithmBarChart({ title, unit, data, digits = 1 }: AlgorithmBarChartProps) {
  return (
    <figure className="flex flex-col gap-2 rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
      <figcaption className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
        {title} <span className="font-normal" style={{ color: "var(--text-secondary)" }}>({unit})</span>
      </figcaption>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 22, right: 8, bottom: 4, left: 0 }} barCategoryGap="28%">
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={{ stroke: "var(--axis)" }}
              tick={{ fill: "var(--text-secondary)", fontSize: 12 }}
              interval={0}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              width={52}
              tick={{ fill: "var(--text-secondary)", fontSize: 11 }}
              tickFormatter={(v: number) => format(v, 0)}
            />
            <Tooltip
              cursor={{ fill: "var(--grid)", opacity: 0.5 }}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const datum = payload[0].payload as BarDatum;
                return (
                  <div
                    className="rounded-md border px-2.5 py-1.5 text-xs shadow"
                    style={{ background: "var(--surface-1)", borderColor: "var(--grid)", color: "var(--text-primary)" }}
                  >
                    <div className="flex items-center gap-1.5 font-medium">
                      <span
                        className="inline-block h-2.5 w-2.5 rounded-sm"
                        style={{ background: ALGORITHM_COLORS[datum.algorithm] }}
                      />
                      {datum.label}
                    </div>
                    <div style={{ color: "var(--text-secondary)" }}>
                      {format(datum.value, digits)} {unit}
                    </div>
                  </div>
                );
              }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={56} isAnimationActive={false}>
              {data.map((datum) => (
                <Cell key={datum.algorithm} fill={ALGORITHM_COLORS[datum.algorithm]} />
              ))}
              <LabelList
                dataKey="value"
                position="top"
                fill="var(--text-primary)"
                fontSize={11}
                formatter={(v) => format(Number(v), digits)}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}
