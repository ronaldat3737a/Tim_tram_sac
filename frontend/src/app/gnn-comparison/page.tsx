"use client";

// MAPPO-MLP vs MAPPO-GNN comparison dashboard (backend/api/routes/gnn_eval.py).
// The live simulation dashboard at "/" is unchanged.

import Link from "next/link";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import HeadToHeadSection from "@/components/gnn/HeadToHeadSection";
import ZeroShotSection from "@/components/gnn/ZeroShotSection";
import styles from "@/components/gnn/gnnViz.module.css";
import { fetchGnnEvalStatus, type GNNEvalStatus, type ModelInfo } from "@/lib/gnnEval";

export default function GnnComparisonPage() {
  const [status, setStatus] = useState<GNNEvalStatus | null>(null);

  useEffect(() => {
    fetchGnnEvalStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  return (
    <div className={`${styles.vizRoot} mx-auto flex w-full max-w-[1400px] flex-1 flex-col gap-8 p-4`}>
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">MAPPO-MLP vs MAPPO-GNN</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            So sánh mô hình cũ (MLP) và mô hình đồ thị mới (GNN) trên cùng tài nguyên, và khả năng co giãn zero-shot.
          </p>
        </div>
        <Link
          href="/"
          className="flex items-center gap-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
        >
          <ArrowLeft className="h-4 w-4" /> Về dashboard mô phỏng
        </Link>
      </header>

      {status && (
        <div className="grid gap-3 sm:grid-cols-2">
          <ModelCard title="MAPPO-MLP (cũ)" model={status.mlp_model} />
          <ModelCard title="MAPPO-GNN (mới)" model={status.gnn_model} />
        </div>
      )}

      <HeadToHeadSection />
      <ZeroShotSection />
    </div>
  );
}

function ModelCard({ title, model }: { title: string; model: ModelInfo | null }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3 text-sm dark:border-gray-800 dark:bg-gray-900">
      <div className="font-medium">{title}</div>
      {model ? (
        <div className="text-xs text-gray-500 dark:text-gray-400">
          {model.parameters.toLocaleString("vi-VN")} tham số · đầu vào{" "}
          {model.input_dim === null ? "không phụ thuộc số trạm (đồ thị)" : `cố định ${model.input_dim} chiều`} · train
          trên {model.trained_on_stations} trạm / {model.trained_on_evs} xe
          {model.trained_decisions !== null && ` · checkpoint sau ${model.trained_decisions.toLocaleString("vi-VN")} quyết định`}
        </div>
      ) : (
        <div className="text-xs text-amber-700 dark:text-amber-400">Chưa có file model.</div>
      )}
    </div>
  );
}
