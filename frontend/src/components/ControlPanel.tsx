"use client";

// Control panel (PROJECT_SPEC.md section 38): Start, Pause, Reset, Algorithm
// selection, Simulation speed. Sends control requests to the backend only --
// no RL decision or simulation logic lives here (section 39).

import { useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import type { Algorithm, SimulationStatusResponse } from "@/lib/types";

const ALGORITHMS: { value: Algorithm; label: string }[] = [
  { value: "nearest_station", label: "Nearest Station" },
  { value: "shortest_time", label: "Shortest Travel Time" },
  { value: "least_queue", label: "Least Queue" },
  { value: "dqn", label: "DQN (trained model)" },
];

interface ControlPanelProps {
  status: SimulationStatusResponse | null;
  onStart: (params: { seed?: number; algorithm: Algorithm; speed: number }) => void;
  onPause: () => void;
  onResume: () => void;
  onReset: (seed?: number) => void;
  onSpeedChange: (speed: number) => void;
  errorMessage: string | null;
}

export default function ControlPanel({
  status,
  onStart,
  onPause,
  onResume,
  onReset,
  onSpeedChange,
  errorMessage,
}: ControlPanelProps) {
  const [algorithm, setAlgorithm] = useState<Algorithm>(status?.algorithm ?? "nearest_station");
  const [seedInput, setSeedInput] = useState("");
  const [speed, setSpeed] = useState(status?.speed ?? 5);

  const parsedSeed = seedInput.trim() === "" ? undefined : Number(seedInput);

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-medium ${
            status?.status === "running"
              ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300"
              : status?.status === "paused"
                ? "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300"
                : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300"
          }`}
        >
          {status?.status ?? "loading…"}
        </span>
        <span className="text-xs text-gray-500 dark:text-gray-400">
          t = {(status?.simulation_time ?? 0).toFixed(0)}s · seed {status?.episode_seed ?? "—"}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium text-gray-500 dark:text-gray-400">Algorithm</label>
        <select
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
          className="rounded-md border border-gray-300 bg-white px-2 py-1 text-sm dark:border-gray-700 dark:bg-gray-800"
        >
          {ALGORITHMS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium text-gray-500 dark:text-gray-400">
          Seed (optional)
        </label>
        <input
          type="number"
          value={seedInput}
          onChange={(e) => setSeedInput(e.target.value)}
          placeholder={String(status?.episode_seed ?? 42)}
          className="rounded-md border border-gray-300 bg-white px-2 py-1 text-sm dark:border-gray-700 dark:bg-gray-800"
        />
      </div>

      <div className="flex flex-col gap-1">
        <label className="text-xs font-medium text-gray-500 dark:text-gray-400">
          Speed: {speed.toFixed(1)} ticks/s
        </label>
        <input
          type="range"
          min={1}
          max={50}
          step={0.5}
          value={speed}
          onChange={(e) => {
            const next = Number(e.target.value);
            setSpeed(next);
            onSpeedChange(next);
          }}
        />
      </div>

      <div className="flex flex-wrap gap-2">
        {status?.status === "running" ? (
          <button
            onClick={onPause}
            className="flex items-center gap-1 rounded-md bg-amber-500 px-3 py-1.5 text-sm font-medium text-white hover:bg-amber-400"
          >
            <Pause className="h-4 w-4" /> Pause
          </button>
        ) : (
          <>
            <button
              onClick={() => onStart({ seed: parsedSeed, algorithm, speed })}
              title="Reset to a fresh episode using the algorithm/seed/speed above and run it"
              className="flex items-center gap-1 rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-500"
            >
              <Play className="h-4 w-4" /> Start
            </button>
            {status?.status === "paused" && (
              <button
                onClick={onResume}
                title="Continue the current episode without resetting it"
                className="flex items-center gap-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
              >
                <Play className="h-4 w-4" /> Resume
              </button>
            )}
          </>
        )}
        <button
          onClick={() => onReset(parsedSeed)}
          className="flex items-center gap-1 rounded-md border border-gray-300 px-3 py-1.5 text-sm font-medium hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
        >
          <RotateCcw className="h-4 w-4" /> Reset
        </button>
      </div>

      {errorMessage && (
        <p className="text-xs text-red-600 dark:text-red-400">{errorMessage}</p>
      )}
    </div>
  );
}
