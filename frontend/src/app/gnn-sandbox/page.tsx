"use client";

// GNN Sandbox: pick any area on the map, generate a road network with
// K-Means-spread stations, then watch MAPPO-GNN dispatch its EVs live
// (backend/api/routes/sandbox.py). Separate from the simulation dashboard.

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import type { LatLngTuple } from "leaflet";
import { SetupPanel, SimulationPanel, type SetupValues } from "@/components/sandbox/SandboxPanels";
import { fetchActiveSandbox, generateSandbox, type GenerateEnvResponse, useSandboxSocket } from "@/lib/sandbox";

// Leaflet touches `window` on load, so the map can only render client-side.
const SandboxMap = dynamic(() => import("@/components/sandbox/SandboxMap"), { ssr: false });

const DEFAULT_SPEED = 5;

export default function GnnSandboxPage() {
  const [center, setCenter] = useState<LatLngTuple | null>(null);
  const [values, setValues] = useState<SetupValues>({ radiusKm: 3, numStations: 20, numEvs: 500, numRestStops: 10 });
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [env, setEnv] = useState<GenerateEnvResponse | null>(null);
  const [previous, setPrevious] = useState<GenerateEnvResponse | null>(null);

  useEffect(() => {
    fetchActiveSandbox()
      .then((active) => {
        if (!active) return;
        setPrevious(active);
        const p = active.sandbox.params;
        setCenter((current) => current ?? [p.lat, p.lng]);
        setValues({ radiusKm: p.radius_km, numStations: p.num_stations, numEvs: p.num_evs, numRestStops: p.num_rest_stops });
      })
      .catch(() => undefined);
  }, []);

  const generate = useCallback(async () => {
    if (!center) return;
    setGenerating(true);
    setError(null);
    try {
      const generated = await generateSandbox({
        lat: center[0],
        lng: center[1],
        radius_km: values.radiusKm,
        num_stations: values.numStations,
        num_evs: values.numEvs,
        num_rest_stops: values.numRestStops,
      });
      setPrevious(generated);
      setEnv(generated);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setGenerating(false);
    }
  }, [center, values]);

  return (
    <div className="relative h-screen w-full overflow-hidden">
      {/* One map per mode: the simulation view mounts its own (fed by its socket). */}
      {!env && (
        <div className="absolute inset-0">
          <SandboxMap mode="setup" center={center} radiusKm={values.radiusKm} onPick={setCenter} env={null} snapshot={null} />
        </div>
      )}

      <div className="pointer-events-none absolute left-4 top-4 z-[1000] flex items-center gap-2">
        <Link
          href="/"
          className="pointer-events-auto rounded-md border border-gray-300 bg-white/95 px-3 py-1.5 text-sm font-medium shadow hover:bg-gray-50 dark:border-gray-700 dark:bg-gray-900/95 dark:hover:bg-gray-800"
        >
          ← Dashboard
        </Link>
        <span className="rounded-md bg-white/90 px-2 py-1 text-sm font-semibold shadow dark:bg-gray-900/90">
          GNN Sandbox
        </span>
      </div>

      {env ? (
        <SimulationView key={env.sandbox.sandbox_id} env={env} onBack={() => setEnv(null)} />
      ) : (
        <div className="absolute right-4 top-4 z-[1000]">
          <SetupPanel
            center={center}
            values={values}
            onChange={setValues}
            onGenerate={generate}
            generating={generating}
            error={error}
            previous={previous}
            onReopen={() => previous && setEnv(previous)}
          />
        </div>
      )}
    </div>
  );
}

/** Simulation mode: owns the sandbox's WebSocket (one live run per mount) and
 *  draws its snapshots over the setup map's container. */
function SimulationView({ env, onBack }: { env: GenerateEnvResponse; onBack: () => void }) {
  const { snapshot, status, connection, lastError, send, reconnect } = useSandboxSocket(env.sandbox.sandbox_id);
  const [speed, setSpeed] = useState(DEFAULT_SPEED);

  // A fresh connection runs at the server's default speed: apply the chosen one.
  useEffect(() => {
    if (connection === "open") send({ action: "speed", value: speed });
    // Only on (re)connection; slider moves are sent by onSpeed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connection, send]);

  return (
    <>
      <div className="absolute inset-0">
        <SandboxMap mode="simulation" center={null} radiusKm={0} onPick={() => undefined} env={env} snapshot={snapshot} />
      </div>
      <div className="absolute right-4 top-4 z-[1000]">
        <SimulationPanel
          env={env}
          status={status}
          connection={connection}
          error={lastError}
          speed={speed}
          onSpeed={(value) => {
            setSpeed(value);
            send({ action: "speed", value });
          }}
          onStart={() => send({ action: "start" })}
          onPause={() => send({ action: "pause" })}
          onReset={() => send({ action: "reset" })}
          onReconnect={reconnect}
          onBack={onBack}
        />
      </div>
    </>
  );
}
