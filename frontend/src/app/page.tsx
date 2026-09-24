"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchMyCarPreview,
  fetchNetworkInfo,
  fetchStatus,
  pauseSimulation,
  resetSimulation,
  resumeSimulation,
  setSimulationSpeed,
  startSimulation,
} from "@/lib/api";
import { useSimulationSocket } from "@/lib/websocket";
import type {
  Algorithm,
  MyCarPreviewResponse,
  NetworkInfo,
  SimulationStatusResponse,
} from "@/lib/types";
import ControlPanel from "@/components/ControlPanel";
import DashboardCharts from "@/components/DashboardCharts";
import MyCarPanel from "@/components/MyCarPanel";
import StationDetailPanel from "@/components/StationDetailPanel";

// Leaflet touches `window` on load, so the map can only render client-side.
const MapComponent = dynamic(() => import("@/components/MapComponent"), { ssr: false });

const STATUS_POLL_INTERVAL_MS = 1000;

export default function Home() {
  const [network, setNetwork] = useState<NetworkInfo | null>(null);
  const [status, setStatus] = useState<SimulationStatusResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedStationId, setSelectedStationId] = useState<number | null>(null);
  const [myCarPreview, setMyCarPreview] = useState<MyCarPreviewResponse | null>(null);
  const [myCarPreviewError, setMyCarPreviewError] = useState<string | null>(null);
  const [myCarPreviewLoading, setMyCarPreviewLoading] = useState(false);
  const { snapshot, connectionState } = useSimulationSocket();

  useEffect(() => {
    fetchNetworkInfo()
      .then(setNetwork)
      .catch((err) => setErrorMessage(String(err)));
  }, []);

  const refreshStatus = useCallback(() => {
    fetchStatus()
      .then(setStatus)
      .catch((err) => setErrorMessage(String(err)));
  }, []);

  useEffect(() => {
    refreshStatus();
    const interval = setInterval(refreshStatus, STATUS_POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [refreshStatus]);

  const withErrorHandling = useCallback(
    async (action: () => Promise<SimulationStatusResponse>) => {
      try {
        setErrorMessage(null);
        const next = await action();
        setStatus(next);
        setMyCarPreview(null); // a new/changed episode invalidates any old preview route
        setMyCarPreviewError(null);
      } catch (err) {
        setErrorMessage(String(err));
      }
    },
    [],
  );

  const vehicles = useMemo(() => snapshot?.vehicles ?? [], [snapshot]);
  const stations = useMemo(() => snapshot?.stations ?? [], [snapshot]);

  const myVehicle = useMemo(
    () => vehicles.find((v) => v.id === status?.my_vehicle_id) ?? null,
    [vehicles, status?.my_vehicle_id],
  );

  // Once "my car" leaves TRAVELING (it got dispatched by the normal
  // automatic loop, or failed/completed) any earlier preview route is
  // stale. Derived directly during render instead of synced via an effect
  // -- no extra render pass needed.
  const visibleMyCarPreview = myVehicle && myVehicle.state === "TRAVELING" ? myCarPreview : null;

  const handleFindStation = useCallback(async () => {
    setMyCarPreviewLoading(true);
    setMyCarPreviewError(null);
    try {
      const preview = await fetchMyCarPreview();
      setMyCarPreview(preview);
      setSelectedStationId(preview.station_id);
    } catch (err) {
      setMyCarPreviewError(String(err));
    } finally {
      setMyCarPreviewLoading(false);
    }
  }, []);

  const selectedStationInfo = useMemo(
    () => network?.stations.find((s) => s.id === selectedStationId) ?? null,
    [network, selectedStationId],
  );
  const selectedLiveStation = useMemo(
    () => stations.find((s) => s.id === selectedStationId) ?? null,
    [stations, selectedStationId],
  );

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-4 p-4">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">EV Charging Dispatch Dashboard</h1>
        <span
          className={`text-xs ${connectionState === "open" ? "text-emerald-600" : "text-gray-400"}`}
        >
          WebSocket: {connectionState}
        </span>
      </header>

      <div className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-[1fr_280px_280px]">
        <div className="h-[60vh] min-h-[360px] overflow-hidden rounded-lg border border-gray-200 dark:border-gray-800 lg:h-auto">
          <MapComponent
            network={network}
            vehicles={vehicles}
            stations={stations}
            myVehicleId={status?.my_vehicle_id ?? null}
            myCarPreview={visibleMyCarPreview}
            onStationSelect={setSelectedStationId}
          />
        </div>

        <div className="flex flex-col gap-4">
          <ControlPanel
            status={status}
            errorMessage={errorMessage}
            onStart={(params: { seed?: number; algorithm: Algorithm; speed: number }) =>
              withErrorHandling(() => startSimulation(params))
            }
            onPause={() => withErrorHandling(pauseSimulation)}
            onResume={() => withErrorHandling(resumeSimulation)}
            onReset={(seed?: number) => withErrorHandling(() => resetSimulation(seed))}
            onSpeedChange={(speed: number) => withErrorHandling(() => setSimulationSpeed(speed))}
          />
          <MyCarPanel
            myVehicle={myVehicle}
            preview={visibleMyCarPreview}
            previewError={myCarPreviewError}
            loading={myCarPreviewLoading}
            onFindStation={handleFindStation}
          />
        </div>

        <div className="lg:sticky lg:top-4 lg:self-start">
          <StationDetailPanel station={selectedStationInfo} liveStation={selectedLiveStation} vehicles={vehicles} />
        </div>
      </div>

      <DashboardCharts status={status} stations={stations} network={network} />
    </div>
  );
}
