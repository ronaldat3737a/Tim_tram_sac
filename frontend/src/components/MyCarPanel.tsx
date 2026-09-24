"use client";

// "My Car" panel: highlights one designated EV and lets the user preview
// (not force-dispatch) what the currently-selected algorithm would suggest
// for it. The vehicle is only ever actually sent to a station through the
// normal automatic dispatch loop, once it naturally needs a decision
// (section 20) -- this button never bypasses that.

import { Car, Navigation } from "lucide-react";
import type { MyCarPreviewResponse, VehicleUpdate } from "@/lib/types";

interface MyCarPanelProps {
  myVehicle: VehicleUpdate | null;
  preview: MyCarPreviewResponse | null;
  previewError: string | null;
  loading: boolean;
  onFindStation: () => void;
}

export default function MyCarPanel({
  myVehicle,
  preview,
  previewError,
  loading,
  onFindStation,
}: MyCarPanelProps) {
  const canPreview = myVehicle?.state === "TRAVELING";

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-800 dark:bg-gray-900">
      <div className="flex items-center gap-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-full bg-lime-400 text-black">
          <Car className="h-4 w-4" />
        </span>
        <h3 className="text-sm font-semibold">Xe của tôi</h3>
        {myVehicle && (
          <span className="ml-auto text-xs text-gray-400">
            EV {myVehicle.id} — {myVehicle.state}
          </span>
        )}
      </div>

      {myVehicle && (
        <div className="text-xs text-gray-500 dark:text-gray-400">
          Battery: {(myVehicle.battery * 100).toFixed(0)}%
        </div>
      )}

      <button
        onClick={onFindStation}
        disabled={!canPreview || loading}
        className="flex items-center justify-center gap-1 rounded-md bg-lime-500 px-3 py-1.5 text-sm font-medium text-black hover:bg-lime-400 disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Navigation className="h-4 w-4" />
        {loading ? "Đang tìm…" : "Tìm trạm sạc"}
      </button>

      {!canPreview && myVehicle && (
        <p className="text-xs text-gray-400">
          Xe hiện đang ở trạng thái {myVehicle.state.toLowerCase()}, không thể xem gợi ý lúc này.
        </p>
      )}

      {previewError && <p className="text-xs text-red-500">{previewError}</p>}

      {preview && (
        <div className="rounded-md bg-lime-50 p-2 text-xs dark:bg-lime-950/40">
          <p>
            Gợi ý: <span className="font-semibold">Station {preview.station_id}</span>
          </p>
          <p className={preview.reachable ? "text-emerald-600" : "text-red-500"}>
            {preview.reachable ? "Đủ pin để tới trạm này" : "Không đủ pin để tới trạm này"}
          </p>
        </div>
      )}
    </div>
  );
}
