// EV state colors of the GNN Sandbox (same as the simulation dashboard). Kept
// out of SandboxMap.tsx so panels can use them without loading Leaflet, which
// needs `window` and so cannot be imported during server rendering.

import type { VehicleUpdate } from "@/lib/types";

export const VEHICLE_COLORS: Record<VehicleUpdate["state"], string> = {
  TRAVELING: "#22c55e",
  WAITING: "#f59e0b",
  CHARGING: "#3b82f6",
  RETURNING_TO_DEPOT: "#8b5cf6",
  COMPLETED: "#9ca3af",
  FAILED: "#ef4444",
};
