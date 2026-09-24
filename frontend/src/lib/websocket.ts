"use client";

// WebSocket client for /ws/simulation (backend/api/websockets.py,
// PROJECT_SPEC.md section 34/35). Only ever parses and stores the messages
// the backend sends -- no simulation or reward computation here (section 39).

import { useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "@/lib/api";
import type { SimulationUpdateMessage } from "@/lib/types";

const RECONNECT_DELAY_MS = 1000;

function websocketUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_WS_URL;
  if (explicit) return explicit;
  return `${API_BASE_URL.replace(/^http/, "ws")}/ws/simulation`;
}

export type WebSocketConnectionState = "connecting" | "open" | "closed";

/** Latest simulation_update snapshot, kept fresh via a self-reconnecting WebSocket. */
export function useSimulationSocket(): {
  snapshot: SimulationUpdateMessage | null;
  connectionState: WebSocketConnectionState;
} {
  const [snapshot, setSnapshot] = useState<SimulationUpdateMessage | null>(null);
  const [connectionState, setConnectionState] =
    useState<WebSocketConnectionState>("connecting");
  const shouldReconnect = useRef(true);

  useEffect(() => {
    shouldReconnect.current = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      setConnectionState("connecting");
      socket = new WebSocket(websocketUrl());

      socket.onopen = () => setConnectionState("open");

      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as SimulationUpdateMessage;
        setSnapshot(message);
      };

      socket.onclose = () => {
        setConnectionState("closed");
        if (shouldReconnect.current) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      shouldReconnect.current = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return { snapshot, connectionState };
}
