import React, { useMemo, useState } from "react";

type LatLon = { lat: number; lon: number };

function parseLatLon(latStr: string, lonStr: string): LatLon | null {
  const lat = Number(latStr);
  const lon = Number(lonStr);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (lat < -90 || lat > 90) return null;
  if (lon < -180 || lon > 180) return null;
  return { lat, lon };
}

export function LatLonJump({
  onGo,
  defaultLat = "",
  defaultLon = "",
}: {
  onGo: (p: LatLon) => void;
  defaultLat?: string;
  defaultLon?: string;
}) {
  const [lat, setLat] = useState(defaultLat);
  const [lon, setLon] = useState(defaultLon);
  const [error, setError] = useState<string | null>(null);

  const canGo = useMemo(() => parseLatLon(lat, lon) !== null, [lat, lon]);

  function handleGo() {
    const p = parseLatLon(lat, lon);
    if (!p) {
      setError("Please enter a valid lat (-90..90) and lon (-180..180).");
      return;
    }
    setError(null);
    onGo(p);
  }

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      gap: 8,
      padding: 12,
      borderRadius: 10,
      background: "rgba(0,0,0,0.55)",
      color: "white",
      width: 260
    }}>
      <div style={{ fontWeight: 600 }}>Go to coordinates</div>

      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 12, opacity: 0.9 }}>Latitude</span>
        <input
          value={lat}
          onChange={(e) => setLat(e.target.value)}
          placeholder="e.g. 36.7783"
          style={{ padding: 8, borderRadius: 8, border: "1px solid #444" }}
        />
      </label>

      <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontSize: 12, opacity: 0.9 }}>Longitude</span>
        <input
          value={lon}
          onChange={(e) => setLon(e.target.value)}
          placeholder="e.g. -119.4179"
          style={{ padding: 8, borderRadius: 8, border: "1px solid #444" }}
        />
      </label>

      <button
        onClick={handleGo}
        disabled={!canGo}
        style={{
          padding: 10,
          borderRadius: 10,
          border: "none",
          cursor: canGo ? "pointer" : "not-allowed",
          background: canGo ? "#2f80ed" : "#555",
          color: "white",
          fontWeight: 600
        }}
      >
        Go
      </button>

      {error && <div style={{ fontSize: 12, color: "#ffb4b4" }}>{error}</div>}
      <div style={{ fontSize: 11, opacity: 0.85 }}>
        Tip: You can paste decimals directly.
      </div>
    </div>
  );
}
