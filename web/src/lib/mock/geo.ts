/** Small geodesy helpers (WGS84, metres). Shared by the mock and the map. */
export type LonLat = [number, number];

const R = 6371008.8;
const rad = (d: number) => (d * Math.PI) / 180;

export function haversineM(a: LonLat, b: LonLat): number {
  const dLat = rad(b[1] - a[1]);
  const dLon = rad(b[0] - a[0]);
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a[1])) * Math.cos(rad(b[1])) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)));
}

export function lineLengthM(coords: LonLat[]): number {
  let s = 0;
  for (let i = 1; i < coords.length; i++) s += haversineM(coords[i - 1], coords[i]);
  return s;
}

/** Point at distance `d` metres along the polyline (clamped), plus bearing in degrees. */
export function pointAlong(coords: LonLat[], d: number): { p: LonLat; bearing: number } {
  let acc = 0;
  for (let i = 1; i < coords.length; i++) {
    const seg = haversineM(coords[i - 1], coords[i]);
    if (acc + seg >= d || i === coords.length - 1) {
      const t = seg > 0 ? Math.max(0, Math.min(1, (d - acc) / seg)) : 0;
      const a = coords[i - 1];
      const b = coords[i];
      const p: LonLat = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
      const y = Math.sin(rad(b[0] - a[0])) * Math.cos(rad(b[1]));
      const x = Math.cos(rad(a[1])) * Math.sin(rad(b[1])) - Math.sin(rad(a[1])) * Math.cos(rad(b[1])) * Math.cos(rad(b[0] - a[0]));
      return { p, bearing: ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360 };
    }
    acc += seg;
  }
  return { p: coords[coords.length - 1], bearing: 0 };
}

/** Sub-polyline between chainages d0..d1 (metres). */
export function sliceLine(coords: LonLat[], d0: number, d1: number): LonLat[] {
  const out: LonLat[] = [pointAlong(coords, d0).p];
  let acc = 0;
  for (let i = 1; i < coords.length; i++) {
    acc += haversineM(coords[i - 1], coords[i]);
    if (acc > d0 && acc < d1) out.push(coords[i]);
  }
  out.push(pointAlong(coords, d1).p);
  return out;
}

/** Offset a point by metres east/north. */
export function offsetM(p: LonLat, east: number, north: number): LonLat {
  return [p[0] + east / (111320 * Math.cos(rad(p[1]))), p[1] + north / 110574];
}

export function inBbox(p: LonLat, bbox: [number, number, number, number]): boolean {
  return p[0] >= bbox[0] && p[0] <= bbox[2] && p[1] >= bbox[1] && p[1] <= bbox[3];
}

export function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
