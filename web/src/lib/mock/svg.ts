/** Tiny SVG placeholders for mock thumbnails (< 2 KB each, PLAN WP9: list thumbnails <= 20 KB). */
const SEV_FILL: Record<string, string> = { small: '#7a6a4f', medium: '#5c4a32', large: '#3b2a1a', unknown_water: '#3d5f7a' };

export function svgDataUrl(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

/** Deterministic blob-ish outline in a w x h box. */
export function blobPath(cx: number, cy: number, rx: number, ry: number, seed: number, n = 14): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    const j = 0.78 + 0.22 * Math.abs(Math.sin(seed * 12.9898 + i * 78.233));
    pts.push([cx + Math.cos(a) * rx * j, cy + Math.sin(a) * ry * j]);
  }
  return pts;
}

export function thumbSvg(id: number, severity: string | null | undefined, water: boolean): string {
  const fill = water ? SEV_FILL.unknown_water : SEV_FILL[severity ?? 'medium'] ?? SEV_FILL.medium;
  const pts = blobPath(48, 36, 26, 15, id).map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 72" width="96" height="72">`
    + `<rect width="96" height="72" fill="#8d8f93"/>`
    + `<rect y="0" width="96" height="72" fill="url(#g)" opacity=".25"/>`
    + `<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient></defs>`
    + `<polygon points="${pts}" fill="${fill}" stroke="#2b2b2b" stroke-width="1.2"/>`
    + (water ? `<ellipse cx="44" cy="33" rx="12" ry="4" fill="#9cc3dd" opacity=".7"/>` : '')
    + `<text x="4" y="68" font-size="9" font-family="sans-serif" fill="#fff">#${id}</text></svg>`;
  return svgDataUrl(svg);
}
