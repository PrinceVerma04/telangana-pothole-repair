/**
 * MapLibre map of potholes over the OpenFreeMap "liberty" basemap (DECISIONS D7).
 * Layers: heatmap at z < 13 (density overview; PLAN asks for H3 heat - heatmap is the client-side stand-in),
 * circles coloured by severity (status palette + label in legend) or priority (single-hue sequential ramp),
 * and an optional IRC:82 stretch-rating line layer. Falls back to a plain background style when the
 * basemap cannot be fetched (offline first visit) and to a message when WebGL is unavailable.
 */
import 'maplibre-gl/dist/maplibre-gl.css';
import { Map as MlMap, NavigationControl, Popup, ScaleControl, type GeoJSONSource, type MapLayerMouseEvent, type MapOptions } from 'maplibre-gl';
import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import type { PotholeListItem, SegmentRating, Severity } from '../lib/types';
import { SEVERITY_COLOR } from './ui';

export const BASEMAP_STYLE = 'https://tiles.openfreemap.org/styles/liberty';
/** What the Map constructor accepts as `style`: a style URL or an inline style document. */
type MapStyle = NonNullable<MapOptions['style']>;
/** Inline style document (maplibre-gl does not re-export StyleSpecification, so derive it). */
type StyleDoc = Exclude<MapStyle, string>;
const OFFLINE_STYLE: StyleDoc = {
  version: 8,
  sources: {},
  layers: [{ id: 'bg', type: 'background', paint: { 'background-color': '#e9ecef' } }],
};

export const PRIORITY_RAMP: [number, string][] = [[0, '#cde2fb'], [40, '#6da7ec'], [70, '#256abf'], [100, '#0d366b']];
export const RATING_COLOR = { Good: '#0ca30c', Fair: '#fab219', Poor: '#d03b3b' } as const;
export type ColorBy = 'severity' | 'priority';
export type Bbox = [number, number, number, number];

export const HYD_CENTER: [number, number] = [78.475, 17.405];

function circleColor(by: ColorBy): unknown {
  if (by === 'priority') return ['interpolate', ['linear'], ['coalesce', ['get', 'priority'], 0], ...PRIORITY_RAMP.flat()];
  const sev = Object.entries(SEVERITY_COLOR) as [Severity, string][];
  return ['match', ['coalesce', ['get', 'severity'], 'none'], ...sev.flat(), '#898781'];
}

function toFc(items: PotholeListItem[]) {
  return {
    type: 'FeatureCollection' as const,
    features: items.map((p) => ({
      type: 'Feature' as const,
      id: p.id,
      geometry: { type: 'Point' as const, coordinates: [p.lon, p.lat] },
      properties: { id: p.id, severity: p.severity ?? null, priority: p.priority ?? 0, status: p.status, road: p.road_name ?? '', depth: p.depth_p95_mm ?? null, water: p.water_filled },
    })),
  };
}
function ratingsFc(r: SegmentRating[] | null | undefined) {
  return {
    type: 'FeatureCollection' as const,
    features: (r ?? []).map((s) => ({ type: 'Feature' as const, geometry: s.geometry, properties: { rating: s.rating, pct: s.pothole_area_pct } })),
  };
}

async function pickStyle(timeoutMs = 6000): Promise<MapStyle> {
  try {
    const ctrl = new AbortController();
    const tm = setTimeout(() => ctrl.abort(), timeoutMs);
    const r = await fetch(BASEMAP_STYLE, { signal: ctrl.signal });
    clearTimeout(tm);
    if (r.ok) return BASEMAP_STYLE;
  } catch { /* offline */ }
  return OFFLINE_STYLE;
}

export interface PotholeMapProps {
  items: PotholeListItem[];
  colorBy?: ColorBy;
  heat?: boolean;
  ratings?: SegmentRating[] | null;
  selectedId?: number | null;
  onSelect?: (id: number) => void;
  onBbox?: (b: Bbox) => void;
  height?: number | string;
  fit?: boolean;
  center?: [number, number];
  zoom?: number;
}

export default function PotholeMap({ items, colorBy = 'severity', heat = true, ratings, selectedId, onSelect, onBbox, height = 480, fit = true, center, zoom }: PotholeMapProps) {
  const { t } = useTranslation();
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MlMap | null>(null);
  const [ready, setReady] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [offlineStyle, setOfflineStyle] = useState(false);
  const cb = useRef({ onSelect, onBbox });
  cb.current = { onSelect, onBbox };
  const fitted = useRef(false);

  useEffect(() => {
    let disposed = false;
    let m: MlMap | null = null;
    (async () => {
      const style = await pickStyle();
      if (disposed || !el.current) return;
      setOfflineStyle(typeof style !== 'string');
      try {
        m = new MlMap({ container: el.current, style, center: center ?? HYD_CENTER, zoom: zoom ?? 11.2, attributionControl: { compact: true } });
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
        return;
      }
      map.current = m;
      m.addControl(new NavigationControl({ showCompass: false }), 'top-right');
      m.addControl(new ScaleControl({ unit: 'metric' }), 'bottom-left');
      m.on('load', () => {
        if (!m) return;
        m.addSource('ratings', { type: 'geojson', data: ratingsFc(null) });
        m.addLayer({ id: 'ratings-line', type: 'line', source: 'ratings', layout: { 'line-cap': 'round' },
          paint: { 'line-color': ['match', ['get', 'rating'], 'Good', RATING_COLOR.Good, 'Fair', RATING_COLOR.Fair, RATING_COLOR.Poor], 'line-width': ['interpolate', ['linear'], ['zoom'], 12, 3, 16, 7], 'line-opacity': 0.85 } });
        m.addSource('potholes', { type: 'geojson', data: toFc([]) });
        m.addLayer({ id: 'ph-heat', type: 'heatmap', source: 'potholes', maxzoom: 13.5,
          paint: {
            'heatmap-weight': ['interpolate', ['linear'], ['coalesce', ['get', 'priority'], 0], 0, 0.2, 100, 1],
            'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 9, 12, 13, 28],
            'heatmap-opacity': ['interpolate', ['linear'], ['zoom'], 11.5, 0.75, 13.5, 0],
            'heatmap-color': ['interpolate', ['linear'], ['heatmap-density'], 0, 'rgba(205,226,251,0)', 0.3, '#86b6ef', 0.6, '#2a78d6', 1, '#0d366b'],
          } });
        m.addLayer({ id: 'ph-circles', type: 'circle', source: 'potholes',
          paint: {
            'circle-color': circleColor('severity') as never,
            'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 3.5, 14, 7, 17, 11],
            'circle-stroke-color': '#ffffff',
            'circle-stroke-width': 1.5,
            'circle-opacity': ['interpolate', ['linear'], ['zoom'], 10.5, 0.55, 12.5, 1],
          } });
        m.addLayer({ id: 'ph-selected', type: 'circle', source: 'potholes', filter: ['==', ['get', 'id'], -1],
          paint: { 'circle-radius': ['interpolate', ['linear'], ['zoom'], 10, 7, 17, 15], 'circle-color': 'rgba(0,0,0,0)', 'circle-stroke-color': '#0b0b0b', 'circle-stroke-width': 2.5 } });
        m.on('click', 'ph-circles', (ev: MapLayerMouseEvent) => {
          const f = ev.features?.[0];
          if (!f) return;
          const id = Number(f.properties?.id);
          const box = document.createElement('div');
          box.className = 'map-pop';
          const h = document.createElement('strong');
          h.textContent = `#${id} · ${String(f.properties?.road ?? '')}`;
          const p = document.createElement('div');
          const d = f.properties?.depth;
          p.textContent = `${t(`severity.${f.properties?.severity ?? 'small'}`)} · P ${Number(f.properties?.priority ?? 0).toFixed(0)}${d != null && d !== 'null' ? ` · ${Math.round(Number(d))} mm` : ''}`;
          const btn = document.createElement('button');
          btn.className = 'btn btn-sm';
          btn.textContent = t('map.openDetail');
          btn.onclick = () => cb.current.onSelect?.(id);
          box.append(h, p, btn);
          new Popup({ closeButton: true, maxWidth: '260px' }).setLngLat(ev.lngLat).setDOMContent(box).addTo(m!);
        });
        m.on('mouseenter', 'ph-circles', () => { m!.getCanvas().style.cursor = 'pointer'; });
        m.on('mouseleave', 'ph-circles', () => { m!.getCanvas().style.cursor = ''; });
        m.on('moveend', () => {
          const b = m!.getBounds();
          cb.current.onBbox?.([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
        });
        setReady(true);
        const b = m.getBounds();
        cb.current.onBbox?.([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
      });
      m.on('error', (e) => { console.warn('map error', e?.error?.message ?? e); });
    })();
    return () => {
      disposed = true;
      m?.remove();
      map.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    (m.getSource('potholes') as GeoJSONSource | undefined)?.setData(toFc(items));
    if (fit && !fitted.current && items.length > 1) {
      fitted.current = true;
      let w = 180, s = 90, e = -180, n = -90;
      for (const p of items) { w = Math.min(w, p.lon); e = Math.max(e, p.lon); s = Math.min(s, p.lat); n = Math.max(n, p.lat); }
      m.fitBounds([[w, s], [e, n]], { padding: 40, duration: 0, maxZoom: 15 });
    }
  }, [items, ready, fit]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    m.setPaintProperty('ph-circles', 'circle-color', circleColor(colorBy) as never);
  }, [colorBy, ready]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    m.setLayoutProperty('ph-heat', 'visibility', heat ? 'visible' : 'none');
  }, [heat, ready]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    (m.getSource('ratings') as GeoJSONSource | undefined)?.setData(ratingsFc(ratings) as never);
  }, [ratings, ready]);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    m.setFilter('ph-selected', ['==', ['get', 'id'], selectedId ?? -1]);
    const p = items.find((x) => x.id === selectedId);
    if (p) m.easeTo({ center: [p.lon, p.lat], zoom: Math.max(m.getZoom(), 15), duration: 400 });
  }, [selectedId, ready, items]);

  return (
    <div className="map-wrap" style={{ height }}>
      {err ? (
        <div className="map-fallback">
          <p>{t('map.noWebgl')}</p>
          <small>{err}</small>
        </div>
      ) : (
        <div ref={el} className="map" data-testid="pothole-map" aria-label={t('map.title')} />
      )}
      {offlineStyle && !err ? <div className="map-note">{t('map.styleFallback')}</div> : null}
    </div>
  );
}

export function MapLegend({ colorBy, ratings }: { colorBy: ColorBy; ratings?: boolean }) {
  const { t } = useTranslation();
  return (
    <div className="legend" aria-label={t('map.legend')}>
      {colorBy === 'severity' ? (
        (Object.keys(SEVERITY_COLOR) as Severity[]).map((s) => (
          <span key={s} className="legend-item"><i style={{ background: SEVERITY_COLOR[s] }} />{t(`severity.${s}`)}</span>
        ))
      ) : (
        <span className="legend-item">
          {t('map.priorityLow')}
          <i className="ramp" style={{ background: `linear-gradient(90deg, ${PRIORITY_RAMP.map(([, c]) => c).join(',')})` }} />
          {t('map.priorityHigh')}
        </span>
      )}
      {ratings ? (Object.keys(RATING_COLOR) as (keyof typeof RATING_COLOR)[]).map((r) => (
        <span key={r} className="legend-item"><i className="line" style={{ background: RATING_COLOR[r] }} />{t(`rating.${r}`)}</span>
      )) : null}
    </div>
  );
}
