/** Depth cross-section (bowl profile with +/- sigma band) and contractor patch-survival curve. */
import { Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useTranslation } from 'react-i18next';
import { depthProfile, type ProfileInput } from '../lib/profile';

const INK2 = 'var(--ink-2)';
const GRID = 'var(--grid)';

export function DepthCrossSection({ p, modelled }: { p: ProfileInput; modelled: boolean }) {
  const { t } = useTranslation();
  const pts = depthProfile(p);
  if (!pts) return <div className="empty">{t('pothole.noDepth')}</div>;
  const maxY = Math.ceil(Math.max(...pts.map((q) => q.hi_mm)) / 10) * 10 + 10;
  return (
    <figure className="chart">
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <ComposedChart data={pts} margin={{ top: 8, right: 12, bottom: 18, left: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="x_m" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(v: number) => v.toFixed(2)} stroke={INK2} fontSize={11}
              label={{ value: t('pothole.widthAxis'), position: 'insideBottom', offset: -10, fontSize: 11, fill: 'var(--ink-2)' }} />
            <YAxis reversed domain={[0, maxY]} stroke={INK2} fontSize={11} width={44}
              label={{ value: t('pothole.depthAxis'), angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--ink-2)' }} />
            <Tooltip formatter={(v: unknown, name: unknown) => [Array.isArray(v) ? `${v[0]}–${v[1]} mm` : `${v} mm`, name === 'band' ? t('pothole.sigmaBand') : t('pothole.depth')]}
              labelFormatter={(l: unknown) => `x = ${Number(l).toFixed(2)} m`} />
            <ReferenceLine y={0} stroke="var(--ink)" strokeWidth={1.5} label={{ value: t('pothole.roadSurface'), position: 'insideTopRight', fontSize: 10, fill: 'var(--ink-2)' }} />
            <Area dataKey="band" stroke="none" fill="#86b6ef" fillOpacity={0.45} isAnimationActive={false} name="band" />
            <Line dataKey="depth_mm" stroke="#1c5cab" strokeWidth={2} dot={false} isAnimationActive={false} name="depth" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <figcaption>{modelled ? t('pothole.crossSectionModelled') : t('pothole.crossSectionFromMap')}</figcaption>
    </figure>
  );
}

export function SurvivalCurve({ data }: { data: { days: number; survival: number }[] }) {
  const { t } = useTranslation();
  if (!data.length) return <div className="empty">{t('common.noData')}</div>;
  return (
    <figure className="chart">
      <div style={{ width: '100%', height: 240 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="days" type="number" domain={[0, 'dataMax']} stroke={INK2} fontSize={11}
              label={{ value: t('contractor.days'), position: 'insideBottom', offset: -10, fontSize: 11, fill: 'var(--ink-2)' }} />
            <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} stroke={INK2} fontSize={11} width={44} />
            <Tooltip formatter={(v: unknown) => [`${Math.round(Number(v) * 1000) / 10}%`, t('contractor.survival')]} labelFormatter={(l: unknown) => `${l} ${t('contractor.days')}`} />
            <Line type="stepAfter" dataKey="survival" stroke="#2a78d6" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <figcaption>{t('contractor.survivalDesc')}</figcaption>
    </figure>
  );
}
