import { useQuery } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate } from 'react-router';
import PotholeMap, { MapLegend } from '../components/PotholeMap';
import { Card, ErrorBox, KpiTile, Loading, PageHeader, SeverityBadge } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { sortUnits, unitName, useAdminUnits } from '../lib/circles';
import { fmtHours, fmtInr, fmtInrCompact, fmtNum } from '../lib/format';
import { priorityWithMonsoon } from '../lib/priority';
import { OPEN_STATUSES } from '../lib/states';
import type { PotholeListItem, Severity, WardStat } from '../lib/types';

/** Used only while the zone list from the API (Api.adminUnits('zone')) is loading or unavailable. */
const FALLBACK_ZONES = [
  { id: 'Khairatabad', key: 'zone.Khairatabad' },
  { id: 'LB Nagar', key: 'zone.LB Nagar' },
];
const WINDOWS = [7, 30, 90];
const HEAVY_RAIN_M = 1.0;

function wardName(w: WardStat, lang: string) {
  return (lang === 'te' ? w.name_te : lang === 'hi' ? w.name_hi : null) ?? w.name_en;
}

export default function DashboardPage() {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const nav = useNavigate();
  // zone = admin-unit code (GHMC-KHT) or 'all': the API's /kpi and /potholes?admin_unit= both accept codes
  const [zone, setZone] = useState('all');
  const zones = useAdminUnits('zone');
  const zoneOpts = useMemo(() => (zones.data?.length
    ? sortUnits(zones.data, L).map((z) => ({ value: z.code || String(z.id), label: unitName(z, L), name: z.name_en }))
    : FALLBACK_ZONES.map((z) => ({ value: z.id, label: t(z.key), name: z.id }))), [zones.data, L, t]);
  const zoneName = zone === 'all' ? null : zoneOpts.find((o) => o.value === zone)?.name ?? zone;
  const [win, setWin] = useState(30);
  const [monsoon, setMonsoon] = useState(false);

  const kpi = useQuery({ queryKey: ['kpi', zone, win], queryFn: () => getApi().kpi(zone, win) });
  const pots = useQuery({
    queryKey: ['potholes', 'open', zone],
    queryFn: () => getApi().listPotholes({ status: [...OPEN_STATUSES, 'candidate'], admin_unit: zone === 'all' ? undefined : zone, limit: 1000 }),
  });
  const wards = useQuery({ queryKey: ['publicWards'], queryFn: () => getApi().publicWards() });
  // breakdowns are only in the detail DTO; fetch them for the top candidates to compute the monsoon what-if
  const top = useMemo(() => (pots.data?.items ?? []).filter((p) => p.status !== 'candidate').slice(0, 15), [pots.data]);
  const details = useQuery({
    queryKey: ['monsoonWhatIf', top.map((p) => p.id).join(',')],
    queryFn: () => Promise.all(top.map((p) => getApi().getPothole(p.id))),
    enabled: monsoon && top.length > 0,
  });

  const whatIf = useMemo(() => {
    if (!details.data) return null;
    const rows = details.data.map((d, i) => ({
      item: top[i],
      now: d.pothole.priority ?? 0,
      rain: priorityWithMonsoon(d.pothole.priority, d.pothole.priority_breakdown, HEAVY_RAIN_M) ?? 0,
    }));
    const nowRank = new Map([...rows].sort((a, b) => b.now - a.now).map((r, i) => [r.item.id, i + 1]));
    return [...rows].sort((a, b) => b.rain - a.rain).slice(0, 10).map((r, i) => ({ ...r, rank: i + 1, was: nowRank.get(r.item.id)! }));
  }, [details.data, top]);

  const league = useMemo(() => {
    const ws = (wards.data ?? []).filter((w) => zoneName == null || w.zone === zoneName);
    return [...ws].sort((a, b) => (b.potholes_per_km ?? 0) - (a.potholes_per_km ?? 0) || b.open_potholes - a.open_potholes);
  }, [wards.data, zoneName]);

  const k = kpi.data;
  const openN = k?.open_potholes ?? 0;
  const perPothole = k && openN ? k.est_cost_inr / openN : null;
  const baselinePer = k && openN ? k.baseline_cost_inr / openN : 7396;
  const saving = k ? k.baseline_cost_inr - k.est_cost_inr : null;
  const sevOrder: Severity[] = ['large', 'medium', 'small', 'unknown_water'];

  return (
    <div>
      <PageHeader title={t('dash.title')} sub={t('dash.subtitle')}
        actions={(
          <div className="filters">
            <label className="inline">{t('common.zone')}
              <select value={zone} onChange={(e) => setZone(e.target.value)} data-testid="zone-select">
                <option value="all">{t('dash.zoneAll')}</option>
                {zoneOpts.map((z) => <option key={z.value} value={z.value}>{z.label}</option>)}
              </select>
            </label>
            <label className="inline">{t('dash.window')}
              <select value={win} onChange={(e) => setWin(Number(e.target.value))}>{WINDOWS.map((w) => <option key={w} value={w}>{t('dash.days', { n: w })}</option>)}</select>
            </label>
          </div>
        )} />

      {kpi.isLoading ? <Loading /> : kpi.error ? <ErrorBox error={kpi.error} onRetry={() => kpi.refetch()} /> : k ? (
        <div className="kpis" data-testid="kpis">
          <KpiTile label={t('dash.kpi.open')} value={fmtNum(k.open_potholes, 0, L)} sub={sevOrder.filter((s) => k.by_severity[s]).map((s) => `${t(`severity.${s}`)} ${k.by_severity[s]}`).join(' · ')} />
          <KpiTile label={t('dash.kpi.newConfirmed')} value={fmtNum(k.confirmed_new, 0, L)} sub={t('dash.inWindow', { n: win })} />
          <KpiTile label={t('dash.kpi.repaired')} value={fmtNum(k.repaired_verified, 0, L)} tone="good" sub={t('dash.inWindow', { n: win })} />
          <KpiTile label={t('dash.kpi.medianRepair')} value={fmtHours(k.median_hours_to_repair, L)} sub={t('dash.slaArterial')} />
          <KpiTile label={t('dash.kpi.slaBreaches')} value={fmtNum(k.sla_breaches, 0, L)} tone={k.sla_breaches ? 'bad' : undefined} />
          <KpiTile label={t('dash.kpi.reopened')} value={fmtNum(k.reopened, 0, L)} tone={k.reopened ? 'warn' : undefined} />
          <KpiTile label={t('dash.kpi.busKm')} value={`${fmtNum(k.bus_km_scanned, 0, L)} km`} />
          <KpiTile label={t('dash.kpi.estCost')} value={fmtInrCompact(k.est_cost_inr, L)} sub={t('dash.forOpen')} />
        </div>
      ) : null}

      <div className="grid-main">
        <Card title={t('dash.mapTitle')} actions={<Link to="/map" className="btn btn-sm">{t('dash.viewMap')}</Link>}>
          {pots.error ? <ErrorBox error={pots.error} /> : (
            <>
              <PotholeMap items={pots.data?.items ?? []} colorBy={monsoon ? 'priority' : 'severity'} height={380} onSelect={(id) => nav(`/potholes/${id}`)} />
              <MapLegend colorBy={monsoon ? 'priority' : 'severity'} />
            </>
          )}
        </Card>

        <Card title={t('dash.costTitle')}>
          {k ? (
            <div className="cost">
              <CostBar label={t('dash.costOurs')} value={k.est_cost_inr} max={Math.max(k.est_cost_inr, k.baseline_cost_inr)} tone="ours" />
              <CostBar label={t('dash.costBaseline')} value={k.baseline_cost_inr} max={Math.max(k.est_cost_inr, k.baseline_cost_inr)} tone="base" />
              <dl className="kv">
                <dt>{t('dash.perPotholeOurs')}</dt><dd className="num">{fmtInr(perPothole, L)}</dd>
                <dt>{t('dash.perPotholeBaseline')}</dt><dd className="num">{fmtInr(baselinePer, L)}</dd>
                <dt>{t('dash.costSaving')}</dt><dd className="num strong">{fmtInrCompact(saving, L)}</dd>
              </dl>
              <p className="muted small">{t('dash.baselineNote')}</p>
            </div>
          ) : <Loading />}
        </Card>
      </div>

      <div className="grid-2">
        <Card title={t('dash.monsoon.title')} actions={(
          <label className="switch">
            <input type="checkbox" checked={monsoon} onChange={(e) => setMonsoon(e.target.checked)} data-testid="monsoon-toggle" />
            <span>{t('dash.monsoon.toggle')}</span>
          </label>
        )}>
          <p className="muted small">{t('dash.monsoon.desc')}</p>
          {!monsoon ? <TopList items={top.slice(0, 10)} /> : details.isLoading ? <Loading /> : whatIf ? (
            <div className="table-wrap">
              <table className="tbl">
                <thead><tr><th>#</th><th>{t('common.road')}</th><th>{t('common.severity')}</th><th className="r">{t('dash.monsoon.current')}</th><th className="r">{t('dash.monsoon.whatIf')}</th><th className="r">{t('dash.monsoon.moved')}</th></tr></thead>
                <tbody>
                  {whatIf.map((r) => (
                    <tr key={r.item.id} onClick={() => nav(`/potholes/${r.item.id}`)} className="click">
                      <td className="num">{r.rank}</td>
                      <td>{r.item.road_name} <span className="muted small">#{r.item.id}</span></td>
                      <td><SeverityBadge s={r.item.severity} /></td>
                      <td className="r num">{fmtNum(r.now, 1, L)}</td>
                      <td className="r num strong">{fmtNum(r.rain, 1, L)}</td>
                      <td className="r num">{r.was === r.rank ? '–' : r.was > r.rank ? `▲ ${r.was - r.rank}` : `▼ ${r.rank - r.was}`}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </Card>

        <Card title={t('dash.league.title')} actions={<Link to="/public/wards" className="btn btn-sm">{t('common.viewAll')}</Link>}>
          {wards.isLoading ? <Loading /> : wards.error ? <ErrorBox error={wards.error} /> : (
            <div className="table-wrap">
              <table className="tbl" data-testid="league">
                <thead><tr><th>#</th><th>{t('common.ward')}</th><th className="r">{t('dash.league.open')}</th><th className="r">{t('dash.league.large')}</th><th className="r">{t('dash.league.perKm')}</th><th className="r">{t('dash.league.sla')}</th><th className="r">{t('dash.league.median')}</th></tr></thead>
                <tbody>
                  {league.map((w, i) => (
                    <tr key={w.admin_unit_id}>
                      <td className="num">{i + 1}</td>
                      <td>{wardName(w, L)}<div className="muted small">{w.zone ? t(`zone.${w.zone}`, { defaultValue: w.zone }) : ''}</div></td>
                      <td className="r num">{fmtNum(w.open_potholes, 0, L)}</td>
                      <td className="r num">{fmtNum(w.large_open, 0, L)}</td>
                      <td className="r num">{fmtNum(w.potholes_per_km, 2, L)}</td>
                      <td className="r num">{w.sla_met_pct == null ? '–' : `${fmtNum(w.sla_met_pct, 0, L)}%`}</td>
                      <td className="r num">{fmtHours(w.median_hours_to_repair, L)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function CostBar({ label, value, max, tone }: { label: string; value: number; max: number; tone: 'ours' | 'base' }) {
  const { i18n } = useTranslation();
  return (
    <div className="costbar">
      <div className="costbar-h"><span>{label}</span><strong className="num">{fmtInrCompact(value, i18n.language)}</strong></div>
      <div className="costbar-track"><div className={`costbar-fill ${tone}`} style={{ width: `${max > 0 ? (100 * value) / max : 0}%` }} /></div>
    </div>
  );
}

function TopList({ items }: { items: PotholeListItem[] }) {
  const { t, i18n } = useTranslation();
  const nav = useNavigate();
  return (
    <div className="table-wrap">
      <table className="tbl">
        <thead><tr><th>#</th><th>{t('common.road')}</th><th>{t('common.severity')}</th><th className="r">{t('common.depth')}</th><th className="r">{t('common.priority')}</th></tr></thead>
        <tbody>
          {items.map((p, i) => (
            <tr key={p.id} className="click" onClick={() => nav(`/potholes/${p.id}`)}>
              <td className="num">{i + 1}</td>
              <td>{p.road_name} <span className="muted small">#{p.id}</span></td>
              <td><SeverityBadge s={p.severity} /></td>
              <td className="r num">{p.depth_p95_mm == null ? '–' : `${fmtNum(p.depth_p95_mm, 0, i18n.language)} mm`}</td>
              <td className="r num strong">{fmtNum(p.priority, 1, i18n.language)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
