import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';
import PotholeMap, { MapLegend, type Bbox, type ColorBy } from '../components/PotholeMap';
import { ErrorBox, PageHeader, SeverityBadge, StatusChip } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { fmtNum } from '../lib/format';
import { OPEN_STATUSES } from '../lib/states';
import { SEVERITIES, type PotholeStatus, type Severity } from '../lib/types';

type StatusFilter = 'open' | 'candidate' | 'repairing' | 'closed' | 'all';
const STATUS_SETS: Record<StatusFilter, PotholeStatus[] | undefined> = {
  open: OPEN_STATUSES,
  candidate: ['candidate'],
  repairing: ['work_ordered', 'acknowledged', 'in_progress', 'repaired_claimed', 'verification_pending'],
  closed: ['verified_closed', 'recurrence_watch'],
  all: undefined,
};
const AGENCIES = ['GHMC', 'NHAI', 'RNB', 'HMDA', 'CMC', 'MMC', 'SCB'];

const roundBbox = (b: Bbox | null): Bbox | null => (b ? (b.map((x) => Math.round(x * 500) / 500) as Bbox) : null);

export default function MapPage() {
  const { t, i18n } = useTranslation();
  const nav = useNavigate();
  const [status, setStatus] = useState<StatusFilter>('open');
  const [sev, setSev] = useState<Severity[]>([]);
  const [minP, setMinP] = useState(0);
  const [agency, setAgency] = useState('');
  const [colorBy, setColorBy] = useState<ColorBy>('severity');
  const [heat, setHeat] = useState(true);
  const [showRatings, setShowRatings] = useState(false);
  const [bbox, setBbox] = useState<Bbox | null>(null);
  const [sel, setSel] = useState<number | null>(null);

  const q = useQuery({
    queryKey: ['potholes', 'map', status, sev.join(','), minP, agency],
    queryFn: () => getApi().listPotholes({ status: STATUS_SETS[status], severity: sev.length ? sev : undefined, min_priority: minP || undefined, agency: agency || undefined, limit: 2000 }),
    placeholderData: keepPreviousData,
  });
  const rb = roundBbox(bbox);
  const ratings = useQuery({
    queryKey: ['ratings', rb?.join(',')],
    queryFn: () => getApi().segmentRatings(rb!),
    enabled: showRatings && !!rb,
    placeholderData: keepPreviousData,
  });
  const items = useMemo(() => q.data?.items ?? [], [q.data]);
  const visible = useMemo(() => items.filter((p) => !bbox || (p.lon >= bbox[0] && p.lon <= bbox[2] && p.lat >= bbox[1] && p.lat <= bbox[3])), [items, bbox]);

  return (
    <div className="map-page">
      <PageHeader title={t('map.title')} sub={t('map.count', { n: q.data?.total ?? 0, v: visible.length })} />
      <div className="filters card card-b" role="search">
        <label className="inline">{t('common.status')}
          <select value={status} onChange={(e) => setStatus(e.target.value as StatusFilter)}>
            {(Object.keys(STATUS_SETS) as StatusFilter[]).map((s) => <option key={s} value={s}>{t(`map.statusSet.${s}`)}</option>)}
          </select>
        </label>
        <fieldset className="inline checks"><legend>{t('common.severity')}</legend>
          {SEVERITIES.map((s) => (
            <label key={s}><input type="checkbox" checked={sev.includes(s)} onChange={(e) => setSev(e.target.checked ? [...sev, s] : sev.filter((x) => x !== s))} />{t(`severity.${s}`)}</label>
          ))}
        </fieldset>
        <label className="inline">{t('map.minPriority')} <span className="num">{minP}</span>
          <input type="range" min={0} max={90} step={5} value={minP} onChange={(e) => setMinP(Number(e.target.value))} />
        </label>
        <label className="inline">{t('common.agency')}
          <select value={agency} onChange={(e) => setAgency(e.target.value)}>
            <option value="">{t('common.all')}</option>
            {AGENCIES.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
        <div className="seg" role="group" aria-label={t('map.colorBy')}>
          <button className={colorBy === 'severity' ? 'on' : ''} onClick={() => setColorBy('severity')}>{t('map.bySeverity')}</button>
          <button className={colorBy === 'priority' ? 'on' : ''} onClick={() => setColorBy('priority')}>{t('map.byPriority')}</button>
        </div>
        <label className="switch"><input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} /><span>{t('map.heat')}</span></label>
        <label className="switch"><input type="checkbox" checked={showRatings} onChange={(e) => setShowRatings(e.target.checked)} /><span>{t('map.ratings')}</span></label>
      </div>
      {q.error ? <ErrorBox error={q.error} onRetry={() => q.refetch()} /> : null}
      <div className="map-layout">
        <div>
          <PotholeMap items={items} colorBy={colorBy} heat={heat} ratings={showRatings ? ratings.data ?? null : null} selectedId={sel}
            onSelect={(id) => nav(`/potholes/${id}`)} onBbox={setBbox} height="min(70vh, 640px)" />
          <MapLegend colorBy={colorBy} ratings={showRatings} />
          {showRatings ? <p className="muted small">{t('map.ratingsHint')}</p> : null}
        </div>
        <aside className="side-list card">
          <div className="card-h"><h2>{t('map.list')}</h2></div>
          <ul>
            {visible.slice(0, 80).map((p) => (
              <li key={p.id} className={sel === p.id ? 'sel' : ''}>
                <button className="li-btn" onClick={() => setSel(p.id)} onDoubleClick={() => nav(`/potholes/${p.id}`)}>
                  {p.thumb_url ? <img src={p.thumb_url} alt="" width={48} height={36} loading="lazy" /> : null}
                  <span className="li-main">
                    <span className="li-title">{p.road_name ?? '–'} <span className="muted small">#{p.id}</span></span>
                    <span className="li-meta"><SeverityBadge s={p.severity} /> <StatusChip s={p.status} /></span>
                  </span>
                  <span className="li-num num">{fmtNum(p.priority, 0, i18n.language)}</span>
                </button>
                {sel === p.id ? <button className="btn btn-sm" onClick={() => nav(`/potholes/${p.id}`)}>{t('map.openDetail')}</button> : null}
              </li>
            ))}
          </ul>
        </aside>
      </div>
    </div>
  );
}
