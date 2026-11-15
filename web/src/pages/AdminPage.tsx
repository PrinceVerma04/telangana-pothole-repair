import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useSearchParams } from 'react-router';
import { Card, ErrorBox, errorText, Loading, PageHeader } from '../components/ui';
import type { GeoJsonFeatureCollection } from '../lib/api';
import { getApi, getApiInfo } from '../lib/apiClient';
import { fmtDateTime, fmtInr, fmtNum } from '../lib/format';
import { COMPONENTS, weightsSumOk, type Component } from '../lib/priority';
import type { PriorityConfig, RateCard } from '../lib/types';

const TABS = ['weights', 'rates', 'sla', 'units', 'jurisdiction', 'audit', 'calibration', 'models'] as const;
type Tab = (typeof TABS)[number];
/**
 * Devices whose certificates are shown. There is no device list endpoint (shared-file request filed), so these are
 * the seeded ids: the mock's (src/lib/mock/seed.ts DEMO_CERTS) and the live seed's (server/app/seed.py DEMO_DEVICES).
 */
const MOCK_DEVICES = ['sim-bus', 'bus-default', 'phone-demo', 'van-oakd-01'];
const LIVE_DEVICES = ['tsrtc-bus-ts09ua1234', 'sim-bus-01', 'demo-phone-01', 'demo-car-01'];
const SLA_DEFAULT = { arterial: 24, other_monsoon: 72, other_offseason: 168, rnb_state_road: 168 };

export default function AdminPage() {
  const { t } = useTranslation();
  const [sp, setSp] = useSearchParams();
  const raw = sp.get('tab');
  const tab: Tab = TABS.includes(raw as Tab) ? (raw as Tab) : 'weights';
  return (
    <div>
      <PageHeader title={t('admin.title')} />
      <div className="tabs" role="tablist">
        {TABS.map((x) => (
          <button key={x} role="tab" aria-selected={tab === x} className={tab === x ? 'on' : ''} onClick={() => setSp({ tab: x })}>
            {t(`admin.tabs.${x}`)}
          </button>
        ))}
      </div>
      {tab === 'weights' ? <WeightsTab /> : null}
      {tab === 'rates' ? <RatesTab /> : null}
      {tab === 'sla' ? <SlaTab /> : null}
      {tab === 'units' ? <UnitsTab /> : null}
      {tab === 'jurisdiction' ? <JurisdictionTab /> : null}
      {tab === 'audit' ? <AuditTab /> : null}
      {tab === 'calibration' ? <CalibrationTab /> : null}
      {tab === 'models' ? <ModelsTab /> : null}
    </div>
  );
}

function usePriorityConfig() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ['priorityConfig'], queryFn: () => getApi().getPriorityConfig() });
  const save = useMutation({
    mutationFn: (c: PriorityConfig) => getApi().putPriorityConfig(c),
    onSuccess: (c) => {
      qc.setQueryData(['priorityConfig'], c);
      void qc.invalidateQueries({ queryKey: ['audit'] });
    },
  });
  return { q, save };
}

function SaveState({ save }: { save: { error: unknown; isSuccess: boolean } }) {
  const { t } = useTranslation();
  if (save.error) return <div className="alert alert-bad">{errorText(save.error, t)}</div>;
  if (save.isSuccess) return <div className="alert alert-good">{t('common.saved')}</div>;
  return null;
}

function WeightsTab() {
  const { t } = useTranslation();
  const { q, save } = usePriorityConfig();
  const [w, setW] = useState<Record<Component, number> | null>(null);
  useEffect(() => {
    if (q.data) setW({ ...q.data.weights });
  }, [q.data]);
  if (q.error) return <ErrorBox error={q.error} />;
  if (q.isLoading || !w || !q.data) return <Loading />;
  const cfg = q.data;
  const sum = COMPONENTS.reduce((s, k) => s + (w[k] ?? 0), 0);
  const ok = weightsSumOk(w);
  return (
    <Card title={t('admin.weights.title', { v: cfg.version })}>
      <p className="muted small">{t('admin.weights.desc')}</p>
      <div className="weights">
        {COMPONENTS.map((k) => (
          <label key={k} className="weight">
            <span><strong>{k}</strong> · {t(`component.${k}`)}</span>
            <input type="range" min={0} max={0.6} step={0.01} value={w[k]} onChange={(e) => setW({ ...w, [k]: Number(e.target.value) })} />
            <input type="number" min={0} max={1} step={0.01} value={w[k]} className="num narrow" onChange={(e) => setW({ ...w, [k]: Number(e.target.value) })} />
            <span className="muted small">{t(`componentDesc.${k}`)}</span>
          </label>
        ))}
      </div>
      <p className={ok ? 'strong' : 'alert alert-warn'}>
        {t('admin.weights.sum', { s: sum.toFixed(2) })}{ok ? '' : ` — ${t('admin.weights.mustSum')}`}
      </p>
      <SaveState save={save} />
      <button className="btn btn-primary" disabled={!ok || save.isPending} onClick={() => save.mutate({ ...cfg, weights: w })}>{t('common.save')}</button>
    </Card>
  );
}

function SlaTab() {
  const { t } = useTranslation();
  const { q, save } = usePriorityConfig();
  const [h, setH] = useState<Record<string, number> | null>(null);
  useEffect(() => {
    if (q.data) setH({ ...(q.data.age_sla?.sla_hours ?? SLA_DEFAULT) });
  }, [q.data]);
  if (q.error) return <ErrorBox error={q.error} />;
  if (q.isLoading || !h || !q.data) return <Loading />;
  const cfg = q.data;
  return (
    <Card title={t('admin.sla.title')}>
      <p className="muted small">{t('admin.sla.desc')}</p>
      <div className="form grid-2 tight">
        {Object.keys(h).map((k) => (
          <label key={k}>
            {t(`admin.sla.${k}`, { defaultValue: k })}
            <input type="number" min={1} value={h[k]} onChange={(e) => setH({ ...h, [k]: Number(e.target.value) })} />
          </label>
        ))}
      </div>
      <p className="muted small">{t('admin.sla.escalation')}</p>
      <SaveState save={save} />
      <button className="btn btn-primary" disabled={save.isPending} onClick={() => save.mutate({ ...cfg, age_sla: { ...(cfg.age_sla ?? {}), sla_hours: h } })}>
        {t('common.save')}
      </button>
    </Card>
  );
}

function RatesTab() {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ['rateCard'], queryFn: () => getApi().getRateCard() });
  const [card, setCard] = useState<RateCard | null>(null);
  useEffect(() => {
    if (q.data) setCard(structuredClone(q.data));
  }, [q.data]);
  const save = useMutation({ mutationFn: (c: RateCard) => getApi().putRateCard(c), onSuccess: (c) => qc.setQueryData(['rateCard'], c) });
  if (q.error) return <ErrorBox error={q.error} />;
  if (q.isLoading || !card) return <Loading />;
  return (
    <Card title={t('admin.rates.title', { agency: card.agency, fy: card.fy, v: card.version })}>
      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr><th>{t('admin.rates.item')}</th><th className="r">{t('admin.rates.rate')}</th><th>{t('admin.rates.unit')}</th><th>{t('admin.rates.provenance')}</th></tr>
          </thead>
          <tbody>
            {Object.entries(card.items).map(([k, v]) => (
              <tr key={k}>
                <td>{t(`rateItem.${k}`, { defaultValue: k })}<div className="muted small">{k}</div></td>
                <td className="r">
                  <input className="num narrow" type="number" min={0} step={k === 'overhead_frac' ? 0.01 : 1} value={v.rate}
                    onChange={(e) => setCard({ ...card, items: { ...card.items, [k]: { ...v, rate: Number(e.target.value) } } })} />
                </td>
                <td>{v.unit}</td>
                <td className="muted small">{v.provenance}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        {t('admin.rates.baseline')}: <strong className="num">{fmtInr(card.baseline.cost_per_pothole_inr, i18n.language)}</strong>{' '}
        <span className="muted small">{card.baseline.provenance}</span>
      </p>
      <SaveState save={save} />
      <button className="btn btn-primary" disabled={save.isPending} onClick={() => save.mutate(card)}>{t('common.save')}</button>
    </Card>
  );
}

function UnitsTab() {
  const { t } = useTranslation();
  const [res, setRes] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const up = useMutation({
    mutationFn: (fc: GeoJsonFeatureCollection) => getApi().uploadAdminUnits(fc),
    onSuccess: (r) => setRes(JSON.stringify(r)),
    onError: (e) => setErr(errorText(e, t)),
  });
  async function onFile(f: File | undefined) {
    setRes(null);
    setErr(null);
    if (!f) return;
    let fc: GeoJsonFeatureCollection;
    try {
      fc = JSON.parse(await f.text()) as GeoJsonFeatureCollection;
    } catch {
      setErr(t('admin.units.badJson'));
      return;
    }
    up.mutate(fc);
  }
  return (
    <Card title={t('admin.units.title')}>
      <p className="muted small">{t('admin.units.desc')}</p>
      <label className="btn">
        {t('admin.units.upload')}
        <input type="file" accept=".geojson,.json,application/geo+json,application/json" hidden
          onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; void onFile(f); }} />
      </label>
      {err ? <div className="alert alert-bad">{err}</div> : null}
      {res ? <div className="alert alert-good small">{t('admin.units.result')}: {res}</div> : null}
    </Card>
  );
}

function JurisdictionTab() {
  const { t, i18n } = useTranslation();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ['jurisdiction'], queryFn: () => getApi().jurisdictionQueue() });
  const [apply, setApply] = useState(true);
  const resolve = useMutation({
    mutationFn: (v: { id: number; agency: string }) => getApi().resolveJurisdiction(v.id, { agency_code: v.agency, apply_to_segment: apply }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['jurisdiction'] }),
  });
  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorBox error={q.error} />;
  const items = q.data ?? [];
  return (
    <Card title={t('admin.jur.title')} actions={(
      <label className="switch"><input type="checkbox" checked={apply} onChange={(e) => setApply(e.target.checked)} /><span>{t('admin.jur.applySegment')}</span></label>
    )}>
      {resolve.error ? <div className="alert alert-bad">{errorText(resolve.error, t)}</div> : null}
      {!items.length ? <div className="empty">{t('common.noData')}</div> : (
        <ul className="plain jur">
          {items.map((j) => (
            <li key={j.id} className={j.resolved ? 'muted' : ''}>
              <div>
                <strong>#{j.id}</strong>{' '}
                {j.pothole_id ? <Link to={`/potholes/${j.pothole_id}`}>{t('common.pothole')} #{j.pothole_id}</Link> : null}{' '}
                <span className="small muted">{fmtDateTime(j.created_at, i18n.language)}</span>
              </div>
              <div className="small">{j.reason}</div>
              {j.resolved ? <span className="chip tone-good">{t('admin.jur.resolved')}</span> : (
                <div className="row-gap">
                  {t('admin.jur.assignTo')}:
                  {j.candidates.map((a) => (
                    <button key={a} className="btn btn-sm" disabled={resolve.isPending} onClick={() => resolve.mutate({ id: j.id, agency: a })}>{a}</button>
                  ))}
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

const AUDIT_ENTITIES = ['pothole', 'work_order', 'priority_config', 'rate_card', 'jurisdiction_queue', 'admin_unit'];

function AuditTab() {
  const { t, i18n } = useTranslation();
  const [entity, setEntity] = useState('');
  const [eid, setEid] = useState('');
  const q = useQuery({ queryKey: ['audit', entity, eid], queryFn: () => getApi().audit(entity || undefined, eid ? Number(eid) : undefined) });
  return (
    <Card title={t('admin.audit.title')}>
      <div className="filters">
        <label className="inline">{t('admin.audit.entity')}
          <select value={entity} onChange={(e) => setEntity(e.target.value)}>
            <option value="">{t('common.all')}</option>
            {AUDIT_ENTITIES.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
        <label className="inline">{t('admin.audit.entityId')}
          <input className="narrow" inputMode="numeric" value={eid} onChange={(e) => setEid(e.target.value.replace(/\D/g, ''))} />
        </label>
      </div>
      {q.isLoading ? <Loading /> : q.error ? <ErrorBox error={q.error} /> : (
        <div className="table-wrap">
          <table className="tbl" data-testid="audit-table">
            <thead>
              <tr><th>{t('admin.audit.at')}</th><th>{t('admin.audit.entity')}</th><th>{t('admin.audit.action')}</th><th>{t('admin.audit.actor')}</th><th>{t('admin.audit.change')}</th><th>{t('common.reason')}</th></tr>
            </thead>
            <tbody>
              {(q.data ?? []).slice(0, 150).map((a) => (
                <tr key={a.id}>
                  <td className="small">{fmtDateTime(a.at, i18n.language)}</td>
                  <td>{a.entity} #{a.entity_id}</td>
                  <td>{a.action}</td>
                  <td>{a.actor ?? '–'}</td>
                  <td className="small mono">{a.before ? JSON.stringify(a.before) : '∅'} → {a.after ? JSON.stringify(a.after) : '∅'}</td>
                  <td className="small">{a.reason ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function CalibrationTab() {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const DEVICES = getApi().mode === 'mock' ? MOCK_DEVICES : LIVE_DEVICES;
  const qs = useQueries({ queries: DEVICES.map((d) => ({ queryKey: ['cert', d], queryFn: () => getApi().deviceCert(d), retry: false })) });
  return (
    <Card title={t('admin.calib.title')}>
      <p className="muted small">{t('admin.calib.desc')}</p>
      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>{t('admin.calib.device')}</th><th>{t('admin.calib.kind')}</th><th>{t('common.status')}</th><th className="r">{t('admin.calib.height')}</th>
              <th className="r">{t('admin.calib.pitch')}</th><th className="r">{t('admin.calib.reproj')}</th><th className="r">{t('admin.calib.truthN')}</th>
              <th className="r">{t('admin.calib.mae')}</th><th>{t('admin.calib.calibratedAt')}</th>
            </tr>
          </thead>
          <tbody>
            {qs.map((q, i) => {
              const c = q.data;
              return (
                <tr key={DEVICES[i]}>
                  <td>{DEVICES[i]}{c ? <div className="muted small">{c.calib_id}</div> : null}</td>
                  {q.isLoading ? <td colSpan={8}><Loading /></td> : !c ? <td colSpan={8} className="muted small">{errorText(q.error, t)}</td> : (
                    <>
                      <td>{c.vehicle_kind}</td>
                      <td>{c.certified ? <span className="badge cert">✓ {t('common.certified')}</span> : <span className="badge uncert">⚠ {t('common.uncertified')}</span>}</td>
                      <td className="r num">{fmtNum(c.h_true_m, 2, L)} m</td>
                      <td className="r num">{fmtNum(c.pitch_deg, 1, L)}°</td>
                      <td className="r num">{c.reproj_rms_px == null ? '–' : `${fmtNum(c.reproj_rms_px, 2, L)} px`}</td>
                      <td className="r num">{c.truth_fit.n_truth}</td>
                      <td className="r num">{c.truth_fit.depth_mae_mm == null ? '–' : `${fmtNum(c.truth_fit.depth_mae_mm, 1, L)} mm`}</td>
                      <td className="small">{fmtDateTime(c.calibrated_at, L)}</td>
                    </>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function ModelsTab() {
  const { t } = useTranslation();
  const q = useQuery({ queryKey: ['health'], queryFn: () => getApi().health() });
  const info = getApiInfo();
  const h = q.data;
  return (
    <Card title={t('admin.models.title')}>
      <p className="muted small">{t('admin.models.desc')}</p>
      {q.isLoading ? <Loading /> : q.error ? <ErrorBox error={q.error} /> : h ? (
        <dl className="kv">
          <dt>{t('admin.models.api')}</dt><dd>{info?.mode} · {h.status} · v{h.version}{h.demo_mode ? ' · DEMO_MODE' : ''}</dd>
          <dt>{t('admin.models.seg')}</dt><dd className="mono">{h.seg_weights ?? '–'}</dd>
          <dt>{t('admin.models.depth')}</dt><dd className="mono">{h.depth_backend ?? '–'}</dd>
          <dt>GPU</dt><dd className="mono">{h.gpu ?? '–'}</dd>
          <dt>DB / storage</dt><dd>{h.db ? '✓' : '✗'} / {h.storage ? '✓' : '✗'}</dd>
        </dl>
      ) : null}
    </Card>
  );
}
