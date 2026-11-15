import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useParams } from 'react-router';
import { DepthCrossSection } from '../components/Charts';
import { FramePanel, PlanView, PriorityBars, Timeline } from '../components/PotholeParts';
import { Card, ErrorBox, errorText, Loading, MethodBadge, PageHeader, Pm, SeverityBadge, StatusChip, WoStatusChip } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { fmtDateTime, fmtInr, fmtNum } from '../lib/format';
import { useSession } from '../lib/session';
import { DEFECT_CAUSES, type DefectCause } from '../lib/types';

export default function PotholePage() {
  const { id } = useParams();
  const pid = Number(id);
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const qc = useQueryClient();
  const role = useSession((s) => s.role);
  const [frameIx, setFrameIx] = useState(0);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [cause, setCause] = useState<DefectCause>('unknown');

  const q = useQuery({ queryKey: ['pothole', pid], queryFn: () => getApi().getPothole(pid), enabled: Number.isFinite(pid) });
  const plan = useMutation({
    mutationFn: () => getApi().planPothole(pid, null),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ['pothole', pid] }); void qc.invalidateQueries({ queryKey: ['potholes'] }); },
  });
  const reject = useMutation({
    mutationFn: () => getApi().rejectPothole(pid, { reason, cause }),
    onSuccess: () => { setRejecting(false); void qc.invalidateQueries({ queryKey: ['pothole', pid] }); void qc.invalidateQueries({ queryKey: ['potholes'] }); },
  });

  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorBox error={q.error} onRetry={() => q.refetch()} />;
  const d = q.data;
  if (!d) return null;
  const p = d.pothole;
  const ms = d.measurements;
  const m = ms[Math.min(frameIx, Math.max(0, ms.length - 1))];
  const staff = role === 'zonal_admin' || role === 'super_admin' || role === 'field_engineer';
  const canReject = staff && !['verified_closed', 'rejected', 'archived'].includes(p.status);
  const modelled = !ms.some((x) => x.depth_map_url);

  return (
    <div className="detail">
      <PageHeader
        title={<>{t('pothole.title', { id: p.id })} <span className="muted">· {d.road?.name ?? '–'}</span></>}
        sub={(
          <span className="row-gap">
            <StatusChip s={p.status} /> <SeverityBadge s={p.severity} /> <MethodBadge m={p.best_method} certified={p.certified_measurement} />
            <span className="muted">{t('common.priority')} <strong className="num">{fmtNum(p.priority, 1, L)}</strong></span>
          </span>
        )}
        actions={staff ? (
          <>
            <button className="btn" onClick={() => plan.mutate()} disabled={plan.isPending}>{d.plan ? t('pothole.replan') : t('pothole.makePlan')}</button>
            {canReject ? <button className="btn btn-danger" onClick={() => setRejecting(true)}>{t('pothole.reject')}</button> : null}
          </>
        ) : null}
      />
      {plan.error ? <ErrorBox error={plan.error} /> : null}
      {rejecting ? (
        <form className="card card-b form" onSubmit={(e) => { e.preventDefault(); reject.mutate(); }}>
          <label>{t('pothole.rejectReason')}<input value={reason} onChange={(e) => setReason(e.target.value)} minLength={3} required placeholder={t('pothole.rejectPh')} /></label>
          <label>{t('pothole.rejectCause')}
            <select value={cause} onChange={(e) => setCause(e.target.value as DefectCause)}>{DEFECT_CAUSES.map((c) => <option key={c} value={c}>{t(`cause.${c}`)}</option>)}</select>
          </label>
          {reject.error ? <div className="alert alert-bad">{errorText(reject.error, t)}</div> : null}
          <div className="row-gap"><button className="btn btn-danger" disabled={reject.isPending}>{t('pothole.rejectConfirm')}</button><button type="button" className="btn" onClick={() => setRejecting(false)}>{t('common.cancel')}</button></div>
        </form>
      ) : null}
      {p.water_filled ? <div className="alert alert-warn">{t('pothole.waterNote')}{p.remeasure_after ? ` ${t('pothole.remeasureAfter', { at: fmtDateTime(p.remeasure_after, L) })}` : ''}</div> : null}
      {!p.certified_measurement ? <div className="alert alert-info small">{t('pothole.uncertifiedNote')}</div> : null}

      <div className="grid-2">
        <Card title={t('pothole.frame')} actions={ms.length > 1 ? (
          <div className="seg">{ms.slice(0, 8).map((x, i) => <button key={x.id} className={i === frameIx ? 'on' : ''} onClick={() => setFrameIx(i)}>{i + 1}</button>)}</div>
        ) : null}>
          <FramePanel m={m} />
        </Card>
        <Card title={t('pothole.crossSection')}>
          <DepthCrossSection p={{ width_m: p.width_m, depth_max_mm: p.depth_max_mm, depth_p95_mm: p.depth_p95_mm, depth_mean_mm: p.depth_mean_mm, depth_sigma_mm: p.depth_sigma_mm }} modelled={modelled} />
        </Card>
      </div>

      <div className="grid-3">
        <Card title={t('pothole.measurement')}>
          <dl className="kv">
            <dt>{t('pothole.depthP95')}</dt><dd><Pm v={p.depth_p95_mm} s={p.depth_sigma_mm} unit="mm" /></dd>
            <dt>{t('pothole.depthMax')}</dt><dd><Pm v={p.depth_max_mm} unit="mm" /></dd>
            <dt>{t('pothole.depthMean')}</dt><dd><Pm v={p.depth_mean_mm} unit="mm" /></dd>
            <dt>{t('pothole.area')}</dt><dd><Pm v={p.area_m2} s={p.area_sigma} unit="m²" d={3} /></dd>
            <dt>{t('pothole.volume')}</dt><dd><Pm v={p.volume_l} s={p.volume_sigma} unit="L" d={1} /></dd>
            <dt>{t('pothole.dims')}</dt><dd className="num">{fmtNum(p.length_m, 2, L)} × {fmtNum(p.width_m, 2, L)} m</dd>
            <dt>{t('pothole.quality')}</dt><dd className="num">{fmtNum(p.quality, 2, L)}</dd>
            <dt>{t('common.method')}</dt><dd><MethodBadge m={p.best_method} certified={p.certified_measurement} /></dd>
            <dt>{t('pothole.passes')}</dt><dd className="num">{t('pothole.passesSeen', { seen: p.passes_seen, total: p.passes_total, trips: p.trips_seen })}</dd>
            <dt>{t('pothole.imu')}</dt><dd className="num">{p.imu_corroborations}</dd>
            <dt>{t('pothole.complaints')}</dt><dd className="num">{p.complaint_count}</dd>
            <dt>{t('common.firstSeen')}</dt><dd>{fmtDateTime(p.first_seen, L)}</dd>
            <dt>{t('common.lastSeen')}</dt><dd>{fmtDateTime(p.last_seen, L)}</dd>
          </dl>
        </Card>
        <Card title={t('pothole.priorityBreakdown')}>
          <PriorityBars breakdown={p.priority_breakdown} priority={p.priority} />
        </Card>
        <Card title={t('pothole.roadInfo')}>
          <dl className="kv">
            <dt>{t('common.road')}</dt><dd>{d.road?.name ?? '–'} <span className="muted small">OSM {d.road?.osm_way_id ?? p.way_id}</span></dd>
            <dt>{t('pothole.class')}</dt><dd>{d.road?.highway ?? '–'}{d.road?.lanes ? ` · ${t('pothole.lanes', { n: d.road.lanes })}` : ''}</dd>
            <dt>{t('pothole.owner')}</dt><dd>{d.road?.owner ?? '–'}</dd>
            <dt>{t('pothole.chainage')}</dt><dd className="num">{fmtNum(p.chainage_m, 0, L)} m · {t('pothole.laneOffset')} {fmtNum(p.lane_offset_m, 2, L)} m</dd>
            <dt>{t('pothole.cause')}</dt><dd>{t(`cause.${p.defect_cause}`)}{p.root_cause_flag ? ` · ${t('pothole.rootCause')}` : ''}</dd>
            <dt>{t('pothole.location')}</dt><dd className="num">{p.lat?.toFixed(5)}, {p.lon?.toFixed(5)}</dd>
          </dl>
          {d.nearby_pois.length ? (
            <>
              <h3 className="h4">{t('pothole.nearby')}</h3>
              <ul className="plain">{d.nearby_pois.map((x, i) => <li key={i}>{t(`poi.${String(x.kind)}`, { defaultValue: String(x.kind) })}: {String(x.name ?? '')} <span className="muted">{String(x.dist_m ?? '')} m</span></li>)}</ul>
            </>
          ) : null}
        </Card>
      </div>

      <div className="grid-2">
        <Card title={t('pothole.plan')}>
          {d.plan ? <PlanView plan={d.plan} /> : <div className="empty">{t('pothole.noPlan')}</div>}
        </Card>
        <div>
          <Card title={t('pothole.timeline')}><Timeline events={d.events} /></Card>
          <Card title={t('pothole.workOrders')}>
            {d.work_orders.length ? (
              <ul className="plain">
                {d.work_orders.map((w) => (
                  <li key={w.id} className="row-gap">
                    <Link to={`/work-orders?id=${w.id}`}>WO #{w.id}</Link> <WoStatusChip s={w.status} />
                    <span className="muted small">{w.contractor_name ?? '–'} · {fmtInr(w.est_cost_inr, L)}{w.run_id ? <> · <Link to={`/runs/${w.run_id}`}>{t('wo.run')} #{w.run_id}</Link></> : null}</span>
                  </li>
                ))}
              </ul>
            ) : <div className="empty">{t('wo.noOrders')}</div>}
          </Card>
        </div>
      </div>

      <Card title={t('pothole.perPass')}>
        <div className="table-wrap">
          <table className="tbl">
            <thead><tr><th>{t('pothole.trip')}</th><th className="r">{t('pothole.frames')}</th><th>{t('common.method')}</th><th className="r">{t('pothole.depthP95')}</th><th className="r">{t('pothole.area')}</th><th className="r">{t('pothole.volume')}</th><th className="r">{t('pothole.quality')}</th></tr></thead>
            <tbody>
              {d.passes.map((x, i) => (
                <tr key={i}>
                  <td>{x.trip_id ?? '–'}</td>
                  <td className="r num">{x.n_frames}/{x.n_frames_total}</td>
                  <td><MethodBadge m={x.method} certified={x.certified} /></td>
                  <td className="r"><Pm v={x.depth_p95_mm} s={x.depth_sigma_mm} unit="mm" /></td>
                  <td className="r"><Pm v={x.area_m2} s={x.area_sigma} unit="m²" d={3} /></td>
                  <td className="r"><Pm v={x.volume_l} s={x.volume_sigma} unit="L" d={1} /></td>
                  <td className="r num">{fmtNum(x.quality, 2, L)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
