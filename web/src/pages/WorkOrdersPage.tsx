import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate, useSearchParams } from 'react-router';
import { Card, downloadUrl, Empty, ErrorBox, errorText, Loading, PageHeader, WoStatusChip } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { sortUnits, unitName, useAdminUnits } from '../lib/circles';
import { fmtDateTime, fmtHours, fmtInr, hoursUntil, todayIst } from '../lib/format';
import { useNow } from '../lib/hooks';
import { useSession } from '../lib/session';
import { allowedWorkOrderActions } from '../lib/states';
import { WORK_ORDER_STATUSES, type WorkOrderOut, type WorkOrderStatus } from '../lib/types';

export default function WorkOrdersPage() {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const qc = useQueryClient();
  const nav = useNavigate();
  const [sp, setSp] = useSearchParams();
  const s = useSession();
  const now = useNow();
  const status = (sp.get('status') ?? '') as WorkOrderStatus | '';
  const hi = Number(sp.get('id') ?? NaN);
  const circles = useAdminUnits('circle');
  const circleList = sortUnits(circles.data ?? [], L);
  const [circlePick, setCircle] = useState<string>('');
  // live: real admin-unit id; API down: a typed circle code (the server accepts id or code)
  const circle = circlePick || (circleList[0] ? String(circleList[0].id) : '');
  const [date, setDate] = useState(todayIst());
  const [pending, setPending] = useState<{ id: number; to: WorkOrderStatus } | null>(null);
  const [bags, setBags] = useState('');
  const [reason, setReason] = useState('');
  const [msg, setMsg] = useState<string | null>(null);

  const q = useQuery({ queryKey: ['workOrders', status], queryFn: () => getApi().listWorkOrders({ status: status || undefined }) });
  const move = useMutation({
    mutationFn: (v: { id: number; to: WorkOrderStatus }) => getApi().transitionWorkOrder(v.id, { to: v.to, bags_used: v.to === 'repaired_claimed' ? Number(bags) : null, reason: reason || null }),
    onSuccess: (wo) => {
      setPending(null); setBags(''); setReason('');
      setMsg(t('wo.transitionDone', { id: wo.id, status: t(`woStatus.${wo.status}`) }));
      void qc.invalidateQueries({ queryKey: ['workOrders'] });
      void qc.invalidateQueries({ queryKey: ['potholes'] });
      void qc.invalidateQueries({ queryKey: ['pothole'] });
    },
  });
  const gen = useMutation({
    mutationFn: () => getApi().generateRun(/^\d+$/.test(circle) ? Number(circle) : circle.trim(), date),
    onSuccess: (run) => { void qc.invalidateQueries({ queryKey: ['workOrders'] }); nav(`/runs/${run.id}`); },
  });
  const staff = s.role === 'zonal_admin' || s.role === 'super_admin' || s.role === 'field_engineer';

  function act(w: WorkOrderOut, to: WorkOrderStatus) {
    setMsg(null);
    if (to === 'repaired_claimed' || to === 'qc_failed' || to === 'cancelled') setPending({ id: w.id, to });
    else move.mutate({ id: w.id, to });
  }

  return (
    <div>
      <PageHeader title={t('wo.title')} actions={(
        <div className="filters">
          <label className="inline">{t('common.status')}
            <select value={status} onChange={(e) => { const n = new URLSearchParams(sp); if (e.target.value) n.set('status', e.target.value); else n.delete('status'); setSp(n); }}>
              <option value="">{t('common.all')}</option>
              {WORK_ORDER_STATUSES.map((x) => <option key={x} value={x}>{t(`woStatus.${x}`)}</option>)}
            </select>
          </label>
          {staff ? (
            <>
              <button className="btn btn-sm" onClick={() => downloadUrl(getApi().exportUrl('measurement-book.csv'), 'measurement-book.csv', s.token).catch((e) => setMsg(errorText(e, t)))}>{t('wo.mbExport')}</button>
              <button className="btn btn-sm" onClick={() => downloadUrl(getApi().exportUrl('potholes.geojson'), 'potholes.geojson', s.token).catch((e) => setMsg(errorText(e, t)))}>{t('wo.geojsonExport')}</button>
            </>
          ) : null}
        </div>
      )} />
      {staff ? (
        <Card title={t('wo.generateRun')}>
          <form className="filters" onSubmit={(e) => { e.preventDefault(); gen.mutate(); }}>
            <label className="inline">{t('common.circle')}
              {circles.error ? (
                <input value={circlePick} onChange={(e) => setCircle(e.target.value)} placeholder={t('wo.circleCode')} required size={10} />
              ) : (
                <select value={circle} onChange={(e) => setCircle(e.target.value)} disabled={circles.isLoading} data-testid="circle-select">
                  {circleList.map((c) => <option key={c.id} value={c.id}>{unitName(c, L)}{c.code ? ` (${c.code})` : ''}</option>)}
                </select>
              )}
            </label>
            <label className="inline">{t('common.date')}<input type="date" value={date} onChange={(e) => setDate(e.target.value)} /></label>
            <button className="btn btn-primary" disabled={gen.isPending || !circle}>{gen.isPending ? t('wo.generating') : t('wo.generateRun')}</button>
            <span className="muted small">{t('wo.generateHint')}</span>
          </form>
          {circles.error ? <div className="muted small">{t('wo.circlesUnavailable')}: {errorText(circles.error, t)}</div> : null}
          {gen.error ? <div className="alert alert-bad">{errorText(gen.error, t)}</div> : null}
        </Card>
      ) : null}
      {msg ? <div className="alert alert-info" role="status">{msg}</div> : null}
      {move.error ? <div className="alert alert-bad">{errorText(move.error, t)}</div> : null}
      {q.isLoading ? <Loading /> : q.error ? <ErrorBox error={q.error} onRetry={() => q.refetch()} /> : !q.data?.length ? <Empty>{t('wo.noOrders')}</Empty> : (
        <div className="wo-list">
          {q.data.map((w) => {
            const left = hoursUntil(w.sla_due, now);
            const actions = allowedWorkOrderActions(w.status, s.role);
            return (
              <article key={w.id} className={`card wo ${w.id === hi ? 'hl' : ''}`} data-testid="wo-card">
                <div className="wo-h">
                  <strong>WO #{w.id}</strong> <WoStatusChip s={w.status} />
                  {w.is_rework ? <span className="chip tone-warn">{t('wo.rework')} ← #{w.parent_work_order_id}</span> : null}
                  {w.sla_breached ? <span className="chip tone-bad">{t('wo.slaBreached')}{w.escalation_level ? ` · ${t(`escalation.${w.escalation_level}`, { defaultValue: w.escalation_level })}` : ''}</span>
                    : left != null && !['verified_closed', 'cancelled', 'qc_failed'].includes(w.status) ? (
                      <span className={`chip ${left < 0 ? 'tone-bad' : left < 6 ? 'tone-warn' : 'tone-neutral'}`}>{left < 0 ? t('wo.overdue', { h: fmtHours(-left, L) }) : t('wo.slaLeft', { h: fmtHours(left, L) })}</span>
                    ) : null}
                </div>
                <div className="wo-grid">
                  <div><span className="muted small">{t('wo.potholes')}</span><div>{w.pothole_ids.map((id) => <Link key={id} to={`/potholes/${id}`} className="pill">#{id}</Link>)}</div></div>
                  <div><span className="muted small">{t('common.contractor')}</span><div>{w.contractor_id ? <Link to={`/contractors/${w.contractor_id}`}>{w.contractor_name ?? `#${w.contractor_id}`}</Link> : '–'}{w.crew ? ` · ${w.crew}` : ''}</div></div>
                  <div><span className="muted small">{t('wo.estBags')}</span><div className="num">{w.est_bags ?? '–'}{w.bags_used != null ? <> · {t('wo.bagsUsed')} <strong>{w.bags_used}</strong></> : null}{w.bag_variance_flag ? <span className="chip tone-warn">{t('wo.bagVariance')}</span> : null}</div></div>
                  <div><span className="muted small">{t('common.cost')}</span><div className="num">{fmtInr(w.est_cost_inr, L)}</div></div>
                  <div><span className="muted small">{t('wo.issuedAt')}</span><div className="small">{fmtDateTime(w.issued_at, L)}</div></div>
                  <div><span className="muted small">{t('wo.sla')}</span><div className="small">{fmtDateTime(w.sla_due, L)}</div></div>
                  {w.run_id ? <div><span className="muted small">{t('wo.run')}</span><div><Link to={`/runs/${w.run_id}`}>#{w.run_id}</Link></div></div> : null}
                  {w.status === 'verified_closed' ? <div><span className="chip tone-good">{t('wo.invoiceEligible')}</span></div> : null}
                </div>
                {w.photos.length ? (
                  <div className="photos">{(w.photos as { kind?: string; url?: string }[]).map((p, i) => p.url ? <figure key={i}><img src={p.url} alt={p.kind} width={96} height={72} loading="lazy" /><figcaption>{t(`wo.${p.kind === 'after' ? 'after' : 'before'}`)}</figcaption></figure> : null)}</div>
                ) : null}
                {actions.length ? (
                  <div className="wo-actions">
                    {actions.map((to) => (
                      <button key={to} className={`btn btn-sm ${to === 'qc_failed' || to === 'cancelled' ? 'btn-danger' : 'btn-primary'}`} disabled={move.isPending} onClick={() => act(w, to)}>{t(`woAction.${to}`)}</button>
                    ))}
                  </div>
                ) : null}
                {pending?.id === w.id ? (
                  <form className="form inline-form" onSubmit={(e) => { e.preventDefault(); move.mutate(pending); }}>
                    {pending.to === 'repaired_claimed' ? <label>{t('wo.bagsUsed')}<input type="number" min={0} required value={bags} onChange={(e) => setBags(e.target.value)} /></label> : null}
                    <label>{t('common.reason')}<input value={reason} onChange={(e) => setReason(e.target.value)} required={pending.to !== 'repaired_claimed'} /></label>
                    <button className="btn btn-primary btn-sm">{t(`woAction.${pending.to}`)}</button>
                    <button type="button" className="btn btn-sm" onClick={() => setPending(null)}>{t('common.cancel')}</button>
                  </form>
                ) : null}
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}
