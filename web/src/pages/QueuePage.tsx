import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate } from 'react-router';
import { Card, Empty, ErrorBox, errorText, Loading, MethodBadge, PageHeader, SeverityBadge, StatusChip } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { fmtDateTime, fmtInr, fmtNum } from '../lib/format';
import { planRepair } from '../lib/planner';
import type { PotholeStatus } from '../lib/types';

const ELIGIBLE: PotholeStatus[] = ['confirmed', 'planned', 'reopened', 'qc_failed'];

export default function QueuePage() {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const qc = useQueryClient();
  const nav = useNavigate();
  const [sel, setSel] = useState<number[]>([]);
  const [contractor, setContractor] = useState<string>('');
  const [crew, setCrew] = useState('');
  const [notes, setNotes] = useState('');
  const [created, setCreated] = useState<number | null>(null);

  const q = useQuery({ queryKey: ['potholes', 'queue'], queryFn: () => getApi().listPotholes({ status: ELIGIBLE, limit: 500 }) });
  const contractors = useQuery({ queryKey: ['contractors'], queryFn: () => getApi().listContractors() });
  const details = useQueries({ queries: sel.map((id) => ({ queryKey: ['pothole', id], queryFn: () => getApi().getPothole(id) })) });

  const totals = useMemo(() => {
    let bags = 0, low = 0, high = 0, cost = 0, n = 0;
    for (const d of details) {
      if (!d.data) continue;
      const pl = d.data.plan ?? planRepair({ ...d.data.pothole, pothole_id: d.data.pothole.id });
      bags += pl.bags; low += pl.bags_low; high += pl.bags_high; cost += pl.est_cost_inr; n++;
    }
    return { bags, low, high, cost, n };
  }, [details]);

  const create = useMutation({
    mutationFn: () => getApi().createWorkOrder({ pothole_ids: sel, contractor_id: contractor ? Number(contractor) : null, crew: crew || null, notes: notes || null }),
    onSuccess: (wo) => {
      setCreated(wo.id);
      setSel([]);
      void qc.invalidateQueries({ queryKey: ['potholes'] });
      void qc.invalidateQueries({ queryKey: ['workOrders'] });
    },
  });

  const items = q.data?.items ?? [];
  const toggle = (id: number) => setSel((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]));

  return (
    <div>
      <PageHeader title={t('queue.title')} sub={t('queue.subtitle')} />
      {created ? (
        <div className="alert alert-good" role="status">
          {t('queue.created', { id: created })} <Link to={`/work-orders?id=${created}`}>{t('common.open')}</Link>
        </div>
      ) : null}
      <div className="queue-layout">
        <Card title={t('queue.eligible', { n: items.length })}>
          {q.isLoading ? <Loading /> : q.error ? <ErrorBox error={q.error} onRetry={() => q.refetch()} /> : !items.length ? <Empty /> : (
            <div className="table-wrap">
              <table className="tbl" data-testid="queue-table">
                <thead>
                  <tr>
                    <th><input type="checkbox" aria-label={t('queue.selectAll')} checked={sel.length > 0 && sel.length === items.length}
                      onChange={(e) => setSel(e.target.checked ? items.map((p) => p.id) : [])} /></th>
                    <th className="r">{t('common.priority')}</th><th>{t('common.road')}</th><th>{t('common.severity')}</th>
                    <th className="r">{t('common.depth')}</th><th className="r">{t('common.area')}</th><th>{t('common.status')}</th><th>{t('common.ward')}</th><th>{t('common.lastSeen')}</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((p) => (
                    <tr key={p.id} className={sel.includes(p.id) ? 'selected' : ''}>
                      <td><input type="checkbox" checked={sel.includes(p.id)} onChange={() => toggle(p.id)} aria-label={`#${p.id}`} /></td>
                      <td className="r num strong">{fmtNum(p.priority, 1, L)}</td>
                      <td><Link to={`/potholes/${p.id}`}>{p.road_name ?? '–'}</Link> <span className="muted small">#{p.id} · {p.agency_code}</span>
                        <div><MethodBadge m={p.best_method} certified={p.certified_measurement} /></div></td>
                      <td><SeverityBadge s={p.severity} /></td>
                      <td className="r num">{p.water_filled ? t('severity.unknown_water') : p.depth_p95_mm == null ? '–' : `${fmtNum(p.depth_p95_mm, 0, L)} ± ${fmtNum(p.depth_sigma_mm, 0, L)} mm`}</td>
                      <td className="r num">{fmtNum(p.area_m2, 2, L)} m²</td>
                      <td><StatusChip s={p.status} /></td>
                      <td>{p.admin_unit ?? '–'}</td>
                      <td className="small">{fmtDateTime(p.last_seen, L)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
        <aside className="card sticky">
          <div className="card-h"><h2>{t('queue.createWo')}</h2></div>
          <div className="card-b form">
            <p className="strong">{t('queue.selectedN', { n: sel.length })}</p>
            {sel.length ? (
              <dl className="kv">
                <dt>{t('queue.totalBags')}</dt><dd className="num">{totals.bags} <span className="muted">({totals.low}–{totals.high})</span></dd>
                <dt>{t('queue.totalCost')}</dt><dd className="num">{fmtInr(totals.cost, L)}</dd>
                <dt>{t('queue.baseline')}</dt><dd className="num">{fmtInr(totals.n * 7396, L)}</dd>
              </dl>
            ) : <p className="muted small">{t('queue.hint')}</p>}
            <label>{t('common.contractor')}
              <select value={contractor} onChange={(e) => setContractor(e.target.value)}>
                <option value="">{t('queue.contractorAuto')}</option>
                {(contractors.data ?? []).map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            </label>
            <label>{t('queue.crew')}<input value={crew} onChange={(e) => setCrew(e.target.value)} placeholder="Crew A" /></label>
            <label>{t('common.notes')}<textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} /></label>
            {create.error ? <div className="alert alert-bad">{errorText(create.error, t)}</div> : null}
            <button className="btn btn-primary" disabled={!sel.length || create.isPending} onClick={() => create.mutate()} data-testid="create-wo">
              {create.isPending ? t('queue.creating') : t('queue.createWo')}
            </button>
            <button className="btn" onClick={() => nav('/work-orders')}>{t('nav.workOrders')}</button>
          </div>
        </aside>
      </div>
    </div>
  );
}
