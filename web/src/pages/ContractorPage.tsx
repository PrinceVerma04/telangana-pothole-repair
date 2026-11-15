import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate, useParams } from 'react-router';
import { SurvivalCurve } from '../components/Charts';
import { Card, ErrorBox, KpiTile, Loading, PageHeader, WoStatusChip } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { fmtInr, fmtNum } from '../lib/format';
import { useSession } from '../lib/session';

export default function ContractorPage() {
  const { id } = useParams();
  const cid = Number(id);
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const nav = useNavigate();
  const s = useSession();
  const list = useQuery({ queryKey: ['contractors'], queryFn: () => getApi().listContractors() });
  const sc = useQuery({ queryKey: ['scorecard', cid], queryFn: () => getApi().contractorScorecard(cid) });
  const wos = useQuery({ queryKey: ['workOrders', 'contractor', cid], queryFn: () => getApi().listWorkOrders({ contractor: cid }) });
  const d = sc.data;
  return (
    <div>
      <PageHeader title={d ? d.name : t('contractor.title')} sub={t('contractor.subtitle')} actions={s.role !== 'contractor' && (list.data?.length ?? 0) > 0 ? (
        <label className="inline">{t('contractor.pick')}
          <select value={cid} onChange={(e) => nav(`/contractors/${e.target.value}`)}>{list.data!.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}</select>
        </label>
      ) : null} />
      {sc.isLoading ? <Loading /> : sc.error ? <ErrorBox error={sc.error} onRetry={() => sc.refetch()} /> : d ? (
        <>
          <div className="kpis">
            <KpiTile label={t('contractor.workOrders')} value={fmtNum(d.work_orders, 0, L)} />
            <KpiTile label={t('contractor.slaMet')} value={d.sla_met_pct == null ? '–' : `${fmtNum(d.sla_met_pct, 1, L)}%`} tone={d.sla_met_pct != null && d.sla_met_pct < 80 ? 'bad' : 'good'} />
            <KpiTile label={t('contractor.verified')} value={fmtNum(d.verified_closed, 0, L)} tone="good" />
            <KpiTile label={t('contractor.qcFailed')} value={fmtNum(d.qc_failed, 0, L)} tone={d.qc_failed ? 'bad' : undefined} />
            <KpiTile label={t('contractor.dlp')} value={fmtNum(d.dlp_events, 0, L)} tone={d.dlp_events ? 'warn' : undefined} />
            <KpiTile label={t('contractor.reopenRate')} value={d.reopen_rate == null ? '–' : `${fmtNum(d.reopen_rate * 100, 1, L)}%`} />
            <KpiTile label={t('contractor.medianFix')} value={d.median_days_to_fix == null ? '–' : `${fmtNum(d.median_days_to_fix, 1, L)} d`} />
          </div>
          <div className="grid-2">
            <Card title={t('contractor.survival')}><SurvivalCurve data={d.survival} /></Card>
            <Card title={t('contractor.recentWo')}>
              {wos.data?.length ? (
                <div className="table-wrap">
                  <table className="tbl">
                    <thead><tr><th>WO</th><th>{t('common.status')}</th><th className="r">{t('wo.estBags')}</th><th className="r">{t('wo.bagsUsed')}</th><th className="r">{t('common.cost')}</th><th>{t('wo.sla')}</th></tr></thead>
                    <tbody>
                      {wos.data.map((w) => (
                        <tr key={w.id}>
                          <td><Link to={`/work-orders?id=${w.id}`}>#{w.id}</Link></td>
                          <td><WoStatusChip s={w.status} /></td>
                          <td className="r num">{w.est_bags ?? '–'}</td>
                          <td className="r num">{w.bags_used ?? '–'}{w.bag_variance_flag ? ' ⚠' : ''}</td>
                          <td className="r num">{fmtInr(w.est_cost_inr, L)}</td>
                          <td>{w.sla_breached ? <span className="chip tone-bad">{t('wo.slaBreached')}</span> : <span className="chip tone-good">{t('contractor.onTime')}</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <div className="empty">{t('wo.noOrders')}</div>}
            </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}
