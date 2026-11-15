import { useQuery } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Link, useParams } from 'react-router';
import { downloadUrl, ErrorBox, errorText, Loading, SeverityBadge } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { circleName, useAdminUnits } from '../lib/circles';
import { fmtDate, fmtDateTime, fmtInr, fmtNum } from '../lib/format';
import { useSession } from '../lib/session';
import type { Severity } from '../lib/types';
import { useState } from 'react';

export default function RunPage() {
  const { id } = useParams();
  const rid = Number(id);
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const token = useSession((s) => s.token);
  const [err, setErr] = useState<string | null>(null);
  const q = useQuery({ queryKey: ['run', rid], queryFn: () => getApi().getRun(rid) });
  const circles = useAdminUnits('circle');
  if (q.isLoading) return <Loading />;
  if (q.error) return <ErrorBox error={q.error} onRetry={() => q.refetch()} />;
  const r = q.data!;
  const tot = r.totals;
  const km = r.route.reduce((s, x) => s + (x.hop_m ?? 0), 0) / 1000;
  return (
    <div className="indent">
      <div className="page-h no-print">
        <h1>{t('run.title', { id: r.id })}</h1>
        <div className="page-actions">
          <button className="btn btn-primary" onClick={() => window.print()}>{t('run.printIndent')}</button>
          <button className="btn" onClick={() => downloadUrl(getApi().indentCsvUrl(r.id), `indent-run-${r.id}.csv`, token).catch((e) => setErr(errorText(e, t)))}>{t('run.downloadCsv')}</button>
        </div>
      </div>
      {err ? <div className="alert alert-bad no-print">{err}</div> : null}
      <section className="sheet">
        <header className="sheet-h">
          <div>
            <div className="sheet-org">{t('run.org')}</div>
            <h2>{t('run.indentTitle')}</h2>
            <div>{t('common.circle')}: <strong>{circleName(r.circle_id, L, r.circle_name, circles.data)}</strong> · {t('run.runDate')}: <strong>{fmtDate(r.run_date, L)}</strong> · {t('run.runNo')}: <strong>{r.id}</strong></div>
          </div>
          <div className="sheet-meta small">{t('run.generatedAt')}: {fmtDateTime(r.generated_at, L)}<br />WO: {r.work_order_ids.map((w) => `#${w}`).join(', ')}</div>
        </header>
        <h3 className="h4">{t('run.materials')}</h3>
        <table className="tbl tbl-tight">
          <tbody>
            <tr><th>{t('run.coldMix')}</th><td className="r num strong">{fmtNum(tot.bags, 0, L)} {t('units.bags')}</td>
              <th>{t('pothole.emulsion')}</th><td className="r num">{fmtNum(tot.emulsion_kg, 1, L)} kg</td></tr>
            <tr><th>{t('pothole.sand')}</th><td className="r num">{fmtNum(tot.sand_kg, 1, L)} kg</td>
              <th>{t('pothole.crm')}</th><td className="r num">{fmtNum(tot.crm_t, 3, L)} t</td></tr>
            <tr><th>{t('run.crewHours')}</th><td className="r num">{fmtNum(tot.crew_h, 1, L)} / {fmtNum(r.crew_hours_cap, 0, L)} h</td>
              <th>{t('run.estCost')}</th><td className="r num strong">{fmtInr(tot.cost_inr, L)}</td></tr>
            <tr><th>{t('run.stops')}</th><td className="r num">{r.route.length}</td><th>{t('run.routeKm')}</th><td className="r num">{fmtNum(km, 2, L)} km</td></tr>
          </tbody>
        </table>
        <h3 className="h4">{t('run.route')}</h3>
        <table className="tbl tbl-tight" data-testid="indent-table">
          <thead><tr><th>{t('run.stop')}</th><th>{t('common.id')}</th><th>{t('common.road')}</th><th>{t('common.severity')}</th><th className="r">{t('run.hop')}</th><th className="r">{t('pothole.cutSize')}</th><th className="r">{t('pothole.designDepth')}</th><th className="r">{t('units.bags')}</th><th className="r">{t('common.cost')}</th><th>{t('run.done')}</th></tr></thead>
          <tbody>
            {r.route.map((s, i) => (
              <tr key={s.pothole_id}>
                <td className="num">{i + 1}</td>
                <td><Link to={`/potholes/${s.pothole_id}`}>#{s.pothole_id}</Link>{s.work_order_id ? <div className="muted small">WO #{String(s.work_order_id)}</div> : null}</td>
                <td>{String(s.road_name ?? '–')}<div className="muted small num">{s.lat.toFixed(5)}, {s.lon.toFixed(5)}</div></td>
                <td>{s.severity ? <SeverityBadge s={s.severity as Severity} /> : '–'}</td>
                <td className="r num">{fmtNum(s.hop_m, 0, L)} m</td>
                <td className="r num">{s.a_cut_m2 != null ? `${fmtNum(Number(s.a_cut_m2), 2, L)} m²` : '–'}</td>
                <td className="r num">{s.d_design_mm != null ? `${String(s.d_design_mm)} mm` : '–'}</td>
                <td className="r num strong">{s.bags != null ? String(s.bags) : '–'}{s.bags_low != null ? <div className="muted small">{String(s.bags_low)}–{String(s.bags_high)} × {String(s.bag_kg ?? 25)} kg</div> : null}</td>
                <td className="r num">{s.est_cost_inr != null ? fmtInr(Number(s.est_cost_inr), L) : '–'}</td>
                <td className="tick">☐</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="small muted">{t('run.note')}</p>
        <div className="signs">
          <div>{t('run.preparedBy')}<span /></div>
          <div>{t('run.storeIssued')}<span /></div>
          <div>{t('run.approvedBy')}<span /></div>
        </div>
      </section>
    </div>
  );
}
