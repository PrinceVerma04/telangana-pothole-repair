/** Building blocks of the pothole detail page: frame + mask overlay, priority breakdown, repair plan, timeline. */
import { useTranslation } from 'react-i18next';
import { fmtDateTime, fmtInr, fmtNum } from '../lib/format';
import { parseBreakdown } from '../lib/priority';
import type { MeasurementOut, PotholeEventOut, RepairPlan } from '../lib/types';
import { StatusChip } from './ui';

/** Frame with the segmentation mask. Real frames when the API returns frame_url/mask_url, else an SVG placeholder. */
export function FramePanel({ m }: { m: MeasurementOut | undefined }) {
  const { t } = useTranslation();
  if (m?.frame_url) {
    return (
      <figure className="frame">
        <div className="frame-stack">
          <img src={m.frame_url} alt={t('pothole.frame')} loading="lazy" />
          {m.mask_url ? <img className="mask" src={m.mask_url} alt={t('pothole.maskOverlay')} /> : null}
        </div>
        <figcaption>{t('pothole.frameCaption', { seq: m.seq ?? '–', range: fmtNum(m.range_m, 1) })}</figcaption>
      </figure>
    );
  }
  const poly = (m?.extras?.mask_polygon_norm as [number, number][] | undefined) ?? [];
  const pts = poly.map(([x, y]) => `${(x * 1280).toFixed(0)},${(y * 720).toFixed(0)}`).join(' ');
  return (
    <figure className="frame">
      <svg viewBox="0 0 1280 720" role="img" aria-label={t('pothole.framePlaceholder')}>
        <defs>
          <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#c9d6e3" /><stop offset="1" stopColor="#e6ebef" /></linearGradient>
          <linearGradient id="asphalt" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#6b6e72" /><stop offset="1" stopColor="#4a4d51" /></linearGradient>
        </defs>
        <rect width="1280" height="300" fill="url(#sky)" />
        <rect y="300" width="1280" height="420" fill="#9a9583" />
        <polygon points="560,300 720,300 1280,720 0,720" fill="url(#asphalt)" />
        <polygon points="636,300 644,300 676,720 604,720" fill="#e8e4d6" opacity=".0" />
        {[0, 1, 2, 3].map((i) => {
          const y0 = 320 + i * 100;
          const y1 = y0 + 55;
          const x = (y: number) => 640 + (y - 300) * 0.01;
          return <polygon key={i} points={`${x(y0) - 3},${y0} ${x(y0) + 3},${y0} ${x(y1) + 6},${y1} ${x(y1) - 6},${y1}`} fill="#efe9d2" />;
        })}
        {pts ? <polygon points={pts} fill="rgba(208,59,59,.38)" stroke="#d03b3b" strokeWidth="4" /> : null}
        <text x="24" y="44" fontSize="28" fill="#0b0b0b" opacity=".7">{t('pothole.framePlaceholder')}</text>
      </svg>
      <figcaption>{t('pothole.maskOverlay')}{m?.seq != null ? ` · seq ${m.seq}` : ''}{m?.range_m != null ? ` · ${fmtNum(m.range_m, 1)} m` : ''}</figcaption>
    </figure>
  );
}

export function PriorityBars({ breakdown, priority }: { breakdown: Record<string, unknown>; priority: number | null | undefined }) {
  const { t, i18n } = useTranslation();
  const rows = parseBreakdown(breakdown);
  if (!rows.length) return <div className="empty">{t('common.noData')}</div>;
  return (
    <div className="pbars">
      <div className="pbars-total">
        <span>{t('common.priority')}</span>
        <strong className="num">{fmtNum(priority, 1, i18n.language)}</strong>
        <span className="muted">/ 100</span>
      </div>
      {rows.map((r) => (
        <div className="pbar" key={r.key}>
          <div className="pbar-h">
            <span className="pbar-k">{r.key}</span>
            <span className="pbar-l">{t(`component.${r.key}`)}</span>
            <span className="pbar-v num">{fmtNum(r.points, 1, i18n.language)} {t('pothole.points')}</span>
          </div>
          <div className="pbar-track" title={`${t('pothole.score')} ${r.score.toFixed(2)} × ${t('pothole.weight')} ${r.weight.toFixed(2)}`}>
            <div className="pbar-fill" style={{ width: `${r.score * 100}%` }} />
          </div>
          <div className="pbar-why muted">{t('pothole.score')} {r.score.toFixed(2)} × {t('pothole.weight')} {r.weight.toFixed(2)}{r.why.length ? ` — ${r.why.join('; ')}` : ''}</div>
        </div>
      ))}
    </div>
  );
}

export function PlanView({ plan }: { plan: RepairPlan }) {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const base = plan.baseline_cost_inr ?? null;
  return (
    <div className="plan">
      <div className="plan-hero">
        <div>
          <div className="muted">{t('pothole.bags')}</div>
          <div className="plan-bags num">{plan.bags}</div>
          <div className="num">{t('pothole.bagsRange', { low: plan.bags_low, high: plan.bags_high, kg: plan.bag_kg })}</div>
        </div>
        <div>
          <div className="muted">{t('pothole.estCost')}</div>
          <div className="plan-bags num">{fmtInr(plan.est_cost_inr, L)}</div>
          {base ? <div className="num">{t('pothole.vsBaseline', { base: fmtInr(base, L), pct: Math.round((100 * plan.est_cost_inr) / base) })}</div> : null}
        </div>
      </div>
      {plan.pending_remeasure ? <div className="alert alert-warn">{t('pothole.pendingRemeasure')}</div> : null}
      {plan.weather_blocked ? <div className="alert alert-warn">{t('pothole.weatherBlocked')}</div> : null}
      <dl className="kv">
        <dt>{t('pothole.spec')}</dt><dd>{t(`spec.${plan.spec_code}`)} <span className="muted">({plan.spec_code})</span></dd>
        <dt>{t('pothole.cutSize')}</dt><dd className="num">{fmtNum(plan.l_cut_m, 2, L)} × {fmtNum(plan.w_cut_m, 2, L)} m = {fmtNum(plan.a_cut_m2, 3, L)} m²</dd>
        <dt>{t('pothole.designDepth')}</dt><dd className="num">{plan.d_design_mm} mm{plan.layered ? ` (${t('pothole.layeredDesc', { crm: plan.d_crm_mm, patch: plan.d_patch_mm })})` : ''}</dd>
        <dt>{t('pothole.mix')}</dt><dd className="num">{fmtNum(plan.mix_kg, 1, L)} ± {fmtNum(plan.mix_kg_sigma, 1, L)} kg</dd>
        <dt>{t('pothole.patchVolume')}</dt><dd className="num">{fmtNum(plan.v_patch_l, 1, L)} L{plan.v_cavity_l != null ? ` · ${t('pothole.cavity')} ${fmtNum(plan.v_cavity_l, 1, L)} L` : ''}</dd>
        <dt>{t('pothole.emulsion')}</dt><dd className="num">{fmtNum(plan.emulsion_kg, 2, L)} kg</dd>
        <dt>{t('pothole.sand')}</dt><dd className="num">{fmtNum(plan.sand_kg, 2, L)} kg</dd>
        {plan.crm_t > 0 ? (<><dt>{t('pothole.crm')}</dt><dd className="num">{fmtNum(plan.crm_t, 3, L)} t</dd></>) : null}
        <dt>{t('pothole.labour')}</dt><dd className="num">{fmtNum(plan.labour_h, 1, L)} h</dd>
        <dt>{t('pothole.rateCard')}</dt><dd>{plan.rate_card_version}</dd>
      </dl>
      <details>
        <summary>{t('pothole.costBreakdown')}</summary>
        <div className="table-wrap">
          <table className="tbl">
            <thead><tr><th>{t('pothole.item')}</th><th className="r">{t('pothole.qty')}</th><th className="r">{t('pothole.rate')}</th><th className="r">{t('pothole.amount')}</th><th>{t('pothole.provenance')}</th></tr></thead>
            <tbody>
              {plan.cost_breakdown.map((c, i) => (
                <tr key={i}>
                  <td>{t(`rateItem.${c.item}`, { defaultValue: c.item })}</td>
                  <td className="r num">{fmtNum(c.qty, 3, L)} {c.unit}</td>
                  <td className="r num">{c.unit === 'frac' ? fmtInr(c.rate_inr, L) : fmtInr(c.rate_inr, L)}</td>
                  <td className="r num">{fmtInr(c.amount_inr, L)}</td>
                  <td className="muted small">{c.provenance ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      {plan.notes.length ? <ul className="notes">{plan.notes.map((n, i) => <li key={i}>{n}</li>)}</ul> : null}
      <p className="muted small">{plan.refs.join(' · ')}</p>
    </div>
  );
}

export function Timeline({ events }: { events: PotholeEventOut[] }) {
  const { t, i18n } = useTranslation();
  const ev = [...events].sort((a, b) => Date.parse(b.ts) - Date.parse(a.ts));
  return (
    <ol className="timeline">
      {ev.map((e, i) => (
        <li key={i}>
          <time>{fmtDateTime(e.ts, i18n.language)}</time>
          <div><StatusChip s={e.to_status} />{e.from_status ? <span className="muted small"> ← {t(`status.${e.from_status}`)}</span> : null}</div>
          {e.reason ? <div className="small">{e.reason}</div> : null}
          {e.actor ? <div className="muted small">{e.actor}</div> : null}
        </li>
      ))}
    </ol>
  );
}
