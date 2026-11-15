import { useQuery } from '@tanstack/react-query';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ErrorBox, Loading, PageHeader } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { fmtHours, fmtInr, fmtNum } from '../lib/format';
import type { WardStat } from '../lib/types';

type SortKey = 'potholes_per_km' | 'open_potholes' | 'sla_met_pct' | 'median_hours_to_repair' | 'repaired_verified_30d';

export default function PublicWardsPage() {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const [sort, setSort] = useState<SortKey>('potholes_per_km');
  const q = useQuery({ queryKey: ['publicWards'], queryFn: () => getApi().publicWards() });
  const rows = useMemo(() => {
    const xs = [...(q.data ?? [])];
    const asc = sort === 'median_hours_to_repair';
    const fill = asc ? Infinity : -Infinity;
    return xs.sort((a, b) => {
      const va = (a[sort] as number | null | undefined) ?? fill;
      const vb = (b[sort] as number | null | undefined) ?? fill;
      return asc ? va - vb : vb - va;
    });
  }, [q.data, sort]);
  const name = (w: WardStat) => (L === 'te' ? w.name_te : L === 'hi' ? w.name_hi : null) ?? w.name_en;
  const H = ({ k, label }: { k: SortKey; label: string }) => (
    <th className="r">
      <button className={`th-btn ${sort === k ? 'on' : ''}`} onClick={() => setSort(k)}>{label}{sort === k ? ' ↓' : ''}</button>
    </th>
  );
  return (
    <div>
      <PageHeader title={t('public.title')} sub={t('public.subtitle')} />
      {q.isLoading ? <Loading /> : q.error ? <ErrorBox error={q.error} onRetry={() => q.refetch()} /> : (
        <div className="card">
          <div className="table-wrap">
            <table className="tbl" data-testid="wards-table">
              <thead>
                <tr>
                  <th>{t('common.ward')}</th>
                  <th>{t('common.zone')}</th>
                  <H k="open_potholes" label={t('public.open')} />
                  <th className="r">{t('public.large')}</th>
                  <H k="potholes_per_km" label={t('public.perKm')} />
                  <H k="repaired_verified_30d" label={t('public.repaired30')} />
                  <H k="median_hours_to_repair" label={t('public.median')} />
                  <H k="sla_met_pct" label={t('public.slaMet')} />
                  <th className="r">{t('public.kmScanned')}</th>
                  <th className="r">{t('public.estCost')}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((w) => (
                  <tr key={w.admin_unit_id}>
                    <td><strong>{name(w)}</strong>{L !== 'en' ? <div className="muted small">{w.name_en}</div> : null}</td>
                    <td>{w.zone ? t(`zone.${w.zone}`, { defaultValue: w.zone }) : '–'}</td>
                    <td className="r num">{fmtNum(w.open_potholes, 0, L)}</td>
                    <td className="r num">{fmtNum(w.large_open, 0, L)}</td>
                    <td className="r num">{fmtNum(w.potholes_per_km, 2, L)}</td>
                    <td className="r num">{fmtNum(w.repaired_verified_30d, 0, L)}</td>
                    <td className="r num">{fmtHours(w.median_hours_to_repair, L)}</td>
                    <td className="r num">{w.sla_met_pct == null ? '–' : `${fmtNum(w.sla_met_pct, 0, L)}%`}</td>
                    <td className="r num">{fmtNum(w.km_scanned, 0, L)}</td>
                    <td className="r num">{fmtInr(w.est_cost_inr, L)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <p className="muted small">{t('public.note')}</p>
    </div>
  );
}
