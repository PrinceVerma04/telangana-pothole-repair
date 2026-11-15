import type { ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import { ApiError, NetworkError } from '../lib/api';
import { fmtPm } from '../lib/format';
import type { Method, PotholeStatus, Severity, WorkOrderStatus } from '../lib/types';

/** Severity uses the reserved status palette and ALWAYS shows its text label (never colour alone). */
export const SEVERITY_COLOR: Record<Severity, string> = {
  small: '#fab219',
  medium: '#ec835a',
  large: '#d03b3b',
  unknown_water: '#2a78d6',
};
const SEV_ICON: Record<Severity, string> = { small: '▪', medium: '◆', large: '▲', unknown_water: '≈' };

export function SeverityBadge({ s }: { s: Severity | null | undefined }) {
  const { t } = useTranslation();
  if (!s) return <span className="badge muted">–</span>;
  return (
    <span className={`badge sev sev-${s}`}>
      <span aria-hidden className="sev-dot" style={{ background: SEVERITY_COLOR[s] }}>{SEV_ICON[s]}</span>
      {t(`severity.${s}`)}
    </span>
  );
}

const STATUS_TONE: Partial<Record<PotholeStatus | WorkOrderStatus, string>> = {
  candidate: 'neutral', confirmed: 'warn', planned: 'info', work_ordered: 'info', issued: 'info', acknowledged: 'info',
  in_progress: 'info', repaired_claimed: 'info', verification_pending: 'info', verified_closed: 'good', qc_failed: 'bad',
  reopened: 'bad', recurrence_watch: 'neutral', rejected: 'muted', archived: 'muted', cancelled: 'muted',
};

export function StatusChip({ s }: { s: PotholeStatus }) {
  const { t } = useTranslation();
  return <span className={`chip tone-${STATUS_TONE[s] ?? 'neutral'}`}>{t(`status.${s}`)}</span>;
}

export function WoStatusChip({ s }: { s: WorkOrderStatus }) {
  const { t } = useTranslation();
  return <span className={`chip tone-${STATUS_TONE[s] ?? 'neutral'}`}>{t(`woStatus.${s}`)}</span>;
}

export function MethodBadge({ m, certified }: { m: Method | null | undefined; certified: boolean }) {
  const { t } = useTranslation();
  return (
    <span className="method">
      {m ? <span className="badge outline">{t(`method.${m}`)}</span> : null}
      {certified ? (
        <span className="badge cert" title={t('pothole.certifiedNote')}>✓ {t('common.certified')}</span>
      ) : (
        <span className="badge uncert" title={t('pothole.uncertifiedNote')}>⚠ {t('common.uncertified')}</span>
      )}
    </span>
  );
}

export function Pm({ v, s, unit, d = 0 }: { v: number | null | undefined; s?: number | null; unit: string; d?: number }) {
  const { i18n } = useTranslation();
  return <span className="num">{fmtPm(v, s, unit, d, i18n.language)}</span>;
}

export function Card({ title, actions, children, className }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${className ?? ''}`}>
      {(title || actions) && (
        <header className="card-h">
          {title ? <h2>{title}</h2> : <span />}
          {actions ? <div className="card-actions">{actions}</div> : null}
        </header>
      )}
      <div className="card-b">{children}</div>
    </section>
  );
}

export function KpiTile({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'good' | 'bad' | 'warn' }) {
  return (
    <div className={`kpi ${tone ? `kpi-${tone}` : ''}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub ? <div className="kpi-sub">{sub}</div> : null}
    </div>
  );
}

export function Loading({ label }: { label?: string }) {
  const { t } = useTranslation();
  return <div className="loading" role="status"><span className="spinner" aria-hidden />{label ?? t('common.loading')}</div>;
}

export function errorText(e: unknown, t: (k: string) => string): string {
  if (e instanceof NetworkError) return t('errors.network');
  if (e instanceof ApiError) {
    if (e.status === 404) return t('errors.notFound');
    if (e.status === 403) return t('errors.forbidden');
    if (e.status === 401) return t('errors.unauthorized');
    return typeof e.detail === 'string' ? e.detail : e.message;
  }
  return e instanceof Error ? e.message : String(e);
}

export function ErrorBox({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { t } = useTranslation();
  return (
    <div className="alert alert-bad" role="alert">
      <strong>{t('errors.generic')}</strong> {errorText(error, t)}
      {onRetry ? <button className="btn btn-sm" onClick={onRetry}>{t('common.retry')}</button> : null}
    </div>
  );
}

export function Empty({ children }: { children?: ReactNode }) {
  const { t } = useTranslation();
  return <div className="empty">{children ?? t('common.noData')}</div>;
}

export function PageHeader({ title, sub, actions }: { title: ReactNode; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="page-h">
      <div>
        <h1>{title}</h1>
        {sub ? <p className="sub">{sub}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </div>
  );
}

/** Download a protected file (bearer token) or a data: URL. */
export async function downloadUrl(url: string, filename: string, token?: string | null): Promise<void> {
  let href = url;
  let revoke = false;
  if (!url.startsWith('data:')) {
    const res = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    href = URL.createObjectURL(await res.blob());
    revoke = true;
  }
  const a = document.createElement('a');
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (revoke) setTimeout(() => URL.revokeObjectURL(href), 5000);
}
