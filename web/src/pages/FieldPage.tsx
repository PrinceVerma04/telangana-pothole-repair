/**
 * Field mode (mobile-first, offline-capable). Work orders are cached in IndexedDB; every action is written to
 * the outbox first (optimistic local status) and synced when online (src/lib/offline/queue.ts).
 */
import { useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Link } from 'react-router';
import { PageHeader, SeverityBadge, WoStatusChip, errorText } from '../components/ui';
import { getApi } from '../lib/apiClient';
import { fmtDateTime, fmtHours, hoursUntil } from '../lib/format';
import { useNow, useOnline } from '../lib/hooks';
import { resizePhoto, type ResizedPhoto } from '../lib/image';
import { fieldDb, type CachedWorkOrder, type OutboxItem } from '../lib/offline/db';
import { getQueue } from '../lib/offline/instance';
import type { FlushResult } from '../lib/offline/queue';
import { requestBackgroundSync } from '../lib/offline/sync';
import { useSession } from '../lib/session';
import { allowedWorkOrderActions } from '../lib/states';
import type { PotholeListItem, WorkOrderStatus } from '../lib/types';

const ACTIVE: WorkOrderStatus[] = ['issued', 'acknowledged', 'in_progress', 'repaired_claimed', 'verification_pending'];

function getPosition(): Promise<{ lat: number; lon: number } | null> {
  return new Promise((res) => {
    if (!('geolocation' in navigator)) return res(null);
    navigator.geolocation.getCurrentPosition((p) => res({ lat: p.coords.latitude, lon: p.coords.longitude }), () => res(null), { enableHighAccuracy: true, timeout: 8000, maximumAge: 60_000 });
  });
}

export default function FieldPage() {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const online = useOnline();
  const role = useSession((s) => s.role);
  const queue = getQueue();
  const db = fieldDb();
  const [wos, setWos] = useState<CachedWorkOrder[]>([]);
  const [outbox, setOutbox] = useState<OutboxItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<{ at: string; r?: FlushResult; err?: string } | null>(null);

  const reload = useCallback(async () => {
    const [w, o] = await Promise.all([db.workOrders.toArray(), queue.all()]);
    setWos(w.sort((a, b) => a.id - b.id));
    setOutbox(o);
  }, [db, queue]);

  const refresh = useCallback(async () => {
    if (!navigator.onLine) return;
    try {
      const [list, pots] = await Promise.all([getApi().listWorkOrders({ status: ACTIVE }), getApi().listPotholes({ limit: 2000 })]);
      const byId = new Map<number, PotholeListItem>(pots.items.map((p) => [p.id, p]));
      const pendingWo = new Set((await queue.all()).map((i) => i.work_order_id));
      await db.transaction('rw', db.workOrders, async () => {
        const existing = await db.workOrders.toArray();
        const keep = new Set(list.map((w) => w.id));
        for (const e of existing) if (!keep.has(e.id) && !pendingWo.has(e.id)) await db.workOrders.delete(e.id);
        for (const w of list) {
          const prev = existing.find((e) => e.id === w.id);
          await db.workOrders.put({
            id: w.id, wo: w, potholes: w.pothole_ids.map((id) => byId.get(id)).filter((x): x is PotholeListItem => !!x), cached_at: new Date().toISOString(),
            local_status: pendingWo.has(w.id) ? prev?.local_status ?? null : null, local_bags_used: pendingWo.has(w.id) ? prev?.local_bags_used ?? null : null,
          });
        }
      });
    } catch (e) {
      setLast({ at: new Date().toISOString(), err: errorText(e, t) });
    }
    await reload();
  }, [db, queue, reload, t]);

  const sync = useCallback(async () => {
    setBusy(true);
    try {
      const r = await queue.flush(getApi(), { force: true });
      setLast({ at: new Date().toISOString(), r });
      if (r.sent) await refresh();
    } finally {
      setBusy(false);
      await reload();
    }
  }, [queue, refresh, reload]);

  useEffect(() => {
    void reload().then(() => refresh());
    return queue.onChange(() => { void reload(); });
  }, [queue, reload, refresh]);
  useEffect(() => { if (online) void sync(); }, [online]); // eslint-disable-line react-hooks/exhaustive-deps

  async function afterEnqueue() {
    await reload();
    if (navigator.onLine) await sync();
    else void requestBackgroundSync();
  }

  const pending = outbox.filter((o) => o.status === 'pending').length;
  const failed = outbox.filter((o) => o.status === 'failed').length;

  return (
    <div className="field">
      <PageHeader title={t('field.title')} sub={t('field.subtitle')} />
      <div className={`syncbar ${online ? 'on' : 'off'}`} data-testid="syncbar">
        <span className={`chip ${online ? 'tone-good' : 'tone-bad'}`}>{online ? t('common.online') : t('common.offline')}</span>
        <span data-testid="pending-count">{t('field.pending', { n: pending })}</span>
        {failed ? <span className="chip tone-bad">{t('field.failed', { n: failed })}</span> : null}
        <button className="btn btn-sm" onClick={() => void sync()} disabled={busy || !online}>{busy ? t('field.syncing') : t('field.syncNow')}</button>
        <button className="btn btn-sm" onClick={() => void refresh()} disabled={!online}>{t('field.refresh')}</button>
        {last ? <span className="muted small">{t('field.lastSync', { at: fmtDateTime(last.at, L) })}{last.r ? ` · ${t('field.syncResult', { sent: last.r.sent, retried: last.r.retried, failed: last.r.failed })}` : ''}{last.err ? ` · ${last.err}` : ''}</span> : null}
      </div>

      {!wos.length ? <div className="empty">{t('field.noOrders')}</div> : (
        <div className="field-list">
          {wos.map((c) => <FieldCard key={c.id} c={c} role={role} outbox={outbox.filter((o) => o.work_order_id === c.id)} onQueued={afterEnqueue} />)}
        </div>
      )}

      {outbox.length ? (
        <section className="card">
          <div className="card-h"><h2>{t('field.outbox')}</h2></div>
          <ul className="outbox">
            {outbox.map((o) => (
              <li key={o.id} className={o.status}>
                <span>WO #{o.work_order_id} · {o.kind === 'photo' ? `${t('field.photo')} (${t(`wo.${o.photo?.kind ?? 'before'}`)}, ${Math.round((o.photo?.bytes ?? 0) / 1024)} KB)` : t(`woStatus.${o.transition?.to ?? 'issued'}`)}</span>
                <span className="muted small">{fmtDateTime(o.created_at, L)}{o.attempts ? ` · ${t('field.attempts', { n: o.attempts })}` : ''}{o.last_error ? ` · ${o.last_error}` : ''}</span>
                <span className="row-gap">
                  {o.status === 'failed' ? <button className="btn btn-sm" onClick={() => void queue.retry(o.id!)}>{t('common.retry')}</button> : null}
                  <button className="btn btn-sm btn-danger" onClick={() => { if (window.confirm(t('field.discardConfirm'))) void queue.discard(o.id!); }}>{t('field.discard')}</button>
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function PhotoPicker({ label, value, onChange }: { label: string; value: ResizedPhoto | null; onChange: (p: ResizedPhoto | null) => void }) {
  const { t } = useTranslation();
  const [err, setErr] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!value) { setUrl(null); return; }
    const u = URL.createObjectURL(value.blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [value]);
  return (
    <div className="photo-pick">
      <label className="btn btn-block">
        {value ? t('field.retake') : label}
        <input type="file" accept="image/*" capture="environment" hidden onChange={async (e) => {
          const f = e.target.files?.[0];
          e.target.value = '';
          if (!f) return;
          setErr(null);
          try { onChange(await resizePhoto(f)); } catch (x) { setErr(String(x)); }
        }} />
      </label>
      {value && url ? (
        <figure><img src={url} alt={label} /><figcaption>{t('field.resizedInfo', { w: value.width, h: value.height, kb: Math.round(value.blob.size / 1024) })}</figcaption></figure>
      ) : null}
      {err ? <div className="alert alert-bad small">{err}</div> : null}
    </div>
  );
}

function FieldCard({ c, role, outbox, onQueued }: { c: CachedWorkOrder; role: ReturnType<typeof useSession.getState>['role']; outbox: OutboxItem[]; onQueued: () => Promise<void> }) {
  const { t, i18n } = useTranslation();
  const L = i18n.language;
  const now = useNow();
  const queue = getQueue();
  const st = c.local_status ?? c.wo.status;
  const [before, setBefore] = useState<ResizedPhoto | null>(null);
  const [after, setAfter] = useState<ResizedPhoto | null>(null);
  const [bags, setBags] = useState<number>(c.wo.est_bags ?? 1);
  const [lot, setLot] = useState('');
  const [msg, setMsg] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const actions = allowedWorkOrderActions(st, role);
  const p0 = c.potholes[0];
  const left = hoursUntil(c.wo.sla_due, now);

  async function run(fn: () => Promise<void>) {
    setWorking(true);
    setMsg(null);
    try {
      await fn();
      setMsg(t('field.queued'));
      await onQueued();
    } catch (e) {
      setMsg(errorText(e, t));
    } finally {
      setWorking(false);
    }
  }

  const start = () => run(async () => {
    if (!before) throw new Error(t('field.needBefore'));
    const pos = await getPosition();
    await queue.enqueuePhoto(c.id, { kind: 'before', blob: before.blob, lat: pos?.lat, lon: pos?.lon });
    await queue.enqueueTransition(c.id, { to: 'in_progress', lat: pos?.lat ?? null, lon: pos?.lon ?? null });
    setBefore(null);
  });
  const claim = () => run(async () => {
    if (!after) throw new Error(t('field.needAfter'));
    if (!(bags >= 0)) throw new Error(t('field.needBags'));
    const pos = await getPosition();
    await queue.enqueuePhoto(c.id, { kind: 'after', blob: after.blob, lat: pos?.lat, lon: pos?.lon });
    await queue.enqueueTransition(c.id, { to: 'repaired_claimed', bags_used: bags, bag_lot_no: lot || null, lat: pos?.lat ?? null, lon: pos?.lon ?? null });
    setAfter(null);
  });
  const simple = (to: WorkOrderStatus) => run(async () => { await queue.enqueueTransition(c.id, { to }); });

  return (
    <article className="card field-card" data-testid="field-card">
      <header className="field-h">
        <strong>WO #{c.id}</strong> <WoStatusChip s={st} />
        {c.local_status ? <span className="chip tone-warn">{t('field.notSynced')}</span> : null}
        {left != null && ACTIVE.slice(0, 3).includes(st) ? <span className={`chip ${left < 0 ? 'tone-bad' : 'tone-neutral'}`}>{left < 0 ? t('wo.overdue', { h: fmtHours(-left, L) }) : t('wo.slaLeft', { h: fmtHours(left, L) })}</span> : null}
      </header>
      <ul className="plain field-pots">
        {c.potholes.map((p) => (
          <li key={p.id}>
            {p.thumb_url ? <img src={p.thumb_url} alt="" width={64} height={48} /> : null}
            <span>
              <Link to={`/potholes/${p.id}`}>#{p.id}</Link> {p.road_name} <SeverityBadge s={p.severity} />
              <span className="muted small"> {p.depth_p95_mm != null ? `${Math.round(p.depth_p95_mm)} mm` : ''} · {p.area_m2.toFixed(2)} m²</span>
            </span>
            <a className="btn btn-sm" href={`https://www.openstreetmap.org/?mlat=${p.lat}&mlon=${p.lon}#map=19/${p.lat}/${p.lon}`} target="_blank" rel="noreferrer">{t('field.navigate')}</a>
          </li>
        ))}
        {!c.potholes.length ? <li className="muted small">{c.wo.pothole_ids.map((x) => `#${x}`).join(', ')}</li> : null}
      </ul>
      <div className="small">{t('wo.estBags')}: <strong>{c.wo.est_bags ?? '–'}</strong>{c.local_bags_used != null ? ` · ${t('wo.bagsUsed')}: ${c.local_bags_used}` : c.wo.bags_used != null ? ` · ${t('wo.bagsUsed')}: ${c.wo.bags_used}` : ''}{p0 ? '' : ''}</div>

      {st === 'acknowledged' && actions.includes('in_progress') ? (
        <div className="field-step">
          <PhotoPicker label={t('field.beforePhoto')} value={before} onChange={setBefore} />
          <button className="btn btn-primary btn-block" disabled={!before || working} onClick={start}>{t('woAction.in_progress')}</button>
        </div>
      ) : st === 'in_progress' && actions.includes('repaired_claimed') ? (
        <div className="field-step">
          <PhotoPicker label={t('field.afterPhoto')} value={after} onChange={setAfter} />
          <label>{t('wo.bagsUsed')}
            <div className="stepper">
              <button type="button" className="btn" onClick={() => setBags((b) => Math.max(0, b - 1))} aria-label="-">−</button>
              <input type="number" min={0} value={bags} onChange={(e) => setBags(Number(e.target.value))} />
              <button type="button" className="btn" onClick={() => setBags((b) => b + 1)} aria-label="+">+</button>
            </div>
          </label>
          <label>{t('field.bagLot')}<input value={lot} onChange={(e) => setLot(e.target.value)} placeholder="LOT-2026-11-…" /></label>
          <button className="btn btn-primary btn-block" disabled={!after || working} onClick={claim}>{t('woAction.repaired_claimed')}</button>
        </div>
      ) : (
        <div className="row-gap">
          {actions.filter((a) => a !== 'in_progress' && a !== 'repaired_claimed' && a !== 'cancelled' && a !== 'qc_failed').map((a) => (
            <button key={a} className="btn btn-primary" disabled={working} onClick={() => simple(a)}>{t(`woAction.${a}`)}</button>
          ))}
        </div>
      )}
      {outbox.length ? <div className="muted small">{t('field.queuedN', { n: outbox.length })}</div> : null}
      {msg ? <div className="alert alert-info small" role="status">{msg}</div> : null}
    </article>
  );
}
