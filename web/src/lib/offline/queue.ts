/**
 * Offline outbox for field mode.
 *
 * Writes (work-order transitions, before/after photos) are queued in IndexedDB and sent in creation order.
 * Ordering is per work order: a photo queued before the "in_progress" transition is uploaded first, and an
 * item that has not succeeded blocks the later items of the SAME work order (other work orders continue).
 *   - success                    -> item deleted
 *   - network error              -> flush stops (we are offline); item stays pending, attempts++
 *   - retryable HTTP (5xx/408/429) -> stays pending with exponential backoff
 *   - other HTTP errors (4xx)    -> status 'failed' (needs the user: retry or discard); blocks its work order
 */
import { ApiError, NetworkError, isRetryable, type Api } from '../api';
import type { WorkOrderStatus, WorkOrderTransition } from '../types';
import type { CachedWorkOrder, FieldDb, OutboxItem } from './db';

export type SyncApi = Pick<Api, 'transitionWorkOrder' | 'uploadWorkOrderPhoto'>;

export interface FlushResult { sent: number; retried: number; failed: number; skipped: number; offline: boolean }

export interface QueueOptions {
  now?: () => number;
  backoffBaseMs?: number;
  backoffMaxMs?: number;
}

export interface PhotoInput {
  kind: 'before' | 'after';
  blob: Blob;
  name?: string;
  lat?: number | null;
  lon?: number | null;
  at?: string;
}

type Listener = () => void;

export class OfflineQueue {
  readonly db: FieldDb;
  private readonly now: () => number;
  private readonly base: number;
  private readonly max: number;
  private inflight: Promise<FlushResult> | null = null;
  private listeners = new Set<Listener>();

  constructor(db: FieldDb, opts: QueueOptions = {}) {
    this.db = db;
    this.now = opts.now ?? Date.now;
    this.base = opts.backoffBaseMs ?? 5_000;
    this.max = opts.backoffMaxMs ?? 5 * 60_000;
  }

  onChange(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }
  private emit() { for (const l of this.listeners) l(); }

  backoffMs(attempts: number): number {
    return Math.min(this.max, this.base * 2 ** Math.max(0, attempts - 1));
  }

  async enqueueTransition(workOrderId: number, t: WorkOrderTransition): Promise<number> {
    const id = await this.db.outbox.add({
      kind: 'transition', work_order_id: workOrderId, transition: { ...t }, created_at: new Date(this.now()).toISOString(),
      attempts: 0, status: 'pending', last_error: null, next_attempt_at: 0,
    });
    await this.db.workOrders.update(workOrderId, {
      local_status: t.to, ...(t.bags_used != null ? { local_bags_used: t.bags_used } : {}),
    } as Partial<CachedWorkOrder>);
    this.emit();
    return id;
  }

  async enqueuePhoto(workOrderId: number, p: PhotoInput): Promise<number> {
    const data = await p.blob.arrayBuffer();
    const id = await this.db.outbox.add({
      kind: 'photo', work_order_id: workOrderId,
      photo: {
        kind: p.kind, data, type: p.blob.type || 'image/jpeg', name: p.name ?? `${p.kind}-${workOrderId}.jpg`, bytes: data.byteLength,
        lat: p.lat ?? null, lon: p.lon ?? null, at: p.at ?? new Date(this.now()).toISOString(),
      },
      created_at: new Date(this.now()).toISOString(), attempts: 0, status: 'pending', last_error: null, next_attempt_at: 0,
    });
    this.emit();
    return id;
  }

  all(): Promise<OutboxItem[]> { return this.db.outbox.orderBy('id').toArray(); }

  async counts(): Promise<{ pending: number; failed: number }> {
    const [pending, failed] = await Promise.all([
      this.db.outbox.where('status').equals('pending').count(),
      this.db.outbox.where('status').equals('failed').count(),
    ]);
    return { pending, failed };
  }

  async retry(id: number): Promise<void> {
    await this.db.outbox.update(id, { status: 'pending', next_attempt_at: 0, last_error: null });
    this.emit();
  }

  async discard(id: number): Promise<void> {
    const it = await this.db.outbox.get(id);
    await this.db.outbox.delete(id);
    if (it?.kind === 'transition') await this.recomputeLocalStatus(it.work_order_id);
    this.emit();
  }

  /** Latest still-queued transition target for a work order, or null. */
  async recomputeLocalStatus(workOrderId: number): Promise<WorkOrderStatus | null> {
    const items = await this.db.outbox.where('work_order_id').equals(workOrderId).sortBy('id');
    const last = [...items].reverse().find((i) => i.kind === 'transition');
    const st = last?.transition?.to ?? null;
    await this.db.workOrders.update(workOrderId, { local_status: st } as Partial<CachedWorkOrder>);
    return st;
  }

  /** Single-flight: concurrent callers share one pass. `force` ignores backoff timers. */
  flush(api: SyncApi, opts: { force?: boolean } = {}): Promise<FlushResult> {
    if (!this.inflight) {
      this.inflight = this.doFlush(api, !!opts.force).finally(() => {
        this.inflight = null;
        this.emit();
      });
    }
    return this.inflight;
  }

  private async send(api: SyncApi, it: OutboxItem): Promise<void> {
    if (it.kind === 'transition' && it.transition) {
      await api.transitionWorkOrder(it.work_order_id, it.transition);
    } else if (it.kind === 'photo' && it.photo) {
      const ph = it.photo;
      await api.uploadWorkOrderPhoto(it.work_order_id, {
        kind: ph.kind, photo: new Blob([ph.data], { type: ph.type }), name: ph.name, lat: ph.lat, lon: ph.lon, at: ph.at,
      });
    } else {
      throw new ApiError(400, 'malformed outbox item');
    }
  }

  private async doFlush(api: SyncApi, force: boolean): Promise<FlushResult> {
    const res: FlushResult = { sent: 0, retried: 0, failed: 0, skipped: 0, offline: false };
    const items = await this.db.outbox.orderBy('id').toArray();
    const blocked = new Set(items.filter((i) => i.status === 'failed').map((i) => i.work_order_id));
    const now = this.now();
    for (const it of items) {
      if (it.status !== 'pending') continue;
      if (blocked.has(it.work_order_id) || (!force && it.next_attempt_at > now)) {
        blocked.add(it.work_order_id);
        res.skipped++;
        continue;
      }
      try {
        await this.send(api, it);
        await this.db.outbox.delete(it.id!);
        res.sent++;
        if (it.kind === 'transition') await this.recomputeLocalStatus(it.work_order_id);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        const attempts = it.attempts + 1;
        blocked.add(it.work_order_id);
        if (isRetryable(e)) {
          await this.db.outbox.update(it.id!, { attempts, last_error: msg, next_attempt_at: now + this.backoffMs(attempts) });
          res.retried++;
          if (e instanceof NetworkError) {
            res.offline = true;
            break;
          }
        } else {
          await this.db.outbox.update(it.id!, { attempts, last_error: msg, status: 'failed' });
          res.failed++;
        }
      }
    }
    return res;
  }
}
