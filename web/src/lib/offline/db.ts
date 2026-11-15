/** IndexedDB (Dexie) schema for field mode: the outbox of pending writes and cached work orders. */
import { Dexie, type Table } from 'dexie';
import type { PotholeListItem, WorkOrderOut, WorkOrderStatus, WorkOrderTransition } from '../types';

export type OutboxStatus = 'pending' | 'failed';

export interface OutboxPhoto {
  kind: 'before' | 'after';
  /** Stored as ArrayBuffer + MIME (portable across IndexedDB implementations that mishandle Blobs). */
  data: ArrayBuffer;
  type: string;
  name: string;
  bytes: number;
  lat?: number | null;
  lon?: number | null;
  at: string;
}

export interface OutboxItem {
  id?: number;
  kind: 'transition' | 'photo';
  work_order_id: number;
  transition?: WorkOrderTransition;
  photo?: OutboxPhoto;
  created_at: string;
  attempts: number;
  status: OutboxStatus;
  last_error?: string | null;
  /** epoch ms; retry backoff */
  next_attempt_at: number;
}

export interface CachedWorkOrder {
  id: number;
  wo: WorkOrderOut;
  potholes: PotholeListItem[];
  cached_at: string;
  /** optimistic status after queued transitions (null = server status) */
  local_status?: WorkOrderStatus | null;
  local_bags_used?: number | null;
}

export class FieldDb extends Dexie {
  outbox!: Table<OutboxItem, number>;
  workOrders!: Table<CachedWorkOrder, number>;

  constructor(name = 'gunthameter-field') {
    super(name);
    this.version(1).stores({
      outbox: '++id, status, work_order_id, created_at',
      workOrders: 'id',
    });
  }
}

let shared: FieldDb | null = null;
export function fieldDb(): FieldDb {
  if (!shared) shared = new FieldDb();
  return shared;
}
