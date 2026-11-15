/** App-wide outbox instance (created in main.tsx after the API is bootstrapped). */
import { fieldDb } from './db';
import { OfflineQueue } from './queue';

let q: OfflineQueue | null = null;
export function setQueue(x: OfflineQueue) { q = x; }
export function getQueue(): OfflineQueue {
  if (!q) q = new OfflineQueue(fieldDb());
  return q;
}
