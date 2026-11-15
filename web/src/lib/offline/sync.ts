/**
 * Background sync wiring for the outbox:
 *  - window 'online' event -> flush
 *  - every `intervalMs` while items are pending -> flush (honours backoff)
 *  - service worker 'sync' (Background Sync API, tag gm-outbox) -> SW posts {type:'gm-flush'} -> flush
 */
import type { OfflineQueue, SyncApi } from './queue';

export const SYNC_TAG = 'gm-outbox';

interface SyncRegistration extends ServiceWorkerRegistration { sync?: { register(tag: string): Promise<void> } }

export async function requestBackgroundSync(): Promise<boolean> {
  try {
    if (!('serviceWorker' in navigator)) return false;
    const reg = (await navigator.serviceWorker.getRegistration()) as SyncRegistration | undefined;
    if (!reg?.sync) return false;
    await reg.sync.register(SYNC_TAG);
    return true;
  } catch {
    return false;
  }
}

export function startAutoSync(queue: OfflineQueue, getApi: () => SyncApi, intervalMs = 30_000): () => void {
  const run = (force = false) => {
    if (typeof navigator !== 'undefined' && navigator.onLine === false) return;
    queue.flush(getApi(), { force }).catch(() => { /* reported via item state */ });
  };
  const onOnline = () => run(true);
  const onMsg = (ev: MessageEvent) => { if ((ev.data as { type?: string })?.type === 'gm-flush') run(true); };
  window.addEventListener('online', onOnline);
  navigator.serviceWorker?.addEventListener('message', onMsg);
  const timer = window.setInterval(async () => {
    const c = await queue.counts();
    if (c.pending > 0) run(false);
  }, intervalMs);
  run(false);
  return () => {
    window.removeEventListener('online', onOnline);
    navigator.serviceWorker?.removeEventListener('message', onMsg);
    window.clearInterval(timer);
  };
}
