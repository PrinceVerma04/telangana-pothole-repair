/**
 * Registers the service worker built by vite-plugin-pwa (injectManifest, `injectRegister: false`, src/sw.ts -> /sw.js).
 * Production builds only: the dev server has no sw.js. A new version waits until every tab is closed
 * (registerType 'prompt'; no forced reload, so a half-filled field form is never lost); the outbox lives in
 * IndexedDB and survives updates anyway.
 */
export const SW_URL = '/sw.js';

export async function registerServiceWorker(enabled: boolean = import.meta.env.PROD): Promise<ServiceWorkerRegistration | null> {
  if (!enabled || typeof navigator === 'undefined' || !('serviceWorker' in navigator)) return null;
  try {
    return await navigator.serviceWorker.register(SW_URL, { scope: '/' });
  } catch (e) {
    console.warn('service worker registration failed', e);
    return null;
  }
}
