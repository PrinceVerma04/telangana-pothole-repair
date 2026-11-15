/// <reference lib="webworker" />
/**
 * Service worker (vite-plugin-pwa injectManifest).
 *  - precaches the app shell (JS/CSS/fonts/icons) -> app opens offline
 *  - basemap style/tiles/glyphs from OpenFreeMap: stale-while-revalidate, 30 days
 *  - GET /v1/* (except auth/exports): network-first with 6 s timeout, cached fallback offline
 *  - Background Sync tag `gm-outbox`: tells open clients to flush the IndexedDB outbox
 */
import { ExpirationPlugin } from 'workbox-expiration';
import { cleanupOutdatedCaches, createHandlerBoundToURL, precacheAndRoute } from 'workbox-precaching';
import { NavigationRoute, registerRoute } from 'workbox-routing';
import { NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';

declare let self: ServiceWorkerGlobalScope;
interface SyncEvent extends ExtendableEvent { readonly tag: string }

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

registerRoute(new NavigationRoute(createHandlerBoundToURL('index.html'), { denylist: [/^\/v1\//, /^\/healthz/, /^\/docs/] }));

registerRoute(
  ({ url }) => url.hostname === 'tiles.openfreemap.org',
  new StaleWhileRevalidate({ cacheName: 'gm-basemap', plugins: [new ExpirationPlugin({ maxEntries: 4000, maxAgeSeconds: 30 * 24 * 3600 })] }),
);

registerRoute(
  ({ url, request }) => request.method === 'GET' && url.origin === self.location.origin && url.pathname.startsWith('/v1/')
    && !url.pathname.startsWith('/v1/auth') && !url.pathname.startsWith('/v1/exports'),
  new NetworkFirst({ cacheName: 'gm-api', networkTimeoutSeconds: 6, plugins: [new ExpirationPlugin({ maxEntries: 400, maxAgeSeconds: 3 * 24 * 3600 })] }),
);

async function notifyClients(): Promise<void> {
  const cs = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
  for (const c of cs) c.postMessage({ type: 'gm-flush' });
}

self.addEventListener('sync', (ev) => {
  const e = ev as SyncEvent;
  if (e.tag === 'gm-outbox') e.waitUntil(notifyClients());
});

self.addEventListener('message', (e) => {
  if ((e.data as { type?: string } | null)?.type === 'SKIP_WAITING') void self.skipWaiting();
});

self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
