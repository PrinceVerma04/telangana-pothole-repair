/**
 * Browser tests (PLAN WP9 acceptance: field mode queues a photo offline and syncs when back online).
 *
 *   npm run build && npm run test:e2e
 *
 * Runs against `vite preview` of the production build (service worker included) with the in-browser mock API
 * (localStorage gm.apiMode=mock), so no backend is needed. Browser: the locally installed Google Chrome by default
 * (no download). PW_CHANNEL=msedge uses Edge; PW_CHANNEL=bundled uses Playwright's own Chromium after
 * `npx playwright install chromium` (~150 MB, e.g. on Linux CI).
 */
import { defineConfig } from '@playwright/test';

const PORT = Number(process.env.PW_PORT ?? 4183);
const channel = process.env.PW_CHANNEL === 'bundled' ? undefined : (process.env.PW_CHANNEL ?? 'chrome');

export default defineConfig({
  testDir: 'e2e',
  outputDir: 'node_modules/.cache/playwright-results',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    channel,
    headless: true,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
