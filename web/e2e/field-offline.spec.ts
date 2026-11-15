/**
 * PLAN WP9 acceptance: "field mode queues a photo offline and syncs when online (Playwright test with network
 * off/on)". Also: the production build registers its service worker and the app shell opens offline, and the
 * language switch works for te / hi / en.
 *
 * The mock API runs in the page. While the context is offline its outbox writes fail with a NetworkError like a real
 * server would (MockApiOptions.offlineAware), so nothing can sync until the network is back.
 */
import { readFileSync } from 'node:fs';
import { expect, test, type Page } from '@playwright/test';

const PHOTO = readFileSync(new URL('../public/pwa-512.png', import.meta.url)); // 512x512 PNG -> resized JPEG

async function useMock(page: Page, lang?: 'te' | 'hi' | 'en') {
  await page.addInitScript((l) => {
    localStorage.setItem('gm.apiMode', 'mock');
    if (l && !localStorage.getItem('gm.lang')) localStorage.setItem('gm.lang', l);
  }, lang);
}

test('field mode: a photo + transition queued offline sync when the network is back', async ({ page, context }) => {
  await useMock(page, 'en');
  await page.goto('/login?next=%2Ffield');
  await page.getByTestId('login-contractor1').click();
  await page.waitForURL('**/field');

  // an acknowledged work order: its card offers the "before photo" step. An in_progress card also has a photo input
  // (the after photo), so select by the step's label, not just by the presence of a file input.
  const pick = page.getByTestId('field-card')
    .filter({ has: page.locator('.field-step .photo-pick label', { hasText: /^Before photo$/ }) }).first();
  await expect(pick).toBeVisible();
  const woTitle = (await pick.locator('header strong').first().textContent())!.trim(); // "WO #8"
  expect(woTitle).toMatch(/^WO #\d+$/);
  // pin the card by its WO number: locators re-evaluate, and the label becomes "Retake" once a photo is chosen
  const card = page.getByTestId('field-card').filter({
    has: page.locator('header strong', { hasText: new RegExp(`^${woTitle}$`) }),
  });
  const outbox = page.locator('ul.outbox > li');
  await expect(outbox).toHaveCount(0);

  // ---- offline: take the before photo and start work
  await context.setOffline(true);
  await expect(page.getByTestId('syncbar')).toHaveClass(/\boff\b/);
  await card.locator('input[type=file]').setInputFiles({ name: 'before.png', mimeType: 'image/png', buffer: PHOTO });
  await expect(card.locator('figure img')).toBeVisible(); // resized preview rendered
  await expect(card.locator('figcaption')).toContainText('512'); // 512 px fits in 1280: not upscaled
  await card.locator('.field-step > button.btn-primary').click();

  await expect(outbox).toHaveCount(2); // photo first, then the in_progress transition
  await expect(page.locator('ul.outbox > li.pending')).toHaveCount(2);
  await expect(outbox.nth(0)).toContainText(/KB/);
  await expect(page.getByTestId('pending-count')).toContainText('2');
  // exact header match: a plain hasText 'WO #1' would also match the cards of WO #12, #13, ...
  const queuedCard = page.getByTestId('field-card').filter({
    has: page.locator('header strong', { hasText: new RegExp(`^${woTitle}$`) }),
  });
  await expect(queuedCard).toHaveCount(1);
  await expect(queuedCard.locator('.chip.tone-warn')).toBeVisible(); // "not synced"

  // stays queued while offline (a manual flush attempt would fail with a NetworkError)
  await page.waitForTimeout(1500);
  await expect(page.locator('ul.outbox > li.pending')).toHaveCount(2);
  const stored = await page.evaluate(() => new Promise<number>((resolve, reject) => {
    const req = indexedDB.open('gunthameter-field');
    req.onerror = () => reject(req.error);
    req.onsuccess = () => {
      const db = req.result;
      const tx = db.transaction('outbox', 'readonly');
      const c = tx.objectStore('outbox').count();
      c.onsuccess = () => { resolve(c.result); db.close(); };
      c.onerror = () => reject(c.error);
    };
  }));
  expect(stored).toBe(2); // persisted in IndexedDB, not only in memory

  // ---- back online: the 'online' event flushes the outbox
  await context.setOffline(false);
  await expect(page.getByTestId('syncbar')).toHaveClass(/\bon\b/);
  await expect(outbox).toHaveCount(0, { timeout: 20_000 });
  await expect(page.getByTestId('pending-count')).toContainText('0');
  // the server-side status moved on: the same card now offers the "after photo" + bags step
  await expect(queuedCard.locator('.stepper')).toBeVisible({ timeout: 15_000 });
  await expect(queuedCard.locator('.chip.tone-warn')).toHaveCount(0);
});

test('production build: service worker controls the page and the app shell opens offline', async ({ page, context }) => {
  await useMock(page, 'en');
  await page.goto('/public/wards');
  const sw = await page.evaluate(async () => (await navigator.serviceWorker.ready).active?.scriptURL ?? '');
  expect(sw).toMatch(/\/sw\.js$/);
  // first load is not controlled yet; reload so the precache serves it
  await page.reload();
  await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);
  await context.setOffline(true);
  await page.reload();
  await expect(page.getByTestId('wards-table')).toBeVisible();
  await context.setOffline(false);
});

test('language switch: Telugu by default, then Hindi and English (persisted)', async ({ page }) => {
  await useMock(page); // no stored language -> default
  await page.goto('/public/wards');
  const h1 = page.locator('h1').first();
  await expect(page.locator('html')).toHaveAttribute('lang', 'te');
  await expect(h1).toHaveText(/[ఀ-౿]/);
  await page.locator('.lang button[lang=hi]').first().click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'hi');
  await expect(h1).toHaveText(/[ऀ-ॿ]/);
  await page.locator('.lang button[lang=en]').first().click();
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
  await expect(h1).toHaveText('Ward report card');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('lang', 'en');
});
