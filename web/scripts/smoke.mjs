#!/usr/bin/env node
/**
 * `npm run smoke` - offline checks of the production build (run `npm run build` first). No network, no browser.
 *
 *  1. build output: index.html, assets, manifest, service worker; every file index.html references exists
 *  2. routes: the PLAN WP9 routes are declared in src/App.tsx and every lazy page has a built chunk
 *  3. i18n: te / hi / en have the same keys and placeholders, nothing empty, Telugu is the default, every
 *     static t('key') in src resolves in all three, bags are a low-high range (never "exact bags")
 *  4. PWA: manifest installability fields + icon files with matching PNG sizes; the service worker precaches the
 *     app shell (every precache URL exists, every built js/css/html is precached), has a fetch route and the
 *     outbox sync handler; the app bundle registers /sw.js
 *  5. thumbnails: list thumbnails <= 20 KB (PLAN WP9) - the mock thumbnail generator (src/lib/mock/svg.ts, loaded
 *     with Node's TypeScript type stripping) and, only if SMOKE_API=http://host:port is set, the live API's
 *     /v1/potholes thumb_url downloads
 *
 * Exit code 1 if any check fails. SKIP is not a failure (reason printed).
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DIST = join(WEB, 'dist');
const SRC = join(WEB, 'src');
const THUMB_MAX = 20 * 1024;

const results = [];
function check(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then((r) => {
      if (r && typeof r === 'object' && r.skip) results.push({ name, status: 'SKIP', detail: r.skip });
      else results.push({ name, status: 'PASS', detail: typeof r === 'string' ? r : '' });
    })
    .catch((e) => results.push({ name, status: 'FAIL', detail: e instanceof Error ? e.message : String(e) }));
}
function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function walk(dir) {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}
const rel = (p) => relative(DIST, p).split('\\').join('/');
const read = (p) => readFileSync(p, 'utf8');
const kb = (n) => `${(n / 1024).toFixed(1)} KB`;

// ------------------------------------------------------------------ 1. build output
if (!existsSync(join(DIST, 'index.html'))) {
  console.error(`FAIL build output: ${DIST}${'\\'}index.html missing - run "npm run build" first`);
  process.exit(1);
}
const distFiles = walk(DIST);
const distSet = new Set(distFiles.map(rel));
const indexHtml = read(join(DIST, 'index.html'));

await check('build: index.html references resolve', () => {
  const refs = [...indexHtml.matchAll(/\s(?:src|href)="\/([^"#?]+)"/g)].map((m) => m[1]);
  assert(refs.length >= 3, `only ${refs.length} references in index.html`);
  const missing = refs.filter((r) => !distSet.has(r));
  assert(!missing.length, `missing in dist: ${missing.join(', ')}`);
  assert(/<script type="module"[^>]*src="\/assets\/index-[\w-]+\.js"/.test(indexHtml), 'no module entry script');
  assert(/rel="stylesheet"[^>]*href="\/assets\/[\w-]+\.css"/.test(indexHtml), 'no stylesheet');
  return `${refs.length} references`;
});

await check('build: html lang, theme colour, viewport', () => {
  assert(/<html lang="te">/.test(indexHtml), 'html lang is not te (Telugu first)');
  assert(/<meta name="theme-color" content="#[0-9a-f]{6}"/i.test(indexHtml), 'theme-color meta missing');
  assert(/<meta name="viewport"[^>]*width=device-width/.test(indexHtml), 'viewport meta missing');
});

const assets = distFiles.filter((p) => rel(p).startsWith('assets/'));
const js = assets.filter((p) => p.endsWith('.js'));
await check('build: bundle sizes', () => {
  assert(js.length > 5, `only ${js.length} js chunks`);
  const total = distFiles.reduce((s, p) => s + statSync(p).size, 0);
  const biggest = js.map((p) => [rel(p), statSync(p).size]).sort((a, b) => b[1] - a[1])[0];
  // vite.config chunkSizeWarningLimit is 1600 kB; maplibre is the largest dependency
  assert(biggest[1] < 1600 * 1024, `chunk ${biggest[0]} is ${kb(biggest[1])} (> 1600 KB)`);
  return `${distFiles.length} files, ${kb(total)} total, largest ${biggest[0]} ${kb(biggest[1])}`;
});

// ------------------------------------------------------------------ 2. routes
const appTsx = read(join(SRC, 'App.tsx'));
await check('routes: PLAN WP9 routes declared', () => {
  const paths = new Set([...appTsx.matchAll(/path:\s*'([^']+)'/g)].map((m) => m[1]));
  if (/index:\s*true/.test(appTsx)) paths.add('(index)');
  const required = ['(index)', 'login', 'map', 'potholes/:id', 'queue', 'work-orders', 'runs/:id', 'field', 'contractors/:id',
    'citizen', 'admin', 'public/wards', '*'];
  const missing = required.filter((r) => !paths.has(r));
  assert(!missing.length, `missing routes: ${missing.join(', ')}`);
  return `${paths.size} routes`;
});

await check('routes: every lazy page has a built chunk', () => {
  const pages = [...appTsx.matchAll(/import\('\.\/pages\/(\w+)'\)/g)].map((m) => m[1]);
  assert(pages.length >= 10, `only ${pages.length} lazy pages found`);
  const names = js.map((p) => rel(p).slice('assets/'.length));
  const missing = pages.filter((pg) => !names.some((n) => n.startsWith(`${pg}-`)));
  assert(!missing.length, `no chunk for: ${missing.join(', ')}`);
  return `${pages.length} lazy pages`;
});

// ------------------------------------------------------------------ 3. i18n
const LANGS = ['te', 'hi', 'en'];
const loc = Object.fromEntries(LANGS.map((l) => [l, JSON.parse(read(join(SRC, 'i18n', `${l}.json`)))]));
function leaves(tree, prefix = '', out = {}) {
  for (const [k, v] of Object.entries(tree)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (typeof v === 'string') out[key] = v;
    else if (v && typeof v === 'object') leaves(v, key, out);
    else out[key] = v;
  }
  return out;
}
const flat = Object.fromEntries(LANGS.map((l) => [l, leaves(loc[l])]));
const ph = (s) => [...String(s).matchAll(/\{\{\s*(\w+)\s*\}\}/g)].map((m) => m[1]).sort().join(',');

await check('i18n: te/hi/en key parity, non-empty strings, placeholder parity', () => {
  const ref = Object.keys(flat.en).sort();
  for (const l of ['te', 'hi']) {
    const keys = Object.keys(flat[l]).sort();
    const onlyRef = ref.filter((k) => !(k in flat[l]));
    const extra = keys.filter((k) => !(k in flat.en));
    assert(!onlyRef.length && !extra.length, `${l}: missing ${onlyRef.slice(0, 5).join(', ')} extra ${extra.slice(0, 5).join(', ')}`);
    const badPh = ref.filter((k) => ph(flat.en[k]) !== ph(flat[l][k]));
    assert(!badPh.length, `${l}: placeholder mismatch in ${badPh.slice(0, 5).join(', ')}`);
  }
  for (const l of LANGS) {
    const empty = Object.entries(flat[l]).filter(([, v]) => typeof v !== 'string' || !v.trim()).map(([k]) => k);
    assert(!empty.length, `${l}: empty/non-string ${empty.slice(0, 5).join(', ')}`);
  }
  return `${ref.length} keys x 3 languages`;
});

await check('i18n: Telugu default, te/hi actually translated', () => {
  const idx = read(join(SRC, 'i18n', 'index.ts'));
  assert(/:\s*'te';?\s*\n\}/.test(idx) || /return[^\n]*:\s*'te'/.test(idx), 'initialLang() does not default to te');
  assert(/code:\s*'te'[\s\S]*code:\s*'hi'[\s\S]*code:\s*'en'/.test(idx), 'language order is not te, hi, en');
  const out = [];
  for (const [l, re] of [['te', /[\u0C00-\u0C7F]/], ['hi', /[\u0900-\u097F]/]]) {
    const vals = Object.values(flat[l]);
    const r = vals.filter((v) => re.test(v)).length / vals.length;
    assert(r > 0.85, `${l}: only ${(100 * r).toFixed(0)} % in native script`);
    out.push(`${l} ${(100 * r).toFixed(0)} %`);
  }
  return out.join(', ');
});

await check('i18n: static t() keys used in src resolve in te/hi/en', () => {
  const srcFiles = walk(SRC).filter((p) => /\.tsx?$/.test(p) && !/\.test\.tsx?$/.test(p));
  const keys = new Set();
  for (const p of srcFiles) for (const m of read(p).matchAll(/\bt\(\s*(['"])([a-zA-Z][\w.+ -]*)\1/g)) keys.add(m[2]);
  const missing = [];
  for (const l of LANGS) for (const k of keys) if (typeof flat[l][k] !== 'string') missing.push(`${l}:${k}`);
  assert(!missing.length, `unresolved: ${missing.slice(0, 8).join(', ')}`);
  return `${keys.size} keys`;
});

await check('i18n: bags are a low-high range, never exact', () => {
  for (const l of LANGS) {
    const s = flat[l]['pothole.bagsRange'];
    assert(typeof s === 'string' && s.includes('{{low}}') && s.includes('{{high}}'), `${l}: pothole.bagsRange is not a range`);
    const exact = Object.entries(flat[l]).filter(([, v]) => /exact bags/i.test(v)).map(([k]) => k);
    assert(!exact.length, `${l}: "exact bags" in ${exact.join(', ')}`);
  }
});

// ------------------------------------------------------------------ 4. PWA
function pngSize(p) {
  const b = readFileSync(p);
  assert(b.subarray(1, 4).toString('latin1') === 'PNG', `${rel(p)} is not a PNG`);
  return `${b.readUInt32BE(16)}x${b.readUInt32BE(20)}`;
}
await check('pwa: manifest is installable', () => {
  assert(/<link rel="manifest" href="\/manifest\.webmanifest">/.test(indexHtml), 'index.html does not link the manifest');
  const m = JSON.parse(read(join(DIST, 'manifest.webmanifest')));
  for (const f of ['name', 'short_name', 'start_url', 'scope', 'theme_color', 'background_color']) assert(m[f], `manifest.${f} missing`);
  assert(['standalone', 'fullscreen', 'minimal-ui'].includes(m.display), `display ${m.display}`);
  assert(m.lang === 'te', `manifest lang ${m.lang} (Telugu first)`);
  const icons = m.icons ?? [];
  for (const want of ['192x192', '512x512']) assert(icons.some((i) => i.sizes === want), `no ${want} icon`);
  assert(icons.some((i) => /maskable/.test(i.purpose ?? '')), 'no maskable icon');
  for (const i of icons) {
    const p = join(DIST, i.src.replace(/^\//, ''));
    assert(existsSync(p), `icon ${i.src} missing`);
    assert(pngSize(p) === i.sizes, `icon ${i.src} is ${pngSize(p)}, manifest says ${i.sizes}`);
  }
  return `${icons.length} icons, shortcuts: ${(m.shortcuts ?? []).map((s) => s.url).join(' ')}`;
});

const swPath = join(DIST, 'sw.js');
await check('pwa: service worker precaches the app shell', () => {
  assert(existsSync(swPath), 'dist/sw.js missing');
  const sw = read(swPath);
  const urls = [...sw.matchAll(/"url":"([^"]+)"/g)].map((m) => m[1]);
  assert(urls.includes('index.html'), 'index.html not precached');
  const missing = urls.filter((u) => !distSet.has(u));
  assert(!missing.length, `precached but not built: ${missing.join(', ')}`);
  const shell = [...distSet].filter((f) => /\.(js|css|html|webmanifest)$/.test(f) && f !== 'sw.js' && !/^workbox-/.test(f));
  const notCached = shell.filter((f) => !urls.includes(f));
  assert(!notCached.length, `built but not precached: ${notCached.slice(0, 5).join(', ')}`);
  const bytes = urls.reduce((s, u) => s + statSync(join(DIST, u)).size, 0);
  assert(bytes < 25 * 1024 * 1024, `precache is ${kb(bytes)}`);
  return `${urls.length} entries, ${kb(bytes)}`;
});

await check('pwa: service worker routes + outbox sync handler', () => {
  const sw = read(swPath);
  assert(/addEventListener\(\s*[`"']fetch[`"']/.test(sw), 'no fetch handler (needed for offline + installability)');
  assert(sw.includes('gm-outbox'), 'Background Sync tag gm-outbox missing');
  assert(/addEventListener\(\s*[`"']sync[`"']/.test(sw), 'no sync handler');
  assert(sw.includes('gm-api') && sw.includes('gm-basemap'), 'runtime caches gm-api / gm-basemap missing');
});

await check('pwa: app registers /sw.js', () => {
  const hit = js.filter((p) => {
    const s = read(p);
    return s.includes('/sw.js') && s.includes('serviceWorker');
  });
  assert(hit.length > 0, 'no built chunk calls navigator.serviceWorker.register("/sw.js")');
  return rel(hit[0]);
});

// ------------------------------------------------------------------ 5. thumbnails
await check('thumbnails: mock list thumbnails <= 20 KB', async () => {
  let mod;
  try {
    mod = await import(pathToFileURL(join(SRC, 'lib', 'mock', 'svg.ts')).href);
  } catch (e) {
    return { skip: `cannot load svg.ts with this Node (${process.version}; type stripping needs >= 22.18): ${e.message}` };
  }
  let max = 0;
  for (let id = 1; id <= 3000; id += 7) {
    for (const sev of ['small', 'medium', 'large', 'unknown_water', null]) {
      for (const water of [false, true]) max = Math.max(max, Buffer.byteLength(mod.thumbSvg(id, sev, water), 'utf8'));
    }
  }
  assert(max <= THUMB_MAX, `largest mock thumbnail ${kb(max)}`);
  return `largest ${kb(max)}`;
});

await check('thumbnails: live API thumb_url <= 20 KB', async () => {
  const api = process.env.SMOKE_API;
  if (!api) return { skip: 'set SMOKE_API=http://127.0.0.1:8000 to check a running API' };
  const res = await fetch(`${api.replace(/\/+$/, '')}/v1/potholes?limit=50`);
  assert(res.ok, `GET /v1/potholes -> ${res.status}`);
  const page = await res.json();
  const thumbs = page.items.map((p) => p.thumb_url).filter(Boolean);
  if (!thumbs.length) return { skip: `API has ${page.total} potholes and no thumbnails to check` };
  let max = 0;
  for (const u of thumbs) {
    const r = await fetch(new URL(u, api));
    assert(r.ok, `${u} -> ${r.status}`);
    max = Math.max(max, (await r.arrayBuffer()).byteLength);
  }
  assert(max <= THUMB_MAX, `largest live thumbnail ${kb(max)}`);
  return `${thumbs.length} thumbnails, largest ${kb(max)}`;
});

// ------------------------------------------------------------------ report
const w = Math.max(...results.map((r) => r.name.length));
for (const r of results) console.log(`${r.status.padEnd(4)}  ${r.name.padEnd(w)}  ${r.detail}`);
const fails = results.filter((r) => r.status === 'FAIL').length;
console.log(`\nsmoke: ${results.length - fails} ok (${results.filter((r) => r.status === 'SKIP').length} skipped), ${fails} failed`);
process.exit(fails ? 1 : 0);
