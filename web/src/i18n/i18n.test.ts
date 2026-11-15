/**
 * i18n coverage: every key the UI asks for must exist in te, hi and en (no silent English fallback),
 * and interpolation placeholders must match across languages. Guards the regression where the locale
 * files were missing and `npm run build` failed.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import en from './en.json';
import hi from './hi.json';
import te from './te.json';
import { DEFECT_CAUSES, METHODS, POTHOLE_STATUSES, ROLES, SEVERITIES, SPEC_CODES, WORK_ORDER_STATUSES } from '../lib/types';
import { COMPONENTS } from '../lib/priority';
import { DEMO_USERS } from '../lib/demoUsers';

type Tree = { [k: string]: string | Tree };
const LOCALES: Record<string, Tree> = { te: te as Tree, hi: hi as Tree, en: en as Tree };
const SRC = join(__dirname, '..');

function files(dir: string): string[] {
  return readdirSync(dir).flatMap((f) => {
    const p = join(dir, f);
    if (statSync(p).isDirectory()) return files(p);
    return /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f) ? [p] : [];
  });
}

function lookup(tree: Tree, key: string): unknown {
  let o: unknown = tree;
  for (const part of key.split('.')) {
    if (o == null || typeof o !== 'object') return undefined;
    o = (o as Tree)[part];
  }
  return o;
}

function leaves(tree: Tree, prefix = ''): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(tree)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (typeof v === 'string') out[key] = v;
    else Object.assign(out, leaves(v, key));
  }
  return out;
}

const placeholders = (s: string) => [...s.matchAll(/\{\{\s*(\w+)\s*\}\}/g)].map((m) => m[1]).sort();

const SOURCES = files(SRC).map((p) => readFileSync(p, 'utf8'));
const STATIC_KEYS = new Set<string>();
for (const src of SOURCES) {
  for (const m of src.matchAll(/\bt\(\s*(['"])([a-zA-Z][\w.+ -]*)\1/g)) STATIC_KEYS.add(m[2]);
  for (const m of src.matchAll(/\b(?:key|descKey|labelKey|titleKey)\s*:\s*'([a-z][a-zA-Z]*\.[\w. +-]+)'/g)) STATIC_KEYS.add(m[1]);
}

const DYNAMIC: Record<string, readonly string[]> = {
  role: ROLES,
  severity: SEVERITIES,
  status: POTHOLE_STATUSES,
  woStatus: WORK_ORDER_STATUSES,
  woAction: WORK_ORDER_STATUSES.filter((s) => s !== 'planned'),
  method: METHODS,
  cause: DEFECT_CAUSES,
  spec: SPEC_CODES,
  component: COMPONENTS,
  componentDesc: COMPONENTS,
  rating: ['Good', 'Fair', 'Poor'],
  'admin.tabs': ['weights', 'rates', 'sla', 'units', 'jurisdiction', 'audit', 'calibration', 'models'],
  'admin.sla': ['arterial', 'other_monsoon', 'other_offseason', 'rnb_state_road'],
  'map.statusSet': ['open', 'candidate', 'repairing', 'closed', 'all'],
  'citizen.status': ['received', ...POTHOLE_STATUSES],
  wo: ['before', 'after'],
  rateItem: ['coldmix_bag_25kg', 'coldmix_bag_50kg', 'tack_emulsion_kg', 'crm_tonne', 'sand_kg', 'labour_hour', 'jet_patcher_m2', 'overhead_frac'],
  poi: ['school', 'hospital', 'bus_stop', 'metro_station', 'marketplace'],
  escalation: ['AE', 'DEE', 'EE', 'ZONAL_COMMISSIONER'],
  zone: ['Khairatabad', 'LB Nagar'],
};

const ALL_KEYS = [
  ...STATIC_KEYS,
  ...DEMO_USERS.map((u) => u.descKey),
  ...Object.entries(DYNAMIC).flatMap(([p, vs]) => vs.map((v) => `${p}.${v}`)),
];

describe('i18n', () => {
  it('finds the keys used by the UI', () => {
    expect(STATIC_KEYS.size).toBeGreaterThan(400);
  });

  for (const [lang, tree] of Object.entries(LOCALES)) {
    it(`${lang}: every used key resolves to a non-empty string`, () => {
      const missing = [...new Set(ALL_KEYS)].filter((k) => {
        const v = lookup(tree, k);
        return typeof v !== 'string' || v.trim() === '';
      });
      expect(missing).toEqual([]);
    });
  }

  it('te and hi have exactly the same keys as en, with the same placeholders', () => {
    const ref = leaves(en as Tree);
    for (const lang of ['te', 'hi']) {
      const l = leaves(LOCALES[lang]);
      expect(Object.keys(l).sort(), lang).toEqual(Object.keys(ref).sort());
      const bad = Object.keys(ref).filter((k) => placeholders(ref[k]).join() !== placeholders(l[k] ?? '').join());
      expect(bad, `${lang} placeholder mismatch`).toEqual([]);
    }
  });

  it('te and hi are actually translated (mostly non-Latin script)', () => {
    const ref = leaves(en as Tree);
    for (const [lang, re] of [['te', /[ఀ-౿]/], ['hi', /[ऀ-ॿ]/]] as const) {
      const l = leaves(LOCALES[lang]);
      const translated = Object.keys(ref).filter((k) => re.test(l[k] ?? '')).length;
      expect(translated / Object.keys(ref).length, lang).toBeGreaterThan(0.85);
    }
  });

  it('never promises exact bag counts', () => {
    for (const [lang, tree] of Object.entries(LOCALES)) {
      const bags = lookup(tree, 'pothole.bagsRange') as string;
      expect(bags, lang).toContain('{{low}}');
      expect(bags, lang).toContain('{{high}}');
    }
  });
});
