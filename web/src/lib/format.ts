/** Display formatting: Indian digit grouping (en-IN lakh/crore), IST times, units and +/- sigma. */
export type Lang = 'te' | 'hi' | 'en';

const nfCache = new Map<string, Intl.NumberFormat>();
function nf(lang: string, digits: number): Intl.NumberFormat {
  const k = `${lang}|${digits}`;
  let f = nfCache.get(k);
  if (!f) {
    f = new Intl.NumberFormat(`${lang}-IN`, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
    nfCache.set(k, f);
  }
  return f;
}

export function fmtNum(x: number | null | undefined, digits = 0, lang: string = 'en'): string {
  if (x == null || !Number.isFinite(x)) return '–';
  return nf(lang, digits).format(x);
}

/** Rs with Indian grouping: 1234567 -> "₹12,34,567". */
export function fmtInr(x: number | null | undefined, lang: string = 'en'): string {
  if (x == null || !Number.isFinite(x)) return '–';
  return `₹${nf(lang, 0).format(Math.round(x))}`;
}

/** Compact INR for tiles: 1.2 L (lakh) / 3.4 Cr (crore). */
export function fmtInrCompact(x: number | null | undefined, lang: string = 'en'): string {
  if (x == null || !Number.isFinite(x)) return '–';
  const a = Math.abs(x);
  if (a >= 1e7) return `₹${nf(lang, 2).format(x / 1e7)} Cr`;
  if (a >= 1e5) return `₹${nf(lang, 2).format(x / 1e5)} L`;
  return fmtInr(x, lang);
}

/** "62 ± 7 mm" (sigma rounded to the same precision). */
export function fmtPm(v: number | null | undefined, sigma: number | null | undefined, unit: string, digits = 0, lang = 'en'): string {
  if (v == null || !Number.isFinite(v)) return '–';
  const s = sigma != null && Number.isFinite(sigma) ? ` ± ${fmtNum(sigma, digits, lang)}` : '';
  return `${fmtNum(v, digits, lang)}${s}${unit ? ` ${unit}` : ''}`;
}

const dtfCache = new Map<string, Intl.DateTimeFormat>();
function dtf(lang: string, opts: Intl.DateTimeFormatOptions, key: string) {
  const k = `${lang}|${key}`;
  let f = dtfCache.get(k);
  if (!f) {
    f = new Intl.DateTimeFormat(`${lang}-IN`, { timeZone: 'Asia/Kolkata', ...opts });
    dtfCache.set(k, f);
  }
  return f;
}

/** Timestamps are stored in UTC and always displayed in IST. */
export function fmtDateTime(iso: string | null | undefined, lang: string = 'en'): string {
  if (!iso) return '–';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '–';
  return `${dtf(lang, { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }, 'dt').format(d)} IST`;
}

export function fmtDate(iso: string | null | undefined, lang: string = 'en'): string {
  if (!iso) return '–';
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00+05:30` : iso);
  if (Number.isNaN(d.getTime())) return '–';
  return dtf(lang, { day: '2-digit', month: 'short', year: 'numeric' }, 'd').format(d);
}

/** Today's date in IST as YYYY-MM-DD. */
export function todayIst(now = Date.now()): string {
  return new Date(now + 5.5 * 3600_000).toISOString().slice(0, 10);
}

/** Hours from now until `iso` (negative = overdue). */
export function hoursUntil(iso: string | null | undefined, now = Date.now()): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isNaN(t) ? null : (t - now) / 3600_000;
}

export function fmtHours(h: number | null | undefined, lang = 'en'): string {
  if (h == null || !Number.isFinite(h)) return '–';
  const a = Math.abs(h);
  if (a >= 48) return `${fmtNum(h / 24, 1, lang)} d`;
  return `${fmtNum(h, a < 10 ? 1 : 0, lang)} h`;
}
