/**
 * Admin units (zones / circles) for pickers.
 *
 * Live mode reads them from the API (`Api.adminUnits`, i.e. GET /v1/public/wards?kind=circle|zone, the only list
 * endpoint that exposes admin units). POST /v1/runs/generate?circle= takes an admin-unit id or code, so the circle
 * picker sends the real id. The static CIRCLES below are only the mock seed's circles (ids 1..3) and the fallback
 * labels when the API is unreachable.
 */
import { useQuery } from '@tanstack/react-query';
import { getApi } from './apiClient';
import type { AdminUnitBrief, AdminUnitKind } from './types';

export const CIRCLES = [
  { id: 1, code: 'GHMC-C17', name: 'Khairatabad', te: 'ఖైరతాబాద్', hi: 'खैरताबाद' },
  { id: 2, code: 'GHMC-C12', name: 'Mehdipatnam', te: 'మెహదీపట్నం', hi: 'मेहदीपट्टनम' },
  { id: 3, code: 'GHMC-C04', name: 'LB Nagar', te: 'ఎల్‌బీ నగర్', hi: 'एलबी नगर' },
];

/** Localised admin-unit name (Telugu / Hindi when the server has it, English otherwise). */
export function unitName(u: Pick<AdminUnitBrief, 'name_en' | 'name_te' | 'name_hi'>, lang: string): string {
  return (lang === 'te' ? u.name_te : lang === 'hi' ? u.name_hi : null) || u.name_en;
}

/** Units sorted by their localised name (stable for the same language). */
export function sortUnits(units: AdminUnitBrief[], lang: string): AdminUnitBrief[] {
  return [...units].sort((a, b) => unitName(a, lang).localeCompare(unitName(b, lang), lang) || a.id - b.id);
}

/**
 * Circle label by id: first the API list (live ids), then the run's own `circle_name`, then the mock list.
 * The mock list is consulted last because live ids 1..3 are zones/circles with other names.
 */
export function circleName(id: number, lang: string, fallback?: string | null, units?: AdminUnitBrief[] | null): string {
  const u = units?.find((x) => x.id === id);
  if (u) return unitName(u, lang);
  if (fallback) return fallback;
  const c = CIRCLES.find((x) => x.id === id);
  if (!c) return `#${id}`;
  return lang === 'te' ? c.te : lang === 'hi' ? c.hi : c.name;
}

/** Admin units of one kind from the active API (mock or live); cached for 10 min. */
export function useAdminUnits(kind: AdminUnitKind) {
  return useQuery({
    queryKey: ['adminUnits', kind],
    queryFn: () => getApi().adminUnits(kind),
    staleTime: 10 * 60_000,
  });
}
