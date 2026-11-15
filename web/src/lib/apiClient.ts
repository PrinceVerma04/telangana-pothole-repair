/**
 * Picks the API implementation once at startup (src/lib/api.ts resolveApiMode):
 * localStorage override `gm.apiMode` > VITE_USE_MOCK > health probe of VITE_API_BASE (unreachable -> mock).
 */
import { HttpApi, MODE_OVERRIDE_KEY, resolveApiMode, type Api, type ApiMode } from './api';
import { MockApi } from './mock/mockApi';
import { getToken, useSession } from './session';
import { safeStorage } from './storage';

export interface ApiInfo { mode: ApiMode; reason: string; baseUrl: string }

let current: Api | null = null;
let info: ApiInfo | null = null;

export const API_BASE: string = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';

export function makeHttpApi(baseUrl = API_BASE): HttpApi {
  return new HttpApi({ baseUrl, getToken, onUnauthorized: () => useSession.getState().logout() });
}

export async function bootstrapApi(): Promise<ApiInfo> {
  const http = makeHttpApi();
  const r = await resolveApiMode({
    envUseMock: import.meta.env.VITE_USE_MOCK as string | undefined,
    override: safeStorage.get(MODE_OVERRIDE_KEY),
    probe: () => http.health(),
  });
  current = r.mode === 'mock' ? new MockApi({ getToken }) : http;
  info = { ...r, baseUrl: API_BASE };
  return info;
}

export function getApi(): Api {
  if (!current) throw new Error('API not bootstrapped');
  return current;
}

export function getApiInfo(): ApiInfo | null { return info; }

/** Tests / storybook-style injection. */
export function setApi(api: Api, i: ApiInfo = { mode: api.mode, reason: 'injected', baseUrl: '' }): void {
  current = api;
  info = i;
}

/** Switch implementation (persists override and reloads, so caches start clean). */
export function switchMode(mode: ApiMode | null): void {
  if (mode) safeStorage.set(MODE_OVERRIDE_KEY, mode);
  else safeStorage.remove(MODE_OVERRIDE_KEY);
  useSession.getState().logout();
  location.reload();
}
