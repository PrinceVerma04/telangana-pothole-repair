/** Auth session (JWT or mock token) - zustand store persisted to localStorage. */
import { create } from 'zustand';
import type { Role, TokenResponse } from './types';
import { safeStorage } from './storage';

export const SESSION_KEY = 'gm.session';

export interface Session {
  token: string | null;
  role: Role | null;
  username: string | null;
  agency_id: number | null;
  contractor_id: number | null;
  expires_at: string | null;
}

interface SessionState extends Session {
  setFromToken: (t: TokenResponse, username: string) => void;
  logout: () => void;
}

const EMPTY: Session = { token: null, role: null, username: null, agency_id: null, contractor_id: null, expires_at: null };

function load(): Session {
  const raw = safeStorage.get(SESSION_KEY);
  if (!raw) return EMPTY;
  try {
    const s = JSON.parse(raw) as Session;
    if (s.expires_at && Date.parse(s.expires_at) < Date.now()) return EMPTY;
    return { ...EMPTY, ...s };
  } catch {
    return EMPTY;
  }
}

export const useSession = create<SessionState>((set) => ({
  ...load(),
  setFromToken: (t, username) => {
    const s: Session = {
      token: t.access_token, role: t.role, username, agency_id: t.agency_id ?? null,
      contractor_id: t.contractor_id ?? null, expires_at: t.expires_at,
    };
    safeStorage.set(SESSION_KEY, JSON.stringify(s));
    set(s);
  },
  logout: () => {
    safeStorage.remove(SESSION_KEY);
    // cached API responses are per-user: drop them (service worker cache 'gm-api')
    try { void globalThis.caches?.delete('gm-api').catch(() => undefined); } catch { /* no CacheStorage */ }
    set(EMPTY);
  },
}));

export const getToken = () => useSession.getState().token;
