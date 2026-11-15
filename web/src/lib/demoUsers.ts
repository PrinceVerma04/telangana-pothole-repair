/**
 * Demo users. Mirrors server/app/seed.py:DEMO_USERS (WP4): same usernames, password `demo1234`.
 * The login screen offers them as a role picker; in http mode they log in against the real API.
 */
import type { Role } from './types';

export interface DemoUser {
  username: string;
  password: string;
  role: Role;
  agency_id?: number | null;
  contractor_id?: number | null;
  /** i18n key for the human description */
  descKey: string;
}

export const DEMO_PASSWORD = 'demo1234';

export const DEMO_USERS: DemoUser[] = [
  { username: 'zonal_khairatabad', password: DEMO_PASSWORD, role: 'zonal_admin', agency_id: 1, descKey: 'login.desc.zonal_admin' },
  { username: 'engineer1', password: DEMO_PASSWORD, role: 'field_engineer', agency_id: 1, descKey: 'login.desc.field_engineer' },
  { username: 'contractor1', password: DEMO_PASSWORD, role: 'contractor', agency_id: 1, contractor_id: 1, descKey: 'login.desc.contractor' },
  { username: 'contractor2', password: DEMO_PASSWORD, role: 'contractor', agency_id: 1, contractor_id: 2, descKey: 'login.desc.contractor' },
  { username: 'citizen1', password: DEMO_PASSWORD, role: 'citizen', agency_id: null, descKey: 'login.desc.citizen' },
  { username: 'admin', password: DEMO_PASSWORD, role: 'super_admin', agency_id: 1, descKey: 'login.desc.super_admin' },
];
