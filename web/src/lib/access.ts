/** Which roles see which route (UI only; the API enforces RBAC itself). */
import type { Role } from './types';

export type RouteKey = 'dashboard' | 'map' | 'queue' | 'workOrders' | 'field' | 'contractors' | 'citizen' | 'admin' | 'publicWards' | 'pothole' | 'run';

const STAFF: Role[] = ['zonal_admin', 'super_admin', 'field_engineer'];

export const ROUTE_ROLES: Record<RouteKey, Role[] | 'public'> = {
  dashboard: STAFF,
  map: [...STAFF, 'contractor'],
  pothole: [...STAFF, 'contractor'],
  queue: STAFF,
  workOrders: [...STAFF, 'contractor'],
  run: [...STAFF, 'contractor'],
  field: ['field_engineer', 'contractor', 'super_admin', 'zonal_admin'],
  contractors: [...STAFF, 'contractor'],
  admin: ['zonal_admin', 'super_admin'],
  citizen: 'public',
  publicWards: 'public',
};

export function canSee(route: RouteKey, role: Role | null | undefined): boolean {
  const r = ROUTE_ROLES[route];
  if (r === 'public') return true;
  return !!role && r.includes(role);
}

/** Landing page after login. */
export function homeFor(role: Role | null | undefined): string {
  switch (role) {
    case 'contractor': return '/field';
    case 'citizen': return '/citizen';
    case 'driver_device': return '/public/wards';
    case 'field_engineer': return '/queue';
    default: return '/';
  }
}
