/** Mirror of server/app/schemas/states.py (frozen). Used by the UI to offer only legal actions and by the mock. */
import type { PotholeStatus, Role, WorkOrderStatus } from './types';

export const WORK_ORDER_TRANSITIONS: Record<WorkOrderStatus, WorkOrderStatus[]> = {
  planned: ['issued', 'cancelled'],
  issued: ['acknowledged', 'cancelled'],
  acknowledged: ['in_progress', 'cancelled'],
  in_progress: ['repaired_claimed'],
  repaired_claimed: ['verification_pending'],
  verification_pending: ['verified_closed', 'qc_failed'],
  qc_failed: [],
  verified_closed: [],
  cancelled: [],
};

export const WO_TO_POTHOLE: Partial<Record<WorkOrderStatus, PotholeStatus>> = {
  issued: 'work_ordered',
  acknowledged: 'acknowledged',
  in_progress: 'in_progress',
  repaired_claimed: 'repaired_claimed',
  verification_pending: 'verification_pending',
  verified_closed: 'verified_closed',
  qc_failed: 'qc_failed',
};

export const WO_TRANSITION_ROLES: Partial<Record<WorkOrderStatus, Role[]>> = {
  issued: ['zonal_admin', 'field_engineer'],
  acknowledged: ['contractor'],
  in_progress: ['contractor'],
  repaired_claimed: ['contractor'],
  verification_pending: ['field_engineer', 'zonal_admin'],
  verified_closed: ['field_engineer', 'zonal_admin'],
  qc_failed: ['field_engineer', 'zonal_admin'],
  cancelled: ['zonal_admin'],
};

export const INVOICE_ELIGIBLE: WorkOrderStatus[] = ['verified_closed'];

const LIVE: PotholeStatus[] = [
  'candidate', 'confirmed', 'planned', 'work_ordered', 'acknowledged', 'in_progress', 'repaired_claimed',
  'verification_pending', 'qc_failed', 'reopened', 'recurrence_watch',
];

export const POTHOLE_TRANSITIONS: Record<PotholeStatus, PotholeStatus[]> = (() => {
  const t: Record<PotholeStatus, PotholeStatus[]> = {
    candidate: ['confirmed', 'rejected', 'archived'],
    confirmed: ['planned', 'rejected', 'archived'],
    planned: ['work_ordered', 'confirmed', 'rejected'],
    work_ordered: ['acknowledged', 'planned', 'rejected'],
    acknowledged: ['in_progress', 'planned'],
    in_progress: ['repaired_claimed'],
    repaired_claimed: ['verification_pending'],
    verification_pending: ['verified_closed', 'qc_failed'],
    qc_failed: ['work_ordered', 'planned'],
    verified_closed: ['recurrence_watch', 'reopened'],
    recurrence_watch: ['reopened', 'archived'],
    reopened: ['confirmed', 'planned'],
    rejected: ['candidate'],
    archived: ['candidate'],
  };
  for (const s of LIVE) if (!t[s].includes('rejected')) t[s] = [...t[s], 'rejected'];
  return t;
})();

/** Open = still needs a repair (counts toward KPIs and the queue). */
export const OPEN_STATUSES: PotholeStatus[] = [
  'confirmed', 'planned', 'work_ordered', 'acknowledged', 'in_progress', 'repaired_claimed',
  'verification_pending', 'qc_failed', 'reopened',
];

export function workOrderCan(from: WorkOrderStatus, to: WorkOrderStatus): boolean {
  return WORK_ORDER_TRANSITIONS[from]?.includes(to) ?? false;
}

export function potholeCan(from: PotholeStatus, to: PotholeStatus): boolean {
  return POTHOLE_TRANSITIONS[from]?.includes(to) ?? false;
}

/** Transitions a role may drive from `from` (super_admin may do all legal ones). */
export function allowedWorkOrderActions(from: WorkOrderStatus, role: Role | null | undefined): WorkOrderStatus[] {
  if (!role) return [];
  return WORK_ORDER_TRANSITIONS[from].filter(
    (to) => role === 'super_admin' || (WO_TRANSITION_ROLES[to] ?? []).includes(role),
  );
}
