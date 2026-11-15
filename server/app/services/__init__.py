"""Ops services (WP8b): DB-aware business logic shared by the routers and the WP6 ops ticks.

Modules (import the one you need; this package does not import them eagerly):

* ``state_machine`` - work-order creation and transitions (RBAC, evidence preconditions, pothole mirroring,
  audit_log, qc_failed -> rework WO), ``work_order_out`` DTO builder.
* ``sla``           - SLA hours per work order, claim time / breach rule, ``sla_tick`` escalation ladder.
* ``verification``  - ``evaluate``: bus-verified repair (PLAN 5.7).
* ``dlp``           - ``watch``: re-detection inside the defect-liability period -> reopened + dlp_event.
* ``runs``          - repair planning of a pothole (``plan_pothole``) and the daily repair run + indent.
* ``kpi``           - ``kpi`` per admin unit and the contractor ``scorecard``.
* ``exports``       - Measurement Book / invoice-eligible work orders / potholes GeoJSON.

Conventions: functions take a SQLAlchemy ``Session`` and **never commit** (the caller commits, so audit rows commit
atomically with the change). ``actor`` is a ``server.app.core.rbac.Principal`` (role-checked) or a string label /
None for the system (trusted caller, no role check). Units: depth mm, area m2, volume litres, money INR; all
timestamps UTC.
"""

__all__ = ["dlp", "exports", "kpi", "runs", "sla", "state_machine", "verification"]
