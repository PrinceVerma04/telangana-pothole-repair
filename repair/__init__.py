"""Repair planning (IRC:82-2015 Sec 10 / Cl. 11.4, IRC:116-2014, IRC:SP:100-2014).

WP8 implements ``repair.planner.plan`` and ``repair.mb_export``; the DTO it
returns is ``repair.types.RepairPlan`` (frozen; see docs/CONTRACTS.md).
Also: ``repair.planner.merge_cut``/``plan_many``, ``repair.loaders.load_rules``/``load_rate_card``.
"""
