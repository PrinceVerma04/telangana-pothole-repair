"""Frozen repair-planning contract (WP0). Pure pydantic; no DB / server imports.

Produced by ``repair.planner.plan(PotholeFused, rules, rate_card, context)`` (PLAN 7).
Units: lengths m, depths mm, volumes litres, masses kg (CRM in tonnes), money INR.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import Field

from measure.types import _Serialisable


class SpecCode(str, Enum):
    IRC116_COLDMIX = "IRC116_COLDMIX"
    IRCSP100_EMULSION = "IRCSP100_EMULSION"
    JET_PATCHER = "JET_PATCHER"
    INFRARED = "INFRARED"
    HOT_MIX = "HOT_MIX"
    CC_MORTAR = "CC_MORTAR"


class PlanContext(_Serialisable):
    """Situational inputs the planner cannot derive from the pothole row."""

    cluster_n_within_300m: int = 0
    raining_now: bool = False
    dry_forecast_48h: bool = True
    infrared_unit_in_zone: bool = False
    hot_mix_plant_km: Optional[float] = None
    in_monsoon_window: bool = False
    agency_pref: SpecCode = SpecCode.IRC116_COLDMIX
    bag_kg: Optional[int] = Field(None, description="override configs/irc_rules.yaml bag_kg_default")
    merged_pothole_ids: list[int] = Field(default_factory=list)


class CostLine(_Serialisable):
    item: str
    qty: float
    unit: str
    rate_inr: float
    amount_inr: float
    provenance: Optional[str] = None


class RepairPlan(_Serialisable):
    pothole_id: Optional[int] = None
    merged_pothole_ids: list[int] = Field(default_factory=list)
    # geometry of the cut
    l_cut_m: float
    w_cut_m: float
    a_cut_m2: float
    d_design_mm: float
    layered: bool
    d_patch_mm: float
    d_crm_mm: float = 0.0
    # volumes (litres) - all four kept for audit (PLAN 7.4)
    v_patch_l: float
    v_crm_l: float = 0.0
    v_cavity_l: Optional[float] = Field(None, description="measured cavity volume; None if water-filled")
    v_cuboid_l: Optional[float] = Field(None, description="L*W*depth_max, shown to expose cuboid over-estimate")
    # materials
    mix_kg: float
    mix_kg_sigma: float
    bag_kg: int
    bags: int
    bags_low: int
    bags_high: int
    emulsion_kg: float
    sand_kg: float
    crm_t: float = 0.0
    labour_h: float
    # method / money
    method: str = Field(..., description="human-readable method name")
    spec_code: SpecCode
    est_cost_inr: float
    cost_breakdown: list[CostLine] = Field(default_factory=list)
    baseline_cost_inr: Optional[float] = None
    rate_card_version: str = "unknown"
    # flags
    pending_remeasure: bool = Field(False, description="water-filled: lower-bound quantities only")
    weather_blocked: bool = False
    notes: list[str] = Field(default_factory=list)
    refs: list[str] = Field(default_factory=list, description="IRC clauses applied")
    extras: dict[str, Any] = Field(default_factory=dict)


__all__ = ["CostLine", "PlanContext", "RepairPlan", "SpecCode"]
