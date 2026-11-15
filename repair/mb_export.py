"""Measurement Book (MB) and daily repair-run indent CSV exports (pure; no DB imports).

Measurement Book columns follow the Telangana PWD / GHMC MB layout:
``Item No, Description, Nos, L (m), B (m), D (m), Quantity, Unit, Rate (INR), Amount (INR)``.
Each repair plan becomes items ``<k>.1, <k>.2, ...`` (one per cost line); a TOTAL row closes the book.

``measurement_book_csv(rows)`` rows: an iterable of ``RepairPlan`` / plan dicts, or mappings
``{"plan": RepairPlan|dict, "work_order_id": int?, "pothole_id": int?, "location": str?, "date": str?}``.

``indent_csv(run)`` run: a mapping ``{"id"?, "circle_id"?, "circle_name"?, "run_date"?, "items": [...]}``
(or a bare list of items/plans); each item is a plan or ``{"plan", "work_order_id"?, "pothole_id"?,
"lat"?, "lon"?, "hop_m"?}`` in route order. ``indent_totals(run)`` returns the totals dict stored on
``repair_run.totals`` (keys of ``RepairRunOut.totals``) and equals the TOTAL row of the CSV.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping
from typing import Any, Optional

from repair.types import RepairPlan

MB_COLUMNS = ["Item No", "Description", "Nos", "L (m)", "B (m)", "D (m)", "Quantity", "Unit", "Rate (INR)", "Amount (INR)"]
INDENT_COLUMNS = [
    "Seq", "Work Order", "Pothole ID", "Lat", "Lon", "Hop (m)", "Spec", "A_cut (m2)", "D design (mm)",
    "Bag (kg)", "Bags", "Bags range", "Mix (kg)", "Emulsion (kg)", "Sand (kg)", "CRM (t)", "Crew (h)",
    "Est. cost (INR)", "Pending remeasure",
]
TOTAL_KEYS = ["bags", "bags_25kg", "bags_50kg", "mix_kg", "crm_t", "emulsion_kg", "sand_kg", "crew_h", "cost_inr", "n_plans"]


def _as_plan(x: Any) -> RepairPlan:
    if isinstance(x, RepairPlan):
        return x
    if isinstance(x, Mapping):
        return RepairPlan.model_validate(x)
    raise TypeError(f"expected RepairPlan or mapping, got {type(x).__name__}")


def _unpack(row: Any) -> tuple[RepairPlan, dict[str, Any]]:
    if isinstance(row, Mapping) and "plan" in row:
        meta = {k: v for k, v in row.items() if k != "plan"}
        return _as_plan(row["plan"]), meta
    return _as_plan(row), {}


def _fmt(x: Optional[float], nd: int) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def _mb_dims(plan: RepairPlan, item: str, unit: str) -> tuple[str, str, str]:
    """L/B/D recorded against each MB line (D in metres)."""
    L, B = _fmt(plan.l_cut_m, 3), _fmt(plan.w_cut_m, 3)
    it = item.lower()
    if it.startswith("cold mix") or "patching" in it:
        return L, B, _fmt(plan.d_patch_mm / 1000.0, 3)
    if "crusher run" in it:
        return L, B, _fmt(plan.d_crm_mm / 1000.0, 3)
    if "emulsion" in it or "sand" in it:
        return L, B, ""
    return "", "", ""


def measurement_book_csv(rows: Iterable[Any]) -> str:
    """Measurement Book CSV text (UTF-8, header row + items + TOTAL)."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(MB_COLUMNS)
    total = 0.0
    for k, row in enumerate(rows, start=1):
        plan, meta = _unpack(row)
        pid = meta.get("pothole_id", plan.pothole_id)
        head = [f"Pothole #{pid}" if pid is not None else "Pothole"]
        if plan.merged_pothole_ids:
            head.append("merged cut " + "+".join(str(i) for i in plan.merged_pothole_ids))
        if meta.get("work_order_id") is not None:
            head.append(f"WO {meta['work_order_id']}")
        if meta.get("location"):
            head.append(str(meta["location"]))
        if meta.get("date"):
            head.append(str(meta["date"]))
        head.append(plan.spec_code.value)
        if plan.pending_remeasure:
            head.append("LOWER BOUND - pending re-measure")
        prefix = " | ".join(head)
        for j, cl in enumerate(plan.cost_breakdown, start=1):
            L, B, D = _mb_dims(plan, cl.item, cl.unit)
            is_ls = cl.unit == "INR"
            w.writerow([
                f"{k}.{j}", f"{prefix}: {cl.item}", "1", L, B, D,
                "1" if is_ls else _fmt(cl.qty, 3), "LS" if is_ls else cl.unit,
                _fmt(cl.amount_inr if is_ls else cl.rate_inr, 2), _fmt(cl.amount_inr, 2),
            ])
            total += cl.amount_inr
    w.writerow(["TOTAL", "", "", "", "", "", "", "", "", _fmt(round(total, 2), 2)])
    return buf.getvalue()


def _items(run: Any) -> tuple[list[tuple[RepairPlan, dict[str, Any]]], dict[str, Any]]:
    if isinstance(run, Mapping):
        items = run.get("items", run.get("plans", []))
        head = {k: v for k, v in run.items() if k not in ("items", "plans")}
    else:
        items, head = list(run), {}
    return [_unpack(x) for x in items], head


def indent_totals(run: Any) -> dict[str, float]:
    """Run totals: bags (all sizes and per size), mix kg, CRM t, emulsion kg, sand kg, crew h, cost INR."""
    items, _ = _items(run)
    t = dict.fromkeys(TOTAL_KEYS, 0.0)
    for plan, _meta in items:
        t["bags"] += plan.bags
        if plan.bag_kg == 25:
            t["bags_25kg"] += plan.bags
        elif plan.bag_kg == 50:
            t["bags_50kg"] += plan.bags
        t["mix_kg"] += plan.mix_kg
        t["crm_t"] += plan.crm_t
        t["emulsion_kg"] += plan.emulsion_kg
        t["sand_kg"] += plan.sand_kg
        t["crew_h"] += plan.labour_h
        t["cost_inr"] += plan.est_cost_inr
        t["n_plans"] += 1
    return {k: round(v, 4) for k, v in t.items()}


def indent_csv(run: Any) -> str:
    """Daily repair-run indent CSV: one row per stop in route order + a TOTAL row."""
    items, head = _items(run)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    title = ["Repair run indent"]
    for key in ("id", "circle_name", "circle_id", "run_date"):
        if head.get(key) is not None:
            title.append(f"{key}={head[key]}")
    w.writerow([" ; ".join(title)])
    w.writerow(INDENT_COLUMNS)
    for seq, (plan, meta) in enumerate(items, start=1):
        w.writerow([
            seq, meta.get("work_order_id", ""), meta.get("pothole_id", plan.pothole_id if plan.pothole_id is not None else ""),
            _fmt(meta.get("lat"), 6), _fmt(meta.get("lon"), 6), _fmt(meta.get("hop_m"), 0), plan.spec_code.value,
            _fmt(plan.a_cut_m2, 4), _fmt(plan.d_design_mm, 0), plan.bag_kg, plan.bags, f"{plan.bags_low}-{plan.bags_high}",
            _fmt(plan.mix_kg, 2), _fmt(plan.emulsion_kg, 3), _fmt(plan.sand_kg, 3), _fmt(plan.crm_t, 4),
            _fmt(plan.labour_h, 2), _fmt(plan.est_cost_inr, 2), "yes" if plan.pending_remeasure else "no",
        ])
    t = indent_totals(run)
    w.writerow([
        "TOTAL", "", "", "", "", "", "", "", "", "", int(t["bags"]),
        f"25kg:{int(t['bags_25kg'])} 50kg:{int(t['bags_50kg'])}", _fmt(t["mix_kg"], 2), _fmt(t["emulsion_kg"], 3),
        _fmt(t["sand_kg"], 3), _fmt(t["crm_t"], 4), _fmt(t["crew_h"], 2), _fmt(t["cost_inr"], 2), "",
    ])
    return buf.getvalue()


__all__ = ["INDENT_COLUMNS", "MB_COLUMNS", "indent_csv", "indent_totals", "measurement_book_csv"]
