"""Staged known-depth target protocol (PLAN 4.12 level 2) -- truth CSV template, validation, error table.

Foam-board "potholes" of known depth and diameter are laid on a closed, flat road and photographed from
3-12 m with the calibrated camera (docs/CALIBRATION_SOP.md section 5). One CSV row = one captured frame of one
target. The truth columns are exactly ``measure.truth.TRUTH_CSV_COLUMNS`` (so ``measure.truth.load_truth_csv``
and ``POST /v1/truth/measurements`` accept the same rows); the staged columns say where the frame was taken
from and which ``measure.cli run --json`` output belongs to it.

Subcommands (run from the repo root with the project venv):

  template  write the protocol's CSV (targets x passes x ranges), nominal truth pre-filled
            python scripts/staged_pothole_truth.py template --out data/truth/template.csv
  validate  check a filled CSV (units, ranges, shape/area/volume consistency, one truth per target)
            python scripts/staged_pothole_truth.py validate data/truth/staged_2026-11-20.csv
  compare   truth vs measurement JSONs -> errors.csv, errors.md, summary.json (per range bin, frame and
            fused-pass level, PLAN pass criteria, k_sigma refit)
            python scripts/staged_pothole_truth.py compare data/truth/staged_2026-11-20.csv --out reports/staged_truth
  rehearse  dry run of the whole chain without a camera: renders every circular target with the synthetic
            depth backend through the production measurement path, then runs compare. Output is labelled
            ``synthetic_rehearsal`` and is NOT a staged-target result.
            python scripts/staged_pothole_truth.py rehearse data/truth/template.csv --out reports/staged_truth_rehearsal

Units: depth mm, area m^2, volume litres, lengths m; ``measured_at`` ISO 8601 with a UTC offset (stored UTC).
Truth rows come from a straightedge + depth gauge (``method=manual_gauge``, sigma 3 mm) or a phone LiDAR scan
(``lidar_phone``). Measurement JSONs are read as data only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ.setdefault("GM_DEVICE", "cpu")

from measure.truth import TRUTH_CSV_COLUMNS, TruthPair, evaluate  # noqa: E402

STAGED_COLUMNS = ["capture_id", "pass_id", "shape", "range_m", "camera_height_m", "pitch_deg", "image",
                  "measurement_json", "detection_index"]
# Field-friendly order; readers use the header names, so the order is free.
COLUMNS = ["capture_id", "pothole_ref", "pass_id", "range_m", "camera_height_m", "pitch_deg", "shape",
           "depth_p95_mm", "depth_max_mm", "length_m", "width_m", "area_m2", "volume_l", "method", "device_id",
           "measured_by", "measured_at", "lat", "lon", "image", "measurement_json", "detection_index", "mesh_key",
           "notes"]
assert set(COLUMNS) == set(TRUTH_CSV_COLUMNS) | set(STAGED_COLUMNS)

NUMERIC = ("depth_p95_mm", "depth_max_mm", "area_m2", "volume_l", "length_m", "width_m", "lat", "lon", "range_m",
           "camera_height_m", "pitch_deg")
TRUTH_KEYS = ("shape", "depth_p95_mm", "depth_max_mm", "length_m", "width_m", "area_m2", "volume_l")
SHAPES = ("circle", "rect", "bowl", "other")
METHODS = ("manual_gauge", "lidar_phone", "stereo")
INDIA_BBOX = (6.0, 37.5, 68.0, 97.5)  # lat_min, lat_max, lon_min, lon_max

# PLAN 4.12 level 2 pass criteria (demo): fused depth MAE <= 10 mm, IRC class accuracy >= 90 %, volume MAPE <= 25 %
CRITERIA = {"depth_mae_mm": ("<=", 10.0), "irc_class_accuracy": (">=", 0.90), "volume_mape": ("<=", 0.25)}


@dataclass(frozen=True)
class Target:
    ref: str
    shape: str
    diameter_m: float
    depth_mm: float
    ranges_m: tuple[float, ...]
    purpose: str

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m ** 2 / 4.0

    @property
    def volume_l(self) -> float:
        return self.area_m2 * self.depth_mm  # 1 m^2 x 1 mm = 1 L


# Circular cut-outs through a foam deck (vertical walls, flat floor = the road). Ranges are tape distances on
# the ground from the point below the lens to the target centre. Limits found with ``rehearse`` (phone_demo
# geometry, h 1.4 m, pitch 12 deg): a 0.18 m hole beyond ~5 m and a 0.45 m hole at 10 m give < 400 mask pixels
# (measure.yaml mask.min_mask_px -> rejected); at 12 m tape range the camera-axis range is ~12.03 m, just past
# the 12 m quality gate (q = 0). Hence the maximum of 11 m.
TARGETS: tuple[Target, ...] = (
    Target("T20-S", "circle", 0.18, 20.0, (3.0, 5.0), "IRC:82 Small (<=25 mm, <=200 mm)"),
    Target("T40-M", "circle", 0.45, 40.0, (3.0, 5.0, 7.0, 9.0), "IRC:82 Medium (25-50 mm, <=500 mm)"),
    Target("T70-L", "circle", 0.45, 70.0, (3.0, 5.0, 7.0, 9.0), "IRC:82 Large by depth (>50 mm)"),
    Target("T40-W", "circle", 0.80, 40.0, (5.0, 7.0, 9.0, 11.0), "IRC:82 Large by width (>500 mm)"),
)


# ----------------------------------------------------------------------------- CSV I/O
def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in COLUMNS})


def read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        header = [h.strip() for h in (rd.fieldnames or [])]
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rd]
    return header, rows


def make_template(targets: Sequence[Target] = TARGETS, passes: int = 2, camera_height_m: float = 1.4,
                  pitch_deg: float = 12.0, device_id: str = "") -> list[dict[str, Any]]:
    rows = []
    for t in targets:
        for p in range(1, passes + 1):
            for r in t.ranges_m:
                rows.append({
                    "capture_id": f"{t.ref}-P{p}-R{r:g}", "pothole_ref": t.ref, "pass_id": f"P{p}", "range_m": r,
                    "camera_height_m": camera_height_m, "pitch_deg": pitch_deg, "shape": t.shape,
                    "depth_p95_mm": t.depth_mm, "depth_max_mm": t.depth_mm, "length_m": t.diameter_m,
                    "width_m": t.diameter_m, "area_m2": round(t.area_m2, 5), "volume_l": round(t.volume_l, 3),
                    "method": "manual_gauge", "device_id": device_id, "measured_by": "", "measured_at": "",
                    "lat": None, "lon": None, "image": "", "measurement_json": "", "detection_index": 0,
                    "mesh_key": "", "notes": f"NOMINAL {t.purpose}; replace with gauge readings",
                })
    return rows


# ----------------------------------------------------------------------------- validation
@dataclass
class Report:
    errors: list[str]
    warnings: list[str]
    rows: list[dict[str, Any]]  # parsed + normalised

    @property
    def ok(self) -> bool:
        return not self.errors


def _num(v: str) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


def _shape_area(shape: str, length_m: Optional[float], width_m: Optional[float]) -> Optional[float]:
    if length_m is None or width_m is None:
        return None
    if shape == "circle":
        return math.pi * length_m * width_m / 4.0
    if shape == "rect":
        return length_m * width_m
    return None


def validate_rows(header: Sequence[str], raw: Sequence[dict[str, str]], base_dir: Optional[Path] = None) -> Report:
    errors: list[str] = []
    warnings: list[str] = []
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        return Report([f"missing columns: {', '.join(missing)}"], [], [])
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    truth_by_ref: dict[str, tuple[str, dict[str, Any]]] = {}
    for i, r in enumerate(raw, start=2):  # line numbers as in a spreadsheet (header = line 1)
        where = f"line {i} ({r.get('capture_id') or '?'})"
        row: dict[str, Any] = dict(r)
        try:
            for k in NUMERIC:
                row[k] = _num(r.get(k, ""))
            row["detection_index"] = int(r["detection_index"]) if r.get("detection_index") else 0
        except ValueError as e:
            errors.append(f"{where}: not a number: {e}")
            continue
        cid, ref = r.get("capture_id", ""), r.get("pothole_ref", "")
        if not cid:
            errors.append(f"{where}: capture_id is empty")
        elif cid in seen_ids:
            errors.append(f"{where}: duplicate capture_id {cid!r}")
        seen_ids.add(cid)
        if not ref:
            errors.append(f"{where}: pothole_ref is empty")
        shape = r.get("shape") or "other"
        row["shape"] = shape
        if shape not in SHAPES:
            errors.append(f"{where}: shape {shape!r} not in {SHAPES}")
        if r.get("method") not in METHODS:
            errors.append(f"{where}: method {r.get('method')!r} not in {METHODS}")
        if row["detection_index"] < 0:
            errors.append(f"{where}: detection_index must be >= 0")

        d95, dmax = row["depth_p95_mm"], row["depth_max_mm"]
        if d95 is None:
            errors.append(f"{where}: depth_p95_mm is required")
        elif d95 <= 0 or d95 > 300:
            errors.append(f"{where}: depth_p95_mm={d95:g} outside (0, 300] mm")
        elif d95 < 1.0:
            errors.append(f"{where}: depth_p95_mm={d95:g} looks like metres (unit is mm)")
        if d95 is not None and dmax is not None and dmax + 1e-9 < d95:
            errors.append(f"{where}: depth_max_mm {dmax:g} < depth_p95_mm {d95:g}")

        for k in ("length_m", "width_m"):
            v = row[k]
            if v is None:
                if shape in ("circle", "rect"):
                    errors.append(f"{where}: {k} is required for shape {shape}")
            elif v <= 0.02 or v > 5.0:
                hint = " (looks like mm/cm: the unit is m)" if v > 5.0 else ""
                errors.append(f"{where}: {k}={v:g} outside (0.02, 5] m{hint}")

        geo_area = _shape_area(shape, row["length_m"], row["width_m"])
        if row["area_m2"] is None:
            if geo_area is not None:
                row["area_m2"] = geo_area
                warnings.append(f"{where}: area_m2 derived from {shape} L x W = {geo_area:.4f} m2")
        elif row["area_m2"] <= 0 or row["area_m2"] > 10:
            errors.append(f"{where}: area_m2={row['area_m2']:g} outside (0, 10] m2 (unit is m2, not cm2)")
        elif geo_area is not None and abs(row["area_m2"] / geo_area - 1) > 0.05:
            errors.append(f"{where}: area_m2 {row['area_m2']:.4f} differs > 5 % from the {shape} L x W area "
                          f"{geo_area:.4f} (typo or unit mistake?)")
        if shape in ("circle", "rect") and row["area_m2"] and d95:
            v_flat = row["area_m2"] * d95
            if row["volume_l"] is None:
                row["volume_l"] = v_flat
                warnings.append(f"{where}: volume_l derived as area x depth = {v_flat:.3f} L (flat floor)")
            else:
                dev = abs(row["volume_l"] / v_flat - 1)
                if dev > 0.25:
                    errors.append(f"{where}: volume_l {row['volume_l']:.3f} L differs {dev:.0%} from area x depth "
                                  f"{v_flat:.3f} L (unit is litres = m2 x mm)")
                elif dev > 0.10:
                    warnings.append(f"{where}: volume_l differs {dev:.0%} from area x depth (floor not flat?)")
        elif row["volume_l"] is not None and row["volume_l"] < 0:
            errors.append(f"{where}: volume_l < 0")

        rng = row["range_m"]
        if rng is None:
            errors.append(f"{where}: range_m (tape, camera foot -> target centre) is required")
        elif not 1.0 <= rng <= 20.0:
            errors.append(f"{where}: range_m={rng:g} outside [1, 20] m")
        elif not 3.0 <= rng <= 12.0:
            warnings.append(f"{where}: range_m={rng:g} outside the 3-12 m quality gate (expect low quality)")
        h = row["camera_height_m"]
        if h is None:
            errors.append(f"{where}: camera_height_m is required")
        elif not 0.5 <= h <= 4.0:
            errors.append(f"{where}: camera_height_m={h:g} outside [0.5, 4] m (unit is m)")
        if row["pitch_deg"] is not None and not 0.0 <= row["pitch_deg"] <= 45.0:
            errors.append(f"{where}: pitch_deg={row['pitch_deg']:g} outside [0, 45]")

        lat, lon = row["lat"], row["lon"]
        if (lat is None) != (lon is None):
            errors.append(f"{where}: give both lat and lon or neither")
        elif lat is not None and not (INDIA_BBOX[0] <= lat <= INDIA_BBOX[1] and INDIA_BBOX[2] <= lon <= INDIA_BBOX[3]):
            errors.append(f"{where}: lat/lon {lat},{lon} outside India")

        ts = r.get("measured_at", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    errors.append(f"{where}: measured_at {ts!r} has no UTC offset (write ...Z or +05:30)")
                else:
                    row["measured_at"] = dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                errors.append(f"{where}: measured_at {ts!r} is not ISO 8601")
        if not r.get("measured_by"):
            warnings.append(f"{where}: measured_by empty -> truth not confirmed by a gauge reading yet")

        mj = r.get("measurement_json", "")
        if mj and base_dir is not None and not (base_dir / mj).is_file():
            warnings.append(f"{where}: measurement_json {mj!r} not found (compare will skip this row)")

        if ref:
            sig = {k: row[k] for k in TRUTH_KEYS}
            if ref in truth_by_ref:
                first_where, first = truth_by_ref[ref]
                for k in TRUTH_KEYS:
                    a, b = first[k], sig[k]
                    same = (a == b) if not (isinstance(a, float) and isinstance(b, float)) else \
                        abs(a - b) <= 0.005 * max(abs(a), abs(b), 1e-9)
                    if not same:
                        errors.append(f"{where}: {ref} {k}={b} differs from {first_where} ({a}); one target = one truth")
            else:
                truth_by_ref[ref] = (where, sig)
        rows.append(row)
    if not raw:
        errors.append("no rows")
    return Report(errors, warnings, rows)


def validate_csv(path: str | Path) -> Report:
    header, raw = read_csv(path)
    return validate_rows(header, raw, Path(path).resolve().parent)


# ----------------------------------------------------------------------------- measurements
def _to_measurement(d: dict[str, Any]) -> Any:
    """FrameMeasurement / PassMeasurement from a dict; unknown keys are dropped (data, never code)."""
    from measure.types import FrameMeasurement, PassMeasurement

    cls = PassMeasurement if "n_frames" in d else FrameMeasurement
    try:
        return cls.from_dict(d)
    except Exception:  # noqa: BLE001 - tolerate JSONs from older/newer writers
        known = {k: v for k, v in d.items() if k in cls.model_fields}
        try:
            return cls.from_dict(known)
        except Exception:  # noqa: BLE001
            return SimpleNamespace(**{k: d.get(k) for k in ("depth_p95_mm", "depth_sigma_mm", "volume_l",
                                                           "volume_sigma", "area_m2", "area_sigma", "width_m",
                                                           "range_m", "quality", "water_filled",
                                                           "reject_reason", "certified", "scale_factor")},
                                   extras=d.get("extras") or {})


def load_measurement(path: Path, index: int = 0) -> tuple[Any, dict[str, Any]]:
    """Accepts ``measure.cli run --json`` output (MeasureResponse), a list, or a single measurement dict."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {}
    if isinstance(doc, dict) and "measurements" in doc:
        items = doc["measurements"]
        meta = {k: doc.get(k) for k in ("calib_id", "certified", "depth_model_version", "source")}
    elif isinstance(doc, list):
        items = doc
    else:
        items = [doc]
    if not items:
        raise ValueError("no measurements in file")
    if index >= len(items):
        raise ValueError(f"detection_index {index} but the file has {len(items)} measurement(s)")
    meta["n_in_file"] = len(items)
    return _to_measurement(items[index]), meta


# ----------------------------------------------------------------------------- compare
def _sev(depth_mm: Optional[float], width_m: Optional[float], water: bool = False) -> Optional[str]:
    from measure import irc

    if width_m is None:
        return None
    return irc.severity(depth_mm, width_m * 1000.0, water).value


def _criteria(stats: Optional[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, (op, thr) in CRITERIA.items():
        v = (stats or {}).get(key)
        ok = None if v is None else (v <= thr if op == "<=" else v >= thr)
        out[key] = {"value": v, "target": f"{op} {thr:g}", "pass": ok}
    vals = [c["pass"] for c in out.values()]
    out["overall"] = None if any(v is None for v in vals) else all(vals)
    return out


def _fmt_pm(v: Optional[float], s: Optional[float], nd: int = 1) -> str:
    if v is None:
        return "-"
    return f"{v:.{nd}f} ± {s:.{nd}f}" if s is not None else f"{v:.{nd}f}"


def compare(csv_path: str | Path, out_dir: str | Path, *, mode: str = "staged", force: bool = False,
            cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from measure.config import load_measure_cfg
    from measure.fuse import fuse_frames
    from measure.truth import fit_k_sigma
    from measure.types import FrameMeasurement

    csv_path = Path(csv_path)
    rep = validate_csv(csv_path)
    if rep.errors and not force:
        raise SystemExit("validation failed:\n  " + "\n  ".join(rep.errors))
    cfg = cfg or load_measure_cfg()
    base = csv_path.resolve().parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    frame_pairs: list[TruthPair] = []
    groups: dict[tuple[str, str], list[tuple[dict[str, Any], Any]]] = {}
    notes: list[str] = []
    calib_ids, model_versions, certified = set(), set(), set()
    for row in rep.rows:
        truth = {k: row[k] for k in ("depth_p95_mm", "volume_l", "area_m2", "width_m")}
        rec: dict[str, Any] = {"capture_id": row["capture_id"], "pothole_ref": row["pothole_ref"],
                               "pass_id": row.get("pass_id") or "", "range_m": row["range_m"],
                               "truth_depth_p95_mm": truth["depth_p95_mm"], "truth_volume_l": truth["volume_l"],
                               "truth_area_m2": truth["area_m2"], "truth_width_m": truth["width_m"],
                               "truth_severity": _sev(truth["depth_p95_mm"], truth["width_m"])}
        mj = row.get("measurement_json") or ""
        if not mj:
            rec["status"] = "not_measured"
            records.append(rec)
            continue
        try:
            m, meta = load_measurement(base / mj, int(row.get("detection_index") or 0))
        except (OSError, ValueError) as e:
            rec["status"] = f"unreadable: {e}"
            records.append(rec)
            continue
        if meta.get("n_in_file", 1) > 1:
            notes.append(f"{row['capture_id']}: {meta['n_in_file']} measurements in file, used index "
                         f"{row.get('detection_index') or 0}")
        cert = getattr(m, "certified", meta.get("certified"))
        certified.add(bool(cert))
        if getattr(m, "calib_id", None) or meta.get("calib_id"):
            calib_ids.add(getattr(m, "calib_id", None) or meta.get("calib_id"))
        if getattr(m, "depth_model_version", None):
            model_versions.add(m.depth_model_version)
        d95 = getattr(m, "depth_p95_mm", None)
        water = bool(getattr(m, "water_filled", False))
        rej = getattr(m, "reject_reason", None)
        rec.update(depth_p95_mm=d95, depth_sigma_mm=getattr(m, "depth_sigma_mm", None),
                   volume_l=getattr(m, "volume_l", None), volume_sigma=getattr(m, "volume_sigma", None),
                   area_m2=getattr(m, "area_m2", None), area_sigma=getattr(m, "area_sigma", None),
                   width_m=getattr(m, "width_m", None), measured_range_m=getattr(m, "range_m", None),
                   quality=getattr(m, "quality", None), scale_factor=getattr(m, "scale_factor", None),
                   water_filled=water, reject_reason=rej, certified=bool(cert),
                   measured_severity=_sev(d95, getattr(m, "width_m", None), water) if (d95 is not None or water)
                   else None,
                   warnings="; ".join((getattr(m, "extras", None) or {}).get("warnings", [])))
        if d95 is not None and truth["depth_p95_mm"] is not None:
            e = d95 - truth["depth_p95_mm"]
            s = rec["depth_sigma_mm"]
            rec.update(depth_err_mm=e, depth_within_2sigma=(abs(e) <= 2 * s) if s else None)
        if rec.get("volume_l") is not None and truth["volume_l"]:
            rec["volume_err_pct"] = 100.0 * (rec["volume_l"] / truth["volume_l"] - 1.0)
        if rec.get("area_m2") and truth["area_m2"]:
            rec["area_err_pct"] = 100.0 * (rec["area_m2"] / truth["area_m2"] - 1.0)
        rec["status"] = ("water_filled_no_depth" if water else f"rejected: {rej}" if rej
                         else "no_depth" if d95 is None else "measured")
        records.append(rec)
        if rej is not None or (getattr(m, "quality", 1.0) or 0.0) <= 0.0:
            continue  # production never uses a rejected frame: listed in the table, kept out of the statistics
        frame_pairs.append(TruthPair(truth=truth, measured=m, range_m=row["range_m"],
                                     meta={"capture_id": row["capture_id"]}))
        if isinstance(m, FrameMeasurement):
            groups.setdefault((row["pothole_ref"], row.get("pass_id") or row["capture_id"]), []).append((row, m))

    # fused passes (PLAN 4.8): every (target, pass_id) group with >= 2 frames
    fused_pairs: list[TruthPair] = []
    fused_rows: list[dict[str, Any]] = []
    for (ref, pid), items in groups.items():
        if len(items) < 2:
            continue
        pm = fuse_frames([m for _, m in items], cfg=cfg)
        r0 = items[0][0]
        truth = {k: r0[k] for k in ("depth_p95_mm", "volume_l", "area_m2", "width_m")}
        rng = min(r["range_m"] for r, _ in items)
        fused_pairs.append(TruthPair(truth=truth, measured=pm, range_m=rng, meta={"target": ref, "pass": pid}))
        fr = {"pothole_ref": ref, "pass_id": pid, "n_frames": pm.n_frames, "n_frames_total": pm.n_frames_total,
              "nearest_range_m": rng, "truth_depth_p95_mm": truth["depth_p95_mm"], "depth_p95_mm": pm.depth_p95_mm,
              "depth_sigma_mm": pm.depth_sigma_mm, "truth_volume_l": truth["volume_l"], "volume_l": pm.volume_l,
              "volume_sigma": pm.volume_sigma, "reject_reason": pm.reject_reason,
              "truth_severity": _sev(truth["depth_p95_mm"], truth["width_m"]),
              "measured_severity": _sev(pm.depth_p95_mm, pm.width_m, pm.water_filled)
              if pm.depth_p95_mm is not None or pm.water_filled else None}
        if pm.depth_p95_mm is not None:
            fr["depth_err_mm"] = pm.depth_p95_mm - truth["depth_p95_mm"]
        if pm.volume_l is not None and truth["volume_l"]:
            fr["volume_err_pct"] = 100.0 * (pm.volume_l / truth["volume_l"] - 1.0)
        fused_rows.append(fr)

    frame_stats = evaluate(frame_pairs) if frame_pairs else {}
    fused_stats = evaluate(fused_pairs) if fused_pairs else {}
    k_fit: Optional[dict[str, Any]] = None
    with_parts = [p for p in frame_pairs if isinstance(p.measured, FrameMeasurement)
                  and "sigma_parts" in p.measured.extras and p.measured.depth_p95_mm is not None]
    if len(with_parts) >= 3:
        try:
            k_fit = fit_k_sigma(with_parts)
        except Exception as e:  # noqa: BLE001 - informative only
            k_fit = {"error": f"{type(e).__name__}: {e}"}
    basis = "fused" if fused_stats.get("all", {}).get("depth_mae_mm") is not None else "frame"
    crit = _criteria((fused_stats if basis == "fused" else frame_stats).get("all"))
    n_meas = sum(1 for r in records if r["status"] != "not_measured" and not r["status"].startswith("unreadable"))
    summary = {
        "mode": mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv": str(csv_path),
        "n_rows": len(records),
        "n_measured": n_meas,
        "n_with_depth": sum(1 for r in records if r["status"] == "measured"),
        "n_water_or_rejected": sum(1 for r in records if r["status"].startswith(("water", "rejected", "no_depth"))),
        "n_not_measured": sum(1 for r in records if r["status"] == "not_measured"),
        "frame": frame_stats,
        "fused": fused_stats,
        "fused_passes": len(fused_pairs),
        "criteria_basis": basis,
        "criteria": crit,
        "k_fit": k_fit,
        "certified_all": bool(certified) and all(certified),
        "calib_ids": sorted(c for c in calib_ids if c),
        "depth_model_versions": sorted(model_versions),
        "validation_warnings": rep.warnings,
        "validation_errors": rep.errors,
        "notes": notes,
    }
    _write_outputs(out, records, fused_rows, summary)
    return summary


def _write_outputs(out: Path, records: list[dict[str, Any]], fused_rows: list[dict[str, Any]],
                   summary: dict[str, Any]) -> None:
    cols: list[str] = []
    for r in records:
        cols += [k for k in r if k not in cols]
    with open(out / "errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({k: _fmt(r.get(k)) for k in cols})
    if fused_rows:
        fcols: list[str] = []
        for r in fused_rows:
            fcols += [k for k in r if k not in fcols]
        with open(out / "fused.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fcols)
            w.writeheader()
            for r in fused_rows:
                w.writerow({k: _fmt(r.get(k)) for k in fcols})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
                                      encoding="utf-8")
    (out / "errors.md").write_text(render_markdown(records, fused_rows, summary), encoding="utf-8")


def _bin_key(label: str) -> tuple[float, str]:
    """Numeric order of measure.truth range-bin labels ('<3 m', '3-5 m', ..., '>=12 m', 'unknown')."""
    m = re.match(r"\s*(<|>=)?\s*([0-9.]+)", label)
    if not m:
        return (math.inf, label)
    v = float(m.group(2))
    return (v - 0.5 if m.group(1) == "<" else v + 0.5 if m.group(1) == ">=" else v, label)


def stats_table(stats: dict[str, Any]) -> list[str]:
    def f(v: Any, pct: bool = False, nd: int = 1) -> str:
        if v is None:
            return "-"
        return f"{100 * v:.{nd}f} %" if pct else f"{v:.{nd}f}"

    lines = ["| range bin | n | depth MAE mm | depth bias mm | volume MAPE | area MAPE | 2σ coverage | IRC class acc. |",
             "|---|---|---|---|---|---|---|---|"]
    order = sorted((k for k in stats if k != "all"), key=_bin_key)
    for k in [*order, "all"]:
        if k not in stats:
            continue
        s = stats[k]
        lines.append(f"| {k} | {s['n']} | {f(s['depth_mae_mm'])} | {f(s['depth_bias_mm'])} | "
                     f"{f(s['volume_mape'], True)} | {f(s['area_mape'], True)} | "
                     f"{f(s['depth_coverage_2sigma'], True, 0)} | {f(s['irc_class_accuracy'], True, 0)} |")
    return lines


def render_markdown(records: list[dict[str, Any]], fused_rows: list[dict[str, Any]], s: dict[str, Any]) -> str:
    title = {"staged": "Staged-target truth comparison",
             "synthetic_rehearsal": "Staged-target protocol REHEARSAL (synthetic renders, no camera)"}.get(
        s["mode"], f"Truth comparison ({s['mode']})")
    L = [f"# {title}", "", f"Generated {s['generated_at']} from `{s['csv']}`.", ""]
    if s["mode"] == "synthetic_rehearsal":
        L += ["**This is a dry run of the protocol and scripts on rendered targets (synthetic depth backend, "
              "production measurement path). It is not a staged-target result and says nothing about a real "
              "camera or depth model.**", ""]
    if not s["certified_all"]:
        L += ["Calibration: **uncertified** (`certified=false` in at least one measurement) - numbers are for "
              "validation only, never Measurement Book billing.", ""]
    L += [f"Rows {s['n_rows']}: measured {s['n_measured']} (with depth {s['n_with_depth']}, water-filled / "
          f"rejected / no depth {s['n_water_or_rejected']}), not measured yet {s['n_not_measured']}. "
          f"Fused passes: {s['fused_passes']}. Rejected frames (quality 0) are listed below but excluded from the "
          f"statistics, as in production.", ""]
    c = s["criteria"]
    L += [f"## PLAN 4.12 level-2 pass criteria (basis: {s['criteria_basis']})", "",
          "| criterion | value | target | pass |", "|---|---|---|---|"]
    for k in CRITERIA:
        v = c[k]["value"]
        vs = "-" if v is None else (f"{100 * v:.1f} %" if k != "depth_mae_mm" else f"{v:.1f} mm")
        L.append(f"| {k} | {vs} | {c[k]['target']} | {'-' if c[k]['pass'] is None else ('yes' if c[k]['pass'] else 'NO')} |")
    L += ["", f"Overall: **{'-' if c['overall'] is None else ('PASS' if c['overall'] else 'FAIL')}**", ""]
    if s["fused"]:
        L += ["## Fused passes (PLAN 4.8), binned by the nearest keyframe range", "", *stats_table(s["fused"]), ""]
    if s["frame"]:
        L += ["## Single frames, binned by tape range", "", *stats_table(s["frame"]), ""]
    kf = s.get("k_fit")
    if kf and "k_sigma" in kf:
        L += [f"k_sigma refit (`measure.truth.fit_k_sigma`, n={kf['n']}): {kf['k_sigma']:.3g} 1/m; 2σ coverage "
              f"{kf['coverage_2sigma_before']:.2f} -> {kf['coverage_2sigma_after']:.2f}"
              f"{' (floor-limited: errors smaller than the k-independent terms)' if kf.get('floor_limited') else ''}.",
              ""]
    L += ["## Per frame", "",
          "| capture | range m | truth p95 mm | measured p95 mm | err mm | in 2σ | truth V L | measured V L | V err % "
          "| area err % | IRC truth -> measured | q | status |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        def g(k: str, nd: int = 1, r: dict[str, Any] = r) -> str:
            v = r.get(k)
            return "-" if v is None else f"{v:+.{nd}f}" if k.endswith(("err_mm", "err_pct")) else f"{v:.{nd}f}"

        in2 = r.get("depth_within_2sigma")
        L.append(f"| {r['capture_id']} | {g('range_m')} | {g('truth_depth_p95_mm')} | "
                 f"{_fmt_pm(r.get('depth_p95_mm'), r.get('depth_sigma_mm'))} | {g('depth_err_mm')} | "
                 f"{'-' if in2 is None else ('yes' if in2 else 'no')} | {g('truth_volume_l', 2)} | "
                 f"{_fmt_pm(r.get('volume_l'), r.get('volume_sigma'), 2)} | {g('volume_err_pct')} | "
                 f"{g('area_err_pct')} | {r.get('truth_severity') or '-'} -> {r.get('measured_severity') or '-'} | "
                 f"{g('quality', 2)} | {r['status']} |")
    if s["notes"]:
        L += ["", "Notes:", *[f"- {n}" for n in s["notes"]]]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- rehearsal
def rehearse(csv_path: str | Path, out_dir: str | Path, *, noise_k: float = 4e-4, scale_sigma: float = 0.10,
             jitter: bool = True, seed: int = 0, limit: Optional[int] = None) -> dict[str, Any]:
    """Render every circular target row with ``measure.synthetic`` (cylindrical Box = foam cut-out) and measure
    it through ``measure.validate.measure_synthetic`` (production path, synthetic depth backend)."""
    import numpy as np

    from measure.config import load_measure_cfg
    from measure.irc import severity
    from measure.synthetic import Box, SceneSpec, add_depth_noise, jitter_mask, render
    from measure.validate import measure_synthetic

    rep = validate_csv(csv_path)
    if rep.errors:
        raise SystemExit("validation failed:\n  " + "\n  ".join(rep.errors))
    cfg = load_measure_cfg()
    out = Path(out_dir)
    (out / "measurements").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cache: dict[tuple, Any] = {}
    filled = []
    now = datetime.now(timezone.utc).isoformat()
    for i, row in enumerate(rep.rows):
        row = dict(row)
        if limit is not None and i >= limit:
            break
        if row["shape"] != "circle":
            row["notes"] = (row.get("notes") or "") + " | rehearsal: only circular targets are rendered"
            filled.append(row)
            continue
        h = row["camera_height_m"]
        pitch = row["pitch_deg"] if row["pitch_deg"] is not None else 12.0
        key = (h, pitch, row["range_m"], row["width_m"], row["depth_p95_mm"])
        if key not in cache:
            spec = SceneSpec(h=h, pitch_deg=pitch,
                             caps=[Box(0.0, row["range_m"], row["width_m"] / 2.0, row["depth_p95_mm"] / 1000.0)])
            cache[key] = (spec, render(spec))
        spec, r = cache[key]
        depth = r.depth
        if noise_k > 0:
            depth, _ = add_depth_noise(r.depth, noise_k, rng, mode="mixed", scale_sigma=scale_sigma)
        mask = jitter_mask(r.masks[0], rng) if jitter else r.masks[0]
        fm = measure_synthetic(spec, depth, mask, r.image, cfg)
        doc = {"source": "synthetic_rehearsal", "measurements": [fm.to_dict()],
               "severities": [severity(fm.depth_p95_mm, fm.width_m * 1000.0, fm.water_filled).value],
               "depth_model_version": fm.depth_model_version, "calib_id": fm.calib_id, "certified": fm.certified,
               "noise": {"k": noise_k, "scale_sigma": scale_sigma, "mask_jitter": jitter, "seed": seed}}
        rel = Path("measurements") / f"{row['capture_id']}.json"
        (out / rel).write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
        row.update(measurement_json=rel.as_posix(), measured_by=row.get("measured_by") or "synthetic-rehearsal",
                   measured_at=row.get("measured_at") or now, image="(rendered)")
        filled.append(row)
    write_csv(out / "filled.csv", filled)
    return compare(out / "filled.csv", out, mode="synthetic_rehearsal", cfg=cfg)


# ----------------------------------------------------------------------------- CLI
def _print_report(rep: Report, path: str | Path) -> None:
    print(f"{path}: {len(rep.rows)} rows, {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    for e in rep.errors:
        print(f"  ERROR   {e}")
    for w in rep.warnings[:50]:
        print(f"  warning {w}")
    if len(rep.warnings) > 50:
        print(f"  ... {len(rep.warnings) - 50} more warnings")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="staged_pothole_truth.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="write the staged-target CSV template")
    t.add_argument("--out", default="data/truth/template.csv")
    t.add_argument("--passes", type=int, default=2, help="approaches per target (fusion groups)")
    t.add_argument("--targets", default=",".join(x.ref for x in TARGETS), help="comma-separated target refs")
    t.add_argument("--camera-height", type=float, default=1.4, help="tape-measured lens height, m")
    t.add_argument("--pitch", type=float, default=12.0, help="certificate pitch, degrees")
    t.add_argument("--device-id", default="")
    v = sub.add_parser("validate", help="check a filled CSV")
    v.add_argument("csv")
    c = sub.add_parser("compare", help="truth vs measurement JSONs -> error tables")
    c.add_argument("csv")
    c.add_argument("--out", default="reports/staged_truth")
    c.add_argument("--force", action="store_true", help="compare even if validation reports errors")
    c.add_argument("--strict", action="store_true", help="exit 2 if the PLAN pass criteria are not met")
    r = sub.add_parser("rehearse", help="synthetic dry run of the protocol (no camera)")
    r.add_argument("csv", nargs="?", default="data/truth/template.csv")
    r.add_argument("--out", default="reports/staged_truth_rehearsal")
    r.add_argument("--noise-k", type=float, default=4e-4, help="sigma_Z = k Z^2 (0 = noise-free)")
    r.add_argument("--scale-sigma", type=float, default=0.10, help="global depth-scale error (height lock cancels it)")
    r.add_argument("--no-jitter", action="store_true", help="do not jitter the mask boundary")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=None, help="only the first N rows")
    a = ap.parse_args(argv)

    if a.cmd == "template":
        refs = [s.strip() for s in a.targets.split(",") if s.strip()]
        unknown = [x for x in refs if x not in {t.ref for t in TARGETS}]
        if unknown:
            ap.error(f"unknown target(s) {unknown}; known: {[t.ref for t in TARGETS]}")
        rows = make_template([t for t in TARGETS if t.ref in refs], a.passes, a.camera_height, a.pitch, a.device_id)
        write_csv(a.out, rows)
        print(f"wrote {a.out}: {len(rows)} rows ({len(refs)} targets x {a.passes} passes x their ranges)")
        return 0
    if a.cmd == "validate":
        rep = validate_csv(a.csv)
        _print_report(rep, a.csv)
        return 0 if rep.ok else 1
    if a.cmd == "compare":
        s = compare(a.csv, a.out, force=a.force)
    else:
        s = rehearse(a.csv, a.out, noise_k=a.noise_k, scale_sigma=a.scale_sigma, jitter=not a.no_jitter,
                     seed=a.seed, limit=a.limit)
    allm = (s["fused"] or s["frame"] or {}).get("all", {})
    print(f"[{s['mode']}] rows {s['n_rows']}, measured {s['n_measured']}, with depth {s['n_with_depth']}, "
          f"fused passes {s['fused_passes']}")
    if allm:
        print(f"  {s['criteria_basis']}: depth MAE {allm.get('depth_mae_mm')} mm, volume MAPE {allm.get('volume_mape')}, "
              f"IRC class accuracy {allm.get('irc_class_accuracy')}")
    print(f"  criteria overall: {s['criteria']['overall']}  -> {Path(a.out) / 'errors.md'}")
    if a.cmd == "compare" and a.strict and s["criteria"]["overall"] is not True:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
