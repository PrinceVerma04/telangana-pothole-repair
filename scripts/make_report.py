"""Validation report: reports/measure_demo + reports/model_eval.json + staged truth -> reports/REPORT.md (WP11).

Reads only files other steps already wrote; runs no model and needs no database:
  * ``reports/measure_demo/`` (``make measure-demo``): summary.json, synthetic_accuracy.csv/json,
    sigma_coverage.json, k_recovery.json, real/<image>.json
  * ``reports/model_eval.json`` (WP3 ``ml/eval.py``): any JSON; known metric names are matched against the PLAN
    targets, every scalar is listed. Missing -> the report says "pending" (and so does stdout).
  * ``reports/staged_truth/summary.json`` (``scripts/staged_pothole_truth.py compare``): the real-camera staged
    target table. Missing -> "pending". A ``synthetic_rehearsal`` summary is shown separately and labelled.

    python scripts/make_report.py [--out reports/REPORT.md]

Units as everywhere: depth mm, area m^2, volume L, lengths m; timestamps UTC (IST shown alongside).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
IST = timezone(timedelta(hours=5, minutes=30), "IST")
PENDING = "pending"

# PLAN §10 WP3 acceptance / §3.4 targets for reports/model_eval.json
MODEL_TARGETS: list[tuple[str, str, Any, str, float]] = [
    # (label, op, matcher, PLAN source, threshold)
    ("box mAP50, RDD-India D40 test", ">=", lambda k: "map50" in k and not _is_5095(k) and "mask" not in k,
     "WP3 hard floor (target 0.75)", 0.70),
    ("box mAP50-95", ">=", lambda k: _is_5095(k) and "mask" not in k, "PLAN 3.4 target", 0.45),
    ("mask IoU, eYantra test", ">=", lambda k: "iou" in k and "mask" in k, "WP3 floor (PLAN 3.4 target 0.70)", 0.65),
    ("FP per 1,000 frames (after verifier)", "<=", lambda k: "fp" in k and ("1k" in k or "1000" in k),
     "PLAN 3.4 target", 5.0),
]
SCALE_CLIP = (0.5, 2.0)        # configs/measure.yaml scale_lock.clip
Q_CONFIRM = 0.6                # CONTRACTS: 1 pass + IMU needs q >= 0.6
AREA_IMPLAUSIBLE_M2 = 2.0      # report heuristic: one pothole larger than this -> check the calibration
LENGTH_IMPLAUSIBLE_M = 3.0
DET_CONF_LOW = 0.25


def _rel(p: Path) -> str:
    """Repo-relative POSIX path for the report (absolute if outside the repo)."""
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _is_5095(k: str) -> bool:
    return bool(re.search(r"map_?50[-_]?95|map5095", k))


def load_json(p: Path) -> Optional[Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _f(v: Any, nd: int = 2, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if pct:
        return f"{100 * float(v):.{nd}f} %"
    return f"{float(v):.{nd}f}"


def _scale_clip(repo: Path) -> tuple[float, float]:
    try:
        import yaml

        c = yaml.safe_load((repo / "configs" / "measure.yaml").read_text(encoding="utf-8"))
        lo, hi = c["scale_lock"]["clip"]
        return float(lo), float(hi)
    except Exception:  # noqa: BLE001 - the documented default
        return SCALE_CLIP


# ----------------------------------------------------------------------------- measurement (synthetic)
def synthetic_section(md: Path) -> tuple[list[str], dict[str, Any]]:
    summ = load_json(md / "summary.json") or {}
    syn = summ.get("synthetic") or {}
    status: dict[str, Any] = {}
    L = ["## 2. Measurement core, synthetic validation (PLAN 4.12 level 1)", ""]
    rows: list[dict[str, str]] = []
    if (md / "synthetic_accuracy.csv").is_file():
        with open(md / "synthetic_accuracy.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not rows and not syn:
        L += [f"**{PENDING}**: run `make measure-demo` (writes `{_rel(md)}`).", ""]
        status["synthetic"] = PENDING
        return L, status
    if rows:
        n_pass = sum(r["pass_volume"] == "True" and r["pass_p95"] == "True" for r in rows)
        max_v = max(abs(float(r["volume_err_pct"])) for r in rows)
        max_p = max(abs(float(r["p95_err_mm"])) for r in rows)
        n2s = sum(r["within_2sigma"] == "True" for r in rows)
    else:
        n_pass, max_v, max_p, n2s = (syn.get("accuracy_pass"), syn.get("max_abs_volume_err_pct"),
                                     syn.get("max_abs_p95_err_mm"), syn.get("within_2sigma"))
    n = len(rows) or syn.get("accuracy_cases")
    status.update(synthetic_pass=n_pass, synthetic_n=n, max_volume_err_pct=max_v, max_p95_err_mm=max_p)
    L += ["Rendered road + cavity at known camera height/pitch; the depth map is the exact rendered Z, then the "
          "production path (`measure.pipeline.measure_frames`: plane + scale lock, ring quadratic, mask integral). "
          "Noise-free, so this checks the geometry, not a depth model.", "",
          f"**{n_pass}/{n} cases within tolerance**; max |volume error| {_f(max_v, 1)} %, max |p95 error| "
          f"{_f(max_p, 1)} mm; {n2s}/{n} within the reported 2σ.", ""]
    if rows:
        L += ["| case | truth L | measured L | err % | tol % | truth p95 mm | p95 mm | pass |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            ok = r["pass_volume"] == "True" and r["pass_p95"] == "True"
            L.append(f"| {r['case']} | {_f(r['truth_volume_l'])} | {_f(r['volume_l'])} ± {_f(r['volume_sigma_l'])} | "
                     f"{float(r['volume_err_pct']):+.1f} | {_f(r['tol_volume_pct'], 0)} | {_f(r['truth_p95_mm'], 1)} | "
                     f"{_f(r['p95_mm'], 1)} | {'yes' if ok else 'NO'} |")
        L.append("")
    cov = load_json(md / "sigma_coverage.json")
    if cov:
        L += [f"**σ coverage** ({cov.get('measured')}/{cov.get('trials')} noisy trials; noise k = {cov.get('k')} 1/m "
              f"'{cov.get('mode')}', global scale error σ {cov.get('scale_sigma')}, mask jitter "
              f"{cov.get('mask_jitter')}):", "",
              "| metric | 2σ coverage | 1σ coverage | bias | RMSE | mean σ | z_rms |", "|---|---|---|---|---|---|---|"]
        mins = []
        for k, m in (cov.get("metrics") or {}).items():
            mins.append(m.get("coverage_2sigma"))
            L.append(f"| {k} | {_f(m.get('coverage_2sigma'))} | {_f(m.get('coverage_1sigma'))} | {_f(m.get('bias'), 3)} | "
                     f"{_f(m.get('rmse'), 3)} | {_f(m.get('mean_sigma'), 3)} | {_f(m.get('z_rms'))} |")
        status["sigma_coverage_min"] = min((x for x in mins if x is not None), default=None)
        L += ["", "Target: 2σ coverage ≥ 0.90. A z_rms well below 1 means σ is conservative (reported σ larger than "
              "the actual error under this noise model); it is to be re-fitted per device from the staged and field "
              "truth sets (`measure.truth.fit_k_sigma`).", ""]
    kr = load_json(md / "k_recovery.json")
    if kr:
        status["k_rel_err"] = kr.get("rel_err")
        L += [f"**k recovery** (`measure.truth.fit_k_sigma`, n = {kr.get('n')}): injected {kr.get('k_true')}, fitted "
              f"{_f(kr.get('k_sigma'), 6)} 1/m (relative error {_f(kr.get('rel_err'), 1, pct=True)}); 2σ coverage "
              f"{_f(kr.get('coverage_2sigma_before'))} -> {_f(kr.get('coverage_2sigma_after'))}.", ""]
    return L, status


# ----------------------------------------------------------------------------- measurement (real images)
def real_section(md: Path, repo: Path) -> tuple[list[str], dict[str, Any]]:
    summ = load_json(md / "summary.json") or {}
    real = summ.get("real") or {}
    items = real.get("items") or []
    L = ["## 3. Real Indian road images (uncertified demo calibration)", ""]
    if not items:
        return L + [f"**{PENDING}**: no real-image results in `{_rel(md)}`.", ""], {"real": PENDING}
    lo, hi = _scale_clip(repo)
    L += [f"Calibration `{real.get('calib')}` (certified = false), detector `{real.get('weights')}`, depth backend "
          f"`{(summ.get('env') or {}).get('depth_backend')}` on {(summ.get('env') or {}).get('device', '?')}.", "",
          "**These numbers show that the chain runs end to end on real frames. They are not measurements.** The "
          "phone/dashcam geometry of these dataset images (true camera height, pitch, focal length) is unknown, so "
          "the uncertified `phone_demo` certificate is wrong for them. The pipeline detects this and says so itself "
          "(flags below).", "",
          "| image | mask | severity | depth p95 mm | area m² | volume L | range m | scale s | q | det conf | flags |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    flags_all: dict[str, int] = {}
    n_meas = n_flagged = 0
    for it in items:
        stem = Path(it["image"]).stem
        per = load_json(md / "real" / f"{stem}.json") or {}
        pm = per.get("measurements") or []
        for j, p in enumerate(it.get("potholes") or []):
            n_meas += 1
            full = pm[j] if j < len(pm) else {}
            ex = full.get("extras") or {}
            det = ex.get("detection") or {}
            flags = []
            if not it.get("certified", False):
                flags.append("uncertified")
            s = p.get("scale_factor")
            if s is not None and not lo <= s <= hi:
                flags.append(f"scale {s:.2f} outside [{lo:g}, {hi:g}]")
            if any("pitch_dev" in w for w in ex.get("warnings", [])):
                flags.append("pitch deviation")
            if (p.get("quality") or 0) < Q_CONFIRM:
                flags.append(f"q < {Q_CONFIRM}")
            if (p.get("area_m2") or 0) > AREA_IMPLAUSIBLE_M2 or (p.get("length_m") or 0) > LENGTH_IMPLAUSIBLE_M:
                flags.append("implausible size")
            if p.get("depth_p95_mm") == 0:
                flags.append("no cavity resolved")
            if det.get("det_conf") is not None and det["det_conf"] < DET_CONF_LOW:
                flags.append(f"det conf < {DET_CONF_LOW}")
            if p.get("water_filled"):
                flags.append("water-filled")
            n_flagged += bool(flags)
            for fl in flags:
                key = re.sub(r"[0-9.]+ outside", "outside", fl)
                flags_all[key] = flags_all.get(key, 0) + 1
            dc = det.get("det_conf")
            L.append(f"| {Path(it['image']).name} | {it.get('mask_source')} | {p.get('severity')} | "
                     f"{_f(p.get('depth_p95_mm'), 1)} ± {_f(p.get('depth_sigma_mm'), 1)} | "
                     f"{_f(p.get('area_m2'), 3)} ± {_f(p.get('area_sigma'), 3)} | {_f(p.get('volume_l'), 1)} ± "
                     f"{_f(p.get('volume_sigma'), 1)} | {_f(p.get('range_m'), 1)} | {_f(s)} | {_f(p.get('quality'))} | "
                     f"{_f(dc) if dc is not None else 'label'} | {', '.join(flags)} |")
    L += ["", f"Flag counts over {n_meas} measurements: "
          + ", ".join(f"{k} {v}/{n_meas}" for k, v in sorted(flags_all.items(), key=lambda kv: -kv[1])) + ".", "",
          f"Heuristics used for the flags: scale outside the `scale_lock.clip` [{lo:g}, {hi:g}] means the "
          f"certificate's height/intrinsics do not fit the image (a certified device would reject the frame); "
          f"q < {Q_CONFIRM} is below the single-pass confirmation threshold; 'implausible size' = area > "
          f"{AREA_IMPLAUSIBLE_M2:g} m² or length > {LENGTH_IMPLAUSIBLE_M:g} m for one pothole. 'label' in the det "
          f"conf column means the bootstrap detector found nothing usable and the dataset outline was used as the mask.",
          ""]
    return L, {"real_measurements": n_meas, "real_flagged": n_flagged, "real_flags": flags_all}


# ----------------------------------------------------------------------------- model metrics
def _flatten(d: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list):
        if all(not isinstance(x, (dict, list)) for x in d) and len(d) <= 8:
            out.append((prefix, ", ".join(str(x) for x in d)))
        else:
            for i, v in enumerate(d[:20]):
                out += _flatten(v, f"{prefix}[{i}]")
    else:
        out.append((prefix, d))
    return out


# Keys that describe targets, thresholds or settings, never a measured value (e.g. ``targets.mask_iou.floor``,
# ``iou_nms``, ``conf``). The generic matcher must not report them as model results.
_NOT_A_RESULT = re.compile(r"(^|\.)(targets?|acceptance|thresholds?|criteria)(\.|\[|$)|floor|max_abs|"
                           r"(^|\.)(conf|iou_nms|imgsz|max_det|limit)$|_ok$")


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _gm_runs_section(doc: dict[str, Any], path: Path) -> tuple[list[str], dict[str, Any]]:
    """``gunthameter.model_eval/1`` (ml/eval.py): runs[] + targets{} + acceptance{} (WP3)."""
    runs = [r for r in doc.get("runs") or [] if isinstance(r, dict)]
    acc = doc.get("acceptance") if isinstance(doc.get("acceptance"), dict) else {}
    L = [f"Source: `{_rel(path)}` (schema `{doc.get('schema')}`, updated {doc.get('updated_at', '-')}).", ""]
    st: dict[str, Any] = {"model": "present", "model_targets": {}, "model_runs": len(runs)}
    if runs:
        L += ["| stage | weights | split (n images) | device | pothole n_gt | pothole box AP50 | AP50-95 | "
              "P / R @ conf 0.25 | best F1 | mask IoU | ms/img mean |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        mt = r.get("metrics") or {}
        pot = mt.get("pothole") or {}
        mask = mt.get("mask") or {}
        tim = mt.get("timing") or {}
        iou = mask.get("pothole_mean_matched_mask_iou")
        iou_s = _f(iou, 3) if iou is not None else (mask.get("status") or "-")
        split = f"{r.get('split', '-')}{' (limit ' + str(r['limit']) + ')' if r.get('limit') else ''}"
        L.append(f"| {r.get('stage', '-')} | `{r.get('weights', '-')}` | {split} ({mt.get('n_images', '-')}) | "
                 f"{r.get('device_name') or r.get('device') or '-'} | {pot.get('n_gt', '-')} | "
                 f"{_f(pot.get('box_ap50'), 4)} | {_f(pot.get('box_ap50_95'), 4)} | "
                 f"{_f(pot.get('p_at_025'), 3)} / {_f(pot.get('r_at_025'), 3)} | {_f(pot.get('best_f1'), 3)} | "
                 f"{iou_s} | {_f(tim.get('ms_per_image_mean'), 0)} |")
    if runs:
        notes = [f"- `{r.get('stage')}` / `{r.get('weights')}`: {n}" for r in runs for n in (r.get("notes") or [])]
        L += ["", "Run notes (from the file):", *notes, ""] if notes else [""]

    # acceptance: only a full test-split evaluation of a *trained* model counts (ml/eval.py acceptance()).
    evaluated = acc.get("evaluated_on")
    ev_run = None
    if evaluated:
        ev_run = next((r for r in runs if f"{r.get('stage')} ({r.get('weights')})" == evaluated), None)
    ev_mt = (ev_run or {}).get("metrics") or {}
    ev_vals = {
        "box mAP50, RDD-India D40 test": _num(acc.get("box_map50_pothole")),
        "box mAP50-95": _num((ev_mt.get("pothole") or {}).get("box_ap50_95")),
        "mask IoU, eYantra test": _num(acc.get("mask_iou_pothole")),
        "FP per 1,000 frames (after verifier)": next((_num(v) for k, v in _flatten(ev_mt)
                                                      if "fp" in k.lower() and ("1k" in k or "1000" in k)
                                                      and _num(v) is not None), None),
    }
    # best baseline (bootstrap / smoke) pothole AP50, shown for context and never as the acceptance value
    base = [(r, _num(((r.get("metrics") or {}).get("pothole") or {}).get("box_ap50"))) for r in runs
            if not r.get("limit") and r.get("split") == "test"]
    base = [(r, v) for r, v in base if v is not None]
    if base:
        br, bv = max(base, key=lambda x: x[1])
        st["model_best_run"] = {"stage": br.get("stage"), "weights": br.get("weights"), "box_ap50_pothole": bv}
    L += ["PLAN WP3 acceptance (evaluated by `ml/eval.py` only on a full test-split run of a trained, non-baseline "
          f"model): **{acc.get('status', 'not stated in file')}**"
          + (f"; evaluated on `{evaluated}`." if evaluated else "."), "",
          "| metric | target | value | status |", "|---|---|---|---|"]
    for lab, op, _, src, thr in MODEL_TARGETS:
        v = ev_vals.get(lab)
        if not evaluated:
            st["model_targets"][lab] = None
            ctx = ""
            if lab.startswith("box mAP50,") and "model_best_run" in st:
                b = st["model_best_run"]
                ctx = f" (best baseline so far: {b['box_ap50_pothole']:.4f}, `{b['weights']}`, stage {b['stage']})"
            L.append(f"| {lab} | {op} {thr:g} ({src}) | -{ctx} | {PENDING}: no trained model evaluated |")
            continue
        if v is None:
            st["model_targets"][lab] = None
            L.append(f"| {lab} | {op} {thr:g} ({src}) | not in file | - |")
            continue
        ok = v >= thr if op == ">=" else v <= thr
        st["model_targets"][lab] = {"key": f"acceptance/{evaluated}", "value": v, "pass": ok}
        L.append(f"| {lab} | {op} {thr:g} ({src}) | {v:.3f} | {'meets' if ok else 'BELOW TARGET'} |")
    st["model_acceptance"] = acc.get("status")
    st["model_evaluated_on"] = evaluated
    tg = doc.get("targets")
    if isinstance(tg, dict) and tg:
        L += ["", "Targets as recorded in the file (thresholds, not results): "
              + "; ".join(f"`{k}` " + ", ".join(f"{kk} {vv}" for kk, vv in v.items() if kk != "source")
                          for k, v in tg.items() if isinstance(v, dict)) + "."]
    return L + [""], st


def model_section(path: Path) -> tuple[list[str], dict[str, Any]]:
    L = ["## 4. Detector / segmenter metrics (`reports/model_eval.json`, WP3)", ""]
    doc = load_json(path)
    if doc is None:
        L += [f"**{PENDING}**: `{_rel(path)}` does not exist yet (WP3 `ml/eval.py` writes it). No model numbers "
              "are claimed anywhere in this report until it does.", "",
              "| metric | target | value | status |", "|---|---|---|---|"]
        L += [f"| {lab} | {op} {thr:g} ({src}) | - | {PENDING} |" for lab, op, _, src, thr in MODEL_TARGETS]
        return L + [""], {"model": PENDING}
    if isinstance(doc, dict) and isinstance(doc.get("runs"), list):
        S, st = _gm_runs_section(doc, path)
        return L + S, st
    flat = [(k, v) for k, v in _flatten(doc) if not _NOT_A_RESULT.search(k.lower())]
    st: dict[str, Any] = {"model": "present", "model_targets": {}}
    L += [f"Source: `{_rel(path)}`.", "", "| metric | target | value (key) | status |", "|---|---|---|---|"]
    for lab, op, match, src, thr in MODEL_TARGETS:
        hit = next(((k, v) for k, v in flat if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and match(re.sub(r"[^a-z0-9_.\-]", "", k.lower()))), None)
        if hit is None:
            L.append(f"| {lab} | {op} {thr:g} ({src}) | not in file | - |")
            st["model_targets"][lab] = None
            continue
        k, v = hit
        ok = v >= thr if op == ">=" else v <= thr
        st["model_targets"][lab] = {"key": k, "value": v, "pass": ok}
        L.append(f"| {lab} | {op} {thr:g} ({src}) | {v:.3f} (`{k}`) | {'meets' if ok else 'BELOW TARGET'} |")
    scal = [(k, v) for k, v in flat if isinstance(v, (int, float, str, bool)) and (not isinstance(v, str) or len(v) <= 100)]
    L += ["", f"All scalar entries ({min(len(scal), 80)} of {len(scal)} shown):", "", "| key | value |", "|---|---|"]
    for k, v in scal[:80]:
        vs = f"{v:.4g}" if isinstance(v, float) else str(v)
        L.append(f"| `{k}` | {vs.replace('|', '/')} |")
    return L + [""], st


# ----------------------------------------------------------------------------- staged truth
def _staged_tables(s: dict[str, Any]) -> list[str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from staged_pothole_truth import CRITERIA, stats_table
    finally:
        sys.path.pop(0)
    c = s.get("criteria") or {}
    L = [f"Pass criteria (PLAN 4.12 level 2, basis: {s.get('criteria_basis')}):", "",
         "| criterion | value | target | pass |", "|---|---|---|---|"]
    for k in CRITERIA:
        e = c.get(k) or {}
        v = e.get("value")
        vs = "-" if v is None else (f"{v:.1f} mm" if k == "depth_mae_mm" else f"{100 * v:.1f} %")
        ok = e.get("pass")
        L.append(f"| {k} | {vs} | {e.get('target', '-')} | {'-' if ok is None else ('yes' if ok else 'NO')} |")
    L += ["", f"Overall: **{'-' if c.get('overall') is None else ('PASS' if c['overall'] else 'FAIL')}** "
          f"({s.get('n_rows')} rows, {s.get('n_with_depth')} with depth, {s.get('fused_passes')} fused passes; "
          f"calibration certified: {s.get('certified_all')}).", ""]
    if s.get("fused"):
        L += ["Fused passes:", "", *stats_table(s["fused"]), ""]
    if s.get("frame"):
        L += ["Single frames:", "", *stats_table(s["frame"]), ""]
    return L


def staged_section(path: Path, rehearsal: Optional[Path]) -> tuple[list[str], dict[str, Any]]:
    L = ["## 5. Staged known-depth targets (PLAN 4.12 level 2)", ""]
    s = load_json(path)
    st: dict[str, Any] = {}
    if s is None or s.get("mode") == "synthetic_rehearsal":
        L += [f"**{PENDING}**: no real-camera staged-target campaign has been run yet. Protocol: "
              "`docs/CALIBRATION_SOP.md` section 5 and `data/truth/README.md`; template `data/truth/template.csv`; "
              f"`python scripts/staged_pothole_truth.py compare <filled.csv> --out {_rel(path.parent)}` writes "
              "the table this section shows.", ""]
        st["staged"] = PENDING
    else:
        L += [f"Source: `{_rel(path)}` (generated {s.get('generated_at')}).", "", *_staged_tables(s)]
        st["staged"] = (s.get("criteria") or {}).get("overall")
    r = load_json(rehearsal) if rehearsal else None
    if r is not None and r.get("mode") == "synthetic_rehearsal":
        L += ["### Protocol rehearsal (synthetic renders - NOT a staged-target result)", "",
              f"`staged_pothole_truth.py rehearse` rendered the template's foam targets with the synthetic depth "
              f"backend and pushed them through the production measurement path and the comparison script "
              f"(source `{_rel(rehearsal)}`, generated {r.get('generated_at')}). It proves the protocol and the "
              "scripts work and shows the purely geometric limits (pixel count, near-wall occlusion); it says nothing "
              "about a real camera or depth model.", "", *_staged_tables(r)]
        st["rehearsal"] = (r.get("criteria") or {}).get("overall")
    return L, st


# ----------------------------------------------------------------------------- report
def build_report(measure_dir: Path, model_eval: Path, truth_summary: Path, rehearsal: Optional[Path],
                 repo: Path = REPO) -> tuple[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    syn_L, syn_s = synthetic_section(measure_dir)
    real_L, real_s = real_section(measure_dir, repo)
    mod_L, mod_s = model_section(model_eval)
    stg_L, stg_s = staged_section(truth_summary, rehearsal)
    status = {**syn_s, **real_s, **mod_s, **stg_s}

    def pf(ok: Optional[bool]) -> str:
        return PENDING if ok is None else ("PASS" if ok else "FAIL")

    syn_ok = None if "synthetic_pass" not in syn_s else syn_s["synthetic_pass"] == syn_s["synthetic_n"]
    cov = syn_s.get("sigma_coverage_min")
    staged = stg_s.get("staged")
    mt = mod_s.get("model_targets") or {}

    def model_cell(label: str) -> tuple[str, str]:
        e = mt.get(label)
        if mod_s.get("model") == PENDING:
            return "-", PENDING
        if e is None:
            if "model_runs" in mod_s and not mod_s.get("model_evaluated_on"):
                b = mod_s.get("model_best_run")
                ctx = (f"baseline {b['box_ap50_pothole']:.4f} (`{Path(str(b['weights'])).name}`)"
                       if b and label.startswith("box mAP50,") else "-")
                return ctx, f"{PENDING} (no trained model)"
            return "not in file", "-"
        return f"{e['value']:.3f}", pf(e["pass"])

    head = ["# RoadFix validation report", "",
            f"Generated {now.isoformat(timespec='seconds')} (UTC) = {now.astimezone(IST).strftime('%Y-%m-%d %H:%M')} "
            "IST by `scripts/make_report.py`. Every number below is copied from an artefact file named in its "
            "section; nothing is typed in by hand.", "",
            "**Nothing here is billing-grade.** All calibrations used so far are `certified=false` demo certificates; "
            "certified numbers need the depot calibration (docs/CALIBRATION_SOP.md) and a truth fit per device.", "",
            "## 1. Status against the PLAN targets", "",
            "| check | target | result | status |", "|---|---|---|---|"]
    rows = [
        ("Synthetic volume (noise-free)", "±3 % plane, ±5 % crown, ±6 % box",
         f"{syn_s.get('synthetic_pass', '-')}/{syn_s.get('synthetic_n', '-')} cases, max {_f(syn_s.get('max_volume_err_pct'), 1)} %",
         pf(syn_ok)),
        ("Synthetic depth p95 (noise-free)", "±2 mm", f"max {_f(syn_s.get('max_p95_err_mm'), 1)} mm",
         pf(None if syn_s.get("max_p95_err_mm") is None else syn_s["max_p95_err_mm"] <= 2.0)),
        ("σ coverage (noisy synthetic)", "2σ coverage ≥ 0.90 (depth, volume, area, width)",
         f"min {_f(cov)}", pf(None if cov is None else cov >= 0.90)),
        ("k_sigma recovery", "informative", _f(syn_s.get("k_rel_err"), 1, pct=True) + " relative error", "-"),
    ]
    sg = None if staged in (None, PENDING) else staged
    rows += [("Staged targets: fused depth MAE / IRC class / volume MAPE", "≤ 10 mm / ≥ 90 % / ≤ 25 %",
              "see section 5" if sg is not None else "-", pf(sg) if staged != PENDING else PENDING)]
    for lab, op, _, src, thr in MODEL_TARGETS:
        val, stt = model_cell(lab)
        rows.append((lab, f"{op} {thr:g} ({src})", val, stt))
    rows.append(("Real images", "chain runs end to end; not a measurement",
                 f"{real_s.get('real_measurements', 0)} measurements, {real_s.get('real_flagged', 0)} flagged "
                 "(section 3)", "demo only"))
    head += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    head += [""]
    text = "\n".join(head + syn_L + real_L + mod_L + stg_L) + "\n"
    return text, status


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="make_report.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/REPORT.md")
    ap.add_argument("--measure-dir", default="reports/measure_demo")
    ap.add_argument("--model-eval", default="reports/model_eval.json")
    ap.add_argument("--truth-summary", default="reports/staged_truth/summary.json")
    ap.add_argument("--rehearsal-summary", default="reports/staged_truth_rehearsal/summary.json",
                    help="synthetic rehearsal summary (shown separately, labelled); '' to omit")
    a = ap.parse_args(argv)

    def rp(p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else Path.cwd() / q

    text, st = build_report(rp(a.measure_dir), rp(a.model_eval), rp(a.truth_summary),
                            rp(a.rehearsal_summary) if a.rehearsal_summary else None)
    out = rp(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    syn = st.get("synthetic")
    print(f"  synthetic: {syn if syn == PENDING else str(st.get('synthetic_pass')) + '/' + str(st.get('synthetic_n'))}")
    if st.get("model") == PENDING:
        print(f"  model metrics: {PENDING}")
    elif "model_runs" in st and not st.get("model_evaluated_on"):
        print(f"  model metrics: {PENDING} (acceptance: {st.get('model_acceptance')}; {st['model_runs']} baseline/"
              f"smoke run(s) listed from {a.model_eval})")
    else:
        print(f"  model metrics: from {a.model_eval}")
    print(f"  staged truth: {PENDING if st.get('staged') == PENDING else st.get('staged')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
