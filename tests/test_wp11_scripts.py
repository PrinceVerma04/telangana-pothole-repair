"""WP11: scripts/staged_pothole_truth.py and scripts/make_report.py on small synthetic inputs (tmp_path only).

No model weights, no database, no geo files: measurements are hand-built FrameMeasurement JSONs or come from the
synthetic depth backend (measure/synthetic.py) through the production measurement path.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from measure.truth import TRUTH_CSV_COLUMNS, load_truth_csv
from measure.types import DepthQuality, FrameMeasurement

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"wp11_{name}", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


spt = _load("staged_pothole_truth")
mr = _load("make_report")


def _target(ref: str):
    return next(t for t in spt.TARGETS if t.ref == ref)


def _rows_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- template / validate
def test_template_roundtrip(tmp_path: Path):
    out = tmp_path / "template.csv"
    assert spt.main(["template", "--out", str(out), "--passes", "1"]) == 0
    rows = _rows_csv(out)
    assert len(rows) == sum(len(t.ranges_m) for t in spt.TARGETS)
    assert list(rows[0].keys()) == spt.COLUMNS
    assert set(TRUTH_CSV_COLUMNS) <= set(rows[0].keys())
    # the truth columns stay readable by measure.truth (and therefore by POST /truth/measurements users)
    truth = load_truth_csv(out)
    assert truth[0]["method"] == "manual_gauge" and truth[0]["depth_p95_mm"] == 20.0
    rep = spt.validate_csv(out)
    assert rep.ok, rep.errors
    assert any("measured_by empty" in w for w in rep.warnings)  # nominal truth is not a gauge reading yet
    t = _target("T40-W")
    r = next(x for x in rows if x["pothole_ref"] == "T40-W")
    assert float(r["area_m2"]) == pytest.approx(t.area_m2, rel=1e-3)
    assert float(r["volume_l"]) == pytest.approx(t.area_m2 * 40.0, rel=1e-3)  # 1 m2 x 1 mm = 1 L


def test_validate_catches_unit_and_consistency_mistakes(tmp_path: Path):
    rows = spt.make_template([_target("T40-M")], passes=1)
    rows[0].update(depth_p95_mm=0.04, depth_max_mm=0.04)          # metres instead of mm
    rows[1].update(width_m=450.0)                                 # mm instead of m
    rows[2].update(measured_at="2026-11-20 10:00:00", measured_by="AE Kapra")  # no UTC offset
    rows[3].update(range_m=25.0, area_m2=1590.0)                  # out of range, cm2
    p = tmp_path / "bad.csv"
    spt.write_csv(p, rows)
    rep = spt.validate_csv(p)
    text = "\n".join(rep.errors)
    assert "looks like metres" in text
    assert "width_m=450" in text and "looks like mm/cm" in text
    assert "no UTC offset" in text
    assert "range_m=25" in text
    assert "area_m2=1590" in text
    assert "one target = one truth" in text                      # same target, different truth
    assert spt.main(["validate", str(p)]) == 1

    good = spt.make_template([_target("T40-M")], passes=1)
    good[0].update(measured_at="2026-11-20T10:00:00+05:30", measured_by="AE Kapra", lat=17.44, lon=78.39)
    spt.write_csv(p, good)
    rep = spt.validate_csv(p)
    assert rep.ok, rep.errors
    assert rep.rows[0]["measured_at"] == "2026-11-20T04:30:00+00:00"  # stored in UTC


# ----------------------------------------------------------------------------- compare
def _fm(depth: float | None, *, range_m: float, area: float, vol: float | None, width: float = 0.45,
        water: bool = False) -> FrameMeasurement:
    return FrameMeasurement(
        depth_p95_mm=depth, depth_max_mm=depth, depth_mean_mm=depth, depth_sigma_mm=None if water else 4.0,
        area_m2=area, area_sigma=0.02 * area, volume_l=vol, volume_sigma=None if vol is None else 0.15 * vol,
        length_m=width, width_m=width, length_sigma_m=0.01, width_sigma_m=0.01, water_filled=water,
        depth_quality=DepthQuality.unknown_water if water else DepthQuality.ok, quality=0.9, range_m=range_m,
        lane_offset_m=0.0, scale_factor=1.0, calib_id="test_cam", certified=False, depth_model_version="test")


def _write_resp(path: Path, fms: list[FrameMeasurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"measurements": [f.to_dict() for f in fms], "severities": [None] * len(fms),
                                "depth_model_version": "test", "calib_id": "test_cam", "certified": False,
                                "elapsed_ms": 0.0}), encoding="utf-8")


def _staged_case(tmp_path: Path) -> Path:
    t = _target("T40-M")  # 40 mm x 0.45 m, ranges 3/5/7/9 m, IRC medium
    rows = spt.make_template([t], passes=2)
    errs = {"T40-M-P1-R3": 2.0, "T40-M-P1-R5": -4.0, "T40-M-P1-R7": 6.0, "T40-M-P1-R9": "water",
            "T40-M-P2-R3": 1.0, "T40-M-P2-R5": -1.0, "T40-M-P2-R7": 3.0, "T40-M-P2-R9": None}
    for r in rows:
        e = errs[r["capture_id"]]
        r.update(measured_by="test", measured_at="2026-11-20T04:30:00Z")
        if e is None:
            continue  # not measured yet
        rel = f"m/{r['capture_id']}.json"
        if e == "water":
            fm = _fm(None, range_m=r["range_m"], area=t.area_m2, vol=None, water=True)
        else:
            fm = _fm(40.0 + e, range_m=r["range_m"], area=t.area_m2 * 1.05, vol=t.volume_l * 1.10)
        _write_resp(tmp_path / rel, [fm])
        r["measurement_json"] = rel
    p = tmp_path / "filled.csv"
    spt.write_csv(p, rows)
    return p


def test_compare_error_table(tmp_path: Path):
    csv_path = _staged_case(tmp_path)
    out = tmp_path / "staged_truth"
    s = spt.compare(csv_path, out)
    assert s["mode"] == "staged"
    assert (s["n_rows"], s["n_measured"], s["n_with_depth"], s["n_not_measured"]) == (8, 7, 6, 1)
    assert s["n_water_or_rejected"] == 1
    fa = s["frame"]["all"]
    assert fa["n"] == 7                                            # the water frame is a pair, but has no depth
    assert fa["depth_mae_mm"] == pytest.approx((2 + 4 + 6 + 1 + 1 + 3) / 6)
    assert fa["depth_bias_mm"] == pytest.approx((2 - 4 + 6 + 1 - 1 + 3) / 6)
    assert fa["volume_mape"] == pytest.approx(0.10, abs=1e-3)  # template truth is rounded to 0.001 L
    assert fa["area_mape"] == pytest.approx(6 * 0.05 / 7, abs=1e-3)  # the water frame keeps its (true) area
    assert fa["depth_coverage_2sigma"] == 1.0                      # every |err| <= 2 x 4 mm
    assert fa["irc_class_accuracy"] == 1.0                          # medium stays medium
    assert set(s["frame"]) >= {"3-5 m", "5-7 m", "7-10 m", "all"}
    assert s["fused_passes"] == 2 and s["criteria_basis"] == "fused"
    assert s["fused"]["all"]["n"] == 2 and s["fused"]["all"]["depth_mae_mm"] < 5.0
    assert s["criteria"]["depth_mae_mm"]["pass"] is True
    assert s["criteria"]["overall"] is True
    assert s["certified_all"] is False and s["calib_ids"] == ["test_cam"]
    # the table files
    rows = _rows_csv(out / "errors.csv")
    by_id = {r["capture_id"]: r for r in rows}
    assert by_id["T40-M-P1-R7"]["depth_err_mm"] == "6"
    assert by_id["T40-M-P1-R9"]["status"] == "water_filled_no_depth"
    assert by_id["T40-M-P2-R9"]["status"] == "not_measured"
    assert len(_rows_csv(out / "fused.csv")) == 2
    md = (out / "errors.md").read_text(encoding="utf-8")
    assert "Staged-target truth comparison" in md and "uncertified" in md and "| T40-M-P1-R5 |" in md
    assert md.index("| 3-5 m |") < md.index("| 7-10 m |")       # range bins in numeric order
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["n_rows"] == 8


def test_compare_refuses_invalid_csv(tmp_path: Path):
    rows = spt.make_template([_target("T20-S")], passes=1)
    rows[0]["depth_p95_mm"] = -5
    p = tmp_path / "bad.csv"
    spt.write_csv(p, rows)
    with pytest.raises(SystemExit, match="validation failed"):
        spt.compare(p, tmp_path / "out")


def test_load_measurement_accepts_list_dict_and_unknown_keys(tmp_path: Path):
    fm = _fm(33.0, range_m=5.0, area=0.2, vol=6.0).to_dict()
    p = tmp_path / "x.json"
    p.write_text(json.dumps([fm, fm]), encoding="utf-8")
    m, meta = spt.load_measurement(p, 1)
    assert m.depth_p95_mm == 33.0 and meta["n_in_file"] == 2
    p.write_text(json.dumps({**fm, "some_future_field": 1}), encoding="utf-8")
    m, _ = spt.load_measurement(p)
    assert isinstance(m, FrameMeasurement) and m.volume_l == 6.0
    with pytest.raises(ValueError, match="detection_index"):
        spt.load_measurement(p, 3)


def test_rehearsal_noise_free_through_production_path(tmp_path: Path):
    t = _target("T70-L")
    rows = [r for r in spt.make_template([t], passes=1) if r["range_m"] in (3.0, 5.0)]
    p = tmp_path / "plan.csv"
    spt.write_csv(p, rows)
    s = spt.rehearse(p, tmp_path / "reh", noise_k=0.0, scale_sigma=0.0, jitter=False)
    assert s["mode"] == "synthetic_rehearsal"
    assert s["n_with_depth"] == 2 and s["fused_passes"] == 1
    fa = s["frame"]["all"]
    assert fa["depth_mae_mm"] < 1.0           # flat floor at 70 mm, exact rendered depth
    assert fa["volume_mape"] < 0.06            # box (vertical walls) tolerance as in measure-demo
    assert fa["irc_class_accuracy"] == 1.0
    md = (tmp_path / "reh" / "errors.md").read_text(encoding="utf-8")
    assert "REHEARSAL" in md and "not a staged-target result" in md
    filled = _rows_csv(tmp_path / "reh" / "filled.csv")
    assert all((tmp_path / "reh" / r["measurement_json"]).is_file() for r in filled)


# ----------------------------------------------------------------------------- make_report
def _measure_dir(tmp_path: Path) -> Path:
    md = tmp_path / "measure_demo"
    (md / "real").mkdir(parents=True)
    cols = ["case", "truth_volume_l", "volume_l", "volume_err_pct", "volume_sigma_l", "truth_p95_mm", "p95_mm",
            "p95_err_mm", "tol_volume_pct", "pass_volume", "pass_p95", "within_2sigma"]
    with open(md / "synthetic_accuracy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(dict(zip(cols, ["car cap @ 7 m", 12.63, 12.73, 0.74, 2.5, 47.5, 47.6, 0.06, 3.0, True, True, True],
                            strict=True)))
        w.writerow(dict(zip(cols, ["bus box @ 7 m", 23.09, 22.0, -4.7, 3.9, 60.0, 60.0, 0.0, 3.0, False, True, True],
                            strict=True)))
    (md / "sigma_coverage.json").write_text(json.dumps({
        "trials": 10, "measured": 10, "k": 4e-4, "mode": "mixed", "scale_sigma": 0.1, "mask_jitter": True,
        "metrics": {"depth_p95_mm": {"coverage_2sigma": 0.95, "coverage_1sigma": 0.8, "bias": 1.0, "rmse": 3.0,
                                     "mean_sigma": 6.0, "z_rms": 0.5}}}), encoding="utf-8")
    (md / "k_recovery.json").write_text(json.dumps({"k_sigma": 5.2e-4, "n": 80, "k_true": 6e-4, "rel_err": 0.13,
                                                    "coverage_2sigma_before": 0.89, "coverage_2sigma_after": 0.95}),
                                        encoding="utf-8")
    pothole = {"depth_p95_mm": 28.6, "depth_sigma_mm": 5.7, "area_m2": 6.7, "area_sigma": 0.4, "volume_l": 38.2,
               "volume_sigma": 38.5, "length_m": 2.9, "width_m": 2.3, "range_m": 6.5, "quality": 0.3,
               "water_filled": False, "scale_factor": 0.24, "severity": "large"}
    (md / "summary.json").write_text(json.dumps({
        "env": {"device": "CPU", "depth_backend": "da2_metric"},
        "real": {"calib": "configs/cameras/phone_demo.json", "weights": "w.pt",
                 "items": [{"image": "x/img1.jpg", "mask_source": "detector", "certified": False,
                            "potholes": [pothole]}]}}), encoding="utf-8")
    (md / "real" / "img1.json").write_text(json.dumps({"measurements": [
        {"extras": {"warnings": ["pitch_dev 16.3 deg > 4.0"], "detection": {"det_conf": 0.2}}}]}), encoding="utf-8")
    return md


def test_make_report_pending_without_model_eval_and_truth(tmp_path: Path):
    md = _measure_dir(tmp_path)
    text, st = mr.build_report(md, tmp_path / "no_model_eval.json", tmp_path / "no_truth.json", None)
    assert st["model"] == "pending" and st["staged"] == "pending"
    assert st["synthetic_pass"] == 1 and st["synthetic_n"] == 2
    assert "1/2 cases within tolerance" in text
    assert "| box mAP50, RDD-India D40 test | >= 0.7" in text and "| - | pending |" in text
    assert "no real-camera staged-target campaign" in text
    flags = st["real_flags"]
    assert flags["uncertified"] == 1 and flags["implausible size"] == 1 and flags["pitch deviation"] == 1
    assert any(k.startswith("scale") for k in flags) and any(k.startswith("det conf") for k in flags)
    assert "Nothing here is billing-grade" in text and "IST" in text


def test_make_report_reads_model_eval_and_staged_summary(tmp_path: Path):
    md = _measure_dir(tmp_path)
    me = tmp_path / "model_eval.json"
    me.write_text(json.dumps({"weights": "runs/y26n/best.pt", "box": {"mAP50": 0.71, "mAP50-95": 0.40},
                              "mask": {"mean_iou": 0.60}, "fp_per_1k_frames": 3.2}), encoding="utf-8")
    csv_path = _staged_case(tmp_path)
    spt.compare(csv_path, tmp_path / "staged_truth")
    text, st = mr.build_report(md, me, tmp_path / "staged_truth" / "summary.json", None)
    tg = st["model_targets"]
    assert tg["box mAP50, RDD-India D40 test"] == {"key": "box.mAP50", "value": 0.71, "pass": True}
    assert tg["box mAP50-95"]["pass"] is False
    assert tg["mask IoU, eYantra test"]["pass"] is False
    assert tg["FP per 1,000 frames (after verifier)"]["pass"] is True
    assert "BELOW TARGET" in text and "runs/y26n/best.pt" in text
    assert st["staged"] is True and "Overall: **PASS**" in text


def test_cli_smoke(tmp_path: Path):
    md = _measure_dir(tmp_path)
    env = {**os.environ, "GM_DEVICE": "cpu", "PYTHONUTF8": "1"}
    r = subprocess.run([PY, str(REPO / "scripts" / "make_report.py"), "--out", str(tmp_path / "REPORT.md"),
                        "--measure-dir", str(md), "--model-eval", str(tmp_path / "none.json"),
                        "--truth-summary", str(tmp_path / "none2.json"), "--rehearsal-summary", ""],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "model metrics: pending" in r.stdout and "staged truth: pending" in r.stdout
    assert (tmp_path / "REPORT.md").read_text(encoding="utf-8").startswith("# GunthaMeter validation report")
    r = subprocess.run([PY, str(REPO / "scripts" / "staged_pothole_truth.py"), "template", "--out",
                        str(tmp_path / "t.csv"), "--targets", "T20-S", "--passes", "1"],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert r.returncode == 0, r.stderr
    assert len(_rows_csv(tmp_path / "t.csv")) == 2


# ----------------------------------------------------------------------------- model_eval.json, real WP3 schema
def _gm_run(stage: str, weights: str, ap50: float, ap5095: float, iou: float | None = None) -> dict:
    mask = ({"pothole_mean_matched_mask_iou": iou} if iou is not None
            else {"status": "n/a", "reason": "detection-only model: no masks; mask IoU is not defined"})
    return {"stage": stage, "weights": weights, "split": "test", "limit": None, "device": "cuda:0",
            "device_name": "NVIDIA GeForce MX330", "conf": 0.001, "iou_nms": 0.7,
            "metrics": {"n_images": 514, "pothole": {"n_gt": 1038, "box_ap50": ap50, "box_ap50_95": ap5095,
                                                     "p_at_025": 0.27, "r_at_025": 0.1, "best_f1": 0.16},
                        "capable_classes": {"map50": ap50, "map50_95": ap5095}, "mask": mask,
                        "timing": {"ms_per_image_mean": 174.6}},
            "notes": [f"{stage} note"]}


_GM_TARGETS = {"box_map50_pothole_rdd_india_test": {"floor": 0.7, "target": 0.75, "source": "PLAN"},
               "mask_iou": {"floor": 0.65, "target": 0.7, "source": "PLAN"}}


def test_make_report_wp3_schema_baseline_only_is_pending(tmp_path: Path):
    """Baseline (bootstrap) runs only: acceptance is NOT VERIFIED -> pending; the targets block
    (e.g. mask_iou.floor 0.65) must never be reported as a measured value."""
    md = _measure_dir(tmp_path)
    me = tmp_path / "model_eval.json"
    me.write_text(json.dumps({
        "schema": "gunthameter.model_eval/1", "updated_at": "2026-11-13T18:16:45Z",
        "runs": [_gm_run("baseline_bootstrap", "weights/bootstrap_pothole_yolo26m.pt", 0.0885, 0.0253),
                 _gm_run("baseline_bootstrap", "weights/bootstrap_pothole_yolov8s.pt", 0.0043, 0.0018)],
        "targets": _GM_TARGETS,
        "acceptance": {"evaluated_on": None, "box_map50_pothole": None, "mask_iou_pothole": None,
                       "status": "NOT VERIFIED: no full test-split evaluation of a trained model yet"}}),
        encoding="utf-8")
    text, st = mr.build_report(md, me, tmp_path / "none.json", None)
    assert st["model"] == "present" and st["model_runs"] == 2 and st["model_evaluated_on"] is None
    assert all(v is None for v in st["model_targets"].values())
    assert st["model_best_run"]["box_ap50_pothole"] == 0.0885
    assert "0.650" not in text and "| meets |" not in text           # the floor is not a result
    assert "NOT VERIFIED" in text and "pending (no trained model)" in text
    assert "| baseline_bootstrap | `weights/bootstrap_pothole_yolo26m.pt` |" in text and "0.0885" in text
    assert "baseline 0.0885 (`bootstrap_pothole_yolo26m.pt`)" in text  # 4 decimals: no half-way rounding
    assert "thresholds, not results" in text

    env = {**os.environ, "GM_DEVICE": "cpu", "PYTHONUTF8": "1"}
    r = subprocess.run([PY, str(REPO / "scripts" / "make_report.py"), "--out", str(tmp_path / "R.md"),
                        "--measure-dir", str(md), "--model-eval", str(me), "--truth-summary",
                        str(tmp_path / "none.json"), "--rehearsal-summary", ""],
                       capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "model metrics: pending (acceptance: NOT VERIFIED" in r.stdout


def test_make_report_wp3_schema_trained_model_uses_acceptance(tmp_path: Path):
    md = _measure_dir(tmp_path)
    me = tmp_path / "model_eval.json"
    trained = _gm_run("trained", "runs/y26n/best.pt", 0.72, 0.41, iou=0.61)
    me.write_text(json.dumps({
        "schema": "gunthameter.model_eval/1",
        "runs": [_gm_run("baseline_bootstrap", "weights/b.pt", 0.09, 0.03), trained],
        "targets": _GM_TARGETS,
        "acceptance": {"evaluated_on": "trained (runs/y26n/best.pt)", "box_map50_pothole": 0.72,
                       "mask_iou_pothole": 0.61, "status": "evaluated (see *_ok flags)"}}), encoding="utf-8")
    text, st = mr.build_report(md, me, tmp_path / "none.json", None)
    tg = st["model_targets"]
    assert tg["box mAP50, RDD-India D40 test"]["value"] == 0.72 and tg["box mAP50, RDD-India D40 test"]["pass"]
    assert tg["box mAP50-95"]["value"] == 0.41 and tg["box mAP50-95"]["pass"] is False
    assert tg["mask IoU, eYantra test"]["value"] == 0.61 and tg["mask IoU, eYantra test"]["pass"] is False
    assert tg["FP per 1,000 frames (after verifier)"] is None
    assert "| box mAP50, RDD-India D40 test | >= 0.7 (WP3 hard floor (target 0.75)) | 0.720 | PASS |" in text


def test_make_report_generic_schema_ignores_target_keys(tmp_path: Path):
    md = _measure_dir(tmp_path)
    me = tmp_path / "model_eval.json"
    me.write_text(json.dumps({"targets": {"mask_iou": {"floor": 0.65}, "map50": 0.75},
                              "box": {"mAP50": 0.31}}), encoding="utf-8")
    _, st = mr.build_report(md, me, tmp_path / "none.json", None)
    tg = st["model_targets"]
    assert tg["box mAP50, RDD-India D40 test"]["value"] == 0.31
    assert tg["mask IoU, eYantra test"] is None
