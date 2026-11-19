"""``make measure-demo``: the minimum viable jury demo (PLAN §10 WP12) — no services, no database.

1. Level-1 synthetic validation (PLAN 4.12) through the production code path:
   noise-free accuracy table, σ-coverage over noisy trials, and k recovery by ``truth.fit_k_sigma``.
2. Real Indian road images (BharatPotHole dashcam + RDD2022-India phone) measured with the real depth backend
   (DA-V2 Metric Outdoor by default, DECISIONS D4) and the bootstrap detector (D9), using the *uncertified*
   ``phone_demo`` calibration. If the detector finds nothing, the dataset's labelled outline is used as the mask
   and the result says so (``mask_source = dataset_label``).

Outputs in ``out``: ``synthetic_accuracy.{csv,json}``, ``sigma_coverage.json``, ``k_recovery.json``,
``real/<image>.json``, ``overlays/<image>.png``, ``summary.json`` and a human-readable ``README.md``.
Every real number is reported with σ; nothing here is billing-grade (``certified=false``).
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

CALIB = "configs/cameras/phone_demo.json"


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _log(msg: str) -> None:
    print(f"[measure-demo] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- synthetic
def run_synthetic(out: Path, trials: int) -> dict[str, Any]:
    from measure.validate import coverage_run, k_recovery_run, noise_free_table

    t0 = time.perf_counter()
    rows = noise_free_table()
    _dump(out / "synthetic_accuracy.json", rows)
    with (out / "synthetic_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(_jsonable(rows))
    n_pass = sum(bool(r["pass_volume"] and r["pass_p95"]) for r in rows)
    _log(f"synthetic accuracy: {n_pass}/{len(rows)} cases within tolerance ({time.perf_counter() - t0:.0f} s)")

    t1 = time.perf_counter()
    cov = coverage_run(trials=trials)
    cov.pop("pairs", None)
    _dump(out / "sigma_coverage.json", cov)
    c2 = {k: v["coverage_2sigma"] for k, v in cov["metrics"].items()}
    _log(f"σ coverage over {cov['measured']}/{trials} noisy trials (2σ): {c2} ({time.perf_counter() - t1:.0f} s)")

    t2 = time.perf_counter()
    kfit = k_recovery_run(trials=max(30, min(80, trials // 2)))
    _dump(out / "k_recovery.json", kfit)
    _log(f"k recovery: true {kfit['k_true']:.1e}, fitted {kfit['k_sigma']:.2e} (rel err {kfit['rel_err']:.1%}) "
         f"({time.perf_counter() - t2:.0f} s)")
    return {"accuracy_cases": len(rows), "accuracy_pass": n_pass,
            "max_abs_volume_err_pct": max(abs(r["volume_err_pct"]) for r in rows),
            "max_abs_p95_err_mm": max(abs(r["p95_err_mm"]) for r in rows),
            "within_2sigma": sum(bool(r["within_2sigma"]) for r in rows),
            "coverage_2sigma": c2, "coverage_trials": trials, "coverage_measured": cov["measured"],
            "k_true": kfit["k_true"], "k_fitted": kfit["k_sigma"], "k_rel_err": kfit["rel_err"],
            "rows": rows}


# ---------------------------------------------------------------------------------------------- real images
def label_pothole_area(img: Path) -> float:
    """Largest labelled pothole (normalised bbox area) in the image's YOLO label file; 0 if none."""
    txt = img.with_suffix(".txt")
    if not txt.is_file():
        return 0.0
    cls = POTHOLE_CLASS.get(img.parent.name, "0")
    best = 0.0
    for line in txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts[:1] != [cls] or len(parts) < 5:
            continue
        xy = np.asarray(parts[1:], float)
        if len(xy) == 4:
            area = xy[2] * xy[3]
        else:
            pts = xy.reshape(-1, 2)
            area = float(np.ptp(pts[:, 0]) * np.ptp(pts[:, 1]))
        best = max(best, float(area))
    return best


def pick_images(images_dir: Path, n: int, min_label_area: float = 0.004) -> list[Path]:
    """Alternate between dataset sub-folders (dashcam and phone images); within each folder, frames with a
    clearly visible labelled pothole come first (largest first), so the demo measures real potholes."""
    def ranked(d: Path) -> list[Path]:
        imgs = sorted(d.glob("*.jpg"))
        area = {p: label_pothole_area(p) for p in imgs}
        good = sorted((p for p in imgs if area[p] >= min_label_area), key=lambda p: -area[p])
        return good + [p for p in imgs if area[p] < min_label_area]

    groups = [ranked(d) for d in sorted(images_dir.iterdir()) if d.is_dir()]
    groups = [g for g in groups if g] or [sorted(images_dir.glob("*.jpg"))]
    picked: list[Path] = []
    i = 0
    while len(picked) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(picked) < n:
                picked.append(g[i])
        i += 1
    return picked


# pothole class id in each demo folder's YOLO labels: BharatPotHole 0 = pothole; the RDD2022 mirror uses
# 4 = D40 (pothole), see datasets/extracted/rdd2022/meta.json "class_ids" (verified visually).
POTHOLE_CLASS = {"rdd2022": "4"}


def label_mask(label_txt: Path, img: np.ndarray, out_png: Path, max_n: int = 5) -> Optional[Path]:
    """Dataset pothole labels -> instance-id PNG; None if the image has no pothole label.

    Polygons are filled as-is; plain boxes become masks through ``measure.detect.box_to_mask`` (the same
    path a box-only detector takes), never a filled rectangle.
    """
    import cv2

    from measure.detect import box_to_mask

    if not label_txt.is_file():
        return None
    H, W = img.shape[:2]
    cls = POTHOLE_CLASS.get(label_txt.parent.name, "0")
    inst = np.zeros((H, W), np.uint8)
    k = 0
    for line in label_txt.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts[:1] != [cls] or len(parts) < 5:
            continue
        xy = np.asarray(parts[1:], float)
        if len(xy) == 4:  # box: cx cy w h (normalised)
            cx, cy, bw, bh = xy
            box = [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]
            m, _ = box_to_mask(img, box, "auto")
        else:
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [np.round(xy.reshape(-1, 2) * [W, H]).astype(np.int32)], 1)
        m = np.asarray(m).astype(bool) & (inst == 0)
        if m.sum() < 50:
            continue
        k += 1
        inst[m] = k
        if k >= max_n:
            break
    if k == 0:
        return None
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if k == 1:
        inst = (inst > 0).astype(np.uint8) * 255  # single instance: plain binary mask
    cv2.imwrite(str(out_png), inst)
    return out_png


def run_real(out: Path, images_dir: Path, n_images: int, weights: str) -> dict[str, Any]:
    import cv2

    from measure.cli import run_measure

    imgs = pick_images(images_dir, n_images) if images_dir.is_dir() else []
    if not imgs:
        _log(f"no images in {images_dir}; skipping the real-image part")
        return {"images": 0, "items": []}
    have_weights = Path(weights).is_file()
    items: list[dict[str, Any]] = []
    for p in imgs:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"image": str(p).replace("\\", "/"), "dataset": p.parent.name}
        try:
            resp: dict[str, Any] = {"measurements": []}
            if have_weights:
                resp = run_measure(str(p), CALIB, detect=weights, overlay=str(out / "overlays" / f"{p.stem}.png"))
                rec["mask_source"] = "detector"
            if not resp["measurements"]:
                img = cv2.imread(str(p))
                m = label_mask(p.with_suffix(".txt"), img, out / "masks" / f"{p.stem}.png")
                if m is not None:
                    resp = run_measure(str(p), CALIB, mask=str(m), overlay=str(out / "overlays" / f"{p.stem}.png"))
                    rec["mask_source"] = "dataset_label"
                    for mm in resp["measurements"]:
                        mm.setdefault("extras", {})["note"] = "mask = dataset label (detector found none)"
            _dump(out / "real" / f"{p.stem}.json", resp)
            rec.update(certified=resp.get("certified"), depth_model_version=resp.get("depth_model_version"),
                       n=len(resp["measurements"]), elapsed_s=time.perf_counter() - t0,
                       overlay=resp.get("overlay_url"), potholes=[])
            for m, sev in zip(resp["measurements"], resp.get("severities", []), strict=True):
                rec["potholes"].append({k: m.get(k) for k in (
                    "depth_p95_mm", "depth_sigma_mm", "depth_max_mm", "area_m2", "area_sigma", "volume_l",
                    "volume_sigma", "length_m", "width_m", "range_m", "quality", "reject_reason", "water_filled",
                    "surface_model", "scale_factor")} | {"severity": sev})
            _log(f"{p.name}: {rec['n']} measured via {rec.get('mask_source', '-')} in {rec['elapsed_s']:.1f} s")
        except Exception as e:  # noqa: BLE001 - one bad image must not kill the demo; recorded
            rec["error"] = f"{type(e).__name__}: {e}"
            _log(f"{p.name}: ERROR {rec['error']}")
        items.append(rec)
    return {"images": len(imgs), "calib": CALIB, "weights": weights if have_weights else None, "items": items}


# ---------------------------------------------------------------------------------------------- report
def _fmt(v: Optional[float], s: Optional[float] = None, nd: int = 1) -> str:
    if v is None:
        return "–"
    return f"{v:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")


def write_readme(out: Path, syn: Optional[dict[str, Any]], real: Optional[dict[str, Any]], env: dict[str, Any]) -> None:
    L = ["# measure-demo report", "",
         f"Generated {env['generated_at']} on {env['device']} (depth backend `{env['depth_backend']}`).", "",
         "**Nothing here is billing-grade.** Real images use the `phone_demo` calibration (`certified=false`): "
         "unknown true camera height/pitch, so absolute depth/volume carry the σ shown and an unquantified "
         "calibration bias. Synthetic numbers use exact ground truth through the production code path.", ""]
    if syn:
        L += ["## 1. Synthetic accuracy (noise-free, PLAN 4.12 level 1)", "",
              f"{syn['accuracy_pass']}/{syn['accuracy_cases']} cases within tolerance; max |volume error| "
              f"{syn['max_abs_volume_err_pct']:.1f} %, max |p95 error| {syn['max_abs_p95_err_mm']:.1f} mm; "
              f"{syn['within_2sigma']}/{syn['accuracy_cases']} within 2σ.", "",
              "| case | truth L | measured L | err % | tol % | truth p95 mm | p95 mm | pass |", "|---|---|---|---|---|---|---|---|"]
        for r in syn["rows"]:
            L.append(f"| {r['case']} | {r['truth_volume_l']:.2f} | {_fmt(r['volume_l'], r['volume_sigma_l'], 2)} | "
                     f"{r['volume_err_pct']:+.1f} | {r['tol_volume_pct']:.0f} | {r['truth_p95_mm']:.1f} | "
                     f"{r['p95_mm']:.1f} | {'yes' if r['pass_volume'] and r['pass_p95'] else 'NO'} |")
        L += ["", f"## 2. σ coverage ({syn['coverage_measured']}/{syn['coverage_trials']} noisy trials measured)", "",
              "Fraction of trials whose error is within the reported 2σ (target ≥ 0.90):", ""]
        L += [f"- {k}: {v:.2f}" if v is not None else f"- {k}: –" for k, v in syn["coverage_2sigma"].items()]
        L += ["", f"k recovery (`truth.fit_k_sigma`): injected {syn['k_true']:.1e}, fitted {syn['k_fitted']:.2e} "
              f"(relative error {syn['k_rel_err']:.1%}).", ""]
    if real and real.get("items"):
        L += ["## 3. Real Indian road images (uncertified demo calibration)", "",
              "| image | mask | severity | depth p95 mm | area m² | volume L | range m | quality | note |",
              "|---|---|---|---|---|---|---|---|---|"]
        for it in real["items"]:
            name = Path(it["image"]).name
            if it.get("error"):
                L.append(f"| {name} | – | – | – | – | – | – | – | error: {it['error'][:80]} |")
                continue
            if not it["potholes"]:
                L.append(f"| {name} | {it.get('mask_source', '–')} | – | – | – | – | – | – | no pothole found |")
            for p in it["potholes"]:
                if p["reject_reason"] or p["depth_p95_mm"] is None:  # never show numbers for a rejected frame
                    L.append(f"| {name} | {it.get('mask_source')} | rejected | – | – | – | {_fmt(p['range_m'])} | "
                             f"{_fmt(p['quality'], None, 2)} | {p['reject_reason'] or 'no depth'} |")
                    continue
                L.append(f"| {name} | {it.get('mask_source')} | {p['severity']} | "
                         f"{_fmt(p['depth_p95_mm'], p['depth_sigma_mm'])} | {_fmt(p['area_m2'], p['area_sigma'], 3)} | "
                         f"{_fmt(p['volume_l'], p['volume_sigma'], 1)} | {_fmt(p['range_m'])} | "
                         f"{_fmt(p['quality'], None, 2)} | {p['reject_reason'] or ''} |")
        L += ["", "Overlays: `overlays/<image>.png`; full measurement JSON (MeasureResponse shape): `real/<image>.json`.", ""]
    (out / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def run_demo(out: str | Path = "reports/measure_demo", *, images_dir: str | Path = "datasets/extracted/measure_demo_imgs",
             n_images: int = 5, weights: Optional[str] = None, trials: int = 200,
             skip_real: bool = False, skip_synthetic: bool = False) -> dict[str, Any]:
    from datetime import datetime, timezone

    from measure.config import load_measure_cfg
    from measure.depth_backends.base import cuda_status

    if not weights:  # D24: the best trained model (ml.common.DEFAULT_SEG_WEIGHTS), as a repo path
        from ml.common import DEFAULT_SEG_WEIGHTS, resolve_repo_path

        weights = str(resolve_repo_path(DEFAULT_SEG_WEIGHTS))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    st = cuda_status()
    env = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "device": st["name"] if st["usable"] else f"CPU ({st['reason'] or 'no CUDA'})",
           "depth_backend": load_measure_cfg()["depth"].get("backend")}
    t0 = time.perf_counter()
    syn = None if skip_synthetic else run_synthetic(out, trials)
    real = None if skip_real else run_real(out, Path(images_dir), n_images, weights)
    summary = {"env": env, "elapsed_s": time.perf_counter() - t0,
               "synthetic": {k: v for k, v in (syn or {}).items() if k != "rows"} or None, "real": real}
    _dump(out / "summary.json", summary)
    write_readme(out, syn, real, env)
    _log(f"done in {summary['elapsed_s']:.0f} s -> {out / 'README.md'}")
    return summary
