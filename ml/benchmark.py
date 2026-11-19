"""Benchmark (D38): every model on every test set of configs/datasets.yaml.

    make benchmark                                   # all models of configs/models.yaml + the model in use
    .venv/bin/python -m ml.benchmark --models models/base_model/pothole_seg11n.pt models/training_runs/detect/x/weights/best.pt

Box test sets are scored with the same COCO matching and 101-point AP as ml.eval (ml.metrics): pothole AP50,
AP50-95, precision / recall at confidence 0.25 and the best F1. Only the model's pothole class counts, whatever it is
called (pothole, potholes, Pothole, D40). Negative sets report the share of pothole-free images that get a pothole
box at 0.25 (false alarms, lower is better); frame sets report frames found / false alarms at 0.25. Results:
reports/benchmark.json and reports/BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ml.metrics import IOU_THRESHOLDS, ap_per_threshold, box_iou, match

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "configs" / "datasets.yaml"
MODELS = REPO / "configs" / "models.yaml"
OUT_JSON = REPO / "reports" / "benchmark.json"
OUT_MD = REPO / "reports" / "BENCHMARK.md"
CONF = 0.25
IMG_EXT = (".jpg", ".jpeg", ".png")
Predictor = Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]  # img, conf -> boxes xyxy, confs


def load_yaml(p: Path) -> Any:
    import yaml

    return yaml.safe_load(p.read_text(encoding="utf-8"))


def pothole_class_ids(names: dict[int, str]) -> list[int]:
    return [k for k, v in names.items() if str(v).strip().lower().rstrip("s") in ("pothole", "d40")]


# ================================================================================================ test sets
def _images(d: Path, prefix: str = "") -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT and p.name.startswith(prefix)) \
        if d.is_dir() else []


def load_testset(spec: dict[str, Any], root: Path = REPO) -> dict[str, Any]:
    """-> {"kind", "items": [(image path, gt boxes (n,4) px or None)], ...} (gt None for negatives/frames)."""
    import cv2

    from ml.eval import rdd_original_label, read_label_file, read_rdd_original

    kind, img_dir = spec["kind"], root / spec["images"]
    items: list[tuple[Path, Optional[np.ndarray]]] = []
    if kind == "yolo":
        ids = {int(i): 0 for i in spec.get("pothole_ids", [0])}
        for p in _images(img_dir, spec.get("prefix", "")):
            lab = root / spec["labels"] / f"{p.stem}.txt"
            if spec.get("require_label") and not lab.is_file():
                continue  # D41: only photos someone marked (no file = not marked yet or "not sure")
            h, w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape[:2]
            _, boxes, _ = read_label_file(lab, (h, w), ids)
            items.append((p, boxes))
    elif kind == "rdd_locked":
        locked = {s.strip() for s in (root / spec["lock"]).read_text().splitlines() if s.strip() and s[0] != "#"}
        from ml.common import class_index

        pid = class_index("pothole")
        for p in _images(img_dir):
            if p.stem not in locked:
                continue
            h, w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape[:2]
            lp = rdd_original_label(p)
            cls, boxes = read_rdd_original(lp, (h, w)) if lp else ([], np.zeros((0, 4)))
            items.append((p, np.asarray([b for c, b in zip(cls, boxes, strict=True) if c == pid]).reshape(-1, 4)))
    elif kind == "negatives":
        items = [(p, None) for p in _images(img_dir, spec.get("prefix", ""))]
    elif kind == "frames":
        lab = json.loads((root / spec["labels"]).read_text(encoding="utf-8"))
        want = set(lab["pothole"]) | set(lab["clean"])
        items = drop_excluded([(p, None) for p in _images(img_dir) if p.name in want], img_dir)
        return {"kind": kind, "items": items, "pothole": set(lab["pothole"]), "clean": set(lab["clean"])}
    else:
        raise ValueError(f"unknown test-set kind {kind!r}")
    return {"kind": kind, "items": drop_excluded(items, img_dir)}


def drop_excluded(items: list, img_dir: Path) -> list:
    """D49: photos listed in <test set>/excluded.txt (corrected as "not sure", or a "clean road" that has a pothole)
    are not scored."""
    f = img_dir.parent / "excluded.txt"
    if not f.is_file():
        return items
    out = set(f.read_text(encoding="utf-8").split())
    return [it for it in items if it[0].name not in out]


# ================================================================================================ scoring
def score_boxes(gts: list[np.ndarray], preds: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    n_gt = int(sum(len(g) for g in gts))
    tps, confs = [], []
    for g, (pb, pc) in zip(gts, preds, strict=True):
        iou = box_iou(pb, g) if len(pb) and len(g) else np.zeros((len(pb), len(g)))
        tps.append(match(iou, pc))
        confs.append(pc)
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRESHOLDS)), bool)
    cf = np.concatenate(confs) if confs else np.zeros(0)
    aps, ex = ap_per_threshold(tp, cf, n_gt)
    return {"ap50": round(float(aps[0]), 4), "ap50_95": round(float(np.nanmean(aps)), 4),
            "p_at_025": round(ex.get("p_at_025", 0.0), 4), "r_at_025": round(ex.get("r_at_025", 0.0), 4),
            "best_f1": round(ex.get("best_f1", 0.0), 4), "n_images": len(gts), "n_gt": n_gt}


def run_on(ts: dict[str, Any], predict: Predictor) -> dict[str, Any]:
    import cv2

    t0 = time.perf_counter()
    if ts["kind"] in ("yolo", "rdd_locked"):
        preds = [predict(cv2.imread(str(p)), 0.001) for p, _ in ts["items"]]
        out = score_boxes([g for _, g in ts["items"]], preds)
    elif ts["kind"] == "negatives":
        flagged = sum(1 for p, _ in ts["items"] if len(predict(cv2.imread(str(p)), CONF)[0]))
        out = {"false_alarm_rate": round(flagged / max(1, len(ts["items"])), 4), "flagged": flagged,
               "n_images": len(ts["items"])}
    else:
        hit = {p.name: len(predict(cv2.imread(str(p)), CONF)[0]) > 0 for p, _ in ts["items"]}
        out = {"found": sum(hit[n] for n in ts["pothole"] if n in hit), "n_pothole_frames": len(ts["pothole"]),
               "false_alarms": sum(hit[n] for n in ts["clean"] if n in hit), "n_clean_frames": len(ts["clean"])}
    out["ms_per_image"] = round((time.perf_counter() - t0) * 1000 / max(1, len(ts["items"])), 1)
    return out


def yolo_predictor(weights: str, device: Optional[str] = None) -> Predictor:
    from ultralytics import YOLO

    if device is None:
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    m = YOLO(weights)
    pid = pothole_class_ids(m.names)
    if not pid:
        raise ValueError(f"{weights}: no pothole class in {m.names}")

    def predict(img: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray]:
        r = m.predict(img, conf=conf, iou=0.7, max_det=100, imgsz=640, classes=pid, verbose=False, device=device)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    return predict


def summary_score(res: dict[str, Any]) -> float:
    """One number to sort by: mean AP50 of the box test sets."""
    v = [r["ap50"] for r in res.values() if "ap50" in r]
    return round(float(np.mean(v)), 4) if v else 0.0


def benchmark(models: list[dict[str, Any]], testsets: list[dict[str, Any]], root: Path = REPO,
              make_predictor: Callable[[str], Predictor] = yolo_predictor,
              log: Callable[[str], None] = lambda m: print(m, flush=True)) -> dict[str, Any]:
    loaded = {t["id"]: load_testset(t, root) for t in testsets}
    empty = [k for k, v in loaded.items() if not v["items"]]
    if empty:
        log(f"skipped (no images / nothing marked yet): {', '.join(empty)}")
        testsets = [t for t in testsets if t["id"] not in empty]
    rows = []
    for m in models:
        pred = make_predictor(str(root / m["path"]) if not Path(m["path"]).is_absolute() else m["path"])
        res = {}
        for t in testsets:
            res[t["id"]] = run_on(loaded[t["id"]], pred)
        rows.append({**m, "results": res, "mean_box_ap50": summary_score(res)})
        log(f"{m['id'][:36]:36s} mean AP50 {rows[-1]['mean_box_ap50']:.3f}  " + "  ".join(
            f"{k}={v.get('ap50', v.get('false_alarm_rate', v.get('found')))}" for k, v in res.items()))
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "conf_for_rates": CONF,
            "testsets": [{k: t.get(k) for k in ("id", "name", "kind", "note", "primary")} for t in testsets],
            "models": rows}


def no_test_worse(cand: dict[str, Any], ref: dict[str, Any], tol_ap: float = 0.01, tol_rate: float = 0.01,
                  tol_frames: int = 0) -> tuple[bool, list[str]]:
    """D38 adoption rule: the candidate is not worse than the reference on ANY test set (small tolerances), and
    better on the mean box AP50. Returns (adopt, reasons)."""
    why = []
    for k, r in ref["results"].items():
        c = cand["results"].get(k, {})
        if "ap50" in r and c.get("ap50", 0) < r["ap50"] - tol_ap:
            why.append(f"{k}: AP50 {c.get('ap50')} < {r['ap50']}")
        if "false_alarm_rate" in r and c.get("false_alarm_rate", 1) > r["false_alarm_rate"] + tol_rate:
            why.append(f"{k}: false alarms {c.get('false_alarm_rate')} > {r['false_alarm_rate']}")
        if "found" in r:
            if c.get("found", 0) < r["found"] - tol_frames:
                why.append(f"{k}: found {c.get('found')} < {r['found']}")
            if c.get("false_alarms", 99) > r["false_alarms"] + tol_frames:
                why.append(f"{k}: false alarms {c.get('false_alarms')} > {r['false_alarms']}")
    if cand["mean_box_ap50"] <= ref["mean_box_ap50"]:
        why.append(f"mean AP50 {cand['mean_box_ap50']} <= {ref['mean_box_ap50']}")
    return not why, why


def to_markdown(rep: dict[str, Any], in_use: Optional[str] = None) -> str:
    ts = rep["testsets"]
    head = ["Model", "role", "mean AP50"]
    for t in ts:
        head.append({"negatives": f"{t['id']} false alarms", "frames": f"{t['id']} found / false"}.get(
            t["kind"], f"{t['id']} AP50"))
    lines = [f"# Benchmark ({rep['at']})", "",
             "Box AP50 (COCO matching, ml.metrics); rates at confidence 0.25. Test sets: configs/datasets.yaml; "
             "models: configs/models.yaml. **in use** = .env SEG_WEIGHTS.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for m in sorted(rep["models"], key=lambda r: -r["mean_box_ap50"]):
        cells = [m["id"] + (" **in use**" if in_use and Path(m["path"]).resolve() == Path(in_use).resolve() else ""),
                 m.get("role", ""), f"{m['mean_box_ap50']:.3f}"]
        for t in ts:
            r = m["results"][t["id"]]
            cells.append(f"{r['false_alarm_rate']:.1%}" if "false_alarm_rate" in r else
                         f"{r['found']}/{r['n_pothole_frames']} · {r['false_alarms']}/{r['n_clean_frames']}"
                         if "found" in r else f"{r['ap50']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Notes:"] + [f"- {t['id']}: {t['name']}" + (f" ({t['note']})" if t.get("note") else "") for t in ts]
    return "\n".join(lines) + "\n"


def in_use_weights() -> Optional[str]:
    try:
        from server.app.core.config import get_settings

        return str(get_settings().SEG_WEIGHTS)
    except Exception:  # noqa: BLE001
        return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="weights to score (default: configs/models.yaml + the model in use)")
    ap.add_argument("--tests", nargs="*", help="test-set ids (default: all)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args(argv)
    ds = load_yaml(DATASETS)
    tests = [t for t in ds["tests"] if not a.tests or t["id"] in a.tests]
    use = in_use_weights()
    if a.models:
        models = [{"id": Path(w).parent.parent.name if Path(w).parent.name == "weights" else Path(w).stem, "path": w,
                   "role": "given"} for w in a.models]
    else:
        models = [m for m in load_yaml(MODELS)["models"] if (REPO / m["path"]).is_file()]
        if use and not any((REPO / m["path"]).resolve() == (REPO / use).resolve() for m in models):
            models.append({"id": Path(use).stem, "path": use, "role": "in use"})
    rep = benchmark(models, tests)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    md = to_markdown(rep, str(REPO / use) if use else None)
    if a.out == OUT_JSON:
        OUT_MD.write_text(md, encoding="utf-8")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
