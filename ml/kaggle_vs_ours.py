"""Kaggle recipe vs ours on OUR dataset, then one benchmark and a conclusion (D45).

    make kaggle-vs-ours                  # all stages, ~20 min on the RTX 5080 (PC stays usable)
    make kaggle-vs-ours STAGES=bench     # re-score only (after the runs exist)

The Kaggle notebook cubeai/pothole-segmentation-with-yolov8 (D44) is `YOLO('yolov8n-seg.pt').train(data, seed=42,
epochs=100)` with every other setting at its default. Here it is applied to the RoadFix base dataset:

1. `seg-data`: the base dataset has boxes only, and the recipe trains a SEGMENTATION model, so every box becomes an
   outline (SAM2.1 -> GrabCut -> ellipse fallback, `measure.detect.box_to_mask`; the method used is counted). Same
   images (symlinks), same leak-free train / val split -> `datasets/base_nagar_netra_pothole_seg`.
2. `train`: two runs with the author's exact settings (`args.yaml` of the Kaggle run, D44):
   - `kaggle_recipe_on_base`: from COCO `yolov8n-seg.pt` = the notebook, our data;
   - `kaggle_model_ft_on_base`: from the author's trained pothole model = the Kaggle model trained further on our data.
   Gentle mode and the display reserve as in `ml/train_det.py` (D40, D20). Runs: models/training_runs/seg/<name>.
3. `bench`: the two runs + the Kaggle original + the model in use + our own recipe on the same data (`det_yolo11n_ft_v1`,
   D38) on every test set of configs/datasets.yaml.
4. `conclude`: reports/KAGGLE_VS_OURS.md: the table, the winner per test set, the D38 adoption rule for every model
   against the one in use, and a recommendation. Nothing is adopted automatically.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from ml import benchmark as B

REPO = Path(__file__).resolve().parents[1]
BASE_DATA = REPO / "datasets" / "base_nagar_netra_pothole"
SEG_DATA = REPO / "datasets" / "base_nagar_netra_pothole_seg"
KAGGLE = REPO / "datasets" / "downloads" / "kaggle_cubeai_pothole_seg_yolov8" / "output"
KAGGLE_ARGS = KAGGLE / "runs" / "segment" / "train" / "args.yaml"
COCO_SEG = REPO / "models" / "yolo_starting_points" / "yolov8n-seg.pt"
KAGGLE_MODEL = REPO / "models" / "downloaded_pretrained" / "kaggle_cubeai_yolov8n_seg.pt"
RUNS = REPO / "models" / "training_runs" / "seg"
REPORT = REPO / "reports" / "KAGGLE_VS_OURS.md"
REPORT_JSON = REPO / "reports" / "kaggle_vs_ours.json"
POTHOLE = 1
IMG_EXT = (".jpg", ".jpeg", ".png")

# training settings taken from the Kaggle run (hyper-parameters only; paths, device and logging are ours)
RECIPE_KEYS = ("epochs", "patience", "batch", "imgsz", "optimizer", "seed", "deterministic", "lr0", "lrf", "momentum",
               "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl", "nbs",
               "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
               "mosaic", "mixup", "copy_paste", "close_mosaic", "cos_lr", "overlap_mask", "mask_ratio", "dropout",
               "label_smoothing", "single_cls", "rect", "amp", "multi_scale", "fraction")
RUNS_SPEC = (("kaggle_recipe_on_base", "coco"), ("kaggle_model_ft_on_base", "kaggle"))


# ================================================================================================ recipe
def kaggle_recipe(args_yaml: Path = KAGGLE_ARGS) -> dict[str, Any]:
    """The author's hyper-parameters (read as data from the untrusted download; only known keys, plain values)."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    raw = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
    out = {}
    for k in RECIPE_KEYS:
        if k in raw and k in DEFAULT_CFG_DICT and isinstance(raw[k], (int, float, str, bool)) and raw[k] is not None:
            out[k] = raw[k]
    if out.get("epochs") != 100 or out.get("seed") != 42:
        raise SystemExit(f"{args_yaml}: not the expected Kaggle run (epochs {out.get('epochs')}, seed {out.get('seed')})")
    return out


# ================================================================================================ seg data
def _box_lines(label: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in label.read_text(encoding="utf-8").splitlines() if label.is_file() else []:
        t = line.split()
        if len(t) == 5:
            out.append((int(float(t[0])), *map(float, t[1:])))
    return out


def build_seg_data(src: Path = BASE_DATA, out: Path = SEG_DATA, masker: Optional[Callable] = None,
                   overwrite: bool = False, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Box labels -> outline labels (one polygon per box), same images and split."""
    import cv2

    from dataset_tools.seg_geometry import mask_to_polygon, yolo_seg_line

    if masker is None:
        from measure.detect import box_to_mask

        masker = box_to_mask
    stats_path = out / "build_stats.json"
    if stats_path.is_file() and not overwrite:
        log(f"[seg-data] {out.relative_to(REPO) if out.is_relative_to(REPO) else out} exists (use --overwrite)")
        return json.loads(stats_path.read_text())
    methods: Counter = Counter()
    counts: Counter = Counter()
    t0 = time.time()
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_p in sorted(p for p in (src / "images" / split).iterdir() if p.suffix.lower() in IMG_EXT):
            dst = out / "images" / split / img_p.name
            if not dst.exists():
                dst.symlink_to(img_p.resolve())
            boxes = _box_lines(src / "labels" / split / f"{img_p.stem}.txt")
            lines = []
            if boxes:
                img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
                h, w = img.shape[:2]
                for c, cx, cy, bw, bh in boxes:
                    box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                    try:
                        mask, how = masker(img, box)
                        poly = mask_to_polygon(mask)
                    except RuntimeError:
                        poly, how = None, "failed"
                    if poly is None or len(poly) < 3:  # keep the pothole: its box as a 4-point outline
                        x0, y0, x1, y1 = box
                        import numpy as np

                        poly, how = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]), "box"
                    methods[how.split(" ")[0]] += 1
                    lines.append(yolo_seg_line(c, poly, w, h))
            (out / "labels" / split / f"{img_p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                      encoding="utf-8")
            counts[f"{split}_images"] += 1
            counts[f"{split}_potholes"] += len(lines)
    meta = yaml.safe_load((src / "data.yaml").read_text())
    (out / "data.yaml").write_text(
        "# D45 base dataset with outlines (ml/kaggle_vs_ours.py): every box -> SAM2 outline, same images and split\n"
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {meta['nc']}\nnames: {json.dumps(meta['names'])}\n", encoding="utf-8")
    stats = {**counts, "outline_method": dict(methods), "seconds": round(time.time() - t0, 1),
             "source": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src)}
    stats_path.write_text(json.dumps(stats, indent=1))
    log(f"[seg-data] {json.dumps(stats)}")
    return stats


# ================================================================================================ train
def train_run(name: str, start: Path, recipe: dict[str, Any], data_yaml: Path = SEG_DATA / "data.yaml") -> Path:
    from ultralytics import YOLO

    from ml.pipeline import checkpoint_globals, unsafe_globals
    from ml.train_det import be_gentle

    if not start.is_file():
        raise SystemExit(f"{start} is missing (deleted in D47): download it again before re-running the train stage "
                         "(yolov8n-seg.pt from the Ultralytics releases; the Kaggle model: tools/undill_checkpoint.py on "
                         "the Kaggle output in datasets/downloads/kaggle_cubeai_pothole_seg_yolov8/)")
    bad = unsafe_globals(checkpoint_globals(start))
    if bad:
        raise SystemExit(f"{start}: unsafe pickle globals {bad} (D17); not loading")
    gentle = be_gentle()
    kwargs: dict[str, Any] = {**recipe, "data": str(data_yaml), "device": "0", "workers": gentle["workers"],
                              "cache": False, "project": str(RUNS), "name": name, "exist_ok": True, "plots": True,
                              "verbose": True}
    try:
        from ml.train_seg import apply_display_reserve

        apply_display_reserve(kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[kaggle_vs_ours] display reserve not applied: {e}", flush=True)
    model = YOLO(str(start))
    try:
        from ml.train_seg import GMCallbacks

        run_dir = RUNS / name
        GMCallbacks(run_dir, run_dir / "gm_report.json", None, {"name": name, "base": str(start)},
                    gm={"batch_requested": recipe.get("batch")}).attach(model)
    except Exception as e:  # noqa: BLE001 - the dashboard feed is optional
        print(f"[kaggle_vs_ours] live dashboard feed off: {e}", flush=True)
    print(f"[kaggle_vs_ours] {name}: start {start.name}, recipe {recipe}, gentle {gentle}", flush=True)
    model.train(**kwargs)
    best = RUNS / name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"{name}: training finished without {best}")
    return best


# ================================================================================================ bench + conclude
def contenders() -> list[dict[str, Any]]:
    in_use = B.in_use_weights()
    out = [{"id": "in_use", "label": "Your model (in use)", "path": in_use, "side": "ours"},
           {"id": "det_yolo11n_ft_v1", "label": "Our recipe on base data (ft_v1)",
            "path": "models/training_runs/detect/det_yolo11n_ft_v1/weights/best.pt", "side": "ours"},
           {"id": "kaggle_cubeai_yolov8n_seg", "label": "Kaggle model as downloaded",
            "path": str(KAGGLE_MODEL.relative_to(REPO)), "side": "kaggle"}]
    for name, _ in RUNS_SPEC:
        out.append({"id": name, "path": f"models/training_runs/seg/{name}/weights/best.pt", "side": "kaggle",
                    "label": {"kaggle_recipe_on_base": "Kaggle recipe on our data (from COCO)",
                              "kaggle_model_ft_on_base": "Kaggle model trained further on our data"}[name]})
    return [m for m in out if m["path"] and (REPO / m["path"]).is_file()]


def higher_is_better(res: dict[str, Any]) -> tuple[str, float, bool]:
    """(metric name, value, higher is better) for one test-set result."""
    if "ap50" in res:
        return "AP50", res["ap50"], True
    if "false_alarm_rate" in res:
        return "false alarms", res["false_alarm_rate"], False
    return "found - false", res["found"] - res["false_alarms"], True


def conclude(rep: dict[str, Any], models: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {m["id"]: m for m in rep["models"]}
    ref = rows.get("in_use")
    tests = [t["id"] for t in rep["testsets"]]
    winners = {}
    for t in tests:
        vals = [(m["id"], *higher_is_better(m["results"][t])) for m in rep["models"] if t in m["results"]]
        _, _, up = vals[0][1:]
        best = (max if up else min)(v[2] for v in vals)
        winners[t] = [v[0] for v in vals if v[2] == best]
    rule = {}
    for m in rep["models"]:
        if ref is not None and m["id"] != "in_use":
            ok, why = B.no_test_worse(m, ref)
            rule[m["id"]] = {"passes": ok, "reasons": why}
    sides = {m["id"]: m["side"] for m in models}
    best_k = max((m for m in rep["models"] if sides.get(m["id"]) == "kaggle"),
                 key=lambda m: m["results"].get("phone_street_marked", {}).get("ap50", -1), default=None)
    best_o = max((m for m in rep["models"] if sides.get(m["id"]) == "ours"),
                 key=lambda m: m["results"].get("phone_street_marked", {}).get("ap50", -1), default=None)
    wins = Counter(sides.get(w[0]) for w in winners.values() if len(w) == 1)
    adopt = [k for k, v in rule.items() if v["passes"]]
    if adopt:
        rec = (f"{', '.join(adopt)} passes the adoption rule (not worse on any test set, better mean AP50): consider "
               "switching to it (make benchmark, then set SEG_WEIGHTS by hand).")
    else:
        rec = ("No model passes the adoption rule against the one in use: keep your model. " +
               (f"On your own phone outlines the best Kaggle-side model is {best_k['id']} "
                f"({best_k['results']['phone_street_marked']['ap50']:.3f}) and the best of ours {best_o['id']} "
                f"({best_o['results']['phone_street_marked']['ap50']:.3f})."
                if best_k and best_o and "phone_street_marked" in best_k["results"] else ""))
    return {"winners": winners, "rule": rule, "wins_by_side": dict(wins), "recommendation": rec}


def to_markdown(rep: dict[str, Any], models: list[dict[str, Any]], con: dict[str, Any], extra: dict[str, Any]) -> str:
    labels = {m["id"]: f"{m['label']} ({m['side']})" for m in models}
    tests = rep["testsets"]
    head = "| Model | " + " | ".join(t["id"] for t in tests) + " | mean AP50 |"
    lines = [f"# Kaggle recipe vs ours on our data ({rep['at']})", "",
             "Notebook `cubeai/pothole-segmentation-with-yolov8` (D44) applied to the RoadFix base dataset (D45). "
             "Box AP50 unless noted; false alarms and phone frames at confidence 0.25. **Bold** = best on that test set.",
             "", head, "|" + "---|" * (len(tests) + 2)]
    for m in rep["models"]:
        cells = []
        for t in tests:
            r = m["results"][t["id"]]
            name, v, _ = higher_is_better(r)
            txt = (f"{v:.3f}" if name == "AP50" else f"{100 * v:.1f} %" if name == "false alarms" else
                   f"{r['found']}/{r['n_pothole_frames']} · {r['false_alarms']}/{r['n_clean_frames']}")
            cells.append(f"**{txt}**" if con["winners"][t["id"]] == [m["id"]] else txt)
        lines.append(f"| {labels.get(m['id'], m['id'])} | " + " | ".join(cells) + f" | {m['mean_box_ap50']:.3f} |")
    lines += ["", "## Adoption rule against the model in use (D38: not worse on any test set, better mean AP50)", ""]
    for k, v in con["rule"].items():
        lines.append(f"- {labels.get(k, k)}: {'PASSES' if v['passes'] else 'fails'}" +
                     ("" if v["passes"] else f" ({'; '.join(v['reasons'][:4])}{' …' if len(v['reasons']) > 4 else ''})"))
    lines += ["", f"Test sets won outright: {con['wins_by_side']}", "", "## Conclusion", "", con["recommendation"], ""]
    if extra.get("seg_data"):
        lines += ["## Training data with outlines", "", f"`{json.dumps(extra['seg_data'])}`", ""]
    if extra.get("recipe"):
        lines += ["## Kaggle settings used", "", f"`{json.dumps(extra['recipe'])}`", ""]
    return "\n".join(lines)


def bench_and_conclude(extra: dict[str, Any]) -> dict[str, Any]:
    models = contenders()
    ds = B.load_yaml(B.DATASETS)
    rep = B.benchmark([{"id": m["id"], "path": m["path"], "role": m["side"]} for m in models], ds["tests"])
    con = conclude(rep, models)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(to_markdown(rep, models, con, extra), encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({"benchmark": rep, "conclusion": con, "models": models, **extra}, indent=1,
                                      default=str))
    return con


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="seg-data,train,bench", help="comma list of seg-data, train, bench")
    ap.add_argument("--overwrite", action="store_true", help="rebuild the outline dataset")
    ap.add_argument("--epochs", type=int, default=None, help="override the recipe's 100 epochs (trial runs only)")
    a = ap.parse_args(argv)
    stages = set(a.stages.split(","))
    extra: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    recipe = kaggle_recipe()
    if a.epochs:
        recipe["epochs"] = a.epochs
    extra["recipe"] = recipe
    if "seg-data" in stages:
        extra["seg_data"] = build_seg_data(overwrite=a.overwrite)
    elif (SEG_DATA / "build_stats.json").is_file():
        extra["seg_data"] = json.loads((SEG_DATA / "build_stats.json").read_text())
    if "train" in stages:
        for name, start in RUNS_SPEC:
            t0 = time.time()
            train_run(name, COCO_SEG if start == "coco" else KAGGLE_MODEL, recipe)
            extra[f"minutes_{name}"] = round((time.time() - t0) / 60, 1)
    if "bench" in stages:
        con = bench_and_conclude(extra)
        print("\n" + REPORT.read_text(encoding="utf-8"))
        print(f"\nreport: {REPORT.relative_to(REPO)}\n{con['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
