"""Train pothole SEGMENTATION models on the merged dataset with hard augmentation, then benchmark (D48).

    make train-merged                       # A then B (~3-4 h, PC stays usable), benchmark, adoption rule
    make train-merged ONLY=A EPOCHS=1 SMOKE=1   # quick pipeline check (never adopts)

Candidates (both trained on datasets/merged_pothole_seg, one class: pothole):
  A  seg11n_from_base_hard: YOLO11n-seg whose backbone + neck start from the model in use (civic YOLO11n); small, for
     phones and the bus box.
  B  seg11s_coco_hard: YOLO11s-seg from COCO weights; bigger, for processing on the PC.
Augmentation (ml/augment/hard.py): strong zoom / perspective / roll (camera height, pitch), lens distortion + field of
view (LensDistort, outlines follow), night driving, lens optics, India weather; mosaic, mixup, copy-paste.
Then every test set of configs/datasets.yaml, including the D48 night / zoom check sets, scores the two candidates, the
model in use and ft_v1. A candidate replaces the model in use only if it passes the D38 rule (not worse on any test
set, better mean AP50). reports/MERGED_TRAINING.md has the table.
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ml import benchmark as B

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "datasets" / "merged_pothole_seg" / "data.yaml"
RUNS = REPO / "models" / "training_runs" / "seg"
BASE = REPO / "models" / "supplied_original" / "weights" / "best.pt"  # D50: the supplied civic YOLO11n
COCO_S = REPO / "models" / "yolo_starting_points" / "yolo11s-seg.pt"
CANDIDATES = REPO / "models" / "adopted_finetunes"
REPORT = REPO / "reports" / "MERGED_TRAINING.md"
SPECS = {"A": {"name": "seg11n_from_base_hard", "cfg": "yolo11n-seg.yaml", "init": BASE,
               "label": "A: YOLO11n-seg from your model, hard aug"},
         "B": {"name": "seg11s_coco_hard", "weights": COCO_S, "label": "B: YOLO11s-seg from COCO, hard aug"}}


def build_model(spec: dict[str, Any]) -> Any:
    from ultralytics import YOLO

    from ml.pipeline import checkpoint_globals, unsafe_globals

    src = spec.get("init") or spec.get("weights")
    if not Path(src).is_file():
        raise SystemExit(f"{src} missing")
    bad = unsafe_globals(checkpoint_globals(Path(src)))
    if bad:
        raise SystemExit(f"{src}: unsafe pickle globals {bad} (D17)")
    if "cfg" in spec:  # new segmentation model; transfer every layer whose shape matches (backbone + neck)
        return YOLO(spec["cfg"]).load(str(src))
    return YOLO(str(src))


def train_one(key: str, epochs: int, fraction: float = 1.0, gpu_frac: float = 0.7) -> Path:
    from ml.augment.hard import HARD_GEOMETRY, attach_lens_distort, hard_transforms
    from ml.train_det import be_gentle

    spec = SPECS[key]
    model = build_model(spec)
    gentle = be_gentle()
    kwargs: dict[str, Any] = dict(
        data=str(DATA), epochs=epochs, imgsz=640, batch=gpu_frac, device="0", workers=gentle["workers"], cache=False,
        project=str(RUNS), name=spec["name"], exist_ok=True, optimizer="auto", cos_lr=True, patience=30, seed=0,
        deterministic=False, plots=True, verbose=True, fraction=fraction, augmentations=hard_transforms(),
        **HARD_GEOMETRY)
    try:
        from ml.train_seg import apply_display_reserve

        apply_display_reserve(kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[train_merged] display reserve not applied: {e}", flush=True)
    attach_lens_distort(model, p=0.3)
    try:
        from ml.train_seg import GMCallbacks

        run_dir = RUNS / spec["name"]
        GMCallbacks(run_dir, run_dir / "gm_report.json", None, {"name": spec["name"], "base": spec.get("label")},
                    gm={"batch_requested": gpu_frac}).attach(model)
    except Exception as e:  # noqa: BLE001
        print(f"[train_merged] live dashboard feed off: {e}", flush=True)
    print(f"[train_merged] {key} {spec['name']}: gentle {gentle}, epochs {epochs}, fraction {fraction}", flush=True)
    model.train(**kwargs)
    best = RUNS / spec["name"] / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"{spec['name']}: no {best}")
    return best


def register(entry: dict[str, Any]) -> None:
    text = B.MODELS.read_text(encoding="utf-8")
    if f"id: {entry['id']}\n" in text:
        return
    B.MODELS.write_text(text.rstrip("\n") + (
        f"\n  - id: {entry['id']}\n    path: {entry['path']}\n    role: candidate\n    label: \"{entry['label']}\"\n"
        f"    note: \"{entry['note']}\"\n"), encoding="utf-8")


def bench(keys: list[str]) -> dict[str, Any]:
    in_use = B.in_use_weights()
    models = [{"id": "in_use", "path": in_use, "role": "in use"}]
    for k in keys:
        p = RUNS / SPECS[k]["name"] / "weights" / "best.pt"
        if p.is_file():
            models.append({"id": SPECS[k]["name"], "path": str(p.relative_to(REPO)), "role": "candidate"})
    ds = B.load_yaml(B.DATASETS)
    rep = B.benchmark(models, ds["tests"])
    ref = next(m for m in rep["models"] if m["id"] == "in_use")
    rule = {m["id"]: dict(zip(("passes", "reasons"), B.no_test_worse(m, ref), strict=True))
            for m in rep["models"] if m["role"] == "candidate"}
    return {"benchmark": rep, "rule": rule}


def write_report(res: dict[str, Any], extra: dict[str, Any]) -> str:
    rep = res["benchmark"]
    md = [B.to_markdown(rep, B.in_use_weights()), "", "## Adoption rule (D38) against the model in use", ""]
    for k, v in res["rule"].items():
        md.append(f"- {k}: {'PASSES' if v['passes'] else 'fails'}" +
                  ("" if v["passes"] else f" ({'; '.join(v['reasons'][:6])}{' …' if len(v['reasons']) > 6 else ''})"))
    md += ["", "## Run", "", f"`{json.dumps(extra, default=str)}`", ""]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    return "\n".join(md)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="A,B", help="A, B or A,B")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 5 %% of the data, no benchmark, never adopts")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--no-adopt", action="store_true")
    a = ap.parse_args(argv)
    keys = [k.strip().upper() for k in a.only.split(",") if k.strip()]
    if not DATA.is_file():
        raise SystemExit(f"{DATA} missing: run make merged-dataset")
    extra: dict[str, Any] = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "epochs": a.epochs}
    if not a.skip_train:
        for k in keys:
            t0 = time.time()
            train_one(k, 1 if a.smoke else a.epochs, 0.05 if a.smoke else 1.0)
            extra[f"minutes_{k}"] = round((time.time() - t0) / 60, 1)
    if a.smoke:
        print("[train_merged] smoke run finished (no benchmark)")
        return 0
    res = bench(keys)
    for k in keys:
        name = SPECS[k]["name"]
        p = RUNS / name / "weights" / "best.pt"
        if p.is_file():
            register({"id": name, "path": str(p.relative_to(REPO)), "label": SPECS[k]["label"],
                      "note": f"D48 merged dataset + hard augmentation; rule: "
                              f"{'passes' if res['rule'].get(name, {}).get('passes') else 'fails'}"})
    passing = [k for k, v in res["rule"].items() if v["passes"]]
    if passing and not a.no_adopt:
        best = max((m for m in res["benchmark"]["models"] if m["id"] in passing), key=lambda m: m["mean_box_ap50"])
        CANDIDATES.mkdir(parents=True, exist_ok=True)
        dst = CANDIDATES / f"{best['id']}.pt"
        shutil.copy2(REPO / best["path"], dst)
        from ml.train_det import set_env_weights

        set_env_weights(str(dst.relative_to(REPO)), f"D48 adopted {datetime.now():%Y-%m-%d}: passed the benchmark rule "
                                                    "(reports/MERGED_TRAINING.md)")
        extra["adopted"] = str(dst.relative_to(REPO))
    print(write_report(res, extra))
    print("ADOPTED: " + extra["adopted"] if extra.get("adopted") else "No candidate passed the rule; model in use kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
