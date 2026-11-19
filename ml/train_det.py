"""Fine-tune the base pothole detector on the base dataset, benchmark it, adopt it only if it is not worse (D38).

    make train                               # NAME=det_<date>, EPOCHS=80, adopts automatically when the rule passes
    make train DATA=combined EPOCHS=60       # D39: base x5 + earlier train/val (make combined-dataset first)
    make negatives-dataset && make train DATA=base_neg EPOCHS=80   # D42: base + earlier images WITHOUT potholes
    make train DATA=combined SMOKE=1         # ~3 min trial: 1 epoch on 10 % of the data, nothing adopted
    .venv/bin/python -m ml.train_det --name det_try --data combined --epochs 60 [--no-adopt]

1. Train: start from weights/base/civic_yolo11n.pt (YOLO11n, 16 Nagar Netra classes, pothole = 1) on
   data/datasets/nagar_netra_pothole (leak-free split). Same head, so nothing is re-initialised. AdamW, low learning
   rate (fine-tune), cosine schedule, batch sized by AutoBatch to `--gpu-frac` of the free GPU memory (D17; the
   display keeps the rest, D20). Output in runs/detect/<name>/ with the live files the training dashboard reads
   (gm_live.json, results.csv).
2. Benchmark: the candidate and the base model on every test set of configs/datasets.yaml (ml.benchmark).
3. Adopt only if it is not worse than the reference (the model in use) on ANY test set, and better on the mean box
   AP50 (ml.benchmark.no_test_worse). Then the weights are copied to weights/candidates/<name>.pt, .env SEG_WEIGHTS
   points to them and configs/models.yaml lists them; otherwise nothing changes. reports/train_<name>.json records why.

On-the-fly augmentation (D40), applied to every training image as it is loaded (labels follow; nothing is written to
disk): Ultralytics mosaic, HSV jitter and left-right flips; the bus-mount geometry of D21 (zoom 0.5-1.5, shift +-15 %,
roll +-5 deg, small perspective; no up-down flip, no shear); and the India road conditions of ml/augment/india.py
(rain / wet asphalt, sun glare, shadows, dusk and haze, motion blur, low-resolution and JPEG artefacts; pixel-only, so
boxes stay valid). `--aug off|light|default|strong`.

System stays responsive (D40): the process runs at nice 10 with low I/O priority, dataloader workers are capped at a
quarter of the CPU threads, PyTorch CPU threads at 4, images are not cached in RAM, and when the monitor hangs off the
training GPU 3 GiB of GPU memory stay free for the desktop (D20 display reserve, ml.train_seg.apply_display_reserve).

Note: the base dataset holds only potholes, so a fine-tuned copy forgets the other 15 civic classes. RoadFix only
uses the pothole class; the user's original multi-class model stays untouched in weights/base/.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ml import benchmark as B

REPO = Path(__file__).resolve().parents[1]
BASE_WEIGHTS = REPO / "weights" / "base" / "civic_yolo11n.pt"
RUNS = REPO / "runs" / "detect"
CANDIDATES = REPO / "weights" / "candidates"
ENV = REPO / ".env"


def set_env_weights(path: str, note: str, env: Path = ENV) -> None:
    lines = env.read_text(encoding="utf-8").splitlines() if env.is_file() else []
    new = f"SEG_WEIGHTS={path}   # {note}"
    for i, line in enumerate(lines):
        if line.startswith("SEG_WEIGHTS="):
            lines[i] = new
            break
    else:
        lines.append(new)
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")


def register_model(entry: dict[str, Any], models: Path = B.MODELS) -> None:
    text = models.read_text(encoding="utf-8")
    if f"id: {entry['id']}\n" in text:
        return
    text = text.rstrip("\n") + (f"\n  - id: {entry['id']}\n    path: {entry['path']}\n    role: {entry['role']}\n"
                                f"    note: \"{entry['note']}\"\n")
    models.write_text(text, encoding="utf-8")


# D21 bus-mount geometry + Ultralytics photometric defaults (fixed forward camera: no up-down flip, no shear)
GEOMETRY = {"scale": 0.5, "translate": 0.15, "degrees": 5.0, "perspective": 0.0005, "shear": 0.0, "flipud": 0.0,
            "fliplr": 0.5, "mosaic": 1.0, "close_mosaic": 10, "mixup": 0.0, "hsv_h": 0.015, "hsv_s": 0.7,
            "hsv_v": 0.4}


def be_gentle() -> dict[str, Any]:
    """Lower CPU and I/O priority of this process (and the dataloader workers it starts) so the desktop stays
    responsive; cap PyTorch's CPU threads. Best effort: never stops training."""
    import os

    out: dict[str, Any] = {}
    try:
        out["nice"] = os.nice(10 - os.nice(0)) if os.nice(0) < 10 else os.nice(0)
    except OSError:
        pass
    try:
        import psutil

        psutil.Process().ionice(psutil.IOPRIO_CLASS_BE, 7)
        out["ionice"] = "best-effort 7 (lowest)"
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.set_num_threads(4)
        out["torch_threads"] = 4
    except Exception:  # noqa: BLE001
        pass
    out["workers"] = max(2, min(8, (os.cpu_count() or 8) // 4))
    return out


def train(name: str, data_yaml: Path, base: Path = BASE_WEIGHTS, epochs: int = 80, gpu_frac: float = 0.8,
          imgsz: int = 640, patience: int = 25, seed: int = 0, fraction: float = 1.0, aug: str = "default") -> Path:
    from ultralytics import YOLO

    from ml.train_seg import GMCallbacks

    run_dir = RUNS / name
    gentle = be_gentle()
    kwargs: dict[str, Any] = dict(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=gpu_frac, device="0", workers=gentle["workers"],
        cache=False, project=str(RUNS), name=name, exist_ok=True, optimizer="AdamW", lr0=0.001, lrf=0.05,
        cos_lr=True, warmup_epochs=1, patience=patience, seed=seed, deterministic=False, plots=True, verbose=True,
        fraction=fraction, **GEOMETRY)
    if aug != "off":
        from ml.augment.india import india_transforms

        kwargs["augmentations"] = india_transforms(aug)
    try:
        from ml.train_seg import apply_display_reserve

        apply_display_reserve(kwargs)  # D20: caps PyTorch GPU memory and scales the batch fraction
    except Exception as e:  # noqa: BLE001
        print(f"[train_det] display reserve not applied: {e}", flush=True)
    print(f"[train_det] gentle mode {gentle}; augmentation: India '{aug}' + geometry {GEOMETRY}", flush=True)
    model = YOLO(str(base))
    try:
        cb = GMCallbacks(run_dir, run_dir / "gm_report.json", None, {"name": name, "base": str(base)},
                         gm={"batch_requested": gpu_frac})
        cb.attach(model)
    except Exception as e:  # noqa: BLE001 - the dashboard feed is optional
        print(f"[train_det] live dashboard feed off: {e}", flush=True)
    model.train(**kwargs)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"training finished without {best}")
    return best


def decide(cand_w: Path, ref_w: Path, name: str, tests: Optional[list[str]] = None) -> dict[str, Any]:
    ds = B.load_yaml(B.DATASETS)
    testsets = [t for t in ds["tests"] if not tests or t["id"] in tests]
    rep = B.benchmark([{"id": "reference", "path": str(ref_w), "role": "in use"},
                       {"id": name, "path": str(cand_w), "role": "candidate"}], testsets)
    ref, cand = rep["models"]
    adopt, why = B.no_test_worse(cand, ref)
    return {"benchmark": rep, "adopt": adopt, "reasons": why}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default=f"det_yolo11n_ft_{datetime.now().strftime('%Y%m%d_%H%M')}")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--data", default="base", help="training set id from configs/datasets.yaml train_sets (base | combined | base_neg | phone)")
    ap.add_argument("--aug", default="default", choices=("off", "light", "default", "strong"),
                    help="India road-condition augmentation strength (geometry augmentation is always on)")
    ap.add_argument("--smoke", action="store_true", help="trial: 1 epoch on 10 %% of the data, 2 test sets, never adopts")
    ap.add_argument("--gpu-frac", type=float, default=0.8, help="AutoBatch share of free GPU memory")
    ap.add_argument("--base", type=Path, default=BASE_WEIGHTS)
    ap.add_argument("--no-adopt", action="store_true", help="benchmark only, never switch the model in use")
    ap.add_argument("--skip-train", action="store_true", help="benchmark an existing runs/detect/<name>")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", a.name):
        raise SystemExit("bad --name")
    ds = B.load_yaml(B.DATASETS)
    sets = ds.get("train_sets") or {"base": ds["base"]["data_yaml"]}
    if a.data not in sets:
        raise SystemExit(f"unknown --data {a.data!r}; one of {sorted(sets)}")
    data_yaml = REPO / sets[a.data]
    if not data_yaml.is_file():
        how = {"combined": "combined-dataset", "base_neg": "negatives-dataset", "phone": "annotate-export"}.get(a.data, "base-dataset")
        raise SystemExit(f"{data_yaml} missing: run make {how}")
    if a.smoke:
        a.epochs, a.no_adopt = 1, True
    t0 = time.time()
    best = RUNS / a.name / "weights" / "best.pt" if a.skip_train else train(
        a.name, data_yaml, a.base, a.epochs, a.gpu_frac, fraction=0.1 if a.smoke else 1.0, aug=a.aug)
    ref = B.in_use_weights() or str(a.base)
    ref_path = Path(ref) if Path(ref).is_absolute() else REPO / ref
    res = decide(best, ref_path, a.name, ["base_test", "phone_street"] if a.smoke else None)
    out: dict[str, Any] = {"name": a.name, "data": a.data, "smoke": a.smoke, "epochs": a.epochs, "aug": a.aug,
                           "geometry": GEOMETRY,
                           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "candidate": str(best.relative_to(REPO)), "reference": ref, "adopt_rule": "not worse on "
                           "any test set (AP50 -0.01, false-alarm rate +0.01, frames +-0) and better mean box AP50",
                           "adopt": res["adopt"], "reasons": res["reasons"], "minutes": round((time.time() - t0) / 60, 1),
                           "benchmark": res["benchmark"]}
    if res["adopt"] and not a.no_adopt:
        CANDIDATES.mkdir(parents=True, exist_ok=True)
        dst = CANDIDATES / f"{a.name}.pt"
        shutil.copy2(best, dst)
        rel = str(dst.relative_to(REPO))
        set_env_weights(rel, f"D38 adopted {datetime.now():%Y-%m-%d}: fine-tuned from {a.base.name}, passed the "
                             f"benchmark rule (reports/train_{a.name}.json)")
        register_model({"id": a.name, "path": rel, "role": "candidate", "note": "fine-tuned from the base, adopted"})
        out["adopted_as"] = rel
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / f"train_{a.name}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("\n" + B.to_markdown(res["benchmark"]))
    print(("ADOPTED: " + out["adopted_as"]) if out.get("adopted_as") else
          ("NOT adopted: " + "; ".join(res["reasons"]) if not res["adopt"] else "passed the rule (--no-adopt given)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
