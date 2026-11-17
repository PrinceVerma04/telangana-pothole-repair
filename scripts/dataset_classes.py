"""Print the classes of the training dataset and how many labelled instances each split has.

    python scripts/dataset_classes.py [--dataset data/datasets/india_pothole_seg4_v2]

Reads only ``data.yaml`` and ``stats.json`` written by ``data/build_dataset.py``; when ``stats.json`` is missing it
counts the YOLO-seg label files itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def count_labels(labels_dir: Path, names: dict[int, str]) -> tuple[int, Counter]:
    c: Counter = Counter()
    n = 0
    for f in labels_dir.glob("*.txt"):
        n += 1
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                c[names.get(int(float(parts[0])), parts[0])] += 1
    return n, c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=Path, default=None,
                    help="dataset folder (default: the training dataset of configs/model/india_pothole_seg.yaml)")
    a = ap.parse_args(argv)
    if a.dataset is None:
        sys.path.insert(0, str(ROOT))
        from ml.common import resolve_dataset_root

        d = resolve_dataset_root()
    else:
        d = a.dataset
    if not (d / "data.yaml").is_file():
        print(f"no dataset at {d} (build it with: make dataset)")
        return 1
    cfg = yaml.safe_load((d / "data.yaml").read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in (cfg.get("names") or {}).items()}
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8")) if (d / "stats.json").is_file() else None
    per_split: dict[str, tuple[int, Counter]] = {}
    for split in ("train", "val", "test"):
        if stats and split in stats.get("splits", {}):
            s = stats["splits"][split]
            per_split[split] = (int(s["images"]), Counter(s.get("instances") or {}))
        elif (d / "labels" / split).is_dir():
            per_split[split] = count_labels(d / "labels" / split, names)
    used = [n for n in names.values() if any(c[n] for _, c in per_split.values())]
    print(f"Dataset {d}")
    print(f"Classes declared: {len(names)}   classes with labelled examples: {len(used)}")
    w = max(len(n) for n in names.values())
    head = "  ".join(f"{s:>7}" for s in per_split)
    print(f"  {'id':>2}  {'class':<{w}}  {head}")
    for i, n in sorted(names.items()):
        row = "  ".join(f"{per_split[s][1][n]:>7}" for s in per_split)
        flag = "" if n in used else "   <- no labels: the model cannot learn this class yet"
        print(f"  {i:>2}  {n:<{w}}  {row}{flag}")
    print(f"  {'':>2}  {'images':<{w}}  " + "  ".join(f"{per_split[s][0]:>7}" for s in per_split))
    return 0


if __name__ == "__main__":
    sys.exit(main())
