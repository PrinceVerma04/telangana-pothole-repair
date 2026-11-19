"""Base dataset (D38): the Nagar Netra pothole set, with a leak-free train/val/test split.

    make base-dataset          # -> datasets/base_nagar_netra_pothole/  (images, labels, data.yaml, split_manifest.json)

Source: ``datasets/supplied_original/pothole_dataset_trained_20260907`` (710 images, YOLO boxes, pothole = class id 1 of
the 16 Nagar Netra names, so the head of the supplied model lines up). The supplied split leaks: 15 test images have a
near-copy (same scene, shot moments apart) in train/val (D36). Here images are grouped by near-duplicate (256-bit
dHash Hamming <= 24, the same rule as our v2 build, with union-find so chains stay together) and every group is put in
ONE split: the split most of its members already had, ties -> train. Test and val therefore never share a scene with
train. The supplied assignment is kept in split_manifest.json for reference. Images are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dataset_tools.seg_geometry import dhash_words, hamming_words  # noqa: E402

SRC = REPO / "datasets" / "supplied_original" / "pothole_dataset_trained_20260907"
OUT = REPO / "datasets" / "base_nagar_netra_pothole"
SPLITS = ("train", "val", "test")
DUP_HAMMING = 24
NAMES = ["garbage_pile", "pothole", "encroachment", "billboard_hoarding", "cow", "manhole_closed", "manhole_open",
         "manhole_broken", "streetlight_working", "streetlight_not_working", "cat", "horse", "dog", "buffalo", "goat",
         "camel"]
POTHOLE_ID = 1


def groups_by_near_duplicate(hashes: np.ndarray, max_hamming: int = DUP_HAMMING) -> list[int]:
    """Union-find over all pairs closer than ``max_hamming`` -> group id per image."""
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    d = hamming_words(hashes, hashes)
    ii, jj = np.nonzero(np.triu(d <= max_hamming, 1))
    for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
        parent[find(i)] = find(j)
    return [find(i) for i in range(n)]


def assign_splits(orig: list[str], groups: list[int]) -> list[str]:
    """Each group goes to the split most of its members had (ties: train > val > test, so test only loses images)."""
    by_group: dict[int, Counter] = {}
    for s, g in zip(orig, groups, strict=True):
        by_group.setdefault(g, Counter())[s] += 1
    rank = {"train": 0, "val": 1, "test": 2}
    choice = {g: min(c.items(), key=lambda kv: (-kv[1], rank[kv[0]]))[0] for g, c in by_group.items()}
    return [choice[g] for g in groups]


def build(src: Path = SRC, out: Path = OUT, overwrite: bool = False) -> dict:
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    items = []
    for s in SPLITS:
        for p in sorted((src / "images" / s).glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                items.append((s, p, src / "labels" / s / f"{p.stem}.txt"))
    hashes = np.stack([dhash_words(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for _, p, _ in items])
    groups = groups_by_near_duplicate(hashes)
    new = assign_splits([s for s, _, _ in items], groups)
    manifest, counts, boxes = [], Counter(), Counter()
    for (s0, p, lab), g, s1 in zip(items, groups, new, strict=True):
        (out / "images" / s1).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s1).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out / "images" / s1 / p.name)
        text = lab.read_text(encoding="utf-8") if lab.is_file() else ""
        (out / "labels" / s1 / f"{p.stem}.txt").write_text(text, encoding="utf-8")
        nb = sum(1 for line in text.splitlines() if line.split() and int(line.split()[0]) == POTHOLE_ID)
        counts[s1] += 1
        boxes[s1] += nb
        manifest.append({"image": p.name, "supplied_split": s0, "split": s1, "group": int(g), "potholes": nb})
    (out / "data.yaml").write_text(
        "# D38 base dataset: Nagar Netra pothole subset, leak-free split (dataset_tools/build_base_dataset.py)\n"
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n", encoding="utf-8")
    moved = Counter(f"{m['supplied_split']}->{m['split']}" for m in manifest if m["supplied_split"] != m["split"])
    stats = {"source": str(src.relative_to(REPO)) if src.is_relative_to(REPO) else str(src), "images": dict(counts),
             "potholes": dict(boxes), "moved": dict(moved), "dup_hamming": DUP_HAMMING,
             "groups": len(set(groups)), "pothole_class_id": POTHOLE_ID}
    (out / "split_manifest.json").write_text(json.dumps({"stats": stats, "images": manifest}, indent=1),
                                            encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--separate-test", action="store_true",
                    help="D43: move the test split to test_datasets/nagar_netra_test after the build")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.src, a.out, a.overwrite), indent=1))
    if a.separate_test:
        from dataset_tools.test_split import BASE_TEST, separate_test_split

        print(json.dumps({"test_moved_to_test_datasets": separate_test_split(a.out, BASE_TEST)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
