"""Merged pothole segmentation set (D48): every training source in one place, re-split train / val, leak-free.

    make merged-dataset          # -> datasets/merged_pothole_seg/ (symlinked images, one class: pothole)

Sources (test parts are never read; every test set stays in test_datasets/):
  - base: datasets/base_nagar_netra_pothole_seg (the supplied Nagar Netra potholes; boxes outlined by SAM2, D45)
  - earlier: datasets/earlier_india_pothole_v2 train + val: RDD-India (rddin_), RDD other countries (rddfx_),
    BharatPotHole dashcam (bh_), Kaggle web photos (ws_), eYantra (ey_)
Labels: potholes only (class 0); cracks, alligator cracks, manholes and clean roads stay as background, so the model
learns that they are NOT potholes. Outlines are kept as polygons (segmentation).

Split: groups never straddle train / val: near-duplicates (256-bit dHash <= 24), frames of one BharatPotHole drive,
one eYantra sequence, Roboflow variants of one photo. About --val-frac of every source goes to val (by groups, seeded),
so the best epoch is chosen on a mix like the tests, not on dashcam frames only.
Leak guard: any image that is a near-copy of ANY test image (test_datasets/*/images) is left out.
Balance: base images are repeated --base-repeat times and web photos --web-repeat times in train (close-range photos
like the phone walk; otherwise ~8,000 dashcam frames dominate).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dataset_tools.seg_geometry import dhash_words, hamming_words  # noqa: E402

BASE = REPO / "datasets" / "base_nagar_netra_pothole_seg"
EARLIER = REPO / "datasets" / "earlier_india_pothole_v2"
TESTS = REPO / "test_datasets"
OUT = REPO / "datasets" / "merged_pothole_seg"
IMG_EXT = (".jpg", ".jpeg", ".png")
DUP_HAMMING = 24
BASE_POTHOLE, EARLIER_POTHOLE = 1, 0


def source_of(name: str) -> str:
    for p in ("rddin", "rddfx", "bh", "ws", "ey"):
        if name.startswith(p + "_"):
            return p
    return "base"


def natural_group(name: str) -> str:
    """Images that must stay together: one BharatPotHole drive, one eYantra sequence, Roboflow variants of one photo."""
    stem = Path(name).stem
    src = source_of(name)
    if src == "bh":
        m = re.match(r"bh_(\d{14})", stem)
        return f"bh:{m.group(1) if m else stem}"
    if src == "ey":
        m = re.match(r"ey_(\d+_\d+)", stem)
        return f"ey:{m.group(1) if m else stem}"
    m = re.split(r"[._]jpe?g[._]rf[._]", stem, maxsplit=1, flags=re.I)
    return f"{src}:{m[0]}"


def pothole_lines(label: Path, pothole_id: int) -> list[str]:
    out = []
    for line in label.read_text(encoding="utf-8").splitlines() if label.is_file() else []:
        t = line.split()
        if len(t) >= 5 and int(float(t[0])) == pothole_id:
            out.append("0 " + " ".join(t[1:]))
    return out


def _hash(paths: list[Path]) -> np.ndarray:
    return np.stack([dhash_words(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]) if paths else \
        np.zeros((0, 4), np.uint64)


def near_pairs(h: np.ndarray, max_hamming: int = DUP_HAMMING, chunk: int = 1024) -> list[tuple[int, int]]:
    """All pairs i < j with Hamming <= max_hamming, in chunks (11k x 11k at once would need ~4 GB)."""
    out = []
    for s in range(0, len(h), chunk):
        d = hamming_words(h[s: s + chunk], h)
        ii, jj = np.nonzero(d <= max_hamming)
        out += [(s + i, j) for i, j in zip(ii.tolist(), jj.tolist(), strict=True) if s + i < j]
    return out


def near_any(h: np.ndarray, ref: np.ndarray, max_hamming: int = DUP_HAMMING, chunk: int = 2048) -> np.ndarray:
    if not len(h) or not len(ref):
        return np.zeros(len(h), bool)
    return np.concatenate([hamming_words(h[s: s + chunk], ref).min(1) <= max_hamming
                           for s in range(0, len(h), chunk)])


def union_groups(keys: list[str], pairs: list[tuple[int, int]]) -> list[int]:
    parent = list(range(len(keys)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in first:
            parent[find(i)] = find(first[k])
        else:
            first[k] = i
    for i, j in pairs:
        parent[find(i)] = find(j)
    return [find(i) for i in range(len(keys))]


def split_groups(sources: list[str], groups: list[int], val_frac: float, seed: int) -> list[str]:
    """~val_frac of the images of every source to val, whole groups at a time (groups keyed by their main source)."""
    rng = random.Random(seed)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    by_src: dict[str, list[int]] = defaultdict(list)
    for g, idx in members.items():
        by_src[Counter(sources[i] for i in idx).most_common(1)[0][0]].append(g)
    split = ["train"] * len(groups)
    for _src, gs in sorted(by_src.items()):
        gs = sorted(gs)
        rng.shuffle(gs)
        want, got = val_frac * sum(len(members[g]) for g in gs), 0
        for g in gs:
            if got >= want:
                break
            for i in members[g]:
                split[i] = "val"
            got += len(members[g])
    return split


def build(out: Path = OUT, base: Path = BASE, earlier: Path = EARLIER, tests: Path = TESTS, val_frac: float = 0.1,
          base_repeat: int = 3, web_repeat: int = 2, seed: int = 0, overwrite: bool = False) -> dict:
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    items: list[tuple[Path, list[str]]] = []
    for split in ("train", "val"):
        for root, pid in ((base, BASE_POTHOLE), (earlier, EARLIER_POTHOLE)):
            d = root / "images" / split
            for p in sorted(x for x in d.iterdir() if x.suffix.lower() in IMG_EXT) if d.is_dir() else []:
                items.append((p, pothole_lines(root / "labels" / split / f"{p.stem}.txt", pid)))
    names = [p.name for p, _ in items]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate file names across sources")
    print(f"[merged] {len(items)} images; hashing ...", flush=True)
    h = _hash([p for p, _ in items])
    test_imgs = sorted(p for d in tests.glob("*/images") for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    leak = near_any(h, _hash(test_imgs))
    keep = [i for i in range(len(items)) if not leak[i]]
    items, h = [items[i] for i in keep], h[keep]
    sources = [source_of(p.name) for p, _ in items]
    groups = union_groups([natural_group(p.name) for p, _ in items], near_pairs(h))
    split = split_groups(sources, groups, val_frac, seed)
    stats: dict = {"train": Counter(), "val": Counter(), "potholes": Counter(), "background": Counter()}

    def put(sp: str, src: Path, name: str, lines: list[str]) -> None:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (out / "images" / sp / name).symlink_to(src.resolve())
        (out / "labels" / sp / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                   encoding="utf-8")

    for (p, lines), src, sp in zip(items, sources, split, strict=True):
        name = p.name if src != "base" else f"nn_{p.name}"
        put(sp, p, name, lines)
        stats[sp][src] += 1
        stats["potholes"][sp] += len(lines)
        stats["background"][sp] += not lines
        reps = (base_repeat if src == "base" else web_repeat if src == "ws" else 1) if sp == "train" else 1
        for r in range(1, reps):
            put(sp, p, f"r{r}_{name}", lines)
            stats[sp][f"{src}_repeat"] += 1
    (out / "data.yaml").write_text(
        "# D48 merged pothole segmentation set (dataset_tools/build_merged_dataset.py): all training sources, one "
        "class, leak-free train/val\n"
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [\"pothole\"]\n",
        encoding="utf-8")
    summary = {"train_images": sum(stats["train"].values()), "val_images": sum(stats["val"].values()),
               "train_by_source": dict(stats["train"]), "val_by_source": dict(stats["val"]),
               "potholes": dict(stats["potholes"]), "background_images": dict(stats["background"]),
               "left_out_near_test": int(leak.sum()), "test_images_guarded": len(test_imgs),
               "groups": len(set(groups)), "val_frac": val_frac, "base_repeat": base_repeat,
               "web_repeat": web_repeat, "seed": seed}
    (out / "build_stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--base-repeat", type=int, default=3)
    ap.add_argument("--web-repeat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, val_frac=a.val_frac, base_repeat=a.base_repeat, web_repeat=a.web_repeat,
                           seed=a.seed, overwrite=a.overwrite), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
