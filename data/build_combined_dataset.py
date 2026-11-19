"""Combined fine-tuning set (D39): base dataset + the earlier dataset's TRAIN/VAL parts, test parts never included.

    make combined-dataset            # -> data/datasets/combined_ft/ (symlinked images, generated box labels)
    make negatives-dataset           # D42 -> data/datasets/base_neg/: base + ONLY the earlier images without potholes

D42 (`--earlier negatives`): the user's own outlines on the phone walk (D41) showed that the earlier POTHOLE labels
(dashcam, small far-away potholes) pull close-range phone recall down (combined fine-tune 7 / 41 outlines found, base
-only fine-tune 25 / 41), while the base-only fine-tune raised false alarms on clean roads (eYantra 24.9 %). This mode
keeps the base potholes and adds only the earlier images with no pothole: clean roads, cracks, alligator cracks,
manholes (all background), so the model learns what is NOT a pothole without learning a different idea of one.

Why: the first fine-tune on the pothole-only base dataset (D38, det_yolo11n_ft_v1) learnt to call clean road
"pothole" (eYantra false alarms 2.8 % -> 24.9 %). The earlier dataset adds what it lacked: Indian dashcam roads with
far-away potholes, cracks and manholes that are NOT potholes, and ~2,000 clean-road images.

- Base train (`data/datasets/nagar_netra_pothole/images/train`) is repeated --base-repeat times (default 5) so the
  ~600 base images are not drowned by the ~10,600 earlier ones.
- Earlier `india_pothole_seg4_v2` train + val: polygons -> boxes, pothole (id 0) -> id 1 (the base model's pothole);
  crack / alligator crack / manhole labels are dropped, so those regions and the empty images are background.
- Validation (best-epoch choice): the base val + --earlier-val images of the earlier val (fixed seed).
- Leakage guard: an earlier train/val image that is a near-duplicate (256-bit dHash <= 24) of ANY test image (base
  test, earlier test incl. the locked RDD-India 514, the phone street frames) is left out. The test parts themselves
  are never read for training.
- Images are symlinks (no copies); names are prefixed (`nn_`, `v2_`, `nnrK_` for repeats) to stay unique.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from data.seg_geometry import dhash_words, hamming_words  # noqa: E402

BASE = REPO / "data" / "datasets" / "nagar_netra_pothole"
V2 = REPO / "data" / "datasets" / "india_pothole_seg4_v2"
PHONE = REPO / "data" / "testsets" / "phone_street_20261118" / "images"
OUT = REPO / "data" / "datasets" / "combined_ft"
POTHOLE_OUT = 1  # base model's pothole id
V2_POTHOLE = 0
IMG_EXT = (".jpg", ".jpeg", ".png")
DUP_HAMMING = 24


def _imgs(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def _hash(paths: list[Path]) -> np.ndarray:
    return np.stack([dhash_words(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in paths]) if paths else \
        np.zeros((0, 4), np.uint64)


def v2_boxes(label: Path) -> list[str]:
    """Earlier label (polygon or box, id 0 = pothole) -> YOLO box lines with id 1; other classes dropped."""
    out = []
    for line in label.read_text(encoding="utf-8").splitlines() if label.is_file() else []:
        t = line.split()
        if len(t) < 5 or int(float(t[0])) != V2_POTHOLE:
            continue
        v = np.asarray(t[1:], float)
        if len(v) == 4:
            cx, cy, w, h = v
        else:
            xy = v[: len(v) // 2 * 2].reshape(-1, 2)
            (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if w > 0 and h > 0:
            out.append(f"{POTHOLE_OUT} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def base_boxes(label: Path) -> list[str]:
    """Base labels kept as they are (pothole = 1)."""
    return [ln for ln in (label.read_text(encoding="utf-8").splitlines() if label.is_file() else []) if ln.strip()]


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def build(out: Path = OUT, base_repeat: int = 5, earlier_val: int = 300, seed: int = 0, overwrite: bool = False,
          base: Path = BASE, v2: Path = V2, phone: Path = PHONE, earlier: str = "all", max_negatives: int = 1250) -> dict:
    if earlier not in ("all", "negatives"):
        raise ValueError("earlier must be 'all' or 'negatives'")
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    tests = _imgs(base / "images" / "test") + _imgs(v2 / "images" / "test") + _imgs(phone)
    th = _hash(tests)
    stats: dict = {"train": Counter(), "val": Counter(), "potholes": Counter(), "dropped_near_test": Counter(),
                   "skipped_earlier_with_potholes": Counter(), "negatives_not_used": Counter()}

    def put(split: str, src: Path, name: str, lines: list[str], tag: str) -> None:
        _link(src, out / "images" / split / name)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                       encoding="utf-8")
        stats[split][tag] += 1
        stats["potholes"][split] += len(lines)

    # base train (repeated) and base val
    for p in _imgs(base / "images" / "train"):
        lines = base_boxes(base / "labels" / "train" / f"{p.stem}.txt")
        for r in range(base_repeat):
            put("train", p, (f"nn_{p.name}" if r == 0 else f"nnr{r}_{p.name}"), lines, "base" if r == 0 else
                "base_repeat")
    for p in _imgs(base / "images" / "val"):
        put("val", p, f"nn_{p.name}", base_boxes(base / "labels" / "val" / f"{p.stem}.txt"), "base")
    # earlier train + val (never its test), minus near-copies of any test image
    rng = random.Random(seed)
    val_pool = _imgs(v2 / "images" / "val")
    if earlier == "negatives":  # D42: only images without a pothole label
        val_pool = [p for p in val_pool if not v2_boxes(v2 / "labels" / "val" / f"{p.stem}.txt")]
    val_pick = set(p.name for p in rng.sample(val_pool, min(earlier_val, len(val_pool))))
    neg_train: list[tuple[Path, str]] = []
    for split in ("train", "val"):
        cand = _imgs(v2 / "images" / split)
        near = (hamming_words(_hash(cand), th).min(1) <= DUP_HAMMING) if len(cand) and len(th) else \
            np.zeros(len(cand), bool)
        for p, is_near in zip(cand, near, strict=True):
            if is_near:
                stats["dropped_near_test"][split] += 1
                continue
            lines = v2_boxes(v2 / "labels" / split / f"{p.stem}.txt")
            if earlier == "negatives" and lines:
                stats["skipped_earlier_with_potholes"][split] += 1
                continue
            dst_split = "val" if (split == "val" and p.name in val_pick) else "train"
            if earlier == "negatives" and dst_split == "train":
                neg_train.append((p, split))
                continue
            put(dst_split, p, f"v2_{p.name}", lines, "earlier_" + p.stem.split("_", 1)[0])
    if neg_train:  # D42: about 30 % background; every eYantra clean road first, then cracks / manholes at random
        ey = [x for x in neg_train if x[0].name.startswith("ey_")]
        rest = [x for x in neg_train if not x[0].name.startswith("ey_")]
        rng.shuffle(rest)
        pick = (ey + rest)[:max_negatives]
        stats["negatives_not_used"]["train"] = len(neg_train) - len(pick)
        for p, _ in pick:
            put("train", p, f"v2_{p.name}", [], "earlier_" + p.stem.split("_", 1)[0])
    names = ["garbage_pile", "pothole", "encroachment", "billboard_hoarding", "cow", "manhole_closed", "manhole_open",
             "manhole_broken", "streetlight_working", "streetlight_not_working", "cat", "horse", "dog", "buffalo",
             "goat", "camel"]
    (out / "data.yaml").write_text(
        f"# D39/D42 fine-tuning set (data/build_combined_dataset.py --earlier {earlier}): base train x"
        f"{base_repeat} + earlier train/val ({'images without potholes only' if earlier == 'negatives' else 'all'}), "
        "test parts excluded\n"
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {json.dumps(names)}\n", encoding="utf-8")
    summary = {"train_images": sum(stats["train"].values()), "val_images": sum(stats["val"].values()),
               "train_by_source": dict(stats["train"]), "val_by_source": dict(stats["val"]),
               "potholes": dict(stats["potholes"]), "dropped_near_test": dict(stats["dropped_near_test"]),
               "earlier": earlier, "skipped_earlier_with_potholes": dict(stats["skipped_earlier_with_potholes"]),
               "negatives_not_used": dict(stats["negatives_not_used"]),
               "background_train_images": sum(1 for f in (out / "labels" / "train").glob("*.txt")
                                              if not f.read_text().strip()),
               "base_repeat": base_repeat, "earlier_val": earlier_val, "seed": seed, "test_images_guarded": len(tests)}
    (out / "build_stats.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--base-repeat", type=int, default=5)
    ap.add_argument("--earlier-val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--earlier", choices=("all", "negatives"), default="all",
                    help="earlier train/val images to add: all (D39) or only those without potholes (D42)")
    ap.add_argument("--max-negatives", type=int, default=1250,
                    help="with --earlier negatives: at most this many background training images (~30 %%)")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.out, a.base_repeat, a.earlier_val, a.seed, a.overwrite, earlier=a.earlier,
                                max_negatives=a.max_negatives), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
