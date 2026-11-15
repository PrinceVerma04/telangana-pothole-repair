"""Derive a training dataset from the built ``india_pothole_seg`` (WP1 output) without rebuilding it (DECISIONS D15).

    .venv\\Scripts\\python.exe data\\derive_dataset.py                       # -> data/datasets/india_pothole_seg4
    .venv\\Scripts\\python.exe data\\derive_dataset.py --keep auto --resplit-class manhole --fractions 0.70,0.15,0.15

Two changes, both recorded in the output's ``stats.json``:

1. **Classes:** only the classes that have labelled examples are kept (``--keep auto``: every class with >= 1 instance
   in the train split; or an explicit comma list). Label ids are renumbered 0..k-1 in the frozen canonical order of
   ``configs/classes.yaml``, so the model trained on it has exactly k outputs. The system-wide class vocabulary
   (``configs/classes.yaml``, edge wire protocol, server) is unchanged: models are mapped to canonical classes by
   name (``ml.common.canonical_class``).
2. **Re-split of one class** (``--resplit-class manhole``): every image group that contains that class is
   re-assigned to train / val / test so the class's *instances* follow ``--fractions`` (default 70/15/15). A group is
   an eYantra drive sequence (``ey_<sequence>_frame…``), so near-identical consecutive frames never end up on both
   sides of a split; any other image is its own group. Groups move whole, including their frames without that class.
   Images that are not in such a group keep their split.

Images are hard-linked (no extra disk space; copied when hard links are not possible) and labels are rewritten. The
input dataset is never modified. Deterministic for a given ``--seed``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.common import class_names  # noqa: E402

SPLITS = ("train", "val", "test")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MARKER = ".gm_derived_dataset"


@dataclass
class Img:
    stem: str
    image: Path
    label: Path
    split: str
    lines: list[tuple[str, str]] = field(default_factory=list)  # (class name, coordinates)

    def count(self, cls: str) -> int:
        return sum(1 for c, _ in self.lines if c == cls)


def group_of(stem: str) -> str:
    """eYantra frames of one drive sequence share a group; everything else is its own group."""
    if stem.startswith("ey_") and "_frame" in stem:
        return stem.split("_frame")[0]
    return stem


def read_dataset(src: Path) -> tuple[dict[int, str], list[Img]]:
    cfg = yaml.safe_load((src / "data.yaml").read_text(encoding="utf-8"))
    names = {int(k): str(v) for k, v in (cfg.get("names") or {}).items()}
    imgs: list[Img] = []
    for split in SPLITS:
        idir = src / "images" / split
        if not idir.is_dir():
            continue
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            lab = src / "labels" / split / f"{p.stem}.txt"
            it = Img(p.stem, p, lab, split)
            if lab.is_file():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    t = line.split(maxsplit=1)
                    if len(t) == 2:
                        it.lines.append((names.get(int(float(t[0])), t[0]), t[1]))
            imgs.append(it)
    return names, imgs


def choose_keep(keep: str, names: dict[int, str], imgs: list[Img]) -> list[str]:
    canon = list(class_names())
    if keep == "auto":
        present = Counter(c for it in imgs if it.split == "train" for c, _ in it.lines)
        chosen = [n for n in canon if present[n] > 0]
    else:
        chosen = [k.strip() for k in keep.split(",") if k.strip()]
        bad = [k for k in chosen if k not in canon]
        if bad:
            raise SystemExit(f"--keep: {bad} are not classes of configs/classes.yaml {canon}")
        chosen = [n for n in canon if n in chosen]  # canonical order
    if not chosen:
        raise SystemExit("no class to keep")
    return chosen


def resplit(imgs: list[Img], cls: str, fractions: tuple[float, float, float], seed: int) -> dict[str, object]:
    """Re-assign the groups containing ``cls`` so its instances follow ``fractions`` (train, val, test)."""
    groups: dict[str, list[Img]] = defaultdict(list)
    for it in imgs:
        groups[group_of(it.stem)].append(it)
    movable = {g: members for g, members in groups.items() if any(m.count(cls) for m in members)}
    before = Counter()
    for members in movable.values():
        for m in members:
            before[m.split] += m.count(cls)
    total = sum(before.values())
    if total == 0:
        return {"class": cls, "instances": 0, "groups": 0, "note": "class absent; nothing re-split"}
    target = {s: f * total for s, f in zip(SPLITS, fractions, strict=True)}
    order = sorted(movable)
    random.Random(seed).shuffle(order)
    counts = [sum(m.count(cls) for m in movable[g]) for g in order]
    assign = best_assignment(counts, [target[s] for s in SPLITS], total) if len(order) <= EXHAUSTIVE_MAX_GROUPS \
        else greedy_assignment(counts, [target[s] for s in SPLITS])
    got: Counter = Counter()
    moved_images = 0
    for g, n, si in zip(order, counts, assign, strict=True):
        dest = SPLITS[si]
        got[dest] += n
        for m in movable[g]:
            if m.split != dest:
                moved_images += 1
            m.split = dest
    return {"class": cls, "instances": total, "groups": len(movable), "images_moved": moved_images,
            "method": "exhaustive" if len(order) <= EXHAUSTIVE_MAX_GROUPS else "greedy",
            "groups_by_split": {s: sorted(g for g, si in zip(order, assign, strict=True) if SPLITS[si] == s)
                                for s in SPLITS},
            "instances_before": dict(before), "instances_after": {s: got[s] for s in SPLITS},
            "fractions_after": {s: round(got[s] / total, 3) for s in SPLITS},
            "target_fractions": dict(zip(SPLITS, fractions, strict=True))}


EXHAUSTIVE_MAX_GROUPS = 10  # 3**10 = 59,049 assignments: instant


def _cost(got: list[float], target: list[float], total: float) -> float:
    c = sum(((g - t) / total) ** 2 for g, t in zip(got, target, strict=True))
    # a split that should receive the class but gets none is far worse than an uneven share
    c += sum(1.0 for g, t in zip(got, target, strict=True) if t > 0 and g == 0)
    return c


def best_assignment(counts: list[int], target: list[float], total: float) -> list[int]:
    """Exact: try every group -> split assignment, keep the one closest to the target shares (few groups only)."""
    import itertools

    best, best_c = None, float("inf")
    for combo in itertools.product(range(len(target)), repeat=len(counts)):
        got = [0.0] * len(target)
        for n, si in zip(counts, combo, strict=True):
            got[si] += n
        c = _cost(got, target, total)
        if c < best_c - 1e-12:
            best, best_c = list(combo), c
    assert best is not None
    return best


def greedy_assignment(counts: list[int], target: list[float]) -> list[int]:
    """Many groups: largest first, each to the split furthest below its target (relative)."""
    got = [0.0] * len(target)
    out = [0] * len(counts)
    for i in sorted(range(len(counts)), key=lambda k: -counts[k]):
        si = max(range(len(target)), key=lambda s: ((target[s] - got[s]) / max(target[s], 1e-9), -s))
        got[si] += counts[i]
        out[i] = si
    return out


def link_or_copy(src: Path, dst: Path) -> str:
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write(out: Path, imgs: list[Img], keep: list[str]) -> dict[str, object]:
    new_id = {n: i for i, n in enumerate(keep)}
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)
    how: Counter = Counter()
    dropped: Counter = Counter()
    per_split: dict[str, dict[str, object]] = {}
    lists: dict[str, list[str]] = defaultdict(list)
    for it in sorted(imgs, key=lambda i: (i.split, i.stem)):
        how[link_or_copy(it.image, out / "images" / it.split / it.image.name)] += 1
        lines = []
        for c, coords in it.lines:
            if c in new_id:
                lines.append(f"{new_id[c]} {coords}")
            else:
                dropped[c] += 1
        (out / "labels" / it.split / f"{it.stem}.txt").write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
        lists[it.split].append(it.image.name)
        s = per_split.setdefault(it.split, {"images": 0, "images_with_labels": 0, "instances": Counter()})
        s["images"] = int(s["images"]) + 1  # type: ignore[call-overload]
        s["images_with_labels"] = int(s["images_with_labels"]) + bool(lines)  # type: ignore[call-overload]
        for c, _ in it.lines:
            if c in new_id:
                s["instances"][c] += 1  # type: ignore[index]
    for split, names in lists.items():
        (out / "splits" / f"{split}.txt").write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
    (out / "data.yaml").write_text(
        "# Written by data/derive_dataset.py (DECISIONS D15). Ordered subset of configs/classes.yaml.\n"
        + yaml.safe_dump({"path": out.resolve().as_posix(), "train": "images/train", "val": "images/val",
                          "test": "images/test", "nc": len(keep), "names": {i: n for i, n in enumerate(keep)}},
                         sort_keys=False), encoding="utf-8")
    return {"files": dict(how), "dropped_instances": dict(dropped),
            "splits": {s: {**v, "instances": {n: v["instances"][n] for n in keep}}  # type: ignore[index]
                       for s, v in sorted(per_split.items(), key=lambda kv: SPLITS.index(kv[0]))}}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "datasets" / "india_pothole_seg")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "datasets" / "india_pothole_seg4")
    ap.add_argument("--keep", default="auto", help="'auto' (classes with train labels) or a comma list of classes")
    ap.add_argument("--resplit-class", default="manhole", help="class to spread over train/val/test ('' = none)")
    ap.add_argument("--fractions", default="0.70,0.15,0.15", help="train,val,test share of that class's instances")
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing derived dataset at --out")
    a = ap.parse_args(argv)

    src, out = a.src.resolve(), a.out.resolve()
    if src == out:
        ap.error("--out must differ from --src (the source dataset is never modified)")
    if not (src / "data.yaml").is_file():
        ap.error(f"{src} has no data.yaml (build it first: tasks.ps1 dataset / make dataset)")
    fr = tuple(float(x) for x in a.fractions.split(","))
    if len(fr) != 3 or any(x < 0 for x in fr) or abs(sum(fr) - 1.0) > 1e-6:
        ap.error("--fractions needs three non-negative numbers that sum to 1, e.g. 0.70,0.15,0.15")
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            ap.error(f"{out} is not empty (use --overwrite)")
        if not (out / MARKER).is_file():
            ap.error(f"{out} was not written by derive_dataset.py; refusing to delete it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    names, imgs = read_dataset(src)
    keep = choose_keep(a.keep, names, imgs)
    split_info = resplit(imgs, a.resplit_class, fr, a.seed) if a.resplit_class else None  # type: ignore[arg-type]
    res = write(out, imgs, keep)
    (out / MARKER).write_text(f"derived from {src}\n", encoding="utf-8")
    stats = {"dataset": out.name, "derived_from": str(src), "derived_at": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "script": "data/derive_dataset.py", "seed": a.seed,
        "names": {str(i): n for i, n in enumerate(keep)},
        "removed_classes": [n for n in names.values() if n not in keep], "resplit": split_info, **res}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")

    print(f"derived {out} from {src}")
    print(f"classes kept ({len(keep)}): {', '.join(keep)}; removed: {', '.join(stats['removed_classes']) or '-'}")
    if split_info:
        print(f"re-split '{split_info['class']}': {split_info.get('groups')} groups, "
              f"{split_info.get('images_moved', 0)} images moved; instances before {split_info.get('instances_before')}"
              f" -> after {split_info.get('instances_after')}")
    for s, v in res["splits"].items():  # type: ignore[attr-defined]
        print(f"  {s:<5} {v['images']:>5} images  " + "  ".join(f"{n} {c}" for n, c in v["instances"].items()))
    print(f"files: {res['files']}; dropped label lines: {res['dropped_instances'] or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
