"""RoadFix WP1 - build the unified YOLO-seg dataset ``india_pothole_seg`` (PLAN 3.3, DECISIONS D6).

Inputs are the unpacked sources in data/interim/ (``python -I data/download.py --all``); nothing under data/raw/ is read.

    make dataset   # = .venv/bin/python data/build_dataset.py --out data/datasets/india_pothole_seg_v2 --sam2 \
                  #   --sam2-max-gpu-gb 10 --negatives-frac 0.2 --include-bharat --include-webseg
    .venv/bin/python data/build_dataset.py --out /tmp/x --limit 20            # quick smoke build (box polygons)

Sources and label mapping (class ids from configs/classes.yaml, frozen):
  * RDD2022-India (720x720 frames of the HF mirror): D40->pothole, D20->alligator_crack, D00/D10->crack; the
    mirror's id 3 ("other corruption": line/crosswalk blur) is dropped. Boxes -> SAM2.1 masks (keep if
    IoU(mask bbox, box) >= 0.6, else the box polygon).
  * RDD2022 non-India frames that contain D40 (Japan/Czech/USA/China): train only, same mapping (--foreign-d40).
    Val/test stay India-only.
  * eYantra Indian roads (labelme polygons, ground-truth masks). Inspected 2026-11-13: its "shallow" class is
    manhole / drain covers and gratings, and its "pothole" class is mostly curb-side drains, loose pavers and
    road-edge drops. Defaults therefore are --eyantra-shallow manhole (class 4 exists to absorb exactly these FPs)
    and --eyantra-pothole exclude_image (frames with that ambiguous label are left out, so neither a wrong
    "pothole" nor a wrong "background" is taught). The literal contract mapping (shallow->pothole_water,
    pothole->pothole) and the PLAN brightness heuristic remain selectable.
  * Negatives: frames without any pothole instance must be >= --negatives-frac of train; fully empty frames are
    capped at --empty-max-frac so they cannot swamp training (RDD-India is ~50 % empty).
  * BharatPotHole: only with --include-bharat (licence unverified); train+valid -> train, test -> val.

Splits (persisted in --splits-dir so re-runs are stable): test = up to --test-n RDD-India frames with D40;
val = eYantra val (GT masks) + --val-frac of the remaining RDD-India frames, stratified (pothole and pothole-free
groups each contribute --val-frac); train = everything else. RDD-India frames are grouped by a 256-bit dHash
(Hamming <= --dup-hamming) so near-duplicates (consecutive frames of one drive) never straddle splits.

Outputs in --out: images/{train,val,test}, labels/{train,val,test} (YOLO-seg polygons), data.yaml, stats.json,
qc/label_studio_tasks.json (2 % sample for human QC), optional aux_road/ (eYantra road masks).
Resumable: SAM2 results are cached per image under data/interim/sam2_cache/.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from data.seg_geometry import (  # noqa: E402
    box_polygon,
    clip_polygon,
    dhash_words,
    hamming_words,
    parse_yolo_seg_line,
    polygon_area,
    rasterize_polygons,
    yolo_box_to_xyxy,
    yolo_seg_line,
)

DEFAULT_OUT = ROOT / "data" / "datasets" / "india_pothole_seg_v2"  # D16; v1 lives in india_pothole_seg until D18 cleanup
DEFAULT_SPLITS = ROOT / "data" / "splits" / "v2"
DEFAULT_INTERIM = ROOT / "data" / "interim"
CLASSES_YAML = ROOT / "configs" / "classes.yaml"
SPLITS = ("train", "val", "test")
POTHOLE_CLASSES = ("pothole", "pothole_water")
BUILD_VERSION = 3
LICENCES = {
    "rdd_india": "CC BY-SA 4.0 (RDD2022)",
    "rdd_foreign": "CC BY-SA 4.0 (RDD2022)",
    "rdd_negpool": "CC BY-SA 4.0 (RDD2022)",
    "eyantra": "GPL-2.0 (Kaggle eyantraiit) - prototype/evaluation only",
    "bharat": "CC BY-SA 4.0 (Kaggle surbhisaswatimohanty/bharatpothole metadata, verified 2026-11-16)",
    "webseg": "Apache-2.0 (Kaggle farzadnekouei/pothole-image-segmentation-dataset; Roboflow origin CC BY 4.0)",
}
UID_PREFIX = {"rdd_india": "rddin_", "rdd_foreign": "rddfx_", "rdd_negpool": "rddng_", "eyantra": "ey_",
              "bharat": "bh_", "webseg": "ws_"}


def log(msg: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ data model
@dataclass
class Inst:
    cls: str
    box: tuple[float, float, float, float] | None = None  # pixel xyxy (box sources)
    poly: np.ndarray | None = None  # pixel polygon (GT polygon sources, or resolved later)
    mask_source: str = ""  # gt_polygon | sam2 | box_fallback | box


@dataclass
class Item:
    uid: str
    source: str
    image: Path
    w: int
    h: int
    insts: list[Inst] = field(default_factory=list)
    fixed_split: str | None = None
    group: str = ""
    aux_road: list[np.ndarray] | None = None
    tags: set[str] = field(default_factory=set)

    @property
    def has_pothole(self) -> bool:
        return any(i.cls in POTHOLE_CLASSES for i in self.insts)

    @property
    def empty(self) -> bool:
        return not self.insts


def load_classes(path: Path = CLASSES_YAML) -> tuple[dict[str, int], dict]:
    cfg = yaml.safe_load(path.read_text())
    names = {str(v): int(k) for k, v in cfg["names"].items()}
    return names, cfg.get("source_maps", {})


def stable_rng(seed: int, *salt: str) -> random.Random:
    h = hashlib.sha1(("|".join([str(seed), *salt])).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def image_size(p: Path) -> tuple[int, int]:
    with Image.open(p) as im:
        return im.size


# ------------------------------------------------------------------------------------------------ collectors
def _read_yolo_boxes(lab: Path, w: int, h: int, id_map: dict[int, str | None], drop: collections.Counter) -> list[Inst]:
    insts = []
    if not lab.exists():
        return insts
    for line in lab.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            if parts:
                drop["malformed_label_lines"] += 1
            continue
        cid = int(float(parts[0]))
        cls = id_map.get(cid)
        if cls is None:
            drop[f"unmapped_class_{cid}"] += 1
            continue
        x1, y1, x2, y2 = yolo_box_to_xyxy(*map(float, parts[1:]), w, h)
        if (x2 - x1) < 3 or (y2 - y1) < 3:
            drop["tiny_boxes"] += 1
            continue
        insts.append(Inst(cls=cls, box=(x1, y1, x2, y2)))
    return insts


def rdd_id_map(interim: Path, source_maps: dict) -> dict[int, str | None]:
    meta = json.loads((interim / "rdd2022" / "meta.json").read_text())
    rdd_map = source_maps["rdd2022"]
    return {int(k): rdd_map.get(code) for k, code in meta["class_ids"].items()}


def _limit(items: list, limit: int | None, seed: int, salt: str, prefer=None) -> list:
    items = sorted(items, key=lambda x: x[0] if isinstance(x, tuple) else str(x))
    if limit is None or len(items) <= limit:
        return items
    rng = stable_rng(seed, "limit", salt)
    rng.shuffle(items)
    if prefer is None:
        return items[:limit]
    a = [x for x in items if prefer(x)]
    b = [x for x in items if not prefer(x)]
    k = min(len(a), math.ceil(limit / 2))
    return a[:k] + b[: limit - k]


def collect_rdd(interim: Path, subset: str, source: str, id_map, limit, seed, drop, fixed_split=None) -> list[Item]:
    base = interim / "rdd2022" / subset
    if not (base / "images").is_dir():
        log(f"{source}: {base} missing - run data/download.py --rdd")
        return []
    index = {}
    idx_path = interim / "rdd2022" / "index.jsonl"
    if idx_path.exists():
        for line in idx_path.read_text().splitlines():
            r = json.loads(line)
            index[r["id"]] = r
    stems = [p.stem for p in (base / "images").glob("*.jpg")]
    d40_id = next((k for k, v in id_map.items() if v == "pothole"), None)

    def has_d40(stem: str) -> bool:
        lab = base / "labels" / f"{stem}.txt"
        return lab.exists() and any(ln.split()[:1] == [str(d40_id)] for ln in lab.read_text().splitlines())

    stems = _limit(stems, limit, seed, source, prefer=has_d40 if source == "rdd_india" else None)
    items = []
    for stem in stems:
        img = base / "images" / f"{stem}.jpg"
        r = index.get(stem)
        w, h = (r["w"], r["h"]) if r and r.get("w") else image_size(img)
        insts = _read_yolo_boxes(base / "labels" / f"{stem}.txt", w, h, id_map, drop)
        prefix = {"rdd_india": "rddin", "rdd_foreign": "rddfx", "rdd_negpool": "rddng"}[source]
        it = Item(uid=f"{prefix}_{stem}", source=source, image=img, w=w, h=h, insts=insts, fixed_split=fixed_split)
        if r and r.get("OTHER"):
            drop["rdd_other_boxes"] += int(r["OTHER"])
            it.tags.add("had_other")
        items.append(it)
    log(f"{source}: {len(items)} images")
    return items


def shallow_heuristic(img_bgr: np.ndarray, poly: np.ndarray) -> str:
    """PLAN 3.3 heuristic (kept for --eyantra-shallow plan_heuristic): bright + smooth inside vs. a ring around
    the polygon => pothole_water, else pothole. NOTE: on eYantra this class is manholes, so it is not the default."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inner = rasterize_polygons([poly], w, h)
    ring = cv2.dilate(inner.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & ~inner
    if inner.sum() < 50 or ring.sum() < 50:
        return "pothole"
    lstd = cv2.blur(gray * gray, (7, 7)) - cv2.blur(gray, (7, 7)) ** 2
    bright = gray[inner].mean() > 1.15 * gray[ring].mean()
    smooth = np.sqrt(np.clip(lstd[inner], 0, None)).mean() < 0.8 * np.sqrt(np.clip(lstd[ring], 0, None)).mean()
    return "pothole_water" if bright and smooth else "pothole"


def collect_eyantra(interim: Path, source_maps: dict, shallow_mode: str, pothole_mode: str, limit, seed, drop,
                    road_aux: bool) -> list[Item]:
    base = interim / "eyantra"
    items = []
    contract = source_maps.get("eyantra", {})
    for split in ("train", "val"):
        pdir = base / split / "polygons"
        if not pdir.is_dir():
            log(f"eyantra: {pdir} missing - run data/download.py --eyantra")
            continue
        stems = _limit([p.stem for p in pdir.glob("*.json")], limit, seed, f"eyantra_{split}")
        for stem in stems:
            img = base / split / "images" / f"{stem}.jpg"
            if not img.exists():
                drop["eyantra_missing_image"] += 1
                continue
            d = json.loads((pdir / f"{stem}.json").read_text())
            w, h = int(d.get("imageWidth") or 0), int(d.get("imageHeight") or 0)
            if not w or not h:
                w, h = image_size(img)
            shapes = [s for s in d.get("shapes", []) if s.get("shape_type", "polygon") == "polygon" and len(s.get("points", [])) >= 3]
            if pothole_mode == "exclude_image" and any(s["label"] == "pothole" for s in shapes):
                drop["eyantra_images_excluded_ambiguous_pothole_label"] += 1
                continue
            insts, road = [], []
            bgr = None
            for s in shapes:
                poly = np.clip(np.array(s["points"], np.float32), 0, [w, h])
                lab = s["label"]
                if lab == "road":
                    road.append(poly)
                    continue
                if lab == "pothole":
                    if pothole_mode == "drop_label":
                        drop["eyantra_pothole_labels_dropped"] += 1
                        continue
                    cls = contract.get("pothole", "pothole")
                elif lab == "shallow":
                    if shallow_mode == "drop":
                        drop["eyantra_shallow_dropped"] += 1
                        continue
                    if shallow_mode == "manhole":
                        cls = "manhole"
                    elif shallow_mode == "contract":
                        cls = contract.get("shallow", "pothole_water")
                    else:  # plan_heuristic
                        bgr = bgr if bgr is not None else cv2.imread(str(img))
                        cls = shallow_heuristic(bgr, poly)
                else:
                    continue  # footpath and anything else: background
                if polygon_area(poly) < 16:
                    drop["tiny_polygons"] += 1
                    continue
                insts.append(Inst(cls=cls, poly=poly, mask_source="gt_polygon"))
            it = Item(uid=f"ey_{stem}", source="eyantra", image=img, w=w, h=h, insts=insts, fixed_split=split,
                      group=f"ey_{stem.split('_frame')[0]}", aux_road=road if road_aux else None)
            items.append(it)
    log(f"eyantra: {len(items)} images ({drop['eyantra_images_excluded_ambiguous_pothole_label']} excluded for the ambiguous 'pothole' label)")
    return items


BH_DRIVE_RE = re.compile(r"^(.+?)_frame_\d+")


def bharat_drive(stem: str) -> str:
    """Drive (source video) of a BharatPotHole frame: ``20250528094319_0060_frame_00012_jpg_rf_<hash>`` ->
    ``20250528094319_0060``; ``signal-2025-06-25-183557_frame_00080...`` -> ``signal-2025-06-25-183557``."""
    m = BH_DRIVE_RE.match(stem)
    return m.group(1) if m else stem.split("_jpg_rf_")[0]


def split_groups(sizes: dict[str, int], fractions: tuple[float, float, float], seed: int, salt: str) -> dict[str, str]:
    """Assign whole groups to train/val/test so each split gets about its share of the items (deterministic).

    Groups are visited in a seeded order; a group goes to test (then val) while that split is below its target and
    the group does not push it more than 10 % past it; everything else is train."""
    total = sum(sizes.values())
    target = {"test": fractions[2] * total, "val": fractions[1] * total}
    got = {"test": 0, "val": 0}
    order = sorted(sizes)
    stable_rng(seed, salt).shuffle(order)
    out = {}
    for g in order:
        n = sizes[g]
        dest = "train"
        for s in ("test", "val"):
            if got[s] < target[s] and got[s] + n <= 1.1 * target[s]:
                dest = s
                break
        if dest != "train":
            got[dest] += n
        out[g] = dest
    return out


def collect_bharat(interim: Path, source_maps: dict, limit, seed, drop, split_mode: str = "drive",
                   fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> list[Item]:
    """BharatPotHole (CC BY-SA 4.0, DECISIONS D16). ``split_mode='drive'`` (default) splits by source video: the
    Roboflow export's own train/valid/test put frames of 111 of its 162 drives into more than one split, so near-
    identical consecutive frames would leak from train into val/test. ``'roboflow'`` keeps the old mapping
    (train+valid -> train, test -> val)."""
    base = interim / "bharatpothole"
    names = ["pothole"]
    dy = base / "data_yaml.txt"
    if dy.exists():
        names = list((yaml.safe_load(dy.read_text()) or {}).get("names", names))
    bmap = source_maps.get("bharatpothole", {"pothole": "pothole"})
    id_map = {i: bmap.get(n) for i, n in enumerate(names)}
    items = []
    roboflow_target = {"train": "train", "valid": "train", "test": "val"}
    for split in ("train", "valid", "test"):
        idir = base / split / "images"
        if not idir.is_dir():
            continue
        stems = _limit([p.stem for p in idir.glob("*.jpg")], limit, seed, f"bharat_{split}")
        for stem in stems:
            img = idir / f"{stem}.jpg"
            w, h = image_size(img)
            insts = _read_yolo_boxes(base / split / "labels" / f"{stem}.txt", w, h, id_map, drop)
            target = roboflow_target[split] if split_mode == "roboflow" else None
            items.append(Item(uid=f"bh_{stem}"[:120], source="bharat", image=img, w=w, h=h, insts=insts,
                              fixed_split=target, group=f"bh:{bharat_drive(stem)}"))
    if split_mode == "drive" and items:
        sizes = collections.Counter(it.group for it in items)
        dest = split_groups(dict(sizes), fractions, seed, "bharat_drives")
        for it in items:
            it.fixed_split = dest[it.group]
    per = collections.Counter(it.fixed_split for it in items)
    log(f"bharat: {len(items)} images from {len({it.group for it in items})} drives, split by {split_mode}: {dict(per)}")
    return items


def collect_webseg(interim: Path, source_maps: dict, limit, seed, drop) -> list[Item]:
    """Human-drawn pothole polygons from the web (Kaggle farzadnekouei, Apache-2.0; DECISIONS D16): the only pothole
    masks in the build that are not SAM2 pseudo-masks. train -> train; valid -> test (a human-mask test subset)."""
    base = interim / "pothole_seg_web"
    if not base.is_dir():
        log(f"webseg: {base} missing - run data/download.py --webseg --fetch")
        return []
    names = ["Pothole"]
    dy = base / "data_yaml.txt"
    if dy.exists():
        names = list((yaml.safe_load(dy.read_text()) or {}).get("names", names))
    wmap = source_maps.get("pothole_seg_web", {"Pothole": "pothole"})
    id_map = {i: wmap.get(n, wmap.get(str(n).lower())) for i, n in enumerate(names)}
    items = []
    for split, target in (("train", "train"), ("valid", "test")):
        idir = base / split / "images"
        if not idir.is_dir():
            continue
        stems = _limit([p.stem for p in idir.glob("*.jpg")], limit, seed, f"webseg_{split}")
        for stem in stems:
            img = idir / f"{stem}.jpg"
            w, h = image_size(img)
            insts = []
            lab = base / split / "labels" / f"{stem}.txt"
            for line in (lab.read_text().splitlines() if lab.exists() else []):
                if len(line.split()) < 7:
                    drop["webseg_short_polygon"] += 1
                    continue
                cid, xy = parse_yolo_seg_line(line)
                cls = id_map.get(cid)
                if cls is None:
                    drop[f"webseg_class_{cid}"] += 1
                    continue
                poly = clip_polygon(xy * np.array([w, h], np.float32), w, h)
                if len(poly) < 3 or polygon_area(poly) < 16:
                    drop["webseg_tiny_polygon"] += 1
                    continue
                insts.append(Inst(cls=cls, poly=poly, mask_source="gt_polygon"))
            items.append(Item(uid=f"ws_{stem}"[:120], source="webseg", image=img, w=w, h=h, insts=insts,
                              fixed_split=target))
    log(f"webseg: {len(items)} images, {sum(len(it.insts) for it in items)} human-drawn polygons")
    return items


# ------------------------------------------------------------------------------------------------ near-duplicate groups
def _dhash_key(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def dhash_lookup(items: list[Item], cache_path: Path | None) -> list[np.ndarray]:
    """256-bit dHash (4 x uint64) per item, cached by repo-relative path so the cache survives a project move."""
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache = {}
    words, dirty = [], False
    for it in items:
        key = _dhash_key(it.image)
        if key not in cache:
            g = cv2.imread(str(it.image), cv2.IMREAD_GRAYSCALE)
            cache[key] = dhash_words(g).tobytes().hex() if g is not None else "0" * 64
            dirty = True
        words.append(np.frombuffer(bytes.fromhex(cache[key]), dtype=np.uint64))
    if cache_path and dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return words


def drop_test_near_duplicates(items: list[Item], split: dict[str, str], cache_path: Path | None,
                              max_hamming: int) -> tuple[set[str], dict]:
    """Train/val images that are near-duplicates (256-bit dHash Hamming <= max_hamming) of any test image, across all
    sources (DECISIONS D16). Web pothole sets are often remixes of RDD2022, and BharatPotHole drives can pass the same
    road twice; dropping such images keeps the test split honest. The test split itself is never changed."""
    test = [it for it in items if split.get(it.uid) == "test"]
    rest = [it for it in items if split.get(it.uid) in ("train", "val")]
    info: dict = {"checked": len(rest), "test_images": len(test), "max_hamming": max_hamming, "dropped": 0}
    if not test or not rest:
        return set(), info
    words = dhash_lookup(test + rest, cache_path)
    ht, hr = np.stack(words[: len(test)]), np.stack(words[len(test):])
    drop: set[str] = set()
    pairs: collections.Counter = collections.Counter()
    examples: list[list] = []
    for s in range(0, len(rest), 512):
        d = hamming_words(hr[s : s + 512], ht)
        j = d.argmin(axis=1)
        dmin = d[np.arange(len(j)), j]
        for k in np.nonzero(dmin <= max_hamming)[0]:
            it, tt = rest[s + int(k)], test[int(j[k])]
            drop.add(it.uid)
            pairs[f"{it.source}->{tt.source}"] += 1
            if len(examples) < 20:
                examples.append([it.uid, tt.uid, int(dmin[k])])
    info.update(dropped=len(drop), by_source_pair=dict(pairs), examples=examples)
    return drop, info


def dhash_groups(items: list[Item], cache_path: Path | None, max_hamming: int) -> tuple[dict[str, str], dict]:
    """Union-find over 256-bit dHash Hamming distance <= max_hamming. Returns uid -> group id and stats.

    Checked on RDD-India (2026-11-13): pairs <= 24/256 bits are consecutive frames of one drive; the 64-bit dHash at
    <= 4 bits chained 320 unrelated 'road + sky' frames into one group."""
    words = dhash_lookup(items, cache_path)
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = 0
    if n > 1:
        hv = np.stack(words)
        for s in range(0, n, 256):
            dist = hamming_words(hv[s : s + 256], hv)
            ii, jj = np.nonzero(dist <= max_hamming)
            for i, j in zip(ii + s, jj, strict=True):
                if j > i:
                    pairs += 1
                    ra, rb = find(int(i)), find(int(j))
                    if ra != rb:
                        parent[ra] = rb
    groups = {it.uid: f"g{find(k)}" for k, it in enumerate(items)}
    sizes = collections.Counter(groups.values())
    return groups, {"hash_bits": 256, "pairs_within_hamming": pairs,
                    "multi_image_groups": sum(1 for v in sizes.values() if v > 1),
                    "images_in_multi_groups": sum(v for v in sizes.values() if v > 1),
                    "largest_group": max(sizes.values()) if sizes else 0, "max_hamming": max_hamming}


# ------------------------------------------------------------------------------------------------ splitting
def assign_splits(items: list[Item], test_n: int, val_frac: float, seed: int) -> dict[str, str]:
    """Fixed splits are kept (eYantra train/val, foreign/bharat train). RDD-India near-duplicate groups go whole into
    one split: test = groups with potholes until ``test_n`` pothole frames (never more than 40 % of them); val =
    ``val_frac`` of the remaining groups, stratified so it gets the same share of pothole and pothole-free groups;
    train = the rest."""
    split: dict[str, str] = {}
    for it in items:
        if it.fixed_split:
            split[it.uid] = it.fixed_split
    india = [it for it in items if it.source == "rdd_india"]
    groups: dict[str, list[Item]] = collections.defaultdict(list)
    for it in india:
        groups[it.group or it.uid].append(it)
    gids = sorted(groups)
    stable_rng(seed, "groups").shuffle(gids)
    pos = [g for g in gids if any(m.has_pothole for m in groups[g])]
    neg = [g for g in gids if not any(m.has_pothole for m in groups[g])]
    n_d40 = sum(1 for it in india if it.has_pothole)
    test_target = min(test_n, int(0.4 * n_d40))
    test_cnt = 0
    rest_pos = []
    for g in pos:
        k = sum(m.has_pothole for m in groups[g])
        if test_cnt + k <= test_target:
            for m in groups[g]:
                split[m.uid] = "test"
            test_cnt += k
        else:
            rest_pos.append(g)
    for bucket in (rest_pos, neg):
        n_b = sum(len(groups[g]) for g in bucket)
        val_target = round(val_frac * n_b)
        val_cnt = 0
        for g in bucket:
            tgt = "val" if val_cnt < val_target else "train"
            for m in groups[g]:
                split[m.uid] = tgt
            if tgt == "val":
                val_cnt += len(groups[g])
    return split


def balance_negatives(train: list[Item], spare: list[Item], neg_frac: float, empty_max_frac: float, seed: int) -> tuple[list[Item], dict]:
    """Cap fully-empty frames at empty_max_frac of train, then make sure frames without a pothole are at least
    neg_frac of train (topping up from ``spare`` empty frames, e.g. the non-India RDD negative pool)."""
    labelled = [it for it in train if not it.empty]
    empty = sorted([it for it in train if it.empty], key=lambda i: i.uid)
    stable_rng(seed, "empty").shuffle(empty)
    # empty-source priority: India + eYantra frames first (domain), foreign pool last
    empty.sort(key=lambda i: i.source == "rdd_negpool")
    cap = int(empty_max_frac / (1 - empty_max_frac) * len(labelled)) if empty_max_frac < 1 else len(empty)
    kept = empty[:cap]
    dropped = len(empty) - len(kept)
    out = labelled + kept

    def neg_count(lst):
        return sum(1 for i in lst if not i.has_pothole)

    added = 0
    spare = sorted(spare, key=lambda i: i.uid)
    stable_rng(seed, "spare").shuffle(spare)
    while out and neg_count(out) / len(out) < neg_frac and spare:
        out.append(spare.pop())
        added += 1
    info = {"empty_candidates": len(empty), "empty_kept": len(kept), "empty_dropped": dropped, "spare_added": added,
            "train_frac_no_pothole": round(neg_count(out) / max(1, len(out)), 4),
            "train_frac_fully_empty": round(sum(1 for i in out if i.empty) / max(1, len(out)), 4)}
    return out, info


# ------------------------------------------------------------------------------------------------ masks
def resolve_masks(items: list[Item], use_sam2: bool, sam_iou: float, cache_dir: Path, device: str, max_gpu_gb: float,
                  min_area_frac: float) -> dict:
    stats = collections.Counter()
    box_items = [it for it in items if any(i.box is not None and i.poly is None for i in it.insts)]
    run_info = {}
    results: dict[str, list[dict]] = {}
    if use_sam2 and box_items:
        from data.sam2_masks import BoxJob, Sam2BoxMasker, Sam2Cache

        cache = Sam2Cache(cache_dir)
        jobs = [BoxJob(it.uid, it.image, [i.box for i in it.insts if i.box is not None]) for it in box_items]
        if any(cache.get(j) is None for j in jobs):
            masker = Sam2BoxMasker(device=device, max_gpu_gb=max_gpu_gb, log=log)
            run_info = masker.run(jobs, cache)
            del masker
        else:
            run_info = {"computed": 0, "cached": len(jobs)}
        for j in jobs:
            results[j.uid] = cache.get(j) or []
    ious = []
    for it in box_items:
        res = results.get(it.uid, [])
        k = 0
        for inst in it.insts:
            if inst.box is None or inst.poly is not None:
                continue
            r = res[k] if k < len(res) else None
            k += 1
            bp = box_polygon(inst.box)
            if not use_sam2:
                inst.poly, inst.mask_source = bp, "box"
                stats["box_no_sam2"] += 1
                continue
            if r is None or r.get("poly") is None:
                inst.poly, inst.mask_source = bp, "box_fallback"
                stats["fallback_no_mask"] += 1
                continue
            ious.append(r["bbox_iou"])
            poly = np.array(r["poly"], np.float32)
            box_area = (inst.box[2] - inst.box[0]) * (inst.box[3] - inst.box[1])
            if r["bbox_iou"] < sam_iou:
                inst.poly, inst.mask_source = bp, "box_fallback"
                stats["fallback_low_iou"] += 1
            elif polygon_area(poly) < min_area_frac * box_area:
                inst.poly, inst.mask_source = bp, "box_fallback"
                stats["fallback_small_mask"] += 1
            else:
                inst.poly, inst.mask_source = poly, "sam2"
                stats["sam2_kept"] += 1
    out = dict(stats)
    out["run"] = run_info
    if ious:
        out["bbox_iou_mean"] = round(float(np.mean(ious)), 4)
        out["bbox_iou_p10"] = round(float(np.percentile(ious, 10)), 4)
    return out


# ------------------------------------------------------------------------------------------------ writing
MARKER = ".gunthameter_dataset"


def claim_out_dir(out: Path) -> None:
    """Create ``out`` or accept it if it is empty or a previous build of ours; refuse anything else.
    Called before anything (e.g. --limit splits) is written into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    ours = (out / "stats.json").exists() or (out / MARKER).exists()
    if any(out.iterdir()) and not ours:
        raise SystemExit(f"refusing to overwrite {out}: not empty and not a previous build (no {MARKER}/stats.json)")
    (out / MARKER).write_text("india_pothole_seg build output (data/build_dataset.py)\n")


def clean_out(out: Path) -> None:
    """Remove a previous build's images/labels (``out`` must have been claimed with claim_out_dir)."""
    if not (out / MARKER).exists():
        raise SystemExit(f"{out} was not claimed by this build")
    for sub in ("images", "labels", "aux_road", "qc"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)


def place_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    shutil.copy2(src, dst)


def write_label_studio(out: Path, items: list[tuple[str, Item]], frac: float, seed: int, names: dict[str, int]) -> int:
    pool = [(s, it) for s, it in items if it.insts]
    stable_rng(seed, "qc").shuffle(pool)
    sample = pool[: max(1, int(frac * len(pool)))] if pool else []
    tasks = []
    for s, it in sample:
        res = []
        for k, inst in enumerate(it.insts):
            pts = [[float(x) * 100 / it.w, float(y) * 100 / it.h] for x, y in inst.poly]
            res.append({"id": f"{it.uid}_{k}", "type": "polygonlabels", "from_name": "label", "to_name": "image",
                        "original_width": it.w, "original_height": it.h,
                        "value": {"points": pts, "polygonlabels": [inst.cls]}, "meta": {"mask_source": inst.mask_source}})
        tasks.append({"data": {"image": f"/data/local-files/?d=images/{s}/{it.uid}{it.image.suffix}", "uid": it.uid,
                               "split": s, "source": it.source},
                      "predictions": [{"model_version": "wp1-pseudo", "result": res}]})
    qc = out / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    (qc / "label_studio_tasks.json").write_text(json.dumps(tasks, indent=1))
    labels = "".join(f'    <Label value="{n}"/>\n' for n in names)
    (qc / "label_studio_config.xml").write_text(
        f'<View>\n  <Image name="image" value="$image"/>\n  <PolygonLabels name="label" toName="image">\n{labels}  </PolygonLabels>\n</View>\n')
    return len(tasks)


def write_dataset(out: Path, split_items: dict[str, list[Item]], names: dict[str, int], link: str) -> dict:
    per = {}
    for s in SPLITS:
        st = {"images": 0, "pothole_images": 0, "no_pothole_images": 0, "fully_empty": 0,
              "instances": collections.Counter(), "mask_source": collections.defaultdict(collections.Counter),
              "by_source": collections.Counter(), "gt_mask_images": 0, "gt_mask_images_with_instances": 0,
              "sam2_mask_images": 0}
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        for it in split_items.get(s, []):
            place_image(it.image, out / "images" / s / f"{it.uid}{it.image.suffix.lower()}", link)
            lines = []
            for inst in it.insts:
                lines.append(yolo_seg_line(names[inst.cls], inst.poly, it.w, it.h))
                st["instances"][inst.cls] += 1
                st["mask_source"][inst.cls][inst.mask_source] += 1
            (out / "labels" / s / f"{it.uid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            if it.aux_road:
                m = rasterize_polygons(it.aux_road, it.w, it.h).astype(np.uint8) * 255
                (out / "aux_road" / s).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "aux_road" / s / f"{it.uid}.png"), m)
            st["images"] += 1
            st["by_source"][it.source] += 1
            st["pothole_images"] += it.has_pothole
            st["no_pothole_images"] += not it.has_pothole
            st["fully_empty"] += it.empty
            if it.source in ("eyantra", "webseg"):  # human-drawn polygons
                st["gt_mask_images"] += 1
                st["gt_mask_images_with_instances"] += bool(it.insts)
            if any(i.mask_source == "sam2" for i in it.insts):
                st["sam2_mask_images"] += 1
        st["instances"] = dict(st["instances"])
        st["mask_source"] = {k: dict(v) for k, v in st["mask_source"].items()}
        st["by_source"] = dict(st["by_source"])
        per[s] = st
    return per


def write_data_yaml(out: Path, names: dict[str, int]) -> Path:
    data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": {i: n for n, i in sorted(names.items(), key=lambda kv: kv[1])},
    }
    p = out / "data.yaml"
    p.write_text("# Written by data/build_dataset.py (WP1). Class order = configs/classes.yaml (frozen).\n"
                 + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return p


def check_with_ultralytics(data_yaml: Path) -> dict:
    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(str(data_yaml), autodownload=False)
    return {"ok": True, "nc": d["nc"], "names": list(d["names"].values()), "train": str(d["train"]), "val": str(d["val"]),
            "test": str(d.get("test"))}


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--interim", type=Path, default=DEFAULT_INTERIM)
    ap.add_argument("--classes", type=Path, default=CLASSES_YAML)
    ap.add_argument("--splits-dir", type=Path, default=None, help="default data/splits/v2 (full build) or <out>/splits (--limit)")
    ap.add_argument("--sam2", action="store_true", help="box -> SAM2.1 masks (else box polygons)")
    ap.add_argument("--sam2-device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--sam2-max-gpu-gb", type=float, default=2.5)
    ap.add_argument("--sam2-iou", type=float, default=0.6, help="keep SAM mask if IoU(mask bbox, box) >= this")
    ap.add_argument("--sam2-min-area-frac", type=float, default=0.08, help="fallback if mask polygon < frac * box area")
    ap.add_argument("--sam2-cache", type=Path, default=None, help="default <interim>/sam2_cache")
    ap.add_argument("--negatives-frac", type=float, default=0.2)
    ap.add_argument("--empty-max-frac", type=float, default=0.15, help="cap on fully-empty frames in train")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-n", type=int, default=500)
    ap.add_argument("--dup-hamming", type=int, default=24, help="near-duplicate threshold, bits of 256-bit dHash")
    ap.add_argument("--foreign-d40", action=argparse.BooleanOptionalAction, default=True,
                    help="add non-India RDD frames with potholes to train")
    ap.add_argument("--include-bharat", action="store_true",
                    help="add BharatPotHole (CC BY-SA 4.0, verified 2026-11-16; DECISIONS D16)")
    ap.add_argument("--bharat-split", default="drive", choices=["drive", "roboflow"],
                    help="drive: whole source videos per split (no leakage); roboflow: its own leaky splits")
    ap.add_argument("--bharat-fractions", default="0.8,0.1,0.1", help="train,val,test share of BharatPotHole frames")
    ap.add_argument("--include-webseg", action="store_true",
                    help="add human-drawn pothole polygons from the web (Apache-2.0; valid -> test; DECISIONS D16)")
    ap.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="drop train/val images that near-duplicate a test image (any source)")
    ap.add_argument("--eyantra-shallow", default="manhole", choices=["manhole", "contract", "plan_heuristic", "drop"])
    ap.add_argument("--eyantra-pothole", default="exclude_image", choices=["exclude_image", "contract", "drop_label"])
    ap.add_argument("--road-aux", action="store_true", help="write eYantra road masks to aux_road/")
    ap.add_argument("--link", default="hardlink", choices=["hardlink", "copy", "symlink"])
    ap.add_argument("--qc-frac", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None, help="max images per source (quick builds)")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--check", action=argparse.BooleanOptionalAction, default=True,
                    help="validate data.yaml with ultralytics check_det_dataset")
    return ap.parse_args(argv)


def build(args) -> dict:
    t0 = time.time()
    out: Path = args.out.resolve()
    interim: Path = args.interim.resolve()
    splits_dir = args.splits_dir or (out / "splits" if args.limit else DEFAULT_SPLITS)
    names, source_maps = load_classes(args.classes)
    drop = collections.Counter()
    claim_out_dir(out)

    # 1. collect
    id_map = rdd_id_map(interim, source_maps) if (interim / "rdd2022" / "meta.json").exists() else {}
    items: list[Item] = []
    if id_map:
        items += collect_rdd(interim, "india", "rdd_india", id_map, args.limit, args.seed, drop)
        if args.foreign_d40:
            items += collect_rdd(interim, "foreign_d40", "rdd_foreign", id_map, args.limit, args.seed, drop, "train")
    else:
        log("rdd: data/interim/rdd2022/meta.json missing - run data/download.py --rdd")
    items += collect_eyantra(interim, source_maps, args.eyantra_shallow, args.eyantra_pothole, args.limit, args.seed,
                             drop, args.road_aux)
    bharat_fracs = tuple(float(x) for x in args.bharat_fractions.split(","))
    if len(bharat_fracs) != 3 or abs(sum(bharat_fracs) - 1.0) > 1e-6:
        raise SystemExit(f"--bharat-fractions {args.bharat_fractions!r}: give three shares that sum to 1")
    if args.include_bharat:
        items += collect_bharat(interim, source_maps, args.limit, args.seed, drop, args.bharat_split, bharat_fracs)
    if args.include_webseg:
        items += collect_webseg(interim, source_maps, args.limit, args.seed, drop)
    spare = collect_rdd(interim, "neg_pool", "rdd_negpool", id_map, args.limit, args.seed, drop, "train") if id_map else []
    if not items:
        raise SystemExit("no input data found under " + str(interim))
    unknown = {i.cls for it in items for i in it.insts} - set(names)
    if unknown:
        raise SystemExit(f"classes not in {args.classes}: {unknown}")

    # 2. near-duplicate groups + splits (persisted)
    dhash_cache = interim / "dhash256_all.json" if not args.limit else None
    india = [it for it in items if it.source == "rdd_india"]
    groups, dup_stats = dhash_groups(india, dhash_cache, args.dup_hamming)
    for it in india:
        it.group = groups[it.uid]
    fp = {"v": BUILD_VERSION, "seed": args.seed, "test_n": args.test_n, "val_frac": args.val_frac, "limit": args.limit,
          "dup_hamming": args.dup_hamming, "bharat_split": [args.bharat_split, list(bharat_fracs)],
          "uids": hashlib.sha1("|".join(sorted(it.uid for it in items)).encode()).hexdigest()}
    split_file = splits_dir / "india_pothole_seg.json"
    split = None
    if split_file.exists():
        saved = json.loads(split_file.read_text())
        if saved.get("fingerprint") == fp:
            split = saved["split"]
            log(f"splits: reusing {split_file}")
        else:
            log(f"splits: {split_file} fingerprint changed -> recomputing")
    if split is None:
        split = assign_splits(items, args.test_n, args.val_frac, args.seed)
        splits_dir.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps({"fingerprint": fp, "split": split}, indent=0))
    dedup_info: dict = {"enabled": bool(args.dedup)}
    if args.dedup:
        leaks, info = drop_test_near_duplicates(items, split, dhash_cache, args.dup_hamming)
        dedup_info.update(info)
        if leaks:
            items = [it for it in items if it.uid not in leaks]
            log(f"dedup: dropped {len(leaks)} train/val images that near-duplicate a test image "
                f"{info.get('by_source_pair')}")
        else:
            log(f"dedup: no train/val image near-duplicates a test image ({info['checked']} checked)")
    by_split: dict[str, list[Item]] = {s: [] for s in SPLITS}
    for it in items:
        by_split[split[it.uid]].append(it)

    # 3. negatives balance (train only)
    by_split["train"], neg_info = balance_negatives(by_split["train"], spare, args.negatives_frac, args.empty_max_frac, args.seed)
    for s in SPLITS:
        by_split[s].sort(key=lambda i: i.uid)
        (splits_dir / f"{s}.txt").write_text("".join(f"{it.uid}\n" for it in by_split[s]))

    # 4. masks
    sam_stats = resolve_masks([it for s in SPLITS for it in by_split[s]], args.sam2, args.sam2_iou,
                              args.sam2_cache or interim / "sam2_cache", args.sam2_device, args.sam2_max_gpu_gb,
                              args.sam2_min_area_frac)

    # 5. write
    clean_out(out)
    per = write_dataset(out, by_split, names, args.link)
    data_yaml = write_data_yaml(out, names)
    n_qc = write_label_studio(out, [(s, it) for s in SPLITS for it in by_split[s]], args.qc_frac, args.seed, names)

    totals = collections.Counter()
    for s in SPLITS:
        totals.update(per[s]["instances"])
    stats = {
        "dataset": out.name,
        "build_version": BUILD_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": None,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "names": {i: n for n, i in sorted(names.items(), key=lambda kv: kv[1])},
        "splits": per,
        "totals": {"images": sum(per[s]["images"] for s in SPLITS), "instances": dict(totals),
                   "pothole_instances": int(totals.get("pothole", 0) + totals.get("pothole_water", 0))},
        "val_mask_images": {
            "gt_polygon_images": per["val"]["gt_mask_images"],
            "gt_polygon_images_with_instances": per["val"]["gt_mask_images_with_instances"],
            "sam2_mask_images": per["val"]["sam2_mask_images"],
            "note": "GT polygons come from eYantra (manhole/cover class + negatives; it has no reliable pothole masks) "
                    "and, with --include-webseg, human-drawn pothole polygons (train + the ws_ test subset); all other "
                    "pothole masks are SAM2.1 pseudo-masks from RDD/BharatPotHole boxes.",
        },
        "sources": {"bharat": {"included": bool(args.include_bharat), "split": args.bharat_split,
                               "fractions": list(bharat_fracs)},
                    "webseg": {"included": bool(args.include_webseg), "valid_goes_to": "test"},
                    "uid_prefix": UID_PREFIX},
        "test_near_duplicates": dedup_info,
        "negatives": neg_info,
        "sam2": sam_stats,
        "dropped": dict(drop),
        "near_duplicates": dup_stats,
        "label_studio_qc_tasks": n_qc,
        "licences": {k: v for k, v in LICENCES.items() if any(per[s]["by_source"].get(k) for s in SPLITS)},
        "class_mapping": {
            "rdd2022": source_maps.get("rdd2022"),
            "rdd_mirror_ids": json.loads((interim / "rdd2022" / "meta.json").read_text())["class_ids"] if id_map else None,
            "eyantra": {"shallow": args.eyantra_shallow, "pothole": args.eyantra_pothole},
        },
    }
    if args.check:
        try:
            stats["ultralytics_check"] = check_with_ultralytics(data_yaml)
        except Exception as e:  # report, do not hide
            stats["ultralytics_check"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    log(f"done in {stats['seconds']} s: {stats['totals']}; val masks {stats['val_mask_images']['gt_polygon_images']} GT + "
        f"{stats['val_mask_images']['sam2_mask_images']} SAM2; check={stats.get('ultralytics_check', {}).get('ok')}")
    return stats


def main(argv=None) -> int:
    stats = build(parse_args(argv))
    return 0 if stats.get("ultralytics_check", {"ok": True}).get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
