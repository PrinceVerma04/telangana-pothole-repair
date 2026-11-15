"""GunthaMeter WP1 - dataset acquisition: verify downloads in data/raw/ and unpack them into data/interim/<name>/.

Run as ``.venv/bin/python -I data/download.py --all`` (``-I``: isolated mode, nothing is imported from the cwd or
from data/raw). The script is self-contained (stdlib + numpy/opencv/pyyaml/huggingface_hub) on purpose.

Everything under data/raw/ is UNTRUSTED. This script never executes anything from it:
  * archives are read with ``zipfile`` member by member (never ``extractall``),
  * only whitelisted extensions are written (images, YOLO txt, labelme json, png masks, yaml/csv text),
  * output file names are computed by us (no path from the archive is used verbatim -> no zip-slip),
  * pickles (Ultralytics ``labels.cache``), ``.npy`` and anything else are skipped,
  * per-member size is capped (decompression-bomb guard).

Idempotent: every unpacked source gets ``data/interim/<name>/.done.json`` holding the raw file's size/mtime and the
extractor version; a re-run with an unchanged raw file is a no-op (``--force`` re-extracts).

Downloads already happened (DECISIONS D11). Network fallbacks are kept and only run with ``--fetch`` when the raw file
is missing:
  rdd     : hf download akiii1234/RoadDamageDetection road_damage_dataset.zip --repo-type dataset (HF mirror, CC BY-SA 4.0)
            fallback Figshare full release / per-country S3 link (403 since 2026-11)
  eyantra : kaggle datasets download -d eyantraiit/semantic-segmentation-datasets-of-indian-roads
  bharat  : kaggle datasets download -d surbhisaswatimohanty/bharatpothole
  gtfs    : Mobility Database mdb-3361 (TGSRTC)            (owned by WP10: verify only)
  osm     : Geofabrik southern-zone-latest.osm.pbf         (owned by WP10: verify only)
  weights : HF checkpoints facebook/sam2.1-hiera-small, depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
EXTRACTOR_VERSION = 4
MAX_MEMBER_BYTES = 64 * 1024 * 1024  # no single image/label in these datasets is anywhere near 64 MB
SAFE_STEM = re.compile(r"[^A-Za-z0-9_-]+")

# RDD2022 HF mirror (akiii1234/RoadDamageDetection) is a flattened, shuffled YOLO export: images/<id>.jpg +
# labels/<id>.txt, no country folders. Class ids were verified on 2026-11-13 by rendering crops of every id
# (scratch montage): 0 longitudinal crack, 1 transverse crack, 2 alligator crack, 3 "other corruption"
# (white-line/crosswalk blur, D43/D44/D50 merged), 4 pothole.
RDD_MIRROR_CLASS_IDS = {0: "D00", 1: "D10", 2: "D20", 3: "OTHER", 4: "D40"}
# Country is not stored in the mirror. RDD2022 cameras have country-specific resolutions; India's smartphone
# captures are 720x720 (Japan/Czech 600x600, USA 640x640, China 512x512, Norway ~3650x2044 / 4040x2035).
RDD_COUNTRY_BY_SIZE = {
    (720, 720): "India",
    (600, 600): "Japan_or_Czech",
    (640, 640): "United_States",
    (512, 512): "China",
}


@dataclass
class Source:
    name: str
    raw: Path
    min_bytes: int
    fetch_cmds: list[list[str]] = field(default_factory=list)
    licence: str = ""
    owner: str = "WP1"


SOURCES: dict[str, Source] = {
    "rdd": Source(
        "rdd2022",
        RAW / "rdd2022" / "road_damage_dataset.zip",
        8_000_000_000,
        [
            ["hf", "download", "akiii1234/RoadDamageDetection", "road_damage_dataset.zip", "--repo-type", "dataset",
             "--local-dir", str(RAW / "rdd2022")],
            # Original sources (kept for provenance; the per-country S3 link returned 403 on 2026-11-12):
            ["curl", "-L", "--fail", "-o", str(RAW / "rdd2022_india" / "RDD2022_India.zip"),
             "https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_India.zip"],
        ],
        "CC BY-SA 4.0 (RDD2022 / CRDDC2022; HF mirror is a re-split derivative)",
    ),
    "eyantra": Source(
        "eyantra",
        RAW / "eyantra" / "semantic-segmentation-datasets-of-indian-roads.zip",
        2_000_000_000,
        [["kaggle", "datasets", "download", "-d", "eyantraiit/semantic-segmentation-datasets-of-indian-roads",
          "-p", str(RAW / "eyantra")]],
        "GPL-2.0 per Kaggle page -> prototype/evaluation only",
    ),
    "bharat": Source(
        "bharatpothole",
        RAW / "bharatpothole" / "bharatpothole.zip",
        500_000_000,
        [["kaggle", "datasets", "download", "-d", "surbhisaswatimohanty/bharatpothole", "-p", str(RAW / "bharatpothole")]],
        "UNVERIFIED (bundled data.yaml claims Roboflow 'CC BY 4.0'); excluded from training unless --include-bharat",
    ),
    "gtfs": Source(
        "gtfs_tgsrtc",
        RAW / "gtfs_tgsrtc" / "tgsrtc_gtfs.zip",
        1_000_000,
        [["curl", "-L", "--fail", "-o", str(RAW / "gtfs_tgsrtc" / "tgsrtc_gtfs.zip"),
          "https://files.mobilitydatabase.org/mdb-3361/latest.zip"]],
        "open data (Mobility Database mdb-3361)",
        owner="WP10",
    ),
    "osm": Source(
        "osm",
        RAW / "osm" / "southern-zone-latest.osm.pbf",
        100_000_000,
        [["curl", "-L", "--fail", "-o", str(RAW / "osm" / "southern-zone-latest.osm.pbf"),
          "https://download.geofabrik.de/asia/india/southern-zone-latest.osm.pbf"]],
        "ODbL",
        owner="WP10",
    ),
}

HF_CHECKPOINTS = [
    "facebook/sam2.1-hiera-small",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
]
LOCAL_WEIGHTS = ["yolo26n-seg.pt", "yolo26s-seg.pt", "bootstrap_pothole_yolo26m.pt", "bootstrap_pothole_yolov8s.pt"]


# --------------------------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(f"[download {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def safe_stem(s: str) -> str:
    """Archive-derived names are reduced to [A-Za-z0-9_-] so they can never escape the target directory."""
    out = SAFE_STEM.sub("_", s).strip("_")
    if not out:
        raise ValueError(f"unusable member name {s!r}")
    return out[:180]


def jpeg_size(head: bytes) -> tuple[int, int] | None:
    """(w, h) from a JPEG header without decoding pixels."""
    if head[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(head)
    while i + 9 < n:
        if head[i] != 0xFF:
            i += 1
            continue
        marker = head[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", head[i + 5 : i + 9])
            return w, h
        seg_len = struct.unpack(">H", head[i + 2 : i + 4])[0]
        i += 2 + seg_len
    return None


def read_member(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"{info.filename}: {info.file_size} B exceeds member cap")
    with z.open(info) as f:
        data = f.read(MAX_MEMBER_BYTES + 1)
    if len(data) > MAX_MEMBER_BYTES:
        raise ValueError(f"{info.filename}: decompressed size exceeds member cap")
    return data


def write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def raw_signature(p: Path) -> dict:
    st = p.stat()
    return {"raw": str(p.relative_to(ROOT)), "bytes": st.st_size, "mtime": int(st.st_mtime), "extractor": EXTRACTOR_VERSION}


def is_done(out: Path, sig: dict) -> bool:
    marker = out / ".done.json"
    if not marker.exists():
        return False
    try:
        old = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all(old.get(k) == v for k, v in sig.items())


def mark_done(out: Path, sig: dict, extra: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / ".done.json").write_text(json.dumps({**sig, **extra, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))


def verify_raw(src: Source, fetch: bool) -> bool:
    if not src.raw.exists() and fetch:
        src.raw.parent.mkdir(parents=True, exist_ok=True)
        for cmd in src.fetch_cmds:
            if shutil.which(cmd[0]) is None:
                log(f"{src.name}: fallback tool {cmd[0]!r} not on PATH, skipping")
                continue
            log(f"{src.name}: fetching: {' '.join(cmd)}")
            if subprocess.run(cmd, check=False).returncode == 0 and src.raw.exists():
                break
    if not src.raw.exists():
        log(f"{src.name}: MISSING {src.raw} (re-run with --fetch, or: {' '.join(src.fetch_cmds[0]) if src.fetch_cmds else '-'})")
        return False
    size = src.raw.stat().st_size
    if size < src.min_bytes:
        log(f"{src.name}: {src.raw} is only {size} B (< {src.min_bytes}); incomplete download?")
        return False
    if src.raw.suffix == ".zip":
        try:
            with zipfile.ZipFile(src.raw) as z:
                n = len(z.infolist())
        except zipfile.BadZipFile as e:
            log(f"{src.name}: corrupt zip: {e}")
            return False
        log(f"{src.name}: OK {src.raw.name} {size / 1e9:.2f} GB, {n} members, licence: {src.licence}")
    else:
        log(f"{src.name}: OK {src.raw.name} {size / 1e9:.2f} GB, licence: {src.licence}")
    return True


# --------------------------------------------------------------------------------------------- RDD2022
def extract_rdd(src: Source, force: bool, neg_pool: int, deep_crc: bool) -> dict:
    """Index the whole mirror (size + label counts per image) and unpack three subsets into data/interim/rdd2022/:
    india/ (720x720 frames), neg_pool/ (label-free non-India frames) and foreign_d40/ (non-India frames with potholes)."""
    out = INTERIM / "rdd2022"
    sig = raw_signature(src.raw) | {"neg_pool": neg_pool}
    if is_done(out, sig) and not force:
        log("rdd: already extracted (data/interim/rdd2022/.done.json matches)")
        return json.loads((out / "meta.json").read_text())
    z = zipfile.ZipFile(src.raw)
    infos = {i.filename: i for i in z.infolist()}
    img_names = sorted(n for n in infos if n.startswith("images/") and n.lower().endswith((".jpg", ".jpeg", ".png")))
    log(f"rdd: indexing {len(img_names)} images (header only)")
    rows = []
    t0 = time.time()
    for k, n in enumerate(img_names):
        stem = Path(n).stem
        sid = safe_stem(stem)
        info = infos[n]
        with z.open(info) as f:
            head = f.read(1 << 16)
        wh = jpeg_size(head) or (0, 0)
        lab_name = f"labels/{stem}.txt"
        counts = collections.Counter()
        lab_ok = lab_name in infos
        if lab_ok:
            for line in read_member(z, infos[lab_name]).decode("utf-8", "replace").splitlines():
                parts = line.split()
                if len(parts) == 5 and parts[0].isdigit():
                    counts[int(parts[0])] += 1
        rows.append({"id": sid, "member": n, "w": wh[0], "h": wh[1], "country": RDD_COUNTRY_BY_SIZE.get(wh, "Norway_or_other"),
                     "has_label_file": lab_ok, **{RDD_MIRROR_CLASS_IDS.get(c, str(c)): v for c, v in counts.items()}})
        if (k + 1) % 5000 == 0:
            log(f"rdd: indexed {k + 1}/{len(img_names)} ({time.time() - t0:.0f} s)")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    india = [r for r in rows if r["country"] == "India"]
    pool = [r for r in rows if r["country"] != "India" and r["has_label_file"] and max(r["w"], r["h"]) <= 1080
            and not any(r.get(c) for c in RDD_MIRROR_CLASS_IDS.values())]
    # deterministic, spread over countries
    pool.sort(key=lambda r: (hash_str(r["id"]), r["id"]))
    pool = pool[:neg_pool]
    # Non-India frames with potholes: train-only extra pothole examples (build_dataset.py --foreign-d40).
    # Norway's 3650x2044+ wide frames are skipped (tiny potholes, very different camera).
    foreign = [r for r in rows if r["country"] != "India" and r.get("D40") and max(r["w"], r["h"]) <= 1080]
    log(f"rdd: India subset {len(india)} images; negative pool {len(pool)} label-free non-India images; "
        f"{len(foreign)} non-India frames with D40")
    for subset, items in (("india", india), ("neg_pool", pool), ("foreign_d40", foreign)):
        for k, r in enumerate(items):
            dst_img = out / subset / "images" / f"{r['id']}.jpg"
            dst_lab = out / subset / "labels" / f"{r['id']}.txt"
            if not dst_img.exists() or force:
                data = read_member(z, infos[r["member"]])
                if deep_crc and zlib.crc32(data) != infos[r["member"]].CRC:
                    raise ValueError(f"CRC mismatch {r['member']}")
                write_bytes(dst_img, data)
            if not dst_lab.exists() or force:
                lab_member = f"labels/{Path(r['member']).stem}.txt"
                write_bytes(dst_lab, read_member(z, infos[lab_member]) if lab_member in infos else b"")
            if (k + 1) % 2000 == 0:
                log(f"rdd: {subset} {k + 1}/{len(items)}")
    sizes = collections.Counter(f"{r['w']}x{r['h']}" for r in rows)
    cls_india = collections.Counter()
    for r in india:
        for c in RDD_MIRROR_CLASS_IDS.values():
            cls_india[c] += r.get(c, 0)
    meta = {
        "source": "HF dataset akiii1234/RoadDamageDetection (RDD2022 re-split derivative, CC BY-SA 4.0)",
        "class_ids": {str(k): v for k, v in RDD_MIRROR_CLASS_IDS.items()},
        "class_id_evidence": "verified visually 2026-11-13 (crop montage per id on 720x720 frames): 4=pothole, 3=line blur",
        "country_rule": {f"{w}x{h}": c for (w, h), c in RDD_COUNTRY_BY_SIZE.items()},
        "n_images": len(rows),
        "n_india": len(india),
        "n_india_with_D40": sum(1 for r in india if r.get("D40")),
        "india_instances": dict(cls_india),
        "n_neg_pool": len(pool),
        "n_foreign_d40": len(foreign),
        "foreign_d40_instances": sum(r.get("D40", 0) for r in foreign),
        "sizes": dict(sizes.most_common()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    mark_done(out, sig, {"n_india": len(india), "n_neg_pool": len(pool)})
    log(f"rdd: done: {json.dumps({k: meta[k] for k in ('n_india', 'n_india_with_D40', 'india_instances')})}")
    return meta


def hash_str(s: str) -> int:
    """Stable (process-independent) string hash for deterministic ordering."""
    return zlib.crc32(s.encode())


# --------------------------------------------------------------------------------------------- eYantra
EY_RE = re.compile(r"^Indian_road_data/Indian_road_data/(Raw_images|Masks)/(train|val)/([^/]+)/([^/]+)$")


def extract_eyantra(src: Source, force: bool) -> dict:
    """Raw_images/<split>/<seq>/<stem>_leftImg8bit.jpg + Masks/.../<stem>_gtFine_{polygons.json,labelTrainIds.png}
    -> interim/eyantra/<split>/{images,polygons,label_ids}/<stem>.{jpg,json,png}. The duplicate numpy export is skipped."""
    out = INTERIM / "eyantra"
    sig = raw_signature(src.raw)
    if is_done(out, sig) and not force:
        log("eyantra: already extracted")
        return json.loads((out / ".done.json").read_text())
    counts = collections.Counter()
    skipped = collections.Counter()
    with zipfile.ZipFile(src.raw) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            m = EY_RE.match(info.filename)
            if not m:
                skipped[info.filename.split("/")[0]] += 1
                continue
            kind, split, _seq, fname = m.groups()
            if kind == "Raw_images" and fname.endswith("_leftImg8bit.jpg"):
                sub, stem, ext = "images", fname[: -len("_leftImg8bit.jpg")], ".jpg"
            elif fname.endswith("_gtFine_polygons.json"):
                sub, stem, ext = "polygons", fname[: -len("_gtFine_polygons.json")], ".json"
            elif fname.endswith("_gtFine_labelTrainIds.png"):
                sub, stem, ext = "label_ids", fname[: -len("_gtFine_labelTrainIds.png")], ".png"
            else:
                skipped["other"] += 1
                continue
            dst = out / split / sub / f"{safe_stem(stem)}{ext}"
            if not dst.exists() or force:
                data = read_member(z, info)
                if ext == ".json":
                    json.loads(data)  # must parse as plain JSON; never evaluated
                write_bytes(dst, data)
            counts[f"{split}/{sub}"] += 1
    extra = {"counts": dict(counts), "skipped": dict(skipped), "licence": src.licence}
    mark_done(out, sig, extra)
    log(f"eyantra: {dict(counts)} (skipped {dict(skipped)})")
    return extra


# --------------------------------------------------------------------------------------------- BharatPotHole
BH_RE = re.compile(r"^BharatPotHole/BharatPotHole/(train|valid|test)/(images|labels)/([^/]+)$")


def extract_bharat(src: Source, force: bool) -> dict:
    """Unpacked so the edge simulator (WP7) can replay its dashcam frames; used for training only with
    build_dataset.py --include-bharat. labels.cache (a pickle) is never extracted."""
    out = INTERIM / "bharatpothole"
    sig = raw_signature(src.raw)
    if is_done(out, sig) and not force:
        log("bharat: already extracted")
        return json.loads((out / ".done.json").read_text())
    counts = collections.Counter()
    skipped = collections.Counter()
    claimed_licence = None
    with zipfile.ZipFile(src.raw) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            if info.filename == "BharatPotHole/BharatPotHole/data.yaml":
                text = read_member(z, info).decode("utf-8", "replace")
                write_bytes(out / "data_yaml.txt", text.encode())  # stored as inert text
                mm = re.search(r"license:\s*(.+)", text)
                claimed_licence = mm.group(1).strip() if mm else None
                continue
            m = BH_RE.match(info.filename)
            if not m:
                skipped[Path(info.filename).suffix or "dir"] += 1
                continue
            split, kind, fname = m.groups()
            p = Path(fname)
            if kind == "images" and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                ext = ".jpg" if p.suffix.lower() in (".jpg", ".jpeg") else ".png"
            elif kind == "labels" and p.suffix == ".txt":
                ext = ".txt"
            else:
                skipped[p.suffix] += 1
                continue
            dst = out / split / kind / f"{safe_stem(p.stem)}{ext}"
            if not dst.exists() or force:
                write_bytes(dst, read_member(z, info))
            counts[f"{split}/{kind}"] += 1
    note = {
        "licence_status": "UNVERIFIED",
        "claimed_by_bundled_data_yaml": claimed_licence,
        "kaggle": "surbhisaswatimohanty/bharatpothole",
        "policy": "excluded from data/datasets unless build_dataset.py --include-bharat; frames usable by the simulator",
    }
    (out / "LICENCE_NOTE.json").write_text(json.dumps(note, indent=2))
    extra = {"counts": dict(counts), "skipped": dict(skipped), **note}
    mark_done(out, sig, extra)
    log(f"bharat: {dict(counts)} (skipped {dict(skipped)}), claimed licence {claimed_licence!r} (unverified)")
    return extra


# --------------------------------------------------------------------------------------------- WP10-owned sources
GTFS_REQUIRED = {"agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt"}


def verify_gtfs(src: Source) -> dict:
    with zipfile.ZipFile(src.raw) as z:
        names = {Path(n).name for n in z.namelist()}
    missing = sorted(GTFS_REQUIRED - names)
    log(f"gtfs: files {sorted(names)}; missing required: {missing or 'none'} (unpacking is WP10's job)")
    return {"files": sorted(names), "missing": missing}


def verify_osm(src: Source) -> dict:
    with src.raw.open("rb") as f:
        head = f.read(64)
    (hlen,) = struct.unpack(">I", head[:4])
    ok = b"OSMHeader" in head[4 : 4 + hlen]
    log(f"osm: PBF header {'OK' if ok else 'NOT FOUND'} (extracts are WP10's job)")
    return {"pbf_header_ok": ok}


# --------------------------------------------------------------------------------------------- weights
def verify_weights(fetch: bool) -> dict:
    res = {}
    for w in LOCAL_WEIGHTS:
        p = ROOT / "weights" / w
        res[w] = p.exists() and p.stat().st_size > 1_000_000
    try:
        from huggingface_hub import scan_cache_dir, snapshot_download
    except ImportError:  # pragma: no cover
        log("weights: huggingface_hub missing")
        return res
    cached = {r.repo_id for r in scan_cache_dir().repos}
    for repo in HF_CHECKPOINTS:
        if repo not in cached and fetch:
            log(f"weights: downloading {repo}")
            snapshot_download(repo)
            cached.add(repo)
        res[repo] = repo in cached
    for k, v in res.items():
        log(f"weights: {k}: {'OK' if v else 'MISSING'}")
    return res


# --------------------------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("all", "rdd", "eyantra", "bharat", "gtfs", "osm", "weights"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--roboflow", nargs="*", help="not available: no ROBOFLOW_API_KEY on the dev box (DECISIONS D6)")
    ap.add_argument("--fetch", action="store_true", help="run the network fallback commands for missing raw files")
    ap.add_argument("--force", action="store_true", help="re-extract even if .done.json matches")
    ap.add_argument("--verify-only", action="store_true", help="check raw files, do not extract")
    ap.add_argument("--rdd-neg-pool", type=int, default=4000, help="label-free non-India RDD frames to unpack as negatives")
    ap.add_argument("--deep-crc", action="store_true", help="CRC-check every extracted RDD member")
    a = ap.parse_args(argv)
    if a.roboflow is not None:
        log("roboflow: skipped - no ROBOFLOW_API_KEY (DECISIONS D6). Set it and use the Roboflow export UI/API manually.")
    chosen = [k for k in ("rdd", "eyantra", "bharat", "gtfs", "osm", "weights") if getattr(a, k) or a.all]
    if not chosen and a.roboflow is None:
        ap.print_help()
        return 2
    report: dict = {}
    ok = True
    for k in chosen:
        if k == "weights":
            report[k] = verify_weights(a.fetch)
            continue
        src = SOURCES[k]
        if not verify_raw(src, a.fetch):
            ok = False
            report[k] = {"ok": False}
            continue
        if a.verify_only:
            report[k] = {"ok": True}
            continue
        if k == "rdd":
            report[k] = extract_rdd(src, a.force, a.rdd_neg_pool, a.deep_crc)
        elif k == "eyantra":
            report[k] = extract_eyantra(src, a.force)
        elif k == "bharat":
            report[k] = extract_bharat(src, a.force)
        elif k == "gtfs":
            report[k] = verify_gtfs(src)
        elif k == "osm":
            report[k] = verify_osm(src)
    INTERIM.mkdir(parents=True, exist_ok=True)
    rep_path = INTERIM / "download_report.json"
    try:
        old = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    except json.JSONDecodeError:
        old = {}
    rep_path.write_text(json.dumps(old | report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
