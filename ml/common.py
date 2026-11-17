"""Shared helpers for WP3 (training, eval, export): repo paths, the frozen class list, dataset-yaml resolution,
device policy and small I/O utilities.

Nothing here imports torch or ultralytics at module import time, so unit tests stay light.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES_YAML = REPO_ROOT / "configs" / "classes.yaml"
ALIASES_YAML = REPO_ROOT / "configs" / "model" / "class_aliases.yaml"
DATASET_CFG = REPO_ROOT / "configs" / "model" / "india_pothole_seg.yaml"
PROFILES_YAML = REPO_ROOT / "configs" / "model" / "train_profiles.yaml"
RUNS_SEG = REPO_ROOT / "runs" / "seg"
REPORTS = REPO_ROOT / "reports"
REGISTRY_DIR = REPO_ROOT / "weights" / "registry"
DEFAULT_SEG_WEIGHTS = "runs/seg/y26m_seg_india_v2ft_geo/weights/best_pothole.pt"  # D24: best trained model (RDD-India AP50 0.2494)


def ultralytics_env() -> None:
    """Call before importing ultralytics: never let it pip-install packages into the venv (CLAUDE.md: never touch
    torch; new packages only on purpose), and keep its console output plain."""
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(p: str | Path) -> str:
    """Repo-relative POSIX path when inside the repo, else the absolute path."""
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def resolve_repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_json_default) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


# ----------------------------------------------------------------------------------------------- classes
@lru_cache(maxsize=1)
def class_names() -> tuple[str, ...]:
    """Frozen class order (configs/classes.yaml). Trained weights depend on it."""
    d = yaml.safe_load(CLASSES_YAML.read_text(encoding="utf-8"))
    names = d["names"]
    return tuple(names[i] for i in sorted(names))


@lru_cache(maxsize=1)
def reporting_classes() -> frozenset[str]:
    d = yaml.safe_load(CLASSES_YAML.read_text(encoding="utf-8"))
    return frozenset(d.get("reporting", ["pothole", "pothole_water"]))


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


@lru_cache(maxsize=1)
def _alias_table() -> dict[str, Optional[str]]:
    """normalised raw model class name -> canonical class (or None = drop)."""
    table: dict[str, Optional[str]] = {_norm(n): n for n in class_names()}
    d = yaml.safe_load(CLASSES_YAML.read_text(encoding="utf-8"))
    for _src, mp in (d.get("source_maps") or {}).items():
        for raw, canon in (mp or {}).items():
            table.setdefault(_norm(raw), canon)
    if ALIASES_YAML.is_file():
        extra = yaml.safe_load(ALIASES_YAML.read_text(encoding="utf-8")) or {}
        for raw, canon in (extra.get("aliases") or {}).items():
            table[_norm(raw)] = canon if canon else None
    return table


def canonical_class(raw_name: str) -> Optional[str]:
    """Map a model's class name to the frozen class list; None if it has no meaning for RoadFix
    (e.g. RDD 'Other', COCO 'person')."""
    return _alias_table().get(_norm(raw_name))


def class_index(name: str) -> Optional[int]:
    try:
        return class_names().index(name)
    except ValueError:
        return None


# ----------------------------------------------------------------------------------------------- dataset
def load_dataset_cfg(cfg_path: str | Path = DATASET_CFG) -> dict[str, Any]:
    return yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))


def resolve_dataset_root(cfg_path: str | Path = DATASET_CFG, cfg: Optional[dict[str, Any]] = None) -> Path:
    """Resolve the dataset `path:` of an Ultralytics data yaml at runtime.

    Order: env ``GM_DATASET_DIR`` > absolute `path` > `path` relative to the yaml's own directory > `path` relative to
    the repo root > ``data/datasets/<name>`` by the yaml's dataset name. The same config therefore works on the Windows
    laptop (C:/gunthameter) and on the Linux RTX 5080 box without editing.
    """
    cfg_path = Path(cfg_path).resolve()
    cfg = cfg if cfg is not None else load_dataset_cfg(cfg_path)
    env = os.environ.get("GM_DATASET_DIR", "").strip()
    cands: list[Path] = []
    if env:
        cands.append(Path(env))
    raw = str(cfg.get("path", "") or "")
    if raw:
        p = Path(raw)
        if p.is_absolute():
            cands.append(p)
        else:
            cands += [(cfg_path.parent / p), (REPO_ROOT / p)]
    cands.append(REPO_ROOT / "data" / "datasets" / (Path(raw).name if raw else "india_pothole_seg4_v2"))
    for c in cands:
        c = c.resolve()
        if (c / str(cfg.get("train", "images/train"))).exists() or (c / "images").is_dir():
            return c
    raise FileNotFoundError(f"dataset not found; tried {[c.as_posix() for c in cands]} (set GM_DATASET_DIR)")


def names_tuple(names: Any) -> tuple[str, ...]:
    """A data yaml's ``names`` (dict id -> name, or list) as a tuple in id order."""
    if isinstance(names, dict):
        return tuple(str(names[k]) for k in sorted(names, key=int))
    return tuple(str(n) for n in names)


def check_names(names: Any) -> None:
    """Raise unless a data yaml's names are an ordered subset of the frozen configs/classes.yaml list.

    The full list is the system-wide vocabulary (edge protocol, server). A training dataset may keep only the classes
    it has labels for (DECISIONS D15, ``data/derive_dataset.py``), renumbered 0..k-1 but in canonical order; models are
    mapped back to canonical classes by name (``canonical_class``)."""
    got = names_tuple(names)
    canon = class_names()
    bad = [n for n in got if n not in canon]
    if bad or not got:
        raise ValueError(f"classes {bad or got} are not in configs/classes.yaml {canon}")
    idx = [canon.index(n) for n in got]
    if idx != sorted(set(idx)):
        raise ValueError(f"class order {got} must follow configs/classes.yaml {canon} (no duplicates)")


def dataset_class_ids(cfg_path: str | Path = DATASET_CFG) -> dict[int, int]:
    """Dataset label id -> canonical class index (configs/classes.yaml), from the data yaml's names."""
    names = names_tuple(load_dataset_cfg(cfg_path)["names"])
    check_names(names)
    canon = class_names()
    return {i: canon.index(n) for i, n in enumerate(names)}


def write_resolved_data_yaml(out_path: str | Path, cfg_path: str | Path = DATASET_CFG,
                             val_list: Optional[list[Path]] = None) -> Path:
    """Write an Ultralytics data yaml with an absolute `path` (machine-specific, generated at run time).

    ``val_list``: optional explicit list of val images (absolute) -> written to a txt next to the yaml; used for
    smoke runs to keep per-epoch validation short.
    """
    cfg = load_dataset_cfg(cfg_path)
    check_names(cfg["names"])
    names = names_tuple(cfg["names"])
    root = resolve_dataset_root(cfg_path, cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = {"path": root.as_posix(), "train": cfg.get("train", "images/train"), "val": cfg.get("val", "images/val"),
         "test": cfg.get("test", "images/test"), "nc": len(names),
         "names": {i: n for i, n in enumerate(names)}}
    if val_list is not None:
        txt = out_path.with_suffix(".val.txt")
        txt.write_text("\n".join(Path(p).resolve().as_posix() for p in val_list) + "\n", encoding="utf-8")
        d["val"] = txt.as_posix()
    out_path.write_text("# generated by ml/common.py at run time; do not commit\n" + yaml.safe_dump(d, sort_keys=False),
                        encoding="utf-8")
    return out_path


def split_images(split: str, cfg_path: str | Path = DATASET_CFG) -> list[Path]:
    cfg = load_dataset_cfg(cfg_path)
    root = resolve_dataset_root(cfg_path, cfg)
    d = root / str(cfg.get(split, f"images/{split}"))
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in d.iterdir() if p.suffix.lower() in exts)


def label_path_for(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


# ----------------------------------------------------------------------------------------------- device
def device_info() -> dict[str, Any]:
    """Device facts for training/eval, using the project's arch-aware CUDA check (DECISIONS D12/D14)."""
    from measure.depth_backends.base import cuda_half_ok, cuda_status, gpu_free_gb

    st = dict(cuda_status())
    usable = bool(st.get("usable"))
    return {"cuda": usable, "name": st.get("name") if usable else "cpu", "capability": st.get("capability"),
            "half_ok": bool(cuda_half_ok()) if usable else False, "free_gb": round(gpu_free_gb(), 2) if usable else 0.0,
            "reason": st.get("reason", "")}


def auto_batch(free_gb: float, imgsz: int = 640, model_size: str = "n") -> int:
    """Conservative batch for YOLO26-seg from free VRAM (Ultralytics halves on OOM by itself as well)."""
    scale = (imgsz / 640.0) ** 2 * {"n": 1.0, "s": 2.0, "m": 4.0}.get(model_size, 2.0)
    per_img = 0.30 * scale  # GB per image, fp32, measured order of magnitude for y26n-seg @640 (incl. workspace)
    usable = max(0.0, free_gb - 0.6)
    b = int(usable / per_img) if per_img > 0 else 1
    for cap in (64, 32, 16, 8, 4, 2, 1):
        if b >= cap:
            return cap
    return 1
