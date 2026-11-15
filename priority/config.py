"""Load ``configs/priority.default.yaml`` (per-agency copies live in ``priority_config``)."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=8)
def _read(path: str, mtime: float) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "weights" not in data:
        raise ValueError(f"{path}: priority config needs a 'weights' mapping")
    return data


def load_priority_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        env = os.environ.get("GUNTHAMETER_CONFIG_DIR")
        path = Path(env) / "priority.default.yaml" if env else REPO_ROOT / "configs" / "priority.default.yaml"
    p = Path(path)
    return copy.deepcopy(_read(str(p.resolve()), p.stat().st_mtime))


def resolve(cfg: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return cfg if cfg is not None else load_priority_config()


def normalise_highway(highway: Optional[str]) -> Optional[str]:
    """``primary_link`` -> ``primary``; lower-case; None stays None."""
    if not highway:
        return None
    h = str(highway).strip().lower()
    return h[:-5] if h.endswith("_link") else h


__all__ = ["load_priority_config", "normalise_highway", "resolve"]
