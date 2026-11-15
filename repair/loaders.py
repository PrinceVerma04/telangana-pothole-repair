"""Config loaders for the repair planner (pure; no DB / server imports).

``load_rules()``      -> dict parsed from ``configs/irc_rules.yaml``
``load_rate_card()``  -> dict parsed from ``configs/rate_card.ghmc_fy2026.yaml``

Both accept an explicit path; otherwise ``$GUNTHAMETER_CONFIG_DIR`` (if set) or the
repo's ``configs/`` directory is used. Results are fresh dicts (callers may mutate).
"""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = "irc_rules.yaml"
DEFAULT_RATE_CARD = "rate_card.ghmc_fy2026.yaml"


def config_dir() -> Path:
    env = os.environ.get("GUNTHAMETER_CONFIG_DIR")
    return Path(env) if env else REPO_ROOT / "configs"


@lru_cache(maxsize=32)
def _read_yaml(path: str, mtime: float) -> dict[str, Any]:  # mtime keys the cache
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at top level")
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return copy.deepcopy(_read_yaml(str(p.resolve()), p.stat().st_mtime))


def load_rules(path: str | Path | None = None) -> dict[str, Any]:
    """IRC constants (severity bins, repair block). Validates the keys the planner needs."""
    rules = load_yaml(path or config_dir() / DEFAULT_RULES)
    rep = rules.get("repair")
    if not isinstance(rep, dict):
        raise ValueError("irc_rules: missing 'repair' block")
    for key in ("cut_margin_m", "min_patch_depth_m", "max_layer_m", "mix_density_kg_m3", "wastage", "bag_kg_default"):
        if key not in rep:
            raise ValueError(f"irc_rules.repair: missing '{key}'")
    return rules


def load_rate_card(path: str | Path | None = None) -> dict[str, Any]:
    """Rate card (INR, excl. GST) with per-item provenance."""
    card = load_yaml(path or config_dir() / DEFAULT_RATE_CARD)
    items = card.get("items")
    if not isinstance(items, dict):
        raise ValueError("rate card: missing 'items' mapping")
    for name, item in items.items():
        if not isinstance(item, dict) or "rate" not in item:
            raise ValueError(f"rate card item '{name}' needs a 'rate'")
    return card


def rate_card_version(card: dict[str, Any]) -> str:
    """Stable identifier stored on every plan, e.g. ``GHMC-2026-27-v1``."""
    agency = card.get("agency", "NA")
    fy = card.get("fy", "NA")
    ver = card.get("version", "0")
    return f"{agency}-{fy}-v{ver}"


__all__ = ["config_dir", "load_rate_card", "load_rules", "load_yaml", "rate_card_version"]
