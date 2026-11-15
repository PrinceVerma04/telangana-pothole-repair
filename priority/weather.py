"""Weather inputs for the M term and the repair weather gate (PLAN 6 M term, PLAN 7.9).

- ``in_monsoon_window(month, cfg, manual)``: Telangana SW monsoon Jun-Sep, NE Oct-Nov (config);
  a manual declaration (admin flag) overrides the calendar either way.
- ``ImdAdapter``: interface for IMD district rainfall / nowcast. Demo ships ``ManualImdAdapter``
  (values set by an admin or tests) and ``NullImdAdapter``. No network access from this module;
  a real IMD client implements the same two methods elsewhere.
- ``monsoon_term(...)``: M = base 0.5; 0.8 in window; 1.0 if 72 h rain > 30 mm; +0.2 hotspot;
  +0.1 water-filled; clip 1.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional, Protocol

from priority.config import resolve

IST = timezone(timedelta(hours=5, minutes=30))


def month_ist(at: Optional[datetime] = None) -> int:
    """Calendar month in IST (UTC stored, IST displayed and used for seasonal rules)."""
    at = at or datetime.now(timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at.astimezone(IST).month


def monsoon_window_name(month: int, cfg: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    for w in resolve(cfg)["monsoon"].get("windows", []):
        lo, hi = int(w["from_month"]), int(w["to_month"])
        inside = lo <= month <= hi if lo <= hi else (month >= lo or month <= hi)  # wrap-around windows
        if inside:
            return str(w.get("name", f"{lo}-{hi}"))
    return None


def in_monsoon_window(month: Optional[int], cfg: Optional[Mapping[str, Any]] = None,
                      manual: Optional[bool] = None) -> bool:
    if manual is not None:
        return bool(manual)
    if month is None:
        return False
    return monsoon_window_name(int(month), cfg) is not None


class ImdAdapter(Protocol):
    def rain_72h_mm(self, district: str, at: Optional[datetime] = None) -> Optional[float]: ...

    def raining_now(self, district: str, at: Optional[datetime] = None) -> Optional[bool]: ...


class NullImdAdapter:
    """No feed: unknown rain (M falls back to the calendar window)."""

    def rain_72h_mm(self, district: str, at: Optional[datetime] = None) -> Optional[float]:
        return None

    def raining_now(self, district: str, at: Optional[datetime] = None) -> Optional[bool]:
        return None


class ManualImdAdapter:
    """Demo adapter: admin-entered district values (``{"Hyderabad": 42.0}``) and nowcast flags."""

    def __init__(self, rain_72h: Optional[Mapping[str, float]] = None,
                 raining: Optional[Mapping[str, bool]] = None) -> None:
        self._rain = {k.lower(): float(v) for k, v in (rain_72h or {}).items()}
        self._now = {k.lower(): bool(v) for k, v in (raining or {}).items()}

    def set_rain(self, district: str, mm_72h: float) -> None:
        self._rain[district.lower()] = float(mm_72h)

    def set_raining(self, district: str, raining: bool) -> None:
        self._now[district.lower()] = bool(raining)

    def rain_72h_mm(self, district: str, at: Optional[datetime] = None) -> Optional[float]:
        return self._rain.get(district.lower())

    def raining_now(self, district: str, at: Optional[datetime] = None) -> Optional[bool]:
        return self._now.get(district.lower())


def monsoon_term(month: Optional[int], rain_72h_mm: Optional[float] = None, in_hotspot: bool = False,
                 water_filled: bool = False, cfg: Optional[Mapping[str, Any]] = None,
                 manual_monsoon: Optional[bool] = None) -> tuple[float, dict[str, Any]]:
    mc = resolve(cfg)["monsoon"]
    window = in_monsoon_window(month, cfg, manual_monsoon)
    heavy = rain_72h_mm is not None and rain_72h_mm > float(mc.get("heavy_rain_72h_mm", 30))
    if heavy:
        base = float(mc.get("heavy_rain", 1.0))
    elif window:
        base = float(mc.get("in_window", 0.8))
    else:
        base = float(mc.get("base", 0.5))
    hot = float(mc.get("hotspot_bonus", 0.2)) if in_hotspot else 0.0
    wat = float(mc.get("water_filled_bonus", 0.1)) if water_filled else 0.0
    m = min(1.0, base + hot + wat)
    return m, {"base": base, "in_window": window, "window": monsoon_window_name(month, cfg) if month else None,
               "manual_monsoon": manual_monsoon, "rain_72h_mm": rain_72h_mm, "heavy_rain": heavy,
               "hotspot_bonus": hot, "water_filled_bonus": wat}


__all__ = ["IST", "ImdAdapter", "ManualImdAdapter", "NullImdAdapter", "in_monsoon_window", "month_ist",
           "monsoon_term", "monsoon_window_name"]
