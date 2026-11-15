"""Measurement overlay PNG: mask + IRC class, depth colormap, cavity-depth heat map, numbers +- sigma.

Text is drawn with PIL + Noto fonts (Telugu / Devanagari shaped by raqm when available); falls back
to English-only if the fonts are missing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from measure.types import FrameMeasurement

IST = timezone(timedelta(hours=5, minutes=30))
FONT_DIR = Path("/usr/share/fonts/truetype/noto")
SEV_COLOURS = {"small": (80, 200, 80), "medium": (0, 170, 255), "large": (40, 40, 230), "unknown_water": (230, 160, 40)}


def _font(name: str, size: int):  # type: ignore[no-untyped-def]
    import os

    from PIL import ImageFont

    bold = "Bold" in name
    win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    # Linux Noto first; on Windows, Nirmala UI covers Latin + Telugu + Devanagari (DECISIONS D12)
    for p in (FONT_DIR / name, win / ("NirmalaB.ttc" if bold else "Nirmala.ttc"), win / "Nirmala.ttc"):
        try:
            if p.exists():
                return ImageFont.truetype(str(p), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fmt(v: Optional[float], s: Optional[float], nd: int = 1, unit: str = "") -> str:
    if v is None:
        return "n/a"
    if s is None:
        return f"{v:.{nd}f}{unit}"
    return f"{v:.{nd}f} ± {s:.{nd}f}{unit}"


def depth_colormap(depth: np.ndarray, vmax: Optional[float] = None) -> np.ndarray:
    import cv2

    d = np.where(np.isfinite(depth), depth, np.nan)
    hi = vmax or float(np.nanpercentile(d, 98))
    lo = float(np.nanpercentile(d, 1))
    x = np.clip((np.log(np.clip(d, 1e-3, None)) - np.log(max(lo, 1e-3))) / max(np.log(hi) - np.log(max(lo, 1e-3)), 1e-6), 0, 1)
    x = np.nan_to_num(x, nan=1.0)
    return cv2.applyColorMap(((1 - x) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def cavity_heatmap(shape: tuple[int, int], maps: dict[str, Any], vmax_mm: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    import cv2

    H, W = shape
    vs, us, dm = maps["vs"], maps["us"], maps["delta_mm"]
    img = np.full((H, W), np.nan)
    img[vs, us] = dm
    pad = 6
    y0, y1 = max(0, vs.min() - pad), min(H, vs.max() + pad + 1)
    x0, x1 = max(0, us.min() - pad), min(W, us.max() + pad + 1)
    crop = img[y0:y1, x0:x1]
    x = np.clip(np.nan_to_num(crop, nan=0.0) / max(vmax_mm, 1e-6), 0, 1)
    col = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    col[np.isnan(crop)] = (60, 60, 60)
    return col, (x0, y0, x1, y1)


def render_overlay(img_bgr: np.ndarray, masks: Sequence[np.ndarray], fms: Sequence[FrameMeasurement],
                   depth: Optional[np.ndarray], out_path: str | Path, *, maps: Optional[Sequence[dict]] = None,
                   title: str = "", notes: Sequence[str] = ()) -> Path:
    import cv2
    from PIL import Image, ImageDraw

    H, W = img_bgr.shape[:2]
    scale = 720.0 / max(H, W) if max(H, W) > 720 else 1.0
    def rs(a: np.ndarray, interp: int = cv2.INTER_AREA) -> np.ndarray:
        return a if scale == 1.0 else cv2.resize(a, (int(W * scale), int(H * scale)), interpolation=interp)

    left = img_bgr.copy()
    for i, (m, fm) in enumerate(zip(masks, fms, strict=False)):
        sev = fm.extras.get("severity", "medium")
        c = SEV_COLOURS.get(sev, (255, 255, 255))
        over = left.copy()
        over[m] = c
        left = cv2.addWeighted(over, 0.40, left, 0.60, 0)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, cnts, -1, c, 2)
        ys, xs = np.nonzero(m)
        cv2.putText(left, f"#{i + 1}", (int(xs.min()), max(14, int(ys.min()) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    left = rs(left)
    right = rs(depth_colormap(depth)) if depth is not None else np.zeros_like(left)
    h, w = left.shape[:2]
    heat_tiles = []
    if maps:
        for i, mp in enumerate(maps):
            if not mp or len(mp.get("vs", [])) == 0:
                continue
            vmax = max(10.0, float(np.percentile(mp["delta_mm"], 99)))
            col, _ = cavity_heatmap((H, W), mp, vmax)
            th = 150
            tw = max(60, min(int(col.shape[1] * th / max(col.shape[0], 1)), 420))
            tile = cv2.resize(col, (tw, th), interpolation=cv2.INTER_NEAREST)
            heat_tiles.append((i, tile, vmax))
    n_lines = 3 + 2 * len(fms) + len(notes)
    panel_h = 34 + 24 * n_lines + (190 if heat_tiles else 0)
    canvas = np.full((h + panel_h, 2 * w, 3), 250, np.uint8)
    canvas[:h, :w] = left
    canvas[:h, w:] = right
    x = 10
    if heat_tiles:
        for i, tile, vmax in heat_tiles:
            if x + tile.shape[1] > 2 * w - 10:
                break
            y = h + panel_h - 175
            canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
            cv2.putText(canvas, f"#{i + 1} cavity depth 0-{vmax:.0f} mm", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (30, 30, 30), 1, cv2.LINE_AA)
            x += tile.shape[1] + 16
    pil = Image.fromarray(canvas[..., ::-1])
    dr = ImageDraw.Draw(pil)
    f_en, f_b = _font("NotoSans-Regular.ttf", 16), _font("NotoSans-Bold.ttf", 17)
    f_te, f_hi = _font("NotoSansTelugu-Regular.ttf", 16), _font("NotoSansDevanagari-Regular.ttf", 16)
    y = h + 8
    now = datetime.now(IST).strftime("%d %b %Y %H:%M IST")
    dr.text((10, y), f"GunthaMeter (గుంత మీటర్)  {title}   {now}", font=f_b, fill=(20, 20, 20))
    dr.text((w + 10, 6), "depth map (near = red, far = blue)", font=f_en, fill=(255, 255, 255))
    y += 26
    for i, fm in enumerate(fms):
        sev = fm.extras.get("severity", "?")
        lab = fm.extras.get("severity_labels", {})
        line = (f"#{i + 1}  IRC:82 {lab.get('en', sev)}   depth p95 {_fmt(fm.depth_p95_mm, fm.depth_sigma_mm, 0, ' mm')}"
                f"   area {_fmt(fm.area_m2, fm.area_sigma, 2, ' m²')}   volume {_fmt(fm.volume_l, fm.volume_sigma, 1, ' L')}"
                f"   L×W {fm.length_m:.2f}×{fm.width_m:.2f} m   range {_fmt(fm.range_m, None, 1, ' m')}   q {fm.quality:.2f}")
        dr.text((10, y), line, font=f_en, fill=(20, 20, 20))
        tx = 10 + dr.textlength(line, font=f_en) + 12
        if lab.get("te"):
            dr.text((tx, y - 2), lab["te"], font=f_te, fill=(20, 20, 120))
            tx += dr.textlength(lab["te"], font=f_te) + 10
        if lab.get("hi"):
            dr.text((tx, y - 2), lab["hi"], font=f_hi, fill=(120, 20, 20))
        y += 22
        rr = fm.reject_reason or ""
        warn = "; ".join(fm.extras.get("warnings", []))[:150]
        meta = (f"     method {fm.method.value}  surface {fm.surface_model.value}  s={_fmt(fm.scale_factor, None, 2)}"
                f"  plane rms {_fmt(fm.plane_rms_mm, None, 1, ' mm')}  water {fm.water_filled}"
                f"  {('REJECTED: ' + rr) if rr else ''}  {warn}")
        dr.text((10, y), meta, font=f_en, fill=(90, 90, 90))
        y += 24
    cert_line = ("certified=false: for prioritisation only, NOT for Measurement Book billing"
                 if not (fms and fms[0].certified) else "certified calibration")
    dr.text((10, y), cert_line, font=f_b, fill=(160, 30, 30))
    y += 24
    for n in notes:
        dr.text((10, y), n, font=f_en, fill=(60, 60, 60))
        y += 22
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    return out
