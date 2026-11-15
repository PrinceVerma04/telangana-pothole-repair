"""Render the PWA PNG icons (192, 512, maskable 512) for web/public. Pure Pillow; no inputs read.

Run: .venv/bin/python web/scripts/make_icons.py web/public
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

NAVY = (11, 61, 110, 255)
GOLD = (242, 201, 76, 255)
WHITE = (255, 255, 255, 255)


def icon(size: int, maskable: bool) -> Image.Image:
    s = 4  # supersample
    n = size * s
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if maskable:
        d.rectangle([0, 0, n, n], fill=NAVY)
        k = 0.62  # keep the motif inside the 80 % safe zone
    else:
        d.rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.19), fill=NAVY)
        k = 0.86
    c = n / 2
    u = n * k / 32  # motif drawn on a 32-unit grid centred in the icon

    def p(x: float, y: float) -> tuple[float, float]:
        return c + (x - 16) * u, c + (y - 16) * u

    d.ellipse([*p(6, 14), *p(26, 24)], fill=GOLD)
    d.ellipse([*p(10, 15.4), *p(22, 20.6)], fill=NAVY)
    d.rectangle([*p(15, 5), *p(17, 14)], fill=WHITE)
    return im.resize((size, size), Image.LANCZOS)


def main(out: str) -> None:
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    icon(192, False).save(o / "pwa-192.png", optimize=True)
    icon(512, False).save(o / "pwa-512.png", optimize=True)
    icon(512, True).save(o / "pwa-maskable-512.png", optimize=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web/public")
