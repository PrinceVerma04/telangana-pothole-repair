"""Seed the demo database (``make seed``): agencies, admin units, CRMP contracts, priority config, rate card,
demo users (password ``demo1234``), devices + calibration certificates, OSM road segments, POIs.

    .venv/bin/python scripts/seed_demo.py                 # data/interim/geo (falls back to tests/fixtures/geo_small)
    .venv/bin/python scripts/seed_demo.py --fixture       # small Ameerpet/Punjagutta fixture (fast)
    .venv/bin/python scripts/seed_demo.py --drop --fixture --sample-potholes 12
    .venv/bin/python scripts/seed_demo.py --db-url sqlite:////tmp/x.db --no-geo

Idempotent. Demo users are listed in ``server/app/seed.py`` (admin, zonal_khairatabad, engineer1, contractor1,
contractor2, citizen1, rnb_admin). Geo files are untrusted input: they are parsed as JSON only, never executed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.app.core.config import get_settings  # noqa: E402
from server.app.db.session import Database  # noqa: E402
from server.app.seed import DEMO_PASSWORD, DEMO_USERS, seed_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-url", default=None, help="DATABASE_URL override (default: .env / sqlite:///data/gunthameter.db)")
    ap.add_argument("--fixture", action="store_true", help="use tests/fixtures/geo_small/ (small, fast)")
    ap.add_argument("--geo-dir", type=Path, default=None, help="directory with road_segments.geojson / pois.geojson")
    ap.add_argument("--no-geo", action="store_true", help="skip roads/POIs/admin-unit layers")
    ap.add_argument("--drop", action="store_true", help="drop and recreate all tables first (DESTROYS DATA)")
    ap.add_argument("--sample-potholes", type=int, default=0, metavar="N",
                    help="also add N synthetic, clearly labelled potholes near Ameerpet (UI work before the simulator)")
    ap.add_argument("--reset-passwords", action="store_true", help=f"reset demo users' passwords to {DEMO_PASSWORD}")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    url = a.db_url or get_settings().DATABASE_URL
    db = Database(url)
    if a.drop:
        db.drop_all()
    db.create_all()
    t0 = time.time()
    with db.session() as s:
        summary = seed_all(s, fixture=a.fixture, geo_dir=a.geo_dir, with_geo=not a.no_geo,
                           sample_potholes=a.sample_potholes, reset_passwords=a.reset_passwords)
    summary["elapsed_s"] = round(time.time() - t0, 2)
    summary["database"] = url
    if not a.quiet:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        print(f"\nDemo users (password '{DEMO_PASSWORD}', demo only): "
              + ", ".join(f"{u['username']} ({u['role']})" for u in DEMO_USERS))
        if summary.get("geo_dir") is None and not a.no_geo:
            print("NOTE: no geo data found yet -> seeded WITHOUT roads/POIs (run WP10 / pass --geo-dir).")
    db.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
