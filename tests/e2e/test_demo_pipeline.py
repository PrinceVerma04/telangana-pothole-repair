"""WP12 end-to-end: the virtual fleet (edge/sim/fleet.py, in-process API) drives the whole pothole lifecycle.

Self-contained (no downloaded data): the ``tests/fixtures/geo_small`` road network is seeded into a fresh SQLite DB,
a tiny labelled frame set is generated (road texture + an elliptical pothole), and ``fleet.main`` runs one bus on the
fixture GTFS route window that lies best inside the fixture box. Depth is the simulator's synthetic road render
(``edge/sim/sim_server.py``), so no torch model is loaded. Flow asserted:

    survey trips -> confirmed (>= 2 trips) -> planned + WO issued -> contractor claims the repair ->
    verification trips (repaired potholes no longer detected) -> verified_closed ->
    DLP check trip re-detects one repair -> reopened + zero-pay rework work order.

The 10 % engineer QC sample is switched off so the bus path decides (it is covered by the WP8b tests).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest

from edge.sim.route import load_routes_json, route_from_record

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "geo_small"


def _label_bank(root: Path, n_pos: int = 6, n_neg: int = 3) -> Path:
    """YOLO-seg frames in <root>/images/val + <root>/labels/val (1280x720, class 0 = pothole)."""
    img_dir, lab_dir = root / "images" / "val", root / "labels" / "val"
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    rng = np.random.default_rng(5)
    W, H = 1280, 720
    for i in range(n_pos + n_neg):
        img = np.clip(rng.normal(105, 18, (H, W, 1)), 0, 255).astype(np.uint8).repeat(3, axis=2)
        img[: H // 3] = (200, 170, 140)  # sky-ish band above the horizon
        text = ""
        if i < n_pos:
            cx, cy = 560 + 40 * i, 540 + 10 * (i % 3)
            ax, ay = 70 + 8 * i, 26 + 3 * i
            cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (45, 45, 50), -1)
            t = np.linspace(0, 2 * math.pi, 24, endpoint=False)
            pts = np.stack([cx + ax * np.cos(t), cy + ay * np.sin(t)], axis=1) / [W, H]
            text = "0 " + " ".join(f"{u:.5f} {v:.5f}" for u, v in pts) + "\n"
        cv2.imwrite(str(img_dir / f"f{i:02d}.jpg"), img)
        (lab_dir / f"f{i:02d}.txt").write_text(text, encoding="utf-8")
    return img_dir


def _best_window(margin_m: float = 150.0, min_len_m: float = 1200.0) -> tuple[str, float, float]:
    """(route_id, start_m, length_m) of the longest fixture-route stretch inside the fixture box (minus a margin)."""
    bb = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))["bbox"]
    dlat = margin_m / 111_000.0
    dlon = margin_m / (111_000.0 * math.cos(math.radians(17.43)))
    lo0, la0, lo1, la1 = bb[0] + dlon, bb[1] + dlat, bb[2] - dlon, bb[3] - dlat
    best = ("", 0.0, 0.0)
    for rec in load_routes_json(FIXTURE / "gtfs_routes.json"):
        if len(rec.get("shape") or []) < 2:
            continue
        r = route_from_record(rec)
        start = None
        c = 0.0
        while c <= r.length_m + 1e-6:
            la, lo = r.latlon_at(c)
            inside = lo0 <= lo <= lo1 and la0 <= la <= la1
            if inside and start is None:
                start = c
            if (not inside or c + 25.0 > r.length_m) and start is not None:
                if c - start > best[2]:
                    best = (str(rec["route_id"]), start, c - start)
                start = None
            c += 25.0
    assert best[2] >= min_len_m, f"no fixture route stretch of {min_len_m} m inside the box: {best}"
    return best


@pytest.fixture
def seeded_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from server.app.core.config import Settings
    from server.app.main import create_app
    from server.app.seed import seed_all
    from server.app.services import verification
    from server.workers import adapters, stages, uninstall

    for var in ("GM_JURISDICTION_POLICY", "GM_JURISDICTION_ADMIN_FALLBACK", "GM_AUTO_PLAN_STATUS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GM_DEVICE", "cpu")
    monkeypatch.setattr(verification, "qc_sampled", lambda _wo_id, _frac: False)
    db = tmp_path / "demo.db"
    settings = Settings(DATABASE_URL=f"sqlite:///{db.as_posix()}", STORAGE_ROOT=str(tmp_path / "store"),
                        DEMO_MODE=True, JWT_SECRET="e2e-demo-secret-not-for-production-0123456789",
                        API_BASE_URL="http://127.0.0.1:8002", _env_file=None)
    app = create_app(settings)
    with app.state.database.session() as s:
        seed_all(s, fixture=True)
    app.state.database.dispose()
    uninstall()
    adapters.clear_caches()
    try:
        yield db
    finally:
        uninstall()
        stages.set_measurer(None)
        stages.reset_runtime()
        adapters.clear_caches()


@pytest.mark.slow
def test_fleet_drives_full_lifecycle(seeded_db: Path, tmp_path: Path) -> None:
    from edge.sim import fleet

    frames = _label_bank(tmp_path / "ds")
    route_id, start_m, length_m = _best_window()
    out = tmp_path / "fleet_summary.json"
    rc = fleet.main([
        "--in-process", "--db", str(seeded_db), "--store", str(tmp_path / "store"),
        "--gtfs", str(FIXTURE / "gtfs_routes.json"), "--frames", str(frames),
        "--routes", route_id, "--buses", "1", "--route-start-km", f"{start_m / 1000:.3f}",
        "--route-km", f"{min(length_m, 2000.0) / 1000:.3f}", "--pothole-every-m", "300",
        "--repairs", "2", "--trips-per-day", "3", "--no-unowned", "--citizen", "1", "--jolts", "1",
        "--p-detect", "0.97", "--min-confirmed", "2", "--summary-json", str(out), "--seed", "11"])
    res = json.loads(out.read_text(encoding="utf-8"))
    s = res["summary"]
    st = s["potholes_by_status"]
    assert s["ever_confirmed"] >= 2, s
    assert len(res["repairs"]) >= 1, res["repairs"]
    assert st.get("verified_closed", 0) + st.get("reopened", 0) >= 1, s  # >= 1 repair proven by the buses
    assert st.get("reopened", 0) >= 1 and s["rework_work_orders"] >= 1, s  # DLP re-detection -> rework
    assert res["dlp_target"]["pothole_id"] in {r["pothole_id"] for r in res["repairs"]}
    assert all(not t.get("rejected") for t in res["trips"]), "the server rejected simulator observations"
    assert rc == 0
