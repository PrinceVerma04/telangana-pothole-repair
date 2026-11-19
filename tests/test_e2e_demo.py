"""D51: dummy-data end-to-end demo helpers and the pipeline dashboard."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import e2e_dashboard as E  # noqa: E402
import e2e_demo as D  # noqa: E402


def test_route_is_cut_to_length_and_follows_the_road(tmp_path: Path):
    db = tmp_path / "r.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE road_segment (osm_way_id INT, seg_idx INT, geom_wkt TEXT)")
    con.execute("INSERT INTO road_segment VALUES (1, 0, 'LINESTRING (78.4000 17.4000, 78.4010 17.4000)')")
    con.execute("INSERT INTO road_segment VALUES (1, 1, 'LINESTRING (78.4010 17.4000, 78.4020 17.4000)')")
    con.commit()
    pts = D.route(db, way=1, length_m=150)
    total = sum(D.metres(a, b) for a, b in zip(pts, pts[1:], strict=False))
    assert abs(total - 150) < 1 and pts[0] == (17.4, 78.4)
    lat, lon, hdg = D.along(pts, 50)
    assert abs(hdg - 90) < 1 and lat == 17.4 and 78.4 < lon < 78.401


def test_frames_are_1280x720():
    assert D.frame_16x9(np.zeros((640, 640, 3), np.uint8)).shape == (720, 1280, 3)
    assert D.frame_16x9(np.zeros((500, 2000, 3), np.uint8), shift=0.3).shape == (720, 1280, 3)


def test_funnel_counts_distinct_api_potholes():
    view = {"frames": 10, "processed": 10, "keyframes": [{"potholes": 1}, {"potholes": 0}],
            "potholes": [{"reliable": True, "api": {"id": 4, "status": "work_ordered", "plan": {"bags_low": 1},
                                                    "work_orders": [{"id": 1}]}},
                         {"reliable": True, "api": {"id": 4, "status": "work_ordered", "plan": {"bags_low": 1},
                                                    "work_orders": [{"id": 1}]}},
                         {"reliable": False, "api": {"id": 5, "status": "candidate"}},
                         {"reliable": False}]}
    f = {s["stage"]: s["n"] for s in E.funnel(view)}
    assert f["Potholes (linked across frames)"] == 4 and f["Measured (reliable size)"] == 2
    assert f["RoadFix potholes (distinct)"] == 2 and f["Confirmed (>= 2 passes)"] == 1 and f["On a work order"] == 1


class FakeApi:
    def __init__(self) -> None:
        self.calls = 0

    def enrich(self, base, view):
        self.calls += 1
        for p in view["potholes"]:
            p["api"] = {"id": 9, "status": "confirmed", "plan": {"bags_low": 2, "bags_high": 3}}
        return {"state": "ok", "base": base, "potholes_on_road": 1}


def test_dashboard_lists_demo_and_real_drives(tmp_path: Path):
    from fastapi.testclient import TestClient

    import phone_capture as pc

    roots = {"demo": tmp_path / "demo", "real": tmp_path / "real"}
    tid = "11111111-0000-4000-8000-000000000001"
    for root in roots.values():
        root.mkdir()
    pc.start_trip(roots["demo"], {"trip_id": tid, "calib_id": "c1", "device_id": "d1"}, "Demo bus A")
    ok, buf = cv2.imencode(".jpg", np.zeros((72, 128, 3), np.uint8))
    pc.put_frame(roots["demo"], tid, {"seq": 1, "captured_at": "2026-11-19T10:00:00+05:30",
                                      "gps": {"lat": 17.4, "lon": 78.4}}, buf.tobytes())
    summ = tmp_path / "s.json"
    summ.write_text(json.dumps({"api": "http://x", "work": {"work_orders": [], "runs": []}, "kpis": []}))
    fake = FakeApi()
    app = E.make_app(roots, state=tmp_path / "state", summary=summ, api_view=fake)
    c = TestClient(app)
    assert c.get("api/drives").status_code == 401
    c.post("api/login", json={"name": "Ann", "code": app.state.access_code})
    d = c.get("api/drives").json()
    assert [x["src"] for x in d["drives"]] == ["demo"] and d["demo"]["api"] == "http://x"
    v = c.get("api/drive", params={"src": "demo", "trip": tid}).json()
    assert v["api_link"]["state"] == "ok" and v["funnel"][0]["n"] == 1 and fake.calls == 1
    assert c.get("api/drive", params={"src": "real", "trip": tid}).status_code == 404
    assert c.get("media", params={"src": "demo", "trip": tid, "name": "../trip.json"}).status_code == 404
    assert c.get("media", params={"src": "demo", "trip": tid, "name": "000001.jpg"}).content[:2] == b"\xff\xd8"
    assert c.get("api/work").json()["api"] == "http://x"
