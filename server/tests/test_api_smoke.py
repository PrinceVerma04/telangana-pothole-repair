"""WP4 acceptance: register device -> presign/upload -> post the edge/schema/examples observations -> list potholes;
duplicate client_uuid counted as duplicate; trip fixes + close; citizen report; tiles; detail; health."""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from measure.types import FrameMeasurement
from server.tests.helpers import DEVICE_ID, example, variant


def test_healthz_and_openapi(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["db"] and body["storage"] and body["demo_mode"]
    assert client.get("/v1/healthz").json()["status"] == "ok"
    spec = client.get("/v1/openapi.json").json()
    for path in ("/v1/ingest/observations", "/v1/potholes", "/v1/edge/register", "/v1/admin/priority-config",
                 "/v1/tiles/potholes/{z}/{x}/{y}.geojson", "/v1/public/wards", "/v1/measure"):
        assert path in spec["paths"], path


def test_register_ingest_duplicate_list(client: TestClient, device_headers: dict[str, str],
                                        login: Any) -> None:
    edge, jolt = example("observation_edge.json"), example("observation_jolt.json")

    # presigned upload of the keyframe (FsStorage demo route)
    key = edge["frame"]["key"]
    r = client.post("/v1/edge/presign", json={"keys": [key]}, headers=device_headers)
    assert r.status_code == 200, r.text
    url = r.json()["urls"][key]
    assert client.put(url, content=b"\xff\xd8\xff fake jpeg").status_code == 200
    assert client.put(url.replace("sig=", "sig=0"), content=b"x").status_code == 403  # tampered signature

    # 1st post: both examples accepted, pipeline runs inline (DEMO_MODE)
    r = client.post("/v1/ingest/observations", json={"observations": [edge, jolt]}, headers=device_headers)
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["accepted"] == 2 and res["duplicates"] == 0 and res["rejected"] == []
    ids = res["observation_ids"]
    assert len(ids) == 2

    # 2nd post (store-and-forward retry): idempotent by client_uuid
    r = client.post("/v1/ingest/observations", json={"observations": [edge, jolt]}, headers=device_headers)
    res2 = r.json()
    assert res2["accepted"] == 0 and res2["duplicates"] == 2 and res2["observation_ids"] == ids
    # duplicate inside one batch is also counted once
    fresh = variant(edge)
    r = client.post("/v1/ingest/observations", json=[fresh, fresh], headers=device_headers)
    assert r.json()["accepted"] == 1 and r.json()["duplicates"] == 1

    # list potholes: staff see the new candidate (1 pass), the public does not
    eng = login("zonal_khairatabad")
    r = client.get("/v1/potholes", headers=eng)
    assert r.status_code == 200, r.text
    page = r.json()
    assert page["total"] == 1
    item = page["items"][0]
    # WP6 locate stage: the pothole lies ahead of the raw fix (17.4375, flat-road projection of the detection +
    # GNSS latency), a few metres away -- not on the fix itself
    from server.app.core.util import haversine_m

    assert item["status"] == "candidate" and 1.0 < haversine_m(item["lat"], item["lon"], 17.4375,
                                                               edge["gps"]["lon"]) < 20.0
    assert item["thumb_url"] and "/v1/files/" in item["thumb_url"]
    assert client.get(item["thumb_url"]).content.startswith(b"\xff\xd8\xff")
    assert client.get("/v1/potholes").json()["total"] == 0  # anonymous: candidates hidden

    # filters
    assert client.get("/v1/potholes?bbox=78.44,17.43,78.45,17.44", headers=eng).json()["total"] == 1
    assert client.get("/v1/potholes?bbox=78.0,17.0,78.1,17.1", headers=eng).json()["total"] == 0
    assert client.get("/v1/potholes?status=confirmed", headers=eng).json()["total"] == 0
    assert client.get("/v1/potholes?status=bogus", headers=eng).status_code == 422
    assert client.get("/v1/potholes?agency=GHMC&admin_unit=GHMC-KHT", headers=eng).json()["total"] == 1

    # detail
    d = client.get(f"/v1/potholes/{item['id']}", headers=eng).json()
    assert d["pothole"]["status"] == "candidate" and d["events"][0]["to_status"] == "candidate"

    # second trip over the same pothole -> confirmed (>= 2 distinct trips), now public
    second = variant(edge, trip_id="11111111-2222-4333-8444-555555555555", seq=7)
    r = client.post("/v1/ingest/observations", json={"observations": [second]}, headers=device_headers)
    assert r.json()["accepted"] == 1
    page = client.get("/v1/potholes").json()
    assert page["total"] == 1 and page["items"][0]["status"] == "confirmed"
    assert page["items"][0]["passes_seen"] == 2

    # per-tile GeoJSON (z16 tile containing the pothole)
    import math

    z = 16
    n = 2 ** z
    x = int((78.4482 + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(17.4375))) / math.pi) / 2 * n)
    fc = client.get(f"/v1/tiles/potholes/{z}/{x}/{y}.geojson").json()
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 1
    assert client.get(f"/v1/tiles/potholes/{z}/{x + 5}/{y}.geojson").json()["features"] == []


def test_trip_fixes_and_close(client: TestClient, device_headers: dict[str, str]) -> None:
    tf = example("trip_fix.json")
    r = client.post("/v1/ingest/trip-fixes", json=tf, headers=device_headers)
    assert r.status_code == 200, r.text
    assert r.json()["accepted"] == 3
    assert client.post("/v1/ingest/trip-fixes", json=tf, headers=device_headers).json()["duplicates"] == 3
    r = client.post("/v1/ingest/trip-close", headers=device_headers,
                    json={"device_id": DEVICE_ID, "trip_id": tf["trip_id"], "ended_at": "2026-07-14T09:00:00+05:30"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["closed"] and body["fixes"] == 3 and 0.0 < body["km"] < 0.1
    r = client.post("/v1/ingest/trip-fixes", json=tf, headers=device_headers)
    assert r.status_code == 409  # closed trip


def test_ingest_validation_and_auth(client: TestClient, device_headers: dict[str, str], login: Any) -> None:
    edge = example("observation_edge.json")
    assert client.post("/v1/ingest/observations", json={"observations": [edge]}).status_code == 401
    bad_schema = variant(edge, source="drone")
    other_dev = variant(edge, device_id="someone-else")
    ok = variant(edge)
    r = client.post("/v1/ingest/observations", json={"observations": [bad_schema, other_dev, ok]},
                    headers=device_headers)
    res = r.json()
    assert res["accepted"] == 1 and len(res["rejected"]) == 2
    assert res["rejected"][0]["error"].startswith("schema:")
    assert "device token" in res["rejected"][1]["error"]
    assert client.post("/v1/ingest/observations", json={"observations": []},
                       headers=device_headers).status_code == 422
    # staff may post for a registered device (simulator); unknown devices are rejected per item
    eng = login("engineer1")
    r = client.post("/v1/ingest/observations", json=[variant(edge), variant(edge, device_id="ghost")], headers=eng)
    assert r.json()["accepted"] == 1 and "unknown device_id" in r.json()["rejected"][0]["error"]
    # citizens / contractors may not ingest
    assert client.post("/v1/ingest/observations", json=[variant(edge)], headers=login("citizen1")).status_code == 403


def test_citizen_report_links_to_pothole(client: TestClient, device_headers: dict[str, str]) -> None:
    edge = example("observation_edge.json")
    client.post("/v1/ingest/observations", json={"observations": [edge]}, headers=device_headers)
    jpeg = b"\xff\xd8\xff\xe0" + b"0" * 100
    r = client.post("/v1/citizen/reports", data={"lat": "17.43752", "lon": "78.44822", "text": "పెద్ద గుంత",
                                                  "lang": "te", "phone": "+919876543210"},
                    files={"photo": ("p.jpg", jpeg, "image/jpeg")})
    assert r.status_code == 201, r.text
    out = r.json()
    assert out["ticket"].startswith("GM-") and out["pothole_id"] is not None
    assert out["status"] == "confirmed"  # 1 pass + citizen report within 25 m (PLAN 5.5)
    t = client.get(f"/v1/citizen/reports/{out['ticket']}").json()
    assert t["pothole_id"] == out["pothole_id"]
    assert client.get("/v1/citizen/reports/GM-000000-XXXXXX").status_code == 404
    r = client.post("/v1/citizen/reports", data={"lat": "17.43", "lon": "78.44"},
                    files={"photo": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 415
    assert client.post("/v1/citizen/reports", data={"lat": "51.5", "lon": "-0.1"}).status_code == 422


def test_edge_config_signed_with_etag(client: TestClient, device_headers: dict[str, str]) -> None:
    import json

    from server.app.core import ed25519

    r = client.get("/v1/edge/config", headers=device_headers)
    assert r.status_code == 200
    etag, cfg = r.headers["etag"], r.json()
    sig = bytes.fromhex(cfg.pop("signature_ed25519"))
    cfg["signature_ed25519"] = None
    pub = bytes.fromhex(client.get("/v1/edge/config/pubkey").json()["public_key_hex"])
    assert ed25519.verify(pub, json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode(), sig)
    r = client.get("/v1/edge/config", headers={**device_headers, "If-None-Match": etag})
    assert r.status_code == 304


def test_measure_503_without_core_and_ok_with_it(client: TestClient, monkeypatch: Any) -> None:
    import cv2

    img = np.full((72, 128, 3), 90, np.uint8)
    ok, jpg = cv2.imencode(".jpg", img)
    files = {"image": ("f.jpg", jpg.tobytes(), "image/jpeg")}
    data = {"calib_id": "phone_demo", "bboxes": "[[40, 40, 80, 60]]"}

    monkeypatch.setitem(sys.modules, "measure.pipeline", None)  # import -> ImportError
    r = client.post("/v1/measure", files=files, data=data)
    assert r.status_code == 503 and "measure.pipeline" in r.json()["detail"]

    calls: list[tuple[int, ...]] = []

    def fake_measure_frame(im: np.ndarray, mask: np.ndarray, cert: Any, cfg: Any) -> FrameMeasurement:
        calls.append(mask.shape)
        return FrameMeasurement(depth_p95_mm=42.0, area_m2=0.2, width_m=0.3, quality=0.8, calib_id=cert.calib_id,
                                depth_model_version="fake-depth", mask_px=int(mask.sum()))

    fake = types.ModuleType("measure.pipeline")
    fake.measure_frame = fake_measure_frame  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "measure.pipeline", fake)
    r = client.post("/v1/measure", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert calls == [(72, 128)] and body["measurements"][0]["depth_p95_mm"] == 42.0
    assert body["calib_id"] == "phone_demo" and body["certified"] is False
    assert body["depth_model_version"] == "fake-depth"
    r = client.post("/v1/measure", files=files, data={"calib_id": "nope"})
    assert r.status_code == 404


def test_segment_rating_and_public_wards(client: TestClient, app: Any, device_headers: dict[str, str]) -> None:
    from server.app.db import models as m

    with app.state.database.session() as s:
        seg = m.RoadSegment(osm_way_id=42, seg_idx=0, highway="primary", lanes=2, length_m=200.0,
                            geom_wkt="LINESTRING (78.4470 17.4370, 78.4490 17.4380)")
        s.add(seg)
        s.flush()
        s.add(m.Pothole(lat=17.4375, lon=78.4480, status="confirmed", area_m2=4.0, road_segment_id=seg.id,
                        severity="large", admin_unit_id=None))
        s.commit()
    rows = client.get("/v1/segments/rating?bbox=78.44,17.43,78.46,17.44").json()
    assert len(rows) == 1 and rows[0]["rating"] == "Poor" and rows[0]["pothole_area_pct"] > 0.5
    assert rows[0]["geometry"]["type"] == "LineString"
    all_bins = client.get("/v1/segments/rating?bbox=78.44,17.43,78.46,17.44&include_good=true").json()
    assert len(all_bins) >= 2 and {b["rating"] for b in all_bins} >= {"Poor", "Good"}
    w = client.get("/v1/public/wards").json()
    assert w["kind"] == "circle" and any(i["code"] == "GHMC-C17" for i in w["items"])
    fc = client.get("/v1/public/wards?geometry=true&kind=zone").json()
    assert fc["type"] == "FeatureCollection" and fc["features"][0]["geometry"]["type"] == "Polygon"
    assert {a["code"] for a in client.get("/v1/public/agencies").json()} >= {"GHMC", "RNB", "NHAI"}
