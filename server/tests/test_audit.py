"""WP4 acceptance: audit_log written on admin config changes (append-only); versioned configs; admin uploads."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import DatabaseError

from server.app.core import audit
from server.app.db import models as m


def _count(app: Any, entity: str) -> int:
    with app.state.database.session() as s:
        return len(s.scalars(select(m.AuditLog).where(m.AuditLog.entity == entity)).all())


def test_priority_config_put_is_versioned_and_audited(client: TestClient, app: Any, login: Any) -> None:
    z = login("zonal_khairatabad")
    cur = client.get("/v1/admin/priority-config", headers=z).json()
    assert cur["agency"] == "GHMC" and cur["version"] == 1 and cur["config"]["weights"]["S"] == 0.30
    n0 = _count(app, "priority_config")
    new = copy.deepcopy(cur["config"])
    new["weights"] = {"S": 0.35, "E": 0.25, "V": 0.20, "T": 0.10, "M": 0.10}
    r = client.put("/v1/admin/priority-config?reason=monsoon%20push", json=new, headers=z)
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 2
    assert client.get("/v1/admin/priority-config", headers=z).json()["config"]["weights"]["S"] == 0.35
    assert _count(app, "priority_config") == n0 + 1
    entries = client.get("/v1/admin/audit?entity=priority_config", headers=z).json()
    e = entries[0]
    assert e["action"] == "update" and e["actor"] == "zonal_khairatabad" and e["reason"] == "monsoon push"
    assert e["before"]["config"]["weights"]["S"] == 0.30 and e["after"]["config"]["weights"]["S"] == 0.35
    hist = client.get("/v1/admin/priority-config/history", headers=z).json()
    assert [h["version"] for h in hist] == [2, 1] and [h["active"] for h in hist] == [True, False]

    bad = copy.deepcopy(new)
    bad["weights"]["S"] = 0.9  # sum != 1
    r = client.put("/v1/admin/priority-config", json=bad, headers=z)
    assert r.status_code == 422 and "sum" in r.json()["detail"]
    assert _count(app, "priority_config") == n0 + 1  # rejected change leaves no audit row


def test_rate_card_put_is_versioned_and_audited(client: TestClient, app: Any, login: Any) -> None:
    z = login("zonal_khairatabad")
    cur = client.get("/v1/admin/rate-card", headers=z).json()
    assert cur["fy"] == "2026-27" and cur["card"]["items"]["coldmix_bag_25kg"]["rate"] == 350
    card = copy.deepcopy(cur["card"])
    card["items"]["coldmix_bag_25kg"]["rate"] = 380
    card["items"]["coldmix_bag_25kg"]["provenance"] = "Khairatabad circle tender 2026 (demo)"
    r = client.put("/v1/admin/rate-card", json=card, headers=z)
    assert r.status_code == 200, r.text
    assert r.json()["version"] == 2 and r.json()["version_label"] == "GHMC/2026-27/v2"
    e = client.get("/v1/admin/audit?entity=rate_card", headers=z).json()[0]
    assert e["before"]["card"]["items"]["coldmix_bag_25kg"]["rate"] == 350
    assert e["after"]["card"]["items"]["coldmix_bag_25kg"]["rate"] == 380
    card["items"]["coldmix_bag_25kg"]["rate"] = -1
    assert client.put("/v1/admin/rate-card", json=card, headers=z).status_code == 422


def test_audit_log_is_append_only(app: Any, seeded: Any) -> None:
    with app.state.database.session() as s:
        row = audit.record(s, "test", 1, "create", actor="pytest", after={"x": 1})
        s.commit()
        row.reason = "tamper"
        with pytest.raises(PermissionError):
            s.flush()
        s.rollback()
        with pytest.raises(PermissionError):
            s.delete(s.get(m.AuditLog, row.id))
            s.flush()
        s.rollback()
    # also enforced in the database (trigger), not only in the ORM
    with app.state.database.engine.connect() as c:
        with pytest.raises(DatabaseError):
            c.execute(text("UPDATE audit_log SET reason='x'"))
        c.rollback()
        with pytest.raises(DatabaseError):
            c.execute(text("DELETE FROM audit_log"))


def test_admin_units_upload_and_jurisdiction_resolve(client: TestClient, app: Any, login: Any) -> None:
    z = login("zonal_khairatabad")
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"code": "GHMC-W097", "kind": "ward", "name_en": "Somajiguda",
                                           "name_te": "సోమాజిగూడ", "parent_code": "GHMC-C17",
                                           "valid_from": "2026-11-01"},
         "geometry": {"type": "Polygon", "coordinates": [[[78.455, 17.42], [78.465, 17.42], [78.465, 17.43],
                                                          [78.455, 17.43], [78.455, 17.42]]]}},
        {"type": "Feature", "properties": {"code": "BAD", "kind": "planet", "name_en": "x"},
         "geometry": {"type": "Point", "coordinates": [78.4, 17.4]}},
    ]}
    r = client.post("/v1/admin/admin-units", json=fc, headers=z)
    assert r.status_code == 200, r.text
    assert r.json()["created"] == 1 and len(r.json()["errors"]) == 1
    assert client.post("/v1/admin/admin-units", json=fc, headers=z).json()["updated"] == 1
    assert client.post("/v1/admin/admin-units", json=fc, headers=login("contractor1")).status_code == 403
    with app.state.database.session() as s:
        w = s.scalars(select(m.AdminUnit).where(m.AdminUnit.code == "GHMC-W097")).one()
        parent = s.get(m.AdminUnit, w.parent_id)
        assert parent.code == "GHMC-C17" and w.min_lon == 78.455 and not w.official
        ph = m.Pothole(lat=17.425, lon=78.46, status="confirmed")
        s.add(ph)
        seg = m.RoadSegment(osm_way_id=7, seg_idx=0, highway="secondary",
                            geom_wkt="LINESTRING (78.459 17.425, 78.461 17.425)")
        s.add(seg)
        s.flush()
        ph.road_segment_id = seg.id
        item = m.JurisdictionQueue(pothole_id=ph.id, reason="segment owner unknown", candidates=["GHMC", "RNB"])
        s.add(item)
        s.commit()
        item_id, ph_id, seg_id = item.id, ph.id, seg.id
    q = client.get("/v1/admin/jurisdiction-queue", headers=z).json()
    assert [i["id"] for i in q] == [item_id] and q[0]["candidates"] == ["GHMC", "RNB"]
    r = client.post(f"/v1/admin/jurisdiction-queue/{item_id}/resolve", json={"agency_code": "GHMC"}, headers=z)
    assert r.status_code == 200 and r.json()["resolved"]
    assert client.post(f"/v1/admin/jurisdiction-queue/{item_id}/resolve", json={"agency_code": "GHMC"},
                       headers=z).status_code == 409
    with app.state.database.session() as s:
        ghmc = s.scalars(select(m.Agency).where(m.Agency.code == "GHMC")).one()
        assert s.get(m.Pothole, ph_id).agency_id == ghmc.id
        assert s.get(m.RoadSegment, seg_id).owner_agency_id == ghmc.id
    a = client.get(f"/v1/admin/audit?entity=jurisdiction_queue&id={item_id}", headers=z).json()
    assert a[0]["action"] == "resolve" and a[0]["after"]["agency_code"] == "GHMC"
    assert client.get("/v1/admin/audit?entity=admin_unit", headers=z).json()
