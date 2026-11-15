"""WP4 acceptance: contractor cannot read another contractor's work orders; role guards; agency scoping."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy import select

from server.app.core.rbac import Principal, scope_query
from server.app.db import models as m
from server.app.schemas.states import Role


def _make_work_orders(app: Any) -> dict[str, int]:
    """Two potholes, one work order each, for contractor1 (CRMP-P2) and contractor2 (CRMP-P7)."""
    with app.state.database.session() as s:
        ghmc = s.scalars(select(m.Agency).where(m.Agency.code == "GHMC")).one()
        ids: dict[str, int] = {}
        for user, code, lat in (("contractor1", "CRMP-P2", 17.4300), ("contractor2", "CRMP-P7", 17.4310)):
            c = s.scalars(select(m.Contract).where(m.Contract.code == code)).one()
            ph = m.Pothole(lat=lat, lon=78.45, status="work_ordered", agency_id=ghmc.id, severity="medium",
                           area_m2=0.2)
            s.add(ph)
            s.flush()
            wo = m.WorkOrder(status="issued", agency_id=ghmc.id, contract_id=c.id, contractor_id=c.contractor_id,
                             est_bags=3, est_cost_inr=1400.0)
            s.add(wo)
            s.flush()
            s.add(m.WorkOrderPothole(work_order_id=wo.id, pothole_id=ph.id))
            ids[f"{user}_wo"], ids[f"{user}_pothole"] = wo.id, ph.id
            ids[f"{user}_contractor"] = c.contractor_id
        s.commit()
    return ids


def _principal(app: Any, username: str) -> Principal:
    with app.state.database.session() as s:
        u = s.scalars(select(m.User).where(m.User.username == username)).one()
        return Principal(kind="user", role=Role(u.role), user_id=u.id, username=u.username, agency_id=u.agency_id,
                         admin_unit_id=u.admin_unit_id, contractor_id=u.contractor_id)


def test_contractor_sees_only_own_work_orders(app: Any, seeded: Any) -> None:
    ids = _make_work_orders(app)
    c1, c2 = _principal(app, "contractor1"), _principal(app, "contractor2")
    assert c1.contractor_id == ids["contractor1_contractor"] != c2.contractor_id
    with app.state.database.session() as s:
        seen1 = set(s.scalars(scope_query(c1, select(m.WorkOrder))).all())
        assert {w.id for w in seen1} == {ids["contractor1_wo"]}
        seen2 = {w.id for w in s.scalars(scope_query(c2, select(m.WorkOrder)))}
        assert seen2 == {ids["contractor2_wo"]}
        # rows hanging off work orders follow the same rule
        assert {r.work_order_id for r in s.scalars(scope_query(c1, select(m.WorkOrderPothole)))} == \
            {ids["contractor1_wo"]}
        # potholes: only those on own work orders
        assert {p.id for p in s.scalars(scope_query(c1, select(m.Pothole)))} == {ids["contractor1_pothole"]}
        # staff of the agency see both; super admin sees everything
        z = _principal(app, "zonal_khairatabad")
        assert {w.id for w in s.scalars(scope_query(z, select(m.WorkOrder)))} == \
            {ids["contractor1_wo"], ids["contractor2_wo"]}
        # an admin of another agency sees none of GHMC's work orders
        rnb = _principal(app, "rnb_admin")
        assert list(s.scalars(scope_query(rnb, select(m.WorkOrder)))) == []


def test_contractor_cannot_read_other_contractors_pothole_or_wo_via_api(client: TestClient, app: Any,
                                                                         login: Any) -> None:
    ids = _make_work_orders(app)
    c1 = login("contractor1")
    other = ids["contractor2_pothole"]
    assert client.get(f"/v1/potholes/{other}", headers=c1).status_code == 404  # hidden, not merely forbidden
    own = client.get(f"/v1/potholes/{ids['contractor1_pothole']}", headers=c1)
    assert own.status_code == 200
    assert [w["id"] for w in own.json()["work_orders"]] == [ids["contractor1_wo"]]
    listed = client.get("/v1/potholes", headers=c1).json()
    assert [i["id"] for i in listed["items"]] == [ids["contractor1_pothole"]]
    c2 = login("contractor2")
    assert client.get(f"/v1/potholes/{ids['contractor1_pothole']}", headers=c2).status_code == 404
    # if WP8's work-order router is mounted, its list must respect the same scope
    r = client.get("/v1/work-orders", headers=c1)
    if r.status_code == 200:
        body = r.json()
        items = body if isinstance(body, list) else body.get("items", [])
        assert {w["id"] for w in items} <= {ids["contractor1_wo"]}


def test_role_guards(client: TestClient, login: Any, seeded: Any) -> None:
    cfg = client.get("/v1/admin/priority-config", headers=login("zonal_khairatabad")).json()["config"]
    assert client.put("/v1/admin/priority-config", json=cfg, headers=login("contractor1")).status_code == 403
    assert client.put("/v1/admin/priority-config", json=cfg, headers=login("engineer1")).status_code == 403
    assert client.get("/v1/admin/audit", headers=login("citizen1")).status_code == 403
    assert client.get("/v1/admin/audit").status_code == 401
    # zonal admin of R&B cannot edit GHMC's config; super admin can
    assert client.put("/v1/admin/priority-config?agency=GHMC", json=cfg,
                      headers=login("rnb_admin")).status_code == 403
    assert client.put("/v1/admin/priority-config?agency=GHMC", json=cfg, headers=login("admin")).status_code == 200
    # only super_admin edits the global default
    assert client.get("/v1/auth/me", headers=login("contractor1")).json()["role"] == "contractor"


def test_reject_requires_engineer_and_scope(client: TestClient, app: Any, login: Any, seeded: Any) -> None:
    with app.state.database.session() as s:
        ghmc = s.scalars(select(m.Agency).where(m.Agency.code == "GHMC")).one()
        rnb = s.scalars(select(m.Agency).where(m.Agency.code == "RNB")).one()
        p1 = m.Pothole(lat=17.43, lon=78.45, status="confirmed", agency_id=ghmc.id)
        p2 = m.Pothole(lat=17.43, lon=78.46, status="confirmed", agency_id=rnb.id)
        s.add_all([p1, p2])
        s.commit()
        pid1, pid2 = p1.id, p2.id
    body = {"reason": "manhole cover, not a pothole", "cause": "utility_cut"}
    assert client.post(f"/v1/potholes/{pid1}/reject", json=body, headers=login("contractor1")).status_code == 403
    assert client.post(f"/v1/potholes/{pid2}/reject", json=body, headers=login("engineer1")).status_code == 404
    r = client.post(f"/v1/potholes/{pid1}/reject", json=body, headers=login("engineer1"))
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    assert client.post(f"/v1/potholes/{pid1}/reject", json=body, headers=login("engineer1")).status_code == 409
    audit = client.get(f"/v1/admin/audit?entity=pothole&id={pid1}", headers=login("zonal_khairatabad")).json()
    assert audit and audit[0]["action"] == "status:confirmed->rejected" and audit[0]["actor"] == "engineer1"


def test_device_token_scope(client: TestClient, device_headers: dict[str, str], seeded: Any) -> None:
    assert client.get("/v1/admin/audit", headers=device_headers).status_code == 403
    assert client.get("/v1/devices/tsrtc-bus-ts09ua1234/cert", headers=device_headers).json()["calib_id"] == \
        "bus_default"
    assert client.get("/v1/devices/sim-bus-01/cert", headers=device_headers).status_code == 403
    r = client.post("/v1/edge/presign", json={"keys": ["frames/someone-else/t/1.jpg"]}, headers=device_headers)
    assert r.status_code == 403
    # re-registration revokes the previous token
    client.post("/v1/edge/register", json={"device_id": "tsrtc-bus-ts09ua1234", "kind": "phone"})
    assert client.get("/v1/edge/config", headers=device_headers).status_code == 401
