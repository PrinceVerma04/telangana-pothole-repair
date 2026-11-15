"""Shared builders for the WP8b ops tests (no tests here). Synthetic in-memory rows only - no geo files, no torch."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from server.app.core.util import now_utc
from server.app.db import models as m
from server.app.services import state_machine as sm

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 64 + b"\xff\xd9"
_WAY = itertools.count(900_000_001)
# Khairatabad circle (GHMC-C17) fallback box: 78.435-78.475 E, 17.40-17.445 N
C17_LAT, C17_LON = 17.4225, 78.455


def unit_id(s: Session, code: str) -> int:
    return s.scalars(select(m.AdminUnit.id).where(m.AdminUnit.code == code)).one()


def contract(s: Session, code: str) -> m.Contract:
    return s.scalars(select(m.Contract).where(m.Contract.code == code)).one()


def make_pothole(app: Any, *, contract_code: Optional[str] = "CRMP-P2", highway: str = "primary",
                 status: str = "confirmed", lat: float = C17_LAT, lon: float = C17_LON, admin_unit: str = "GHMC-C17",
                 contractor_id: Optional[int] = None, length_m: float = 0.50, width_m: float = 0.40,
                 depth_p95_mm: float = 60.0, confirmed_at: Optional[datetime] = None) -> int:
    """A measured pothole (PLAN 7.10 worked-example geometry) on its own road segment."""
    with app.state.database.session() as s:
        ghmc = s.scalars(select(m.Agency).where(m.Agency.code == "GHMC")).one()
        seg = m.RoadSegment(osm_way_id=next(_WAY), seg_idx=0, name=f"Test {highway} road", highway=highway, lanes=2,
                            surface="BT", length_m=200.0, owner_agency_id=ghmc.id,
                            geom_wkt=f"LINESTRING({lon - 0.001} {lat}, {lon + 0.001} {lat})")
        s.add(seg)
        s.flush()
        c = contract(s, contract_code) if contract_code else None
        ph = m.Pothole(lat=lat, lon=lon, status=status, agency_id=ghmc.id, admin_unit_id=unit_id(s, admin_unit),
                       road_segment_id=seg.id, way_id=seg.osm_way_id, direction=0, severity="large",
                       length_m=length_m, width_m=width_m, length_sigma_m=0.03, width_sigma_m=0.03,
                       area_m2=round(3.14159 / 4 * length_m * width_m, 4), area_sigma=0.02,
                       depth_p95_mm=depth_p95_mm, depth_max_mm=depth_p95_mm + 8, depth_mean_mm=depth_p95_mm * 0.6,
                       depth_sigma_mm=8.0, volume_l=6.0, volume_sigma=1.5, quality=0.8, method="mono",
                       best_method="mono", contract_id=c.id if c else None,
                       contractor_id=contractor_id if contractor_id is not None else (c.contractor_id if c else None),
                       confirmed_at=confirmed_at or now_utc() - timedelta(hours=2), first_seen=now_utc(),
                       last_seen=now_utc(), passes_seen=2, trips_seen=2)
        s.add(ph)
        s.commit()
        return ph.id


def photo(client: TestClient, wo_id: int, kind: str, headers: dict[str, str]) -> Any:
    return client.post(f"/v1/work-orders/{wo_id}/photos", headers=headers,
                       data={"kind": kind, "lat": str(C17_LAT), "lon": str(C17_LON)},
                       files={"file": (f"{kind}.jpg", JPEG, "image/jpeg")})


def move(client: TestClient, wo_id: int, to: str, headers: dict[str, str], **extra: Any) -> Any:
    return client.patch(f"/v1/work-orders/{wo_id}/transition", headers=headers, json={"to": to, **extra})


def drive_to_claimed(s: Session, wo: m.WorkOrder, at: datetime, bags: int = 3) -> None:
    """Service-level fast path (system actor): issued -> acknowledged -> in_progress -> repaired_claimed."""
    s.add(m.RepairEvent(work_order_id=wo.id, kind="before_photo", photo_key="repairs/test/before.jpg", at=at))
    sm.transition_work_order(s, wo, "acknowledged", actor="test", now=at)
    sm.transition_work_order(s, wo, "in_progress", actor="test", now=at + timedelta(minutes=30))
    s.add(m.RepairEvent(work_order_id=wo.id, kind="after_photo", photo_key="repairs/test/after.jpg",
                        at=at + timedelta(hours=1)))
    sm.transition_work_order(s, wo, "repaired_claimed", actor="test", bags_used=bags, now=at + timedelta(hours=1))


def passes_provider(passes: list[tuple[Any, datetime]]) -> Any:
    def _clean(_pothole: m.Pothole, _since: datetime) -> list[tuple[Any, datetime]]:
        return list(passes)
    return _clean


def audit_actions(s: Session, entity: str, entity_id: int) -> list[str]:
    return list(s.scalars(select(m.AuditLog.action).where(m.AuditLog.entity == entity, m.AuditLog.entity_id == entity_id)
                          .order_by(m.AuditLog.id)))
