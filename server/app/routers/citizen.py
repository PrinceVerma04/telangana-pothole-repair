"""/v1/citizen: multipart pothole report (photo optional) + ticket tracking (no login needed).

Privacy (DPDP Act 2023): the reporter's phone is stored only as a SHA-256 hash; photos are stored under
``citizen/<yyyy>/<mm>/<ticket>.<ext>`` and served only through signed URLs. A report is linked to the nearest live
pothole within 25 m by the ``complaint`` pipeline hook (PLAN 5.5: 1 pass + citizen report -> confirmed).
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select

from server.app import pipeline
from server.app.core.rbac import OptionalPrincipalDep
from server.app.core.security import OtpStore, sha256_hex
from server.app.core.util import now_utc, to_ist
from server.app.db import models as m
from server.app.deps import DbDep, SettingsDep, StorageDep
from server.app.schemas.api import CitizenReportOut

log = logging.getLogger("gunthameter.citizen")
router = APIRouter(prefix="/citizen", tags=["citizen"])

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I: read out over the phone
_IMAGE_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
_MAGIC = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG\r\n\x1a\n": "image/png"}


def new_ticket() -> str:
    """``GM-YYMMDD-XXXXXX`` (IST date + 6 unambiguous characters, ~1e9 per day)."""
    day = to_ist(now_utc()).strftime("%y%m%d")  # type: ignore[union-attr]
    return f"GM-{day}-" + "".join(secrets.choice(_ALPHABET) for _ in range(6))


def _sniff(data: bytes) -> Optional[str]:
    for magic, ct in _MAGIC.items():
        if data.startswith(magic):
            return ct
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _out(c: m.Complaint, pothole_status: Optional[str]) -> CitizenReportOut:
    return CitizenReportOut(ticket=c.ticket, status=pothole_status or c.status, pothole_id=c.pothole_id,
                            created_at=c.created_at, lat=c.lat, lon=c.lon)


@router.post("/reports", response_model=CitizenReportOut, status_code=status.HTTP_201_CREATED,
             summary="Report a pothole (multipart: lat, lon, optional text/lang/phone/photo). Returns a ticket.")
async def create_report(
    db: DbDep, settings: SettingsDep, storage: StorageDep, p: OptionalPrincipalDep,
    lat: Annotated[float, Form(ge=6.0, le=38.0)],
    lon: Annotated[float, Form(ge=68.0, le=98.0)],
    text: Annotated[Optional[str], Form(max_length=2000)] = None,
    lang: Annotated[Optional[str], Form(pattern="^(te|hi|en|ur)$")] = None,
    phone: Annotated[Optional[str], Form(pattern=r"^\+?91?[6-9]\d{9}$")] = None,
    photo: Annotated[Optional[UploadFile], File(description="JPEG/PNG/WebP, <= MAX_UPLOAD_MB")] = None,
) -> CitizenReportOut:
    ticket = new_ticket()
    photo_key = None
    if photo is not None and photo.filename:
        data = await photo.read(settings.MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"photo larger than {settings.MAX_UPLOAD_MB} MB")
        ct = _sniff(data)
        if ct is None:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "photo must be JPEG, PNG or WebP")
        ist = to_ist(now_utc())
        photo_key = f"citizen/{ist:%Y}/{ist:%m}/{ticket}.{_IMAGE_TYPES[ct]}"  # type: ignore[str-format]
        storage.put(photo_key, data, ct)
    c = m.Complaint(ticket=ticket, source="web", lat=lat, lon=lon, text=text, lang=lang, photo_key=photo_key,
                    status="received", created_at=now_utc(), user_id=p.user_id if p is not None else None,
                    phone_hash=sha256_hex(OtpStore.normalise_phone(phone)) if phone else None)
    db.add(c)
    db.commit()
    try:
        pipeline.process_complaint(c.id, db)
    except Exception:
        db.rollback()
        log.exception("complaint hooks failed for %s", ticket)
    db.refresh(c)
    ph = db.get(m.Pothole, c.pothole_id) if c.pothole_id else None
    return _out(c, ph.status if ph else None)


@router.get("/reports/{ticket}", response_model=CitizenReportOut,
            summary="Track a report by ticket. status = linked pothole's status, else the report status.")
def get_report(ticket: str, db: DbDep) -> CitizenReportOut:
    c = db.scalars(select(m.Complaint).where(m.Complaint.ticket == ticket.strip().upper())).first()
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"ticket {ticket} not found")
    ph = db.get(m.Pothole, c.pothole_id) if c.pothole_id else None
    return _out(c, ph.status if ph else None)
