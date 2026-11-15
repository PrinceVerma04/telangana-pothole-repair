"""/v1/edge: device registration, signed OTA config (ETag), presigned uploads, FsStorage PUT upload route."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from measure.types import CalibCert
from server.app.core import audit
from server.app.core.config import CONFIGS_DIR, Settings
from server.app.core.rbac import OptionalPrincipalDep, RoleDep
from server.app.core.security import config_public_key_hex, create_device_token, sha256_hex, sign_config
from server.app.core.util import now_utc
from server.app.db import models as m
from server.app.deps import DbDep, SettingsDep, StorageDep
from server.app.schemas.api import DeviceRegister, DeviceRegistered, EdgeConfig, PresignRequest, PresignResponse
from server.app.schemas.states import Role
from server.app.storage import FsStorage, StorageError, validate_key

router = APIRouter(prefix="/edge", tags=["edge"])

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
DEVICE_KEY_PREFIXES = ("frames", "imu", "masks", "depth", "video")

ConfigReader = RoleDep(Role.driver_device, Role.zonal_admin, Role.field_engineer)
Uploader = RoleDep(Role.driver_device, Role.field_engineer, Role.zonal_admin)


def load_calib_cert(db: Session, calib_id: str) -> Optional[m.CalibCertRow]:
    """DB row for ``calib_id``; imported from ``configs/cameras/<calib_id>.json`` on first use."""
    row = db.scalars(select(m.CalibCertRow).where(m.CalibCertRow.calib_id == calib_id)).first()
    if row is not None:
        return row
    if not _SAFE_ID.match(calib_id):
        return None
    path = CONFIGS_DIR / "cameras" / f"{calib_id}.json"
    if not path.is_file():
        for p in (CONFIGS_DIR / "cameras").glob("*.json"):  # file name may differ from calib_id
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("calib_id") == calib_id:
                    path = p
                    break
            except (OSError, ValueError):
                continue
        else:
            return None
    cert = CalibCert.load(path)
    row = m.CalibCertRow(calib_id=cert.calib_id, json=cert.to_dict(), certified=cert.certified, issued_at=now_utc())
    db.add(row)
    db.flush()
    return row


@router.post("/register", response_model=DeviceRegistered,
             summary="Register (or re-register) a vehicle device; returns a device bearer token")
def register(body: DeviceRegister, db: DbDep, settings: SettingsDep, p: OptionalPrincipalDep) -> DeviceRegistered:
    if not settings.open_device_registration and (p is None or p.role not in (Role.super_admin, Role.zonal_admin)):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "device registration needs zonal_admin/super_admin")
    if not _SAFE_ID.match(body.device_id):
        raise HTTPException(422, "device_id must match [A-Za-z0-9_.-]{1,64}")
    agency = db.scalars(select(m.Agency).where(m.Agency.code == body.agency_code)).first()
    if agency is None:
        raise HTTPException(422, f"unknown agency_code {body.agency_code!r}")
    cert_row = None
    if body.calib_id:
        cert_row = load_calib_cert(db, body.calib_id)
        if cert_row is None:
            raise HTTPException(422, f"unknown calib_id {body.calib_id!r} (not in DB or configs/cameras/)")
    dev = db.scalars(select(m.Device).where(m.Device.device_id == body.device_id)).first()
    before = None
    if dev is None:
        dev = m.Device(device_id=body.device_id, kind=body.kind, agency_id=agency.id, vehicle_reg=body.vehicle_reg)
        db.add(dev)
        db.flush()
    else:
        before = {"kind": dev.kind, "agency_id": dev.agency_id, "vehicle_reg": dev.vehicle_reg,
                  "calib_cert_id": dev.calib_cert_id}
        dev.kind, dev.agency_id = body.kind, agency.id
        dev.vehicle_reg = body.vehicle_reg or dev.vehicle_reg
    if cert_row is not None:
        dev.calib_cert_id = cert_row.id
        if cert_row.device_id is None:
            cert_row.device_id = dev.id
    tok, _exp = create_device_token(settings, device_pk=dev.id, device_id=dev.device_id)
    dev.token_hash = sha256_hex(tok)
    dev.last_seen_at = now_utc()
    audit.record(db, "device", dev.id, "register" if before is None else "re-register", actor=p or "system:edge",
                 before=before, after={"device_id": dev.device_id, "kind": dev.kind, "agency": agency.code,
                                       "calib_id": body.calib_id})
    db.commit()
    cfg, etag = build_edge_config(db, settings)
    return DeviceRegistered(device_id=dev.device_id, token=tok, config_etag=etag)


def _weights_sha256(settings: Settings, path: Path, cache: dict[str, Any]) -> str:
    if not path.is_file():
        return ""
    st = path.stat()
    k = f"{path}:{st.st_mtime_ns}:{st.st_size}"
    if k not in cache:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache.clear()
        cache[k] = h.hexdigest()
    return cache[k]


_SHA_CACHE: dict[str, Any] = {}


def build_edge_config(db: Session, settings: Settings) -> tuple[EdgeConfig, str]:
    reg = db.scalars(select(m.ModelRegistry).where(m.ModelRegistry.active.is_(True), m.ModelRegistry.task == "seg",
                                                   m.ModelRegistry.format.in_(("litert16", "litert8", "tflite",
                                                                               "onnx", "engine", "pt")))
                     .order_by(m.ModelRegistry.created_at.desc())).first()
    if reg is not None:
        version, url, sha = reg.id, reg.path or "", reg.sha256 or ""
    else:
        wp = settings.seg_weights_path
        version, url, sha = 1, f"{settings.API_BASE_URL}/v1/edge/model/{wp.name}", _weights_sha256(settings, wp,
                                                                                                     _SHA_CACHE)
    cfg = EdgeConfig(version=version, seg_model_url=url, seg_model_sha256=sha)
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    cfg.signature_ed25519 = sign_config(settings, payload)
    etag = hashlib.sha256(payload).hexdigest()[:20]
    return cfg, etag


@router.get("/config", response_model=EdgeConfig,
            summary="Signed OTA config. ETag/If-None-Match; signature = ed25519 over canonical JSON without the "
                    "signature field (public key: GET /v1/edge/config/pubkey)")
def get_config(request: Request, response: Response, db: DbDep, settings: SettingsDep, p: ConfigReader) -> Any:
    cfg, etag = build_edge_config(db, settings)
    quoted = f'"{etag}"'
    if request.headers.get("if-none-match") in (etag, quoted):
        return Response(status_code=304, headers={"ETag": quoted})
    if p.kind == "device":
        dev = db.get(m.Device, p.device_pk)
        if dev is not None:
            dev.last_seen_at = now_utc()
            db.commit()
    response.headers["ETag"] = quoted
    response.headers["X-Config-Pubkey"] = config_public_key_hex(settings)
    return cfg


@router.get("/config/pubkey", summary="ed25519 public key (hex) for verifying the edge config signature")
def get_pubkey(settings: SettingsDep) -> dict[str, str]:
    return {"alg": "ed25519", "public_key_hex": config_public_key_hex(settings),
            "signed_message": "json.dumps(config_without_signature, sort_keys=True, separators=(',',':'))"}


@router.post("/presign", response_model=PresignResponse, summary="Presigned PUT URLs for frames / IMU windows")
def presign(body: PresignRequest, storage: StorageDep, p: Uploader) -> PresignResponse:
    urls: dict[str, str] = {}
    for key in body.keys:
        try:
            validate_key(key)
        except StorageError as e:
            raise HTTPException(422, str(e)) from e
        if p.kind == "device":
            parts = key.split("/")
            if len(parts) < 3 or parts[0] not in DEVICE_KEY_PREFIXES or parts[1] != p.device_id:
                raise HTTPException(403, f"device may only upload under <{'|'.join(DEVICE_KEY_PREFIXES)}>/"
                                         f"{p.device_id}/...: {key}")
        urls[key] = storage.presign(key, "PUT", content_type=body.content_type)
    return PresignResponse(urls=urls)


@router.put("/upload/{key:path}", summary="FsStorage target of presigned PUT URLs (demo; S3 uploads go to S3)")
async def upload(key: str, request: Request, storage: StorageDep, settings: SettingsDep,
                 exp: int = Query(...), sig: str = Query(...)) -> dict[str, Any]:
    if not isinstance(storage, FsStorage):
        raise HTTPException(400, "storage backend is not fs; PUT to the S3 presigned URL instead")
    try:
        validate_key(key)
    except StorageError as e:
        raise HTTPException(422, str(e)) from e
    if not storage.verify("PUT", key, exp, sig):
        raise HTTPException(403, "bad or expired signature")
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    clen = request.headers.get("content-length")
    if clen and clen.isdigit() and int(clen) > limit:
        raise HTTPException(413, f"upload larger than {settings.MAX_UPLOAD_MB} MB")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, f"upload larger than {settings.MAX_UPLOAD_MB} MB")
    storage.put(key, bytes(data), request.headers.get("content-type"))
    return {"key": key, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
