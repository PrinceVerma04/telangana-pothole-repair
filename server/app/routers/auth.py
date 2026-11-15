"""/v1/auth: OTP (citizens / field staff by phone) and password login.

Demo users (seeded by scripts/seed_demo.py, password ``demo1234``): admin, zonal_khairatabad, engineer1,
contractor1, contractor2, citizen1 - see ``server/app/seed.py:DEMO_USERS``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import or_, select

from server.app.core.config import Settings
from server.app.core.rbac import PrincipalDep
from server.app.core.security import OtpStore, create_access_token, verify_password
from server.app.db import models as m
from server.app.deps import DbDep, SettingsDep
from server.app.schemas.api import OtpRequest, TokenRequest, TokenResponse
from server.app.schemas.states import Role

router = APIRouter(prefix="/auth", tags=["auth"])


def _otp_store(request: Request) -> OtpStore:
    st = getattr(request.app.state, "otp_store", None)
    if st is None:
        st = request.app.state.otp_store = OtpStore()
    return st


def _issue(settings: Settings, u: m.User) -> TokenResponse:
    tok, exp = create_access_token(settings, user_id=u.id, role=u.role, agency_id=u.agency_id,
                                   admin_unit_id=u.admin_unit_id, contractor_id=u.contractor_id, username=u.username)
    return TokenResponse(access_token=tok, role=Role(u.role), agency_id=u.agency_id, contractor_id=u.contractor_id,
                         expires_at=exp)


@router.post("/otp", summary="Send an OTP to an Indian mobile number (demo: fixed code, returned in the response)")
def send_otp(body: OtpRequest, request: Request, db: DbDep, settings: SettingsDep) -> dict[str, Any]:
    code = _otp_store(request).issue(settings, body.phone)
    phone = OtpStore.normalise_phone(body.phone)
    # SMS adapter stub: production sends via the notification pipeline; never store the code itself
    db.add(m.Notification(channel="sms", to_role=None, subject="GunthaMeter OTP",
                          body=f"OTP sent to ******{phone[-4:]}", entity="otp", sent=False))
    db.commit()
    out: dict[str, Any] = {"sent": True, "ttl_s": settings.OTP_TTL_S}
    if settings.DEMO_MODE:
        out["demo_code"] = code
        out["note"] = "DEMO_MODE: fixed OTP, never in production"
    return out


@router.post("/token", response_model=TokenResponse, summary="Exchange phone+OTP or username+password for a JWT")
def token(body: TokenRequest, request: Request, db: DbDep, settings: SettingsDep) -> TokenResponse:
    if body.username and body.password is not None:
        u = db.scalars(select(m.User).where(m.User.username == body.username)).first()
        if u is None or not u.active or not verify_password(body.password, u.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
        return _issue(settings, u)
    if body.phone and body.otp:
        if not _otp_store(request).verify(settings, body.phone, body.otp):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired OTP")
        phone = OtpStore.normalise_phone(body.phone)
        u = db.scalars(select(m.User).where(or_(m.User.phone == phone, m.User.phone == phone[-10:]))
                       .order_by(m.User.id)).first()
        if u is None:  # first OTP login -> citizen account
            u = m.User(username=f"citizen-{phone[-10:]}", phone=phone, role=Role.citizen.value, active=True)
            db.add(u)
            db.commit()
        if not u.active:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "account disabled")
        return _issue(settings, u)
    raise HTTPException(422, "give username+password or phone+otp")


@router.post("/login", response_model=TokenResponse, include_in_schema=True,
             summary="OAuth2 password form (Swagger 'Authorize' button)")
def login_form(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbDep,
               settings: SettingsDep) -> TokenResponse:
    u = db.scalars(select(m.User).where(m.User.username == form.username)).first()
    if u is None or not u.active or not verify_password(form.password, u.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid username or password")
    return _issue(settings, u)


@router.get("/me", summary="Who am I (role, agency, admin unit, contractor)")
def me(p: PrincipalDep, db: DbDep) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": p.kind, "role": p.role.value, "username": p.username, "user_id": p.user_id,
                           "agency_id": p.agency_id, "admin_unit_id": p.admin_unit_id,
                           "contractor_id": p.contractor_id, "device_id": p.device_id}
    if p.agency_id:
        a = db.get(m.Agency, p.agency_id)
        out["agency_code"] = a.code if a else None
    if p.user_id:
        u = db.get(m.User, p.user_id)
        out["display_name"] = u.display_name if u else None
        out["lang"] = u.lang if u else None
    return out
