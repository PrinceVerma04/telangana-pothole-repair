"""Passwords (argon2id), JWT HS256 access tokens, device tokens, demo OTP, signed URLs, config signing.

Token claims (both kinds are HS256 JWTs signed with ``JWT_SECRET``):
  user   : {"typ": "user", "sub": "<user.id>", "role", "agency_id", "admin_unit_id", "contractor_id", "exp", "iat"}
  device : {"typ": "device", "sub": "device:<device_id>", "role": "driver_device", "did": <device.id>, "exp", "iat"}
A device token is also checked against ``device.token_hash`` (sha256) so re-registration revokes the old one.

OTP: with ``DEMO_MODE=1`` the fixed ``OTP_DEMO_CODE`` (default 123456) is accepted for any phone and is returned
by ``POST /v1/auth/otp`` (demo only). Otherwise a random 6-digit code is kept (hashed) in-process for
``OTP_TTL_S`` with at most 5 attempts; the SMS adapter is a ``notification`` row (no real SMS in the hackathon).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from server.app.core import ed25519
from server.app.core.config import Settings

_ph = PasswordHasher()  # argon2id, library defaults (t=3, m=64 MiB, p=4)
JWT_ALG = "HS256"


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return _ph.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def sha256_hex(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode()
    return hashlib.sha256(s).hexdigest()


# ------------------------------------------------------------------ JWT
class TokenError(Exception):
    pass


def create_access_token(settings: Settings, *, user_id: int, role: str, agency_id: Optional[int] = None,
                        admin_unit_id: Optional[int] = None, contractor_id: Optional[int] = None,
                        username: Optional[str] = None, ttl_min: Optional[int] = None) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=ttl_min or settings.JWT_TTL_MIN)
    claims = {"typ": "user", "sub": str(user_id), "role": role, "agency_id": agency_id,
              "admin_unit_id": admin_unit_id, "contractor_id": contractor_id, "usr": username,
              "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    return jwt.encode(claims, settings.JWT_SECRET, algorithm=JWT_ALG), exp


def create_device_token(settings: Settings, *, device_pk: int, device_id: str) -> tuple[str, datetime]:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=settings.DEVICE_TOKEN_TTL_DAYS)
    claims = {"typ": "device", "sub": f"device:{device_id}", "role": "driver_device", "did": device_pk,
              "jti": secrets.token_hex(8), "iat": int(now.timestamp()), "exp": int(exp.timestamp())}
    return jwt.encode(claims, settings.JWT_SECRET, algorithm=JWT_ALG), exp


def decode_token(settings: Settings, token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[JWT_ALG], options={"require": ["exp", "sub"]})
    except jwt.ExpiredSignatureError as e:
        raise TokenError("token expired") from e
    except jwt.PyJWTError as e:
        raise TokenError(f"invalid token: {e}") from e


# ------------------------------------------------------------------ OTP
@dataclass
class _Otp:
    code_hash: str
    expires: float
    attempts: int = 0


class OtpStore:
    """In-process OTP store (single API process in the demo; production: Redis/DB with the same API)."""

    MAX_ATTEMPTS = 5

    def __init__(self) -> None:
        self._d: dict[str, _Otp] = {}
        self._lock = threading.Lock()

    @staticmethod
    def normalise_phone(phone: str) -> str:
        digits = "".join(ch for ch in phone if ch.isdigit())
        return "+91" + digits[-10:] if len(digits) >= 10 else phone

    def issue(self, settings: Settings, phone: str) -> str:
        phone = self.normalise_phone(phone)
        code = settings.OTP_DEMO_CODE if settings.DEMO_MODE else f"{secrets.randbelow(10**6):06d}"
        with self._lock:
            self._d[phone] = _Otp(sha256_hex(phone + ":" + code), time.time() + settings.OTP_TTL_S)
        return code

    def verify(self, settings: Settings, phone: str, code: str) -> bool:
        phone = self.normalise_phone(phone)
        if settings.DEMO_MODE and hmac.compare_digest(code, settings.OTP_DEMO_CODE):
            return True
        with self._lock:
            rec = self._d.get(phone)
            if rec is None or rec.expires < time.time() or rec.attempts >= self.MAX_ATTEMPTS:
                self._d.pop(phone, None)
                return False
            rec.attempts += 1
            ok = hmac.compare_digest(rec.code_hash, sha256_hex(phone + ":" + code))
            if ok:
                self._d.pop(phone, None)
            return ok


# ------------------------------------------------------------------ signed URLs (FsStorage presign)
def sign_url(secret: str, method: str, key: str, exp: int) -> str:
    msg = f"{method.upper()}\n{key}\n{exp}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_url_signature(secret: str, method: str, key: str, exp: int | str, sig: str) -> bool:
    try:
        exp_i = int(exp)
    except (TypeError, ValueError):
        return False
    if exp_i < int(time.time()):
        return False
    return hmac.compare_digest(sign_url(secret, method, key, exp_i), sig or "")


# ------------------------------------------------------------------ OTA config signing (ed25519)
def config_signing_secret(settings: Settings) -> bytes:
    """Demo: deterministic 32-byte ed25519 seed derived from JWT_SECRET. Production: load from a KMS/HSM."""
    return hashlib.sha256(b"gunthameter-edge-config-v1:" + settings.JWT_SECRET.encode()).digest()


def config_public_key_hex(settings: Settings) -> str:
    return ed25519.public_key(config_signing_secret(settings)).hex()


def sign_config(settings: Settings, payload: bytes) -> str:
    return ed25519.sign(config_signing_secret(settings), payload).hex()
