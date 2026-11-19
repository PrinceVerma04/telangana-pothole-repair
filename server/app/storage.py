"""Object storage behind one interface (DECISIONS D3): ``FsStorage`` (demo, ``app_data/store/``) | ``S3Storage``.

    st = storage_from_settings(settings)
    st.put("frames/dev/trip/42.jpg", jpeg_bytes, "image/jpeg")
    st.get("frames/dev/trip/42.jpg") -> bytes
    st.presign("frames/dev/trip/42.jpg")            # PUT URL for the edge device (15 min)
    st.url("frames/dev/trip/42.jpg")                # signed GET URL for the web UI (24 h)
    st.local_path("frames/...")                      # FsStorage only: Path for cv2.imread in workers

FsStorage presigned URLs point at this API: ``PUT /v1/edge/upload/{key}?exp=&sig=`` and
``GET /v1/files/{key}?exp=&sig=``; the signature is HMAC-SHA256(JWT_SECRET, "METHOD\\nkey\\nexp").
Keys are restricted to ``[A-Za-z0-9._-/]``, relative, without ``..`` segments (path traversal guard).
"""

from __future__ import annotations

import mimetypes
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable
from urllib.parse import quote

from server.app.core.config import Settings
from server.app.core.security import sign_url, verify_url_signature

_KEY_RE = re.compile(r"^[A-Za-z0-9._\-/]{1,512}$")


class StorageError(Exception):
    pass


class StorageNotFound(StorageError, KeyError):
    pass


def validate_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY_RE.match(key):
        raise StorageError(f"invalid storage key: {key!r}")
    if key.startswith("/") or key.endswith("/"):
        raise StorageError(f"invalid storage key: {key!r}")
    parts = key.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise StorageError(f"invalid storage key: {key!r}")
    return key


def guess_content_type(key: str) -> str:
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


@runtime_checkable
class Storage(Protocol):
    backend: str

    def put(self, key: str, data: bytes, content_type: Optional[str] = None) -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def delete(self, key: str) -> None: ...
    def presign(self, key: str, method: str = "PUT", expires_s: Optional[int] = None,
                content_type: Optional[str] = None) -> str: ...
    def url(self, key: str, expires_s: Optional[int] = None) -> str: ...
    def ping(self) -> bool: ...


class FsStorage:
    backend = "fs"

    def __init__(self, root: str | Path, *, api_base: str = "http://localhost:8000", secret: str = "dev",
                 presign_ttl_s: int = 900, url_ttl_s: int = 86400) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.api_base = api_base.rstrip("/")
        self.secret = secret
        self.presign_ttl_s = presign_ttl_s
        self.url_ttl_s = url_ttl_s

    def local_path(self, key: str) -> Path:
        validate_key(key)
        p = (self.root / key).resolve()
        if self.root not in p.parents:
            raise StorageError(f"key escapes storage root: {key!r}")
        return p

    def put(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:  # noqa: ARG002
        p = self.local_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".up-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return key

    def get(self, key: str) -> bytes:
        p = self.local_path(key)
        if not p.is_file():
            raise StorageNotFound(key)
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        try:
            return self.local_path(key).is_file()
        except StorageError:
            return False

    def delete(self, key: str) -> None:
        self.local_path(key).unlink(missing_ok=True)

    def _signed(self, path: str, method: str, key: str, expires_s: int) -> str:
        exp = int(time.time()) + int(expires_s)
        sig = sign_url(self.secret, method, key, exp)
        return f"{self.api_base}{path}/{quote(key)}?exp={exp}&sig={sig}"

    def presign(self, key: str, method: str = "PUT", expires_s: Optional[int] = None,
                content_type: Optional[str] = None) -> str:  # noqa: ARG002
        validate_key(key)
        method = method.upper()
        if method == "PUT":
            return self._signed("/v1/edge/upload", "PUT", key, expires_s or self.presign_ttl_s)
        if method == "GET":
            return self._signed("/v1/files", "GET", key, expires_s or self.url_ttl_s)
        raise StorageError(f"unsupported presign method {method}")

    def url(self, key: str, expires_s: Optional[int] = None) -> str:
        return self.presign(key, "GET", expires_s)

    def verify(self, method: str, key: str, exp: Any, sig: str) -> bool:
        return verify_url_signature(self.secret, method, key, exp, sig)

    def ping(self) -> bool:
        try:
            probe = self.root / ".ping"
            probe.write_bytes(b"ok")
            probe.unlink()
            return True
        except OSError:
            return False


class S3Storage:
    """S3 / MinIO adapter (production). ``boto3`` is imported lazily (optional extra ``s3``)."""

    backend = "s3"

    def __init__(self, bucket: str, *, endpoint: Optional[str] = None, access_key: Optional[str] = None,
                 secret_key: Optional[str] = None, region: str = "ap-south-1", presign_ttl_s: int = 900,
                 url_ttl_s: int = 86400, client: Any = None) -> None:
        self.bucket = bucket
        self.presign_ttl_s = presign_ttl_s
        self.url_ttl_s = url_ttl_s
        if client is None:
            try:
                import boto3  # type: ignore[import-not-found]
            except ImportError as e:  # pragma: no cover - depends on optional extra
                raise StorageError("S3 backend needs boto3: pip install -e '.[s3]'") from e
            client = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access_key,
                                  aws_secret_access_key=secret_key, region_name=region)
        self.client = client

    def put(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        validate_key(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data,
                               ContentType=content_type or guess_content_type(key))
        return key

    def get(self, key: str) -> bytes:
        validate_key(key)
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as e:
            if type(e).__name__ in ("NoSuchKey", "ClientError"):
                raise StorageNotFound(key) from e
            raise

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=validate_key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=validate_key(key))

    def presign(self, key: str, method: str = "PUT", expires_s: Optional[int] = None,
                content_type: Optional[str] = None) -> str:
        validate_key(key)
        method = method.upper()
        if method == "PUT":
            params = {"Bucket": self.bucket, "Key": key, "ContentType": content_type or guess_content_type(key)}
            return self.client.generate_presigned_url("put_object", Params=params,
                                                      ExpiresIn=expires_s or self.presign_ttl_s)
        if method == "GET":
            return self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key},
                                                      ExpiresIn=expires_s or self.url_ttl_s)
        raise StorageError(f"unsupported presign method {method}")

    def url(self, key: str, expires_s: Optional[int] = None) -> str:
        return self.presign(key, "GET", expires_s)

    def ping(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


def storage_from_settings(settings: Settings) -> Storage:
    if settings.STORAGE_BACKEND.lower() == "s3":
        return S3Storage(settings.S3_BUCKET, endpoint=settings.S3_ENDPOINT, access_key=settings.S3_ACCESS_KEY,
                         secret_key=settings.S3_SECRET_KEY, presign_ttl_s=settings.PRESIGN_TTL_S,
                         url_ttl_s=settings.URL_TTL_S)
    return FsStorage(settings.STORAGE_ROOT, api_base=settings.API_BASE_URL, secret=settings.JWT_SECRET,
                     presign_ttl_s=settings.PRESIGN_TTL_S, url_ttl_s=settings.URL_TTL_S)
