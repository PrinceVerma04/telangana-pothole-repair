"""HTTP client for the RoadFix API as an edge device / staff user would use it (``edge/protocol.md``).

Works with a live server (``httpx.Client``) or in-process (``fastapi.testclient.TestClient``, an httpx client).
Retries: idempotent calls (presign, PUT upload, observation batches keyed by ``client_uuid``, trip fixes keyed by
``ts``, trip close, GETs) are retried on transport errors, 429 and 5xx with exponential backoff, re-sending the
SAME payload so the server deduplicates. State transitions are never retried blindly.

``Outbox`` is the store-and-forward queue: observations wait in it (with their client_uuid fixed at creation)
until a flush succeeds, exactly like the Room queue on the Android device.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Iterable, Optional

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}


class ApiError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        super().__init__(f"{method} {url} -> {status}: {body[:400]}")
        self.status = status
        self.body = body


class ApiClient:
    def __init__(self, base_url: str = "http://localhost:8000", http: Optional[httpx.Client] = None, *,
                 timeout_s: float = 900.0, retries: int = 4, backoff_s: float = 0.5) -> None:
        self.base = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0))
        self.retries = retries
        self.backoff_s = backoff_s
        self.stats = {"requests": 0, "retries": 0}

    def close(self) -> None:
        self.http.close()

    # ------------------------------------------------------------------ core
    def url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base}{path if path.startswith('/') else '/' + path}"

    def request(self, method: str, path: str, *, token: Optional[str] = None, retry: bool = False,
                ok: Iterable[int] = (200, 201), **kw: Any) -> httpx.Response:
        url = self.url(path)
        headers = dict(kw.pop("headers", None) or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        attempts = 1 + (self.retries if retry else 0)
        last: Optional[Exception] = None
        for i in range(attempts):
            if i:
                self.stats["retries"] += 1
                time.sleep(self.backoff_s * (2 ** (i - 1)))
            self.stats["requests"] += 1
            try:
                r = self.http.request(method, url, headers=headers, **kw)
            except httpx.TransportError as e:
                last = e
                continue
            if r.status_code in RETRY_STATUS and i + 1 < attempts:
                last = ApiError(method, url, r.status_code, r.text)
                continue
            if r.status_code not in tuple(ok):
                raise ApiError(method, url, r.status_code, r.text)
            return r
        assert last is not None
        raise last

    def get(self, path: str, token: Optional[str] = None, **params: Any) -> Any:
        return self.request("GET", path, token=token, retry=True,
                            params={k: v for k, v in params.items() if v is not None}).json()

    # ------------------------------------------------------------------ auth / devices
    def login(self, username: str, password: str) -> str:
        r = self.request("POST", "/v1/auth/token", json={"username": username, "password": password}, retry=True)
        return str(r.json()["access_token"])

    def register_device(self, device_id: str, *, kind: str = "sim", calib_id: Optional[str] = None,
                        agency_code: str = "GHMC", vehicle_reg: Optional[str] = None,
                        token: Optional[str] = None) -> str:
        body = {"device_id": device_id, "kind": kind, "agency_code": agency_code, "calib_id": calib_id,
                "vehicle_reg": vehicle_reg}
        r = self.request("POST", "/v1/edge/register", json=body, token=token, retry=True)
        return str(r.json()["token"])

    def edge_config(self, token: str, etag: Optional[str] = None) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        h = {"If-None-Match": etag} if etag else {}
        r = self.request("GET", "/v1/edge/config", token=token, retry=True, ok=(200, 304), headers=h)
        if r.status_code == 304:
            return None, etag
        return r.json(), r.headers.get("ETag")

    # ------------------------------------------------------------------ uploads + ingest
    def presign(self, token: str, keys: list[str], content_type: str = "image/jpeg") -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(keys), 50):
            r = self.request("POST", "/v1/edge/presign", token=token, retry=True,
                             json={"keys": keys[i:i + 50], "content_type": content_type})
            out.update(r.json()["urls"])
        return out

    def put_object(self, url: str, data: bytes, content_type: str = "image/jpeg") -> dict[str, Any]:
        return self.request("PUT", url, retry=True, content=data, headers={"Content-Type": content_type}).json()

    def upload_frames(self, token: str, frames: dict[str, bytes]) -> int:
        if not frames:
            return 0
        urls = self.presign(token, list(frames))
        for key, data in frames.items():
            self.put_object(urls[key], data)
        return len(frames)

    def post_observations(self, token: str, observations: list[dict[str, Any]]) -> dict[str, Any]:
        r = self.request("POST", "/v1/ingest/observations", token=token, retry=True,
                         json={"observations": observations})
        return r.json()

    def post_trip_fixes(self, token: str, batch: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/ingest/trip-fixes", token=token, retry=True, json=batch).json()

    def trip_close(self, token: str, device_id: str, trip_id: str, ended_at_iso: str,
                   km: Optional[float] = None) -> dict[str, Any]:
        body: dict[str, Any] = {"device_id": device_id, "trip_id": trip_id, "ended_at": ended_at_iso}
        if km is not None:
            body["km"] = km
        return self.request("POST", "/v1/ingest/trip-close", token=token, retry=True, json=body).json()

    def citizen_report(self, lat: float, lon: float, *, text: str, lang: str = "te",
                       photo: Optional[bytes] = None, token: Optional[str] = None) -> dict[str, Any]:
        data = {"lat": f"{lat:.7f}", "lon": f"{lon:.7f}", "text": text, "lang": lang}
        files = {"photo": ("report.jpg", photo, "image/jpeg")} if photo else None
        return self.request("POST", "/v1/citizen/reports", token=token, data=data, files=files).json()

    # ------------------------------------------------------------------ staff / contractor operations
    def potholes(self, token: str, **params: Any) -> dict[str, Any]:
        return self.get("/v1/potholes", token, **params)

    def plan(self, token: str, pothole_id: int) -> dict[str, Any]:
        return self.request("POST", f"/v1/potholes/{pothole_id}/plan", token=token).json()

    def create_work_order(self, token: str, pothole_ids: list[int], *, contractor_id: Optional[int] = None,
                          contract_id: Optional[int] = None, crew: Optional[str] = None) -> dict[str, Any]:
        body = {"pothole_ids": pothole_ids, "contractor_id": contractor_id, "contract_id": contract_id, "crew": crew}
        return self.request("POST", "/v1/work-orders", token=token,
                            json={k: v for k, v in body.items() if v is not None}).json()

    def transition(self, token: str, wo_id: int, to: str, **extra: Any) -> dict[str, Any]:
        return self.request("PATCH", f"/v1/work-orders/{wo_id}/transition", token=token,
                            json={"to": to, **{k: v for k, v in extra.items() if v is not None}}).json()

    def wo_photo(self, token: str, wo_id: int, kind: str, jpeg: bytes, lat: float, lon: float) -> dict[str, Any]:
        return self.request("POST", f"/v1/work-orders/{wo_id}/photos", token=token,
                            data={"kind": kind, "lat": f"{lat:.7f}", "lon": f"{lon:.7f}"},
                            files={"file": (f"{kind}.jpg", jpeg, "image/jpeg")}).json()


class Outbox:
    """In-memory store-and-forward queue of observations (each keeps its client_uuid across retries)."""

    def __init__(self) -> None:
        self.q: deque[dict[str, Any]] = deque()
        self.sent = 0
        self.duplicates = 0
        self.rejected: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.q)

    def put(self, obs: dict[str, Any]) -> None:
        self.q.append(obs)

    def flush(self, client: ApiClient, token: str, batch: int = 20) -> list[dict[str, Any]]:
        """Send everything queued, ``batch`` per request; an item leaves the queue only after the server answered."""
        results = []
        while self.q:
            chunk = [self.q[i] for i in range(min(batch, len(self.q)))]
            res = client.post_observations(token, chunk)
            for _ in chunk:
                self.q.popleft()
            self.sent += int(res.get("accepted", 0))
            self.duplicates += int(res.get("duplicates", 0))
            self.rejected.extend(res.get("rejected") or [])
            results.append(res)
        return results
