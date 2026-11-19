"""Runtime settings (WP4). Reads environment variables and the project ``.env`` (keys of ``.env.example``).

Relative paths (``sqlite:///app_data/gunthameter.db``, ``STORAGE_ROOT=app_data/store``) are resolved against the project
root, so the API, seed script, workers and tests agree no matter which directory they are started from.

Usage::

    from server.app.core.config import get_settings
    s = get_settings()           # cached singleton
    s2 = Settings(DATABASE_URL="sqlite:////tmp/x.db", _env_file=None)   # tests
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = PROJECT_ROOT / "configs"
APP_VERSION = "0.1.0"


def _abs(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # runtime mode
    DEMO_MODE: bool = True
    DEMO_LITE: bool = True

    # database (DECISIONS D1)
    DATABASE_URL: str = "sqlite:///app_data/gunthameter.db"
    DB_ECHO: bool = False

    # storage (D3)
    STORAGE_BACKEND: str = "fs"
    STORAGE_ROOT: str = "app_data/store"
    S3_ENDPOINT: Optional[str] = None
    S3_BUCKET: str = "gunthameter"
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    PRESIGN_TTL_S: int = 900
    URL_TTL_S: int = 24 * 3600
    MAX_UPLOAD_MB: int = 15

    # queue (D2)
    REDIS_URL: Optional[str] = None

    # models
    SEG_WEIGHTS: str = "models/base_model/pothole_seg11n.pt"  # D50 (.env sets it too)
    VERIFIER_WEIGHTS: str = ""  # D24: no second model
    VERIFIER: str = "off"
    DEPTH_BACKEND: str = "off"  # D27: size-only (no depth model); da2_metric to measure depth again
    DEPTH_MODEL_ID: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
    CUDA_DEVICE: int = 0

    # auth
    JWT_SECRET: str = "change-me-in-production"
    JWT_TTL_MIN: int = 720
    DEVICE_TOKEN_TTL_DAYS: int = 365
    OTP_DEMO_CODE: str = "123456"
    OTP_TTL_S: int = 300

    # web
    API_BASE_URL: str = "http://localhost:8000"
    WEB_ORIGIN: str = "http://localhost:5173"

    # behaviour switches
    PIPELINE_INLINE: Optional[bool] = None  # default = DEMO_MODE
    OPEN_DEVICE_REGISTRATION: Optional[bool] = None  # default = DEMO_MODE

    @field_validator("S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY", "REDIS_URL", mode="before")
    @classmethod
    def _empty_none(cls, v):  # type: ignore[no-untyped-def]
        return None if (v is None or str(v).strip() == "") else v

    @model_validator(mode="after")
    def _resolve(self) -> "Settings":
        url = self.DATABASE_URL
        for prefix in ("sqlite:///", "sqlite+pysqlite:///"):
            if url.startswith(prefix):
                rest = url[len(prefix):]
                if rest and rest != ":memory:" and not rest.startswith("/"):
                    url = prefix + str(_abs(rest))
        object.__setattr__(self, "DATABASE_URL", url)
        object.__setattr__(self, "STORAGE_ROOT", str(_abs(self.STORAGE_ROOT)))
        return self

    # derived ------------------------------------------------------------------
    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def pipeline_inline(self) -> bool:
        return self.DEMO_MODE if self.PIPELINE_INLINE is None else self.PIPELINE_INLINE

    @property
    def open_device_registration(self) -> bool:
        return self.DEMO_MODE if self.OPEN_DEVICE_REGISTRATION is None else self.OPEN_DEVICE_REGISTRATION

    @property
    def cors_origins(self) -> list[str]:
        origins = {o.strip() for o in self.WEB_ORIGIN.split(",") if o.strip()}
        origins |= {"http://localhost:5173", "http://127.0.0.1:5173"}
        return sorted(origins)

    @property
    def seg_weights_path(self) -> Path:
        return _abs(self.SEG_WEIGHTS)

    @property
    def storage_root_path(self) -> Path:
        return Path(self.STORAGE_ROOT)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
