"""Test helpers shared by the server tests (example payloads from edge/schema/examples)."""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from server.app.core.config import PROJECT_ROOT

EXAMPLES = PROJECT_ROOT / "edge" / "schema" / "examples"
DEVICE_ID = "tsrtc-bus-ts09ua1234"


def example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def variant(obs: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """Copy of an example observation with a fresh client_uuid (and optional overrides)."""
    o = copy.deepcopy(obs)
    o["client_uuid"] = str(uuid.uuid4())
    for k, v in changes.items():
        o[k] = v
    return o
