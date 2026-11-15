"""WP6 workers: real pipeline stages, hooks and ops ticks for the in-process pipeline (DECISIONS D2, no Celery).

    from server.workers import install
    install(settings, storage)     # idempotent; registers stages + hooks into server.app.pipeline

``install()`` registers ``verify, depth, measure, locate, match, fuse, cluster, score, plan`` (server/workers/stages.py)
with ``pipeline.register_stage`` and the ``trip_close`` / ``complaint`` hooks with ``pipeline.register_hook`` (the
pipeline's default complaint hook is replaced by one that also re-scores). ``settings`` / ``storage`` (optional) are
what hooks and ops ticks use when no request context is available; pass the API's own.
``uninstall()`` restores whatever was registered before (tests).

Poller / ops CLI: ``python -m server.workers.run`` (see run.py).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

_STATE: dict[str, Any] = {"installed": False, "previous": {}, "hooks": []}


def install(settings: Any = None, storage: Any = None, *, force: bool = False) -> dict[str, Any]:
    from server.app import pipeline
    from server.workers import stages

    stages.configure_runtime(settings, storage)
    if _STATE["installed"] and not force:
        return status()
    real: dict[str, Callable[..., Any]] = {
        "verify": stages.stage_verify, "depth": stages.stage_depth, "measure": stages.stage_measure,
        "locate": stages.stage_locate, "match": stages.stage_match, "fuse": stages.stage_fuse,
        "cluster": stages.stage_cluster, "score": stages.stage_score, "plan": stages.stage_plan,
    }
    if not _STATE["installed"]:
        _STATE["previous"] = {n: pipeline.get_stage(n) for n in pipeline.STAGE_ORDER}
    for name in pipeline.STAGE_ORDER:
        pipeline.register_stage(name, real[name])
    hooks = pipeline._HOOKS  # noqa: SLF001 - only to drop the default complaint hook (superseded, not edited)
    added: list[tuple[str, Callable[..., Any]]] = []
    if pipeline.default_complaint_hook in hooks["complaint"]:
        hooks["complaint"].remove(pipeline.default_complaint_hook)
        added.append(("-complaint", pipeline.default_complaint_hook))
    for event, fn in (("complaint", stages.complaint_hook), ("trip_close", stages.trip_close_hook)):
        if fn not in hooks[event]:
            pipeline.register_hook(event, fn)
            added.append((event, fn))
    _STATE["hooks"] = added if not _STATE["installed"] else _STATE["hooks"] + added
    _STATE["installed"] = True
    return status()


def uninstall() -> None:
    """Restore the stages / hooks that were registered before ``install()`` (used by tests)."""
    from server.app import pipeline
    from server.workers import stages

    if not _STATE["installed"]:
        return
    for name, fn in _STATE["previous"].items():
        pipeline.register_stage(name, fn)
    hooks = pipeline._HOOKS  # noqa: SLF001
    for event, fn in _STATE["hooks"]:
        if event.startswith("-"):
            ev = event[1:]
            if fn not in hooks[ev]:
                hooks[ev].insert(0, fn)
        elif fn in hooks[event]:
            hooks[event].remove(fn)
    stages.reset_runtime()
    _STATE.update(installed=False, previous={}, hooks=[])


def installed() -> bool:
    return bool(_STATE["installed"])


def status() -> dict[str, Any]:
    from server.app import pipeline

    return {"installed": _STATE["installed"],
            "stages": {n: getattr(pipeline.get_stage(n), "__module__", "?") for n in pipeline.STAGE_ORDER},
            "hooks": {e: [getattr(f, "__name__", str(f)) for f in fs] for e, fs in pipeline._HOOKS.items()}}  # noqa: SLF001


def configure(settings: Optional[Any] = None, storage: Optional[Any] = None) -> None:
    from server.workers import stages

    stages.configure_runtime(settings, storage)


__all__ = ["configure", "install", "installed", "status", "uninstall"]
