"""DB-backed job poller + ops ticks (DECISIONS D2: production path when the API does not run the pipeline inline).

    python -m server.workers.run                 # poll forever: job table + pending observations, ops ticks every 5 min
    python -m server.workers.run --once          # drain what is queued now, then exit
    python -m server.workers.run --tick          # run the ops ticks once and exit (cron / Task Scheduler)
    python -m server.workers.run --tick --ticks sla_tick,verify_repairs --now 2026-11-13T10:00:00+05:30

Jobs (``job`` table, ``kind`` + JSON ``payload``)::

    process_observation {"observation_id": 7}      -> pipeline.process_observation
    trip_close          {"trip_id": 3}             -> pipeline.on_trip_close (hooks)
    complaint           {"complaint_id": 5}        -> pipeline.process_complaint (hooks)
    ops_tick            {"ticks": ["sla_tick"]}    -> ops.run_ticks

A job is claimed with a conditional UPDATE (status queued -> running), so several pollers may share one database.
Failures are retried up to ``MAX_ATTEMPTS`` times, then the job is ``failed`` with the error. Observations still
``pending`` (posted while PIPELINE_INLINE was off) are processed too unless ``--no-scan-pending``.
Exit code of ``--tick``: 0 when every tick is ok or skipped, 1 when a tick raised.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from server.app.core.util import now_utc
from server.app.db import models as m

log = logging.getLogger("gunthameter.workers.run")

MAX_ATTEMPTS = 3
JOB_KINDS = ("process_observation", "trip_close", "complaint", "ops_tick")
_STOP = {"flag": False}


def enqueue(db: Session, kind: str, payload: Optional[dict[str, Any]] = None) -> m.Job:
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind {kind!r}; expected one of {JOB_KINDS}")
    job = m.Job(kind=kind, payload=payload or {}, status="queued", attempts=0, created_at=now_utc())
    db.add(job)
    db.flush()
    return job


def claim_next(db: Session, exclude: Optional[set[int]] = None) -> Optional[m.Job]:
    """Atomically move the oldest queued job to running (safe with several pollers). ``exclude``: job ids already
    tried in this drain (a failed job is retried on the next drain, not in a tight loop)."""
    for _ in range(5):
        q = select(m.Job.id).where(m.Job.status == "queued")
        if exclude:
            q = q.where(m.Job.id.not_in(exclude))
        jid = db.scalar(q.order_by(m.Job.created_at, m.Job.id).limit(1))
        if jid is None:
            db.commit()
            return None
        res = db.execute(update(m.Job).where(m.Job.id == jid, m.Job.status == "queued")
                         .values(status="running", started_at=now_utc(), attempts=m.Job.attempts + 1))
        db.commit()
        if res.rowcount == 1:
            return db.get(m.Job, jid)
    return None


def execute_job(db: Session, job: m.Job, settings: Any, storage: Any) -> Any:
    from server.app import pipeline

    p = job.payload or {}
    if job.kind == "process_observation":
        res = pipeline.process_observation(int(p["observation_id"]), db, settings=settings, storage=storage)
        return {"status": res.status, "stages": res.stages, "pothole_ids": res.pothole_ids}
    if job.kind == "trip_close":
        return [repr(x) if isinstance(x, Exception) else x for x in pipeline.on_trip_close(int(p["trip_id"]), db)]
    if job.kind == "complaint":
        return [repr(x) if isinstance(x, Exception) else x for x in pipeline.process_complaint(int(p["complaint_id"]), db)]
    if job.kind == "ops_tick":
        from server.workers import ops

        return ops.run_ticks(db, None, p.get("ticks"))
    raise ValueError(f"unknown job kind {job.kind!r}")


def run_job(db: Session, job: m.Job, settings: Any, storage: Any) -> str:
    try:
        execute_job(db, job, settings, storage)
        db.commit()
        job = db.get(m.Job, job.id)
        job.status, job.finished_at, job.error = "done", now_utc(), None
    except Exception as e:
        db.rollback()
        log.exception("job %s (%s) failed", job.id, job.kind)
        job = db.get(m.Job, job.id)
        job.error = f"{type(e).__name__}: {e}"[:2000]
        job.status = "failed" if job.attempts >= MAX_ATTEMPTS else "queued"
        job.finished_at = now_utc() if job.status == "failed" else None
    db.commit()
    return job.status


def scan_pending(db: Session, settings: Any, storage: Any, limit: int = 100) -> int:
    """Process observations that are still ``pending`` and were never processed."""
    from server.app import pipeline

    ids = list(db.scalars(select(m.Observation.id).where(m.Observation.status == "pending",
                                                         m.Observation.processed_at.is_(None))
                          .order_by(m.Observation.id).limit(limit)))
    for oid in ids:
        try:
            pipeline.process_observation(oid, db, settings=settings, storage=storage)
        except Exception:
            db.rollback()
            log.exception("pending observation %s failed; marked processed to avoid a retry loop", oid)
            obs = db.get(m.Observation, oid)
            if obs is not None:
                obs.processed_at = now_utc()
                db.commit()
    return len(ids)


def drain(db: Session, settings: Any, storage: Any, *, max_jobs: int = 1000, scan: bool = True) -> dict[str, int]:
    counts = {"done": 0, "failed": 0, "requeued": 0, "pending_observations": 0}
    tried: set[int] = set()
    for _ in range(max_jobs):
        if _STOP["flag"]:
            break
        job = claim_next(db, tried)
        if job is None:
            break
        tried.add(job.id)
        st = run_job(db, job, settings, storage)
        counts["requeued" if st == "queued" else st] += 1
    if scan and not _STOP["flag"]:
        counts["pending_observations"] = scan_pending(db, settings, storage)
    return counts


def _parse_now(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    from server.app.core.util import as_utc

    return as_utc(datetime.fromisoformat(s))


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m server.workers.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--once", action="store_true", help="drain the queue once and exit")
    ap.add_argument("--tick", action="store_true", help="run the ops ticks once and exit")
    ap.add_argument("--ticks", default=None, help="comma-separated tick names (default: all)")
    ap.add_argument("--now", default=None, help="ISO timestamp for the ticks (default: now, UTC)")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    ap.add_argument("--ops-every", type=float, default=300.0, help="seconds between ops ticks while polling (0=off)")
    ap.add_argument("--max-jobs", type=int, default=1000, help="jobs per drain")
    ap.add_argument("--no-scan-pending", action="store_true", help="do not process pending observations")
    ap.add_argument("--database-url", default=None, help="override DATABASE_URL")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from server.app.core.config import Settings, get_settings
    from server.app.db.session import Database, set_database
    from server.app.storage import storage_from_settings
    from server.workers import install, ops

    settings = Settings(DATABASE_URL=a.database_url) if a.database_url else get_settings()
    database = Database(settings.DATABASE_URL)
    if settings.is_sqlite:
        database.create_all()
    set_database(database)
    storage = storage_from_settings(settings)
    install(settings, storage)
    names = [t.strip() for t in a.ticks.split(",") if t.strip()] if a.ticks else None
    try:
        if a.tick:
            with database.session() as db:
                t0 = time.perf_counter()
                res = ops.run_ticks(db, _parse_now(a.now), names)
            print(json.dumps({"ticks": res, "elapsed_s": round(time.perf_counter() - t0, 3)}, default=str, indent=2))
            return 1 if ops.tick_errors(res) else 0
        signal.signal(signal.SIGINT, lambda *_: _STOP.update(flag=True))
        last_ops = 0.0
        while True:
            with database.session() as db:
                c = drain(db, settings, storage, max_jobs=a.max_jobs, scan=not a.no_scan_pending)
                if any(c.values()):
                    log.info("drained %s", c)
                if not a.once and a.ops_every > 0 and time.monotonic() - last_ops >= a.ops_every:
                    res = ops.run_ticks(db, None, names)
                    last_ops = time.monotonic()
                    for err in ops.tick_errors(res):
                        log.error("ops tick %s", err)
            if a.once or _STOP["flag"]:
                return 0
            time.sleep(a.interval)
    finally:
        database.dispose()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
