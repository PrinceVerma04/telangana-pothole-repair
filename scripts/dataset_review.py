"""Multi-user dataset review: keep / reject / needs-fix for every image of the training dataset (DECISIONS D22).

    make review                                    # LAN server on http://<this-pc>:8770 (prints the access code)
    .venv/bin/python scripts/dataset_review.py serve --host 0.0.0.0 --port 8770
    .venv/bin/python scripts/dataset_review.py export   # write data/review/exports/* from the decisions so far
    .venv/bin/python scripts/dataset_review.py status   # counts in the terminal

Workflow (user decisions 2026-11-17):
  * Reviewers sign in with a name and the shared access code (data/review/access_code.txt, or GM_REVIEW_CODE).
  * Work is split: each reviewer is handed a batch of images nobody else holds (claims expire after 30 min).
  * Keep / Needs fix are final after one review. A Reject goes to a *second look* by a different reviewer: reject
    again -> rejected; keep -> kept; needs fix -> needs fix. No single person removes data alone.
  * Test images can be reviewed, but their rejects are only listed (exports/rejected_test_pending.txt): the test split
    is locked and changes only with a deliberate new lock. Train/val rejects go to exports/rejected_trainval.txt, which
    `data/derive_dataset.py --exclude` applies on the next dataset build (`make derive-dataset-reviewed`).
  * Every decision is kept (append-only history); the latest decision per image and round counts.

Images are served from the dataset folder only to signed-in reviewers; nothing is uploaded anywhere.
"""


import argparse
import csv
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "datasets" / "india_pothole_seg4_v2"
STATE = REPO / "data" / "review"
DB_PATH = STATE / "review.sqlite"
EXPORTS = STATE / "exports"
PAGE = Path(__file__).with_name("dataset_review.html")
SPLITS = ("train", "val", "test")
LOCKED_SPLITS = {"test"}
BATCH = 20
CLAIM_TTL_S = 30 * 60
COOKIE = "gm_review"
IST = timezone(timedelta(hours=5, minutes=30))
SOURCES = {"bh": "BharatPotHole (India dashcam)", "rddin": "RDD2022 India", "rddfx": "RDD2022 other countries",
           "rddng": "RDD2022 negatives", "ws": "Web, human masks", "ey": "eYantra Indian roads"}
DECISIONS = ("keep", "reject", "fix")
REASONS = {
    "wrong_class": "Wrong class", "bad_mask": "Bad mask / box", "missing_labels": "Missing labels",
    "not_road": "Not a road", "not_india": "Not India-like", "blurry_dark": "Blurry / dark",
    "duplicate": "Duplicate", "overlay_text": "Watermark / overlay", "other": "Other",
}
STATUSES = ("unreviewed", "keep", "fix", "second_look", "rejected")


# ================================================================================================ dataset index
def source_of(stem: str) -> str:
    return stem.split("_", 1)[0]


def read_index(dataset: Path = DATASET) -> dict[str, dict[str, Any]]:
    """stem -> {split, source, file, n, classes{name: count}} from the YOLO-seg dataset (labels read once)."""
    import yaml

    names = {int(k): v for k, v in (yaml.safe_load((dataset / "data.yaml").read_text())["names"]).items()}
    idx: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        idir = dataset / "images" / split
        if not idir.is_dir():
            continue
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lab = dataset / "labels" / split / f"{p.stem}.txt"
            cls: Counter = Counter()
            if lab.is_file():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        cls[names.get(int(line.split()[0]), "?")] += 1
            idx[p.stem] = {"split": split, "source": source_of(p.stem), "file": p.name, "n": sum(cls.values()),
                           "classes": dict(cls)}
    return idx


def read_polygons(stem: str, meta: dict[str, Any], dataset: Path = DATASET) -> list[dict[str, Any]]:
    import yaml

    names = {int(k): v for k, v in (yaml.safe_load((dataset / "data.yaml").read_text())["names"]).items()}
    lab = dataset / "labels" / meta["split"] / f"{stem}.txt"
    out = []
    if lab.is_file():
        for line in lab.read_text(encoding="utf-8").splitlines():
            v = line.split()
            if len(v) >= 7:
                out.append({"cls": names.get(int(v[0]), "?"), "pts": [float(x) for x in v[1:]]})
    return out


# ================================================================================================ database
SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT, image TEXT NOT NULL, round INTEGER NOT NULL, reviewer TEXT NOT NULL,
  decision TEXT NOT NULL, reasons TEXT NOT NULL DEFAULT '[]', note TEXT NOT NULL DEFAULT '', split TEXT, source TEXT,
  created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS reviews_image ON reviews(image, round, id);
CREATE TABLE IF NOT EXISTS claims (
  image TEXT NOT NULL, round INTEGER NOT NULL, reviewer TEXT NOT NULL, claimed_at REAL NOT NULL,
  PRIMARY KEY (image, round));
CREATE TABLE IF NOT EXISTS reviewers (name TEXT PRIMARY KEY, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def db(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=15, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=15000")
    con.executescript(SCHEMA)
    try:
        yield con
    finally:
        con.close()


def current(con: sqlite3.Connection) -> dict[str, dict[int, sqlite3.Row]]:
    """image -> {round: latest review row}."""
    out: dict[str, dict[int, sqlite3.Row]] = defaultdict(dict)
    for r in con.execute("SELECT r.* FROM reviews r JOIN (SELECT image, round, MAX(id) AS id FROM reviews "
                         "GROUP BY image, round) m ON r.id = m.id"):
        out[r["image"]][r["round"]] = r
    return out


def status_of(rounds: dict[int, Any]) -> str:
    r1, r2 = rounds.get(1), rounds.get(2)
    if r1 is None:
        return "unreviewed"
    if r1["decision"] != "reject":
        return "keep" if r1["decision"] == "keep" else "fix"
    if r2 is None:
        return "second_look"
    return {"reject": "rejected", "keep": "keep", "fix": "fix"}[r2["decision"]]


def norm_name(name: str) -> str:
    name = " ".join(str(name).split())[:40]
    if not name or any(c in name for c in "<>\"'`\\/"):
        raise ValueError("use a plain name (letters, digits, spaces, . - _)")
    return name


# ================================================================================================ queue (claims)
def matches(meta: dict[str, Any], f: dict[str, str]) -> bool:
    if f.get("split") and meta["split"] != f["split"]:
        return False
    if f.get("source") and meta["source"] != f["source"]:
        return False
    c = f.get("cls")
    if c == "__none__":
        return meta["n"] == 0
    if c and not meta["classes"].get(c):
        return False
    return True


def next_item(con: sqlite3.Connection, idx: dict[str, dict[str, Any]], who: str, f: dict[str, str],
              now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The reviewer's next image: an open claim of theirs, else a new batch (second looks first, then unreviewed)."""
    now = time.time() if now is None else now
    con.execute("BEGIN IMMEDIATE")
    try:
        con.execute("DELETE FROM claims WHERE claimed_at < ?", (now - CLAIM_TTL_S,))
        cur = current(con)
        mine = [(r["image"], r["round"]) for r in con.execute(
            "SELECT image, round FROM claims WHERE reviewer = ? ORDER BY round DESC, claimed_at, image", (who,))]

        def still_open(img: str, rnd: int) -> bool:
            st = status_of(cur.get(img, {}))
            return img in idx and ((rnd == 1 and st == "unreviewed") or
                                   (rnd == 2 and st == "second_look" and cur[img][1]["reviewer"] != who))

        mine = [(i, r) for i, r in mine if still_open(i, r)]
        if not mine:
            taken = {(r["image"], r["round"]) for r in con.execute("SELECT image, round FROM claims")}
            pick: list[tuple[str, int]] = []
            for img, meta in idx.items():  # second looks first: they block final decisions
                if len(pick) >= BATCH:
                    break
                rr = cur.get(img, {})
                if (status_of(rr) == "second_look" and rr[1]["reviewer"] != who and (img, 2) not in taken
                        and matches(meta, f)):
                    pick.append((img, 2))
            if len(pick) < BATCH:
                cands = [img for img, meta in idx.items() if img not in cur and (img, 1) not in taken
                         and matches(meta, f)]
                # spread over the dataset instead of walking it alphabetically (one source at a time)
                cands.sort(key=lambda s: hashlib.sha1(f"{who}|{s}".encode()).hexdigest())
                pick += [(img, 1) for img in cands[:BATCH - len(pick)]]
            con.executemany("INSERT OR REPLACE INTO claims(image, round, reviewer, claimed_at) VALUES (?,?,?,?)",
                            [(i, r, who, now) for i, r in pick])
            mine = pick
        else:
            con.executemany("UPDATE claims SET claimed_at = ? WHERE image = ? AND round = ?",
                            [(now, i, r) for i, r in mine])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    if not mine:
        return None
    img, rnd = mine[0]
    return {"image": img, "round": rnd, "upcoming": [i for i, _ in mine[1:6]], "batch_left": len(mine)}


def decide(con: sqlite3.Connection, idx: dict[str, dict[str, Any]], who: str, image: str, decision: str,
           reasons: list[str], note: str) -> dict[str, Any]:
    """Record a decision. Round is derived: first review, a second look of someone else's reject, or a reviewer
    changing their own latest decision (allowed until the next round exists)."""
    if image not in idx:
        raise ValueError("unknown image")
    if decision not in DECISIONS:
        raise ValueError("decision must be keep, reject or fix")
    reasons = [r for r in reasons if r in REASONS]
    if decision in ("reject", "fix") and not reasons:
        raise ValueError("pick at least one reason for reject / needs fix")
    note = str(note or "")[:500]
    con.execute("BEGIN IMMEDIATE")
    try:
        rr = current(con).get(image, {})
        r1, r2 = rr.get(1), rr.get(2)
        if r1 is None:
            rnd = 1
        elif r2 is None and r1["decision"] == "reject" and r1["reviewer"] != who:
            rnd = 2
        elif r2 is None and r1["reviewer"] == who:
            rnd = 1  # change own first review
        elif r2 is not None and r2["reviewer"] == who:
            rnd = 2  # change own second look
        else:
            raise PermissionError("this image already has a final decision by other reviewers")
        meta = idx[image]
        con.execute("INSERT INTO reviews(image, round, reviewer, decision, reasons, note, split, source, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)", (image, rnd, who, decision, json.dumps(reasons), note,
                                                   meta["split"], meta["source"], utc_now()))
        con.execute("DELETE FROM claims WHERE image = ? AND round = ?", (image, rnd))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return {"image": image, "round": rnd, "status": status_of(current(con).get(image, {}))}


# ================================================================================================ stats + export
def stats(con: sqlite3.Connection, idx: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cur = current(con)
    tot: Counter = Counter()
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_source: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    for img, meta in idx.items():
        st = status_of(cur.get(img, {}))
        tot[st] += 1
        by_split[meta["split"]][st] += 1
        by_source[meta["source"]][st] += 1
        if st in ("rejected", "fix", "second_look"):
            last = cur[img].get(2) or cur[img].get(1)
            reasons.update(json.loads(last["reasons"]))
    today = datetime.now(IST).date()
    people: dict[str, Counter] = defaultdict(Counter)
    for r in con.execute("SELECT reviewer, decision, round, created_at FROM reviews"):
        p = people[r["reviewer"]]
        p["total"] += 1
        p[r["decision"]] += 1
        p["second_looks"] += r["round"] == 2
        if datetime.fromisoformat(r["created_at"]).astimezone(IST).date() == today:
            p["today"] += 1
    active = {r["reviewer"] for r in con.execute("SELECT reviewer FROM claims WHERE claimed_at > ?",
                                                 (time.time() - 10 * 60,))}
    recent = [dict(r) for r in con.execute("SELECT image, round, reviewer, decision, reasons, created_at, split "
                                           "FROM reviews ORDER BY id DESC LIMIT 25")]
    return {"total": len(idx), "status": {s: tot[s] for s in STATUSES},
            "by_split": {s: {k: by_split[s][k] for k in STATUSES} for s in SPLITS if s in by_split},
            "by_source": {s: {k: by_source[s][k] for k in STATUSES} for s in sorted(by_source)},
            "reasons": dict(reasons.most_common()),
            "reviewers": sorted(({"name": n, **dict(c), "active": n in active} for n, c in people.items()),
                                key=lambda r: -r["total"]),
            "recent": recent}


def export(con: sqlite3.Connection, idx: dict[str, dict[str, Any]], out: Optional[Path] = None) -> dict[str, Any]:
    out = out or EXPORTS
    out.mkdir(parents=True, exist_ok=True)
    cur = current(con)
    stamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M IST")
    lists: dict[str, list[str]] = {"rejected_trainval": [], "rejected_test_pending": [], "needs_fix": []}
    for img in sorted(cur):
        meta = idx.get(img)
        st = status_of(cur[img])
        last = cur[img].get(2) or cur[img].get(1)
        why = ",".join(json.loads(last["reasons"])) if last else ""
        split = meta["split"] if meta else (last["split"] if last else "?")
        if st == "rejected":
            lists["rejected_test_pending" if split in LOCKED_SPLITS else "rejected_trainval"].append(f"{img}  # {why}")
        elif st == "fix":
            lists["needs_fix"].append(f"{img}  # {split} {why}")
    head = {"rejected_trainval": "train/val images rejected by two reviewers; applied by "
                                 "`data/derive_dataset.py --exclude` (make derive-dataset-reviewed)",
            "rejected_test_pending": "TEST images rejected by two reviewers; NOT applied (test split is locked, D22)",
            "needs_fix": "images whose labels need fixing (kept in the dataset until fixed)"}
    for k, rows in lists.items():
        (out / f"{k}.txt").write_text(f"# {head[k]}\n# {len(rows)} images, written {stamp}\n" +
                                      "".join(r + "\n" for r in rows), encoding="utf-8")
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "image", "split", "source", "round", "reviewer", "decision", "reasons", "note", "created_at_utc"])
    for r in con.execute("SELECT * FROM reviews ORDER BY id"):
        w.writerow([r["id"], r["image"], r["split"], r["source"], r["round"], r["reviewer"], r["decision"],
                    ";".join(json.loads(r["reasons"])), r["note"], r["created_at"]])
    (out / "decisions.csv").write_text(buf.getvalue(), encoding="utf-8")
    return {"dir": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
            **{k: len(v) for k, v in lists.items()}, "decisions": buf.getvalue().count("\n") - 1}


# ================================================================================================ auth
def _secret_file(name: str, make: callable) -> str:
    p = STATE / name
    if not p.is_file():
        STATE.mkdir(parents=True, exist_ok=True)
        p.write_text(make() + "\n", encoding="utf-8")
        os.chmod(p, 0o600)
    return p.read_text(encoding="utf-8").strip()


def access_code() -> str:
    return os.environ.get("GM_REVIEW_CODE") or _secret_file(
        "access_code.txt", lambda: "-".join(secrets.choice(["road", "bus", "pit", "lane", "kerb", "tar", "gravel",
                                                            "patch", "crack", "drain"]) for _ in range(2))
        + f"-{secrets.randbelow(900) + 100}")


def sign(name: str, key: str) -> str:
    mac = hmac.new(key.encode(), name.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{name}|{mac}"


def verify(token: Optional[str], key: str) -> Optional[str]:
    if not token or "|" not in token:
        return None
    name, _ = token.rsplit("|", 1)
    return name if hmac.compare_digest(sign(name, key), token) else None


# ================================================================================================ web app
def make_app(dataset: Path = DATASET) -> Any:
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

    idx = read_index(dataset)
    key = _secret_file("secret.key", lambda: secrets.token_hex(32))
    code = access_code()
    app = FastAPI(title="RoadFix dataset review", docs_url=None, redoc_url=None, openapi_url=None)

    def who(req: Request) -> str:
        name = verify(req.cookies.get(COOKIE), key)
        if not name:
            raise HTTPException(401, "sign in first")
        return name

    @app.get("/", response_class=HTMLResponse)
    def page() -> Response:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.post("/api/login")
    def login(body: dict = Body(...)) -> Response:
        if not hmac.compare_digest(str(body.get("code", "")).strip(), code):
            time.sleep(0.5)
            raise HTTPException(403, "wrong access code")
        try:
            name = norm_name(body.get("name", ""))
        except ValueError as e:
            raise HTTPException(400, str(e))
        with db() as con:
            con.execute("INSERT INTO reviewers(name, first_seen, last_seen) VALUES (?,?,?) ON CONFLICT(name) DO "
                        "UPDATE SET last_seen = excluded.last_seen", (name, utc_now(), utc_now()))
        r = JSONResponse({"name": name})
        r.set_cookie(COOKIE, sign(name, key), httponly=True, samesite="strict", max_age=30 * 86400)
        return r

    @app.post("/api/logout")
    def logout() -> Response:
        r = JSONResponse({"ok": True})
        r.delete_cookie(COOKIE)
        return r

    @app.get("/api/me")
    def me(req: Request) -> dict:
        return {"name": who(req)}

    @app.get("/api/meta")
    def meta(req: Request) -> dict:
        who(req)
        import yaml

        names = list(yaml.safe_load((dataset / "data.yaml").read_text())["names"].values())
        return {"dataset": dataset.name, "total": len(idx), "splits": list(SPLITS), "locked_splits": sorted(LOCKED_SPLITS),
                "sources": {s: SOURCES.get(s, s) for s in sorted({m["source"] for m in idx.values()})},
                "classes": names, "reasons": REASONS, "batch": BATCH}

    @app.get("/api/next")
    def nxt(req: Request, split: str = "", source: str = "", cls: str = "") -> dict:
        name = who(req)
        with db() as con:
            item = next_item(con, idx, name, {"split": split, "source": source, "cls": cls})
        return {"item": item}

    @app.post("/api/release")
    def release(req: Request) -> dict:
        name = who(req)  # filters changed: hand the unfinished batch back to the pool
        with db() as con:
            n = con.execute("DELETE FROM claims WHERE reviewer = ?", (name,)).rowcount
        return {"released": n}

    @app.get("/api/image/{stem}")
    def image(req: Request, stem: str) -> dict:
        name = who(req)
        m = idx.get(stem)
        if not m:
            raise HTTPException(404, "unknown image")
        with db() as con:
            hist = [dict(r) | {"reasons": json.loads(r["reasons"])} for r in
                    con.execute("SELECT round, reviewer, decision, reasons, note, created_at FROM reviews "
                                "WHERE image = ? ORDER BY id", (stem,))]
            rr = current(con).get(stem, {})
        st = status_of(rr)
        r1, r2 = rr.get(1), rr.get(2)
        can = (r1 is None or (r2 is None and (r1["reviewer"] == name or r1["decision"] == "reject"))
               or (r2 is not None and r2["reviewer"] == name))
        return {"image": stem, **m, "locked": m["split"] in LOCKED_SPLITS, "polygons": read_polygons(stem, m, dataset),
                "status": st, "history": hist, "can_decide": bool(can),
                "first": dict(r1) | {"reasons": json.loads(r1["reasons"])} if r1 else None}

    @app.get("/img/{stem}")
    def img(req: Request, stem: str) -> Response:
        who(req)
        m = idx.get(stem)
        p = dataset / "images" / m["split"] / m["file"] if m else None
        if not p or not p.is_file():
            raise HTTPException(404, "image not found")
        return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})

    @app.post("/api/decide")
    def post_decide(req: Request, body: dict = Body(...)) -> dict:
        name = who(req)
        try:
            with db() as con:
                return decide(con, idx, name, str(body.get("image", "")), str(body.get("decision", "")),
                              list(body.get("reasons") or []), str(body.get("note") or ""))
        except PermissionError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.get("/api/stats")
    def get_stats(req: Request) -> dict:
        who(req)
        with db() as con:
            return stats(con, idx)

    @app.get("/api/list")
    def listing(req: Request, status: str = "rejected", split: str = "", source: str = "", reviewer: str = "",
                reason: str = "", page: int = 0) -> dict:
        who(req)
        with db() as con:
            cur = current(con)
        rows = []
        for img, m in idx.items():
            rr = cur.get(img, {})
            st = status_of(rr)
            if status and st != status:
                continue
            if (split and m["split"] != split) or (source and m["source"] != source):
                continue
            last = rr.get(2) or rr.get(1)
            if reviewer and not any(r["reviewer"] == reviewer for r in rr.values()):
                continue
            rs = json.loads(last["reasons"]) if last else []
            if reason and reason not in rs:
                continue
            rows.append({"image": img, "split": m["split"], "source": m["source"], "status": st, "n": m["n"],
                         "reasons": rs, "by": [rr[k]["reviewer"] for k in sorted(rr)],
                         "at": last["created_at"] if last else None})
        rows.sort(key=lambda r: r["at"] or "", reverse=True)
        per = 48
        return {"total": len(rows), "page": page, "per_page": per, "rows": rows[page * per:(page + 1) * per]}

    @app.post("/api/export")
    def post_export(req: Request) -> dict:
        who(req)
        with db() as con:
            return export(con, idx)

    @app.get("/api/decisions.csv")
    def csv_download(req: Request) -> Response:
        who(req)
        with db() as con:
            export(con, idx)
        return FileResponse(Path(EXPORTS) / "decisions.csv", filename="dataset_review_decisions.csv", media_type="text/csv")

    app.state.access_code = code
    app.state.index_size = len(idx)
    return app


def lan_ip() -> str:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 9))  # TEST-NET address: no packet is sent, only the route is resolved
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("serve", help="run the review web app")
    sv.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = office LAN; 127.0.0.1 = this PC only")
    sv.add_argument("--port", type=int, default=8770)
    sub.add_parser("export", help="write data/review/exports/* from the decisions")
    sub.add_parser("status", help="print review progress")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        app = make_app()
        url = f"http://{lan_ip() if a.host == '0.0.0.0' else a.host}:{a.port}"
        print(f"\nRoadFix dataset review: {url}\n  images: {app.state.index_size} ({DATASET.name})\n"
              f"  access code: {app.state.access_code}   (give it to reviewers; file: data/review/access_code.txt)\n"
              f"  Ctrl+C stops the server; decisions are saved as they are made ({DB_PATH.relative_to(REPO)})\n",
              flush=True)
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
        return 0
    idx = read_index()
    with db() as con:
        if a.cmd == "export":
            print(json.dumps(export(con, idx), indent=1))
        else:
            s = stats(con, idx)
            print(f"{s['total']} images: " + ", ".join(f"{k} {v}" for k, v in s["status"].items()))
            for r in s["reviewers"]:
                print(f"  {r['name']}: {r.get('total', 0)} decisions ({r.get('today', 0)} today)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
