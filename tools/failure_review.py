"""Failure review (D49): the photos where a model fails, with editable ground truth, inside the annotation page.

Photo sets `fail:<model>:<test set>` list the test photos where that model misses a true pothole or raises a false
alarm (confidence 0.25, lenient matching: IoU >= 0.3 or the box at least half inside the true outline), plus every
photo already corrected. The predictions come from the test-results cache (reports/test_results, `make results`).

Ground truth is shared by all models of one test set: corrections are saved as marks of collection `gt:<test set>`
(the phone walk keeps its own collection `phone_street_20261118`, D41). The photo opens with the current labels as
editable outlines. `make annotate-export` writes the corrections:
  - yolo test sets: test_datasets/<set>/labels/<stem>.txt (the original file is kept once in labels_original/);
    "no pothole" = empty label; "not sure" = listed in <set>/excluded.txt (not scored);
  - eYantra clean roads (false-alarm test): a photo marked "potholes" or "not sure" goes to excluded.txt, so it no
    longer counts as a clean road.
The RDD-India locked set (project rule) and the generated check sets are not editable here; rebuild the check sets
after corrections with `make check-sets` (overwrite).
Every save is appended to datasets/labelling/annotate/events.jsonl (who, set, photo, status) for live monitoring.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reports" / "test_results"
EVENTS = REPO / "datasets" / "labelling" / "annotate" / "events.jsonl"
CONF = 0.25
FAIL_SETS = ("phone_street_marked", "base_test", "web_test", "bharat_test", "eyantra_negatives")
PHONE_CID = "phone_street_20261118"
_cache: dict[str, Any] = {}


def _json(path: Path) -> Any:
    st = path.stat().st_mtime
    hit = _cache.get(str(path))
    if hit and hit[0] == st:
        return hit[1]
    data = json.loads(path.read_text())
    _cache[str(path)] = (st, data)
    return data


def index(results: Path = RESULTS) -> Optional[dict[str, Any]]:
    f = results / "index.json"
    return _json(f) if f.is_file() else None


def testset(set_id: str, results: Path = RESULTS) -> Optional[dict[str, Any]]:
    idx = index(results)
    return next((t for t in (idx or {}).get("testsets", []) if t["id"] == set_id), None)


def models(results: Path = RESULTS) -> list[dict[str, Any]]:
    return [{"id": m["id"], "label": m.get("label", m["id"])} for m in (index(results) or {}).get("models", [])]


def preds(model: str, set_id: str, results: Path = RESULTS) -> Optional[list]:
    f = results / "preds" / model / f"{set_id}.json"
    return _json(f)["preds"] if f.is_file() else None


def mark_cid(set_id: str) -> str:
    return PHONE_CID if set_id == "phone_street_marked" else f"gt:{set_id}"


def lenient(gt: list, dets: list) -> tuple[int, int]:
    """(missed true potholes, false detections); boxes normalised [x0, y0, x1, y1] (aspect does not matter for the
    inside test; IoU on normalised boxes is a close stand-in at the 0.3 level)."""
    found = [False] * len(gt)
    ok = [False] * len(dets)
    for k, g in enumerate(gt):
        for j, d in enumerate(dets):
            b = d["b"]
            ix = max(0.0, min(b[2], g[2]) - max(b[0], g[0])) * max(0.0, min(b[3], g[3]) - max(b[1], g[1]))
            ad = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
            ag = (g[2] - g[0]) * (g[3] - g[1])
            if ix / (ad + ag - ix + 1e-9) >= 0.3 or ix / ad >= 0.5:
                found[k] = ok[j] = True
    return found.count(False), ok.count(False)


def failures(model: str, set_id: str, results: Path = RESULTS) -> dict[str, dict[str, int]]:
    """name -> {missed, false, gt} for the photos where `model` fails on the ORIGINAL ground truth (memoised until
    the cache files change)."""
    pf = results / "preds" / model / f"{set_id}.json"
    key = ("fail", model, set_id, str(results), pf.stat().st_mtime if pf.is_file() else 0,
           (results / "index.json").stat().st_mtime if (results / "index.json").is_file() else 0)
    if key not in _cache:
        _cache[key] = _failures(model, set_id, results)
    return _cache[key]


def _failures(model: str, set_id: str, results: Path) -> dict[str, dict[str, int]]:
    t, p = testset(set_id, results), preds(model, set_id, results)
    if t is None or p is None:
        return {}
    out = {}
    for it, dets in zip(t["items"], p, strict=True):
        dets = [d for d in dets if d["c"] >= CONF]
        name = Path(it["img"]).name
        if t["kind"] == "negatives":
            miss, fa, n = 0, len(dets), 0
        elif t["kind"] == "frames":
            continue
        else:
            gt = it.get("gt") or []
            miss, fa = lenient(gt, dets)
            n = len(gt)
        if miss or fa:
            out[name] = {"missed": miss, "false": fa, "gt": n}
    return out


def item_of(set_id: str, name: str, results: Path = RESULTS) -> Optional[dict[str, Any]]:
    t = testset(set_id, results)
    return next((it for it in (t or {}).get("items", []) if Path(it["img"]).name == name), None)


def image_dir(set_id: str, results: Path = RESULTS) -> Optional[Path]:
    t = testset(set_id, results)
    return (REPO / t["items"][0]["img"]).parent if t and t["items"] else None


def original_shapes(set_id: str, name: str, results: Path = RESULTS) -> tuple[str, list[dict[str, Any]]]:
    """The current labels of a test photo as editable shapes (polygons when the label has an outline)."""
    it = item_of(set_id, name, results) or {}
    gt, polys = it.get("gt") or [], it.get("gt_poly") or []
    shapes = []
    for k, g in enumerate(gt):
        poly = polys[k] if k < len(polys) else None
        if poly and len(poly) >= 3:
            shapes.append({"type": "polygon", "pts": [[round(x, 5), round(y, 5)] for x, y in poly]})
        else:
            shapes.append({"type": "box", "pts": [[g[0], g[1]], [g[2], g[3]]]})
    return ("potholes" if shapes else "clean"), shapes


def suggestions(set_id: str, name: str, results: Path = RESULTS) -> list[dict[str, Any]]:
    """Every model's detections on this photo (from the cache), with outlines for segmentation models."""
    t = testset(set_id, results)
    if t is None:
        return []
    i = next((k for k, it in enumerate(t["items"]) if Path(it["img"]).name == name), None)
    out = []
    for m in models(results):
        p = preds(m["id"], set_id, results)
        if p is None or i is None:
            continue
        boxes = [{"conf": d["c"], "pts": [[d["b"][0], d["b"][1]], [d["b"][2], d["b"][3]]],
                  **({"poly": d["p"]} if d.get("p") else {})} for d in p[i] if d["c"] >= CONF]
        out.append({"model": m["id"], "label": m["label"], "boxes": boxes})
    return out


def log_event(ev: dict[str, Any], path: Path = EVENTS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **ev}) + "\n")


# ================================================================================================ export
def export_corrections(marks_by_set: dict[str, dict[str, Any]], root: Path = REPO) -> dict[str, Any]:
    """marks_by_set: test-set id -> {image name: row(status, shapes)} (newest mark per photo)."""
    from ml import benchmark as B

    specs = {t["id"]: t for t in B.load_yaml(B.DATASETS)["tests"]}
    rep = {}
    for set_id, marks in marks_by_set.items():
        spec = specs.get(set_id)
        if spec is None or set_id == "phone_street_marked" or not marks:
            continue
        img_dir = root / spec["images"]
        set_dir = img_dir.parent
        excl_f = set_dir / "excluded.txt"
        excluded = set(excl_f.read_text().split()) if excl_f.is_file() else set()
        counts = {"potholes": 0, "clean": 0, "unsure": 0}
        for name, r in marks.items():
            st = r["status"]
            counts[st] += 1
            if spec["kind"] == "negatives":
                (excluded.add if st in ("potholes", "unsure") else excluded.discard)(name)
                continue
            if st == "unsure":
                excluded.add(name)
                continue
            excluded.discard(name)
            lab_dir = root / spec["labels"]
            orig_dir = lab_dir.parent / "labels_original"
            lab = lab_dir / f"{Path(name).stem}.txt"
            orig_dir.mkdir(exist_ok=True)
            backup = orig_dir / lab.name
            if not backup.exists():  # keep the delivered label once (an absent label stays recorded as empty)
                backup.write_text(lab.read_text() if lab.is_file() else "")
            cls = int(spec.get("pothole_ids", [0])[0])
            pids = {int(i) for i in spec.get("pothole_ids", [0])}
            keep = [ln for ln in (lab.read_text().splitlines() if lab.is_file() else [])
                    if ln.split() and int(float(ln.split()[0])) not in pids]  # cracks / manholes stay as they were
            lines = list(keep)
            for s in json.loads(r["shapes"]) if st == "potholes" else []:
                if s["type"] == "box":
                    (x0, y0), (x1, y1) = s["pts"]
                    x0, x1, y0, y1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
                    lines.append(f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
                else:
                    lines.append(f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in s["pts"]))
            lab.write_text("\n".join(lines) + ("\n" if lines else ""))
        if excluded:
            excl_f.write_text("\n".join(sorted(excluded)) + "\n")
        elif excl_f.exists():
            excl_f.unlink()
        rep[set_id] = {**counts, "excluded_total": len(excluded)}
    return rep
