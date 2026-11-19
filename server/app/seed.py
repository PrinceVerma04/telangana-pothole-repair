"""Demo seed (WP4 + WP10 data): agencies, admin units, CRMP contracts, configs, users, devices, roads, POIs.

Idempotent: running it twice changes nothing (rows are matched by their natural keys). Used by
``tools/seed_demo.py`` (``make seed``) and by the API tests.

Demo users - password for all of them is ``demo1234`` (DEMO ONLY; change or disable before any real use):

=====================  ==============  =======================================================
username               role            scope
=====================  ==============  =======================================================
admin                  super_admin     everything
zonal_khairatabad      zonal_admin     GHMC, Khairatabad zone (GHMC-KHT) and its circles/wards
engineer1              field_engineer  GHMC, Khairatabad circle (GHMC-C17)
contractor1            contractor      contractor of CRMP-P2 (own work orders only)
contractor2            contractor      contractor of CRMP-P7 (own work orders only)
citizen1               citizen         public data + own reports (also: OTP login, code 123456)
rnb_admin              zonal_admin     Roads & Buildings Dept (RNB)
=====================  ==============  =======================================================

Geo inputs (GeoJSON, WGS84) come from WP10: ``datasets/extracted/geo/{road_segments,pois}.geojson`` and
``gis/layers/{ghmc_zones,circles,wards,crmp_packages}.geojson``; ``fixture=True`` (or a missing big extract) uses
``tests/fixtures/geo_small/`` instead. A layer file found in the chosen geo directory wins over ``gis/layers``.
If no geo data exists yet, roads/POIs are skipped and minimal approximate admin units are created so the RBAC
demo (zonal admin, engineer) still works.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from measure.types import CalibCert
from server.app import configstore
from server.app.core import audit
from server.app.core.config import CONFIGS_DIR, PROJECT_ROOT
from server.app.core.security import hash_password
from server.app.core.util import now_utc
from server.app.db import models as m

log = logging.getLogger("gunthameter.seed")

DEMO_PASSWORD = "demo1234"
SEED_ACTOR = "system:seed"
GEO_DIR = PROJECT_ROOT / "datasets" / "extracted" / "geo"
FIXTURE_GEO_DIR = PROJECT_ROOT / "tests" / "fixtures" / "geo_small"
LAYERS_DIR = PROJECT_ROOT / "gis" / "layers"

DEMO_USERS: list[dict[str, Any]] = [
    {"username": "admin", "role": "super_admin", "agency": "GHMC", "phone": "+919000000001",
     "display_name": "Super admin (demo)", "lang": "en"},
    {"username": "zonal_khairatabad", "role": "zonal_admin", "agency": "GHMC", "admin_unit": "GHMC-KHT",
     "phone": "+919000000002", "display_name": "Zonal Commissioner, Khairatabad (demo)", "lang": "te"},
    {"username": "engineer1", "role": "field_engineer", "agency": "GHMC", "admin_unit": "GHMC-C17",
     "phone": "+919000000003", "display_name": "Asst. Engineer, Khairatabad circle (demo)", "lang": "te"},
    {"username": "contractor1", "role": "contractor", "agency": "GHMC", "contract": "CRMP-P2",
     "phone": "+919000000004", "display_name": "CRMP Package 2 site manager (demo)", "lang": "en"},
    {"username": "contractor2", "role": "contractor", "agency": "GHMC", "contract": "CRMP-P7",
     "phone": "+919000000005", "display_name": "CRMP Package 7 site manager (demo)", "lang": "hi"},
    {"username": "citizen1", "role": "citizen", "phone": "+919000000006", "display_name": "Citizen (demo)",
     "lang": "te"},
    {"username": "rnb_admin", "role": "zonal_admin", "agency": "RNB", "phone": "+919000000007",
     "display_name": "Executive Engineer, R&B Hyderabad (demo)", "lang": "en"},
]

DEMO_DEVICES: list[dict[str, Any]] = [
    {"device_id": "tsrtc-bus-ts09ua1234", "kind": "phone", "vehicle_reg": "TS09UA1234", "calib_id": "bus_default"},
    {"device_id": "sim-bus-01", "kind": "sim", "vehicle_reg": "TS10SIM0001", "calib_id": "sim_bus"},
    {"device_id": "sim-bus-02", "kind": "sim", "vehicle_reg": "TS10SIM0002", "calib_id": "sim_bus"},
    {"device_id": "sim-bus-03", "kind": "sim", "vehicle_reg": "TS10SIM0003", "calib_id": "sim_bus"},
    {"device_id": "demo-phone-01", "kind": "phone", "vehicle_reg": None, "calib_id": "phone_demo"},
    {"device_id": "demo-car-01", "kind": "phone", "vehicle_reg": "TS07DEMO01", "calib_id": "car_default"},
]

# Used only when no admin-unit layer exists yet (approximate boxes around Khairatabad / Ameerpet-Punjagutta).
FALLBACK_ADMIN_UNITS: dict[str, Any] = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"code": "GHMC-KHT", "kind": "zone", "name_en": "Khairatabad",
                                       "name_te": "ఖైరతాబాద్", "name_hi": "खैरताबाद", "agency": "GHMC",
                                       "official": False, "valid_from": "2026-11-01"},
     "geometry": {"type": "Polygon", "coordinates": [[[78.38, 17.37], [78.50, 17.37], [78.50, 17.46],
                                                      [78.38, 17.46], [78.38, 17.37]]]}},
    {"type": "Feature", "properties": {"code": "GHMC-C17", "kind": "circle", "name_en": "Khairatabad",
                                       "name_te": "ఖైరతాబాద్", "name_hi": "खैरताबाद", "agency": "GHMC",
                                       "parent_code": "GHMC-KHT", "official": False, "valid_from": "2026-11-01"},
     "geometry": {"type": "Polygon", "coordinates": [[[78.435, 17.40], [78.475, 17.40], [78.475, 17.445],
                                                      [78.435, 17.445], [78.435, 17.40]]]}},
]}


# =========================================================================== helpers
def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _read_geojson(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:  # untrusted data: parsed only
        d = json.load(fh)
    if not isinstance(d, dict) or d.get("type") != "FeatureCollection":
        log.warning("%s is not a FeatureCollection; ignored", path)
        return None
    return d


def resolve_geo_dir(fixture: bool = False, geo_dir: Optional[Path] = None) -> Optional[Path]:
    if geo_dir is not None:
        return geo_dir if geo_dir.is_dir() else None
    if fixture:
        return FIXTURE_GEO_DIR if FIXTURE_GEO_DIR.is_dir() else None
    if (GEO_DIR / "road_segments.geojson").is_file():
        return GEO_DIR
    if FIXTURE_GEO_DIR.is_dir():
        return FIXTURE_GEO_DIR
    return None


def find_layer(name: str, geo_dir: Optional[Path]) -> Optional[Path]:
    for d in ([geo_dir] if geo_dir else []) + [geo_dir / "layers" if geo_dir else None, LAYERS_DIR]:
        if d is not None and (d / name).is_file():
            return d / name
    return None


def _agency_ids(db: Session) -> dict[str, int]:
    return {a.code: a.id for a in db.scalars(select(m.Agency))}


# =========================================================================== reference data
def seed_agencies(db: Session) -> int:
    cfg = _read_yaml(CONFIGS_DIR / "agencies.yaml")
    n = 0
    for a in cfg.get("agencies", []):
        row = db.scalars(select(m.Agency).where(m.Agency.code == a["code"])).first()
        if row is None:
            row = m.Agency(code=a["code"])
            db.add(row)
            n += 1
        row.name_en, row.name_te, row.name_hi, row.kind = a["name_en"], a.get("name_te"), a.get("name_hi"), a["kind"]
    db.flush()
    return n


def _tag_agency(fc: dict[str, Any], codes: Iterable[str], default: str = "GHMC") -> dict[str, Any]:
    codes = set(codes)
    for f in fc.get("features", []):
        props = f.setdefault("properties", {})
        if not props.get("agency"):
            prefix = str(props.get("code", "")).split("-")[0].upper()
            props["agency"] = prefix if prefix in codes else default
    return fc


def seed_admin_units(db: Session, geo_dir: Optional[Path], use_layers: bool = True) -> dict[str, Any]:
    from shapely.geometry import shape

    from server.app.routers.admin import upsert_admin_units

    codes = _agency_ids(db)
    ghmc = db.scalars(select(m.Agency).where(m.Agency.code == "GHMC")).first()
    out: dict[str, Any] = {"files": [], "created": 0, "updated": 0, "errors": 0}
    for name in ("ghmc_zones.geojson", "circles.geojson", "wards.geojson") if use_layers else ():
        path = find_layer(name, geo_dir)
        fc = _read_geojson(path)
        if fc is None:
            continue
        before = {(u.agency_id, u.kind, u.code): u.geom_wkt for u in db.scalars(select(m.AdminUnit))}
        changed = {"type": "FeatureCollection", "features": []}
        for f in _tag_agency(fc, codes)["features"]:
            props = f.get("properties") or {}
            key = (codes.get(props.get("agency")), props.get("kind"), str(props.get("code")))
            try:
                same = key in before and shape(f.get("geometry") or {}).wkt == before[key]
            except Exception:
                same = False
            if not same:  # new unit or changed boundary (unchanged ones are skipped: no audit noise)
                changed["features"].append(f)
        res = upsert_admin_units(db, changed, default_agency=ghmc, actor=SEED_ACTOR) if changed["features"] \
            else {"created": 0, "updated": 0, "errors": []}
        out["files"].append(str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path))
        out["created"] += res["created"]
        out["updated"] += res["updated"]
        out["errors"] += len(res["errors"])
        for e in res["errors"][:5]:
            log.warning("admin unit %s: %s", e.get("code"), e.get("error"))
    if db.scalar(select(m.AdminUnit.id).limit(1)) is None:
        res = upsert_admin_units(db, FALLBACK_ADMIN_UNITS, default_agency=ghmc, actor=SEED_ACTOR)
        out["fallback"] = True
        out["created"] += res["created"]
    return out


def seed_contracts(db: Session, geo_dir: Optional[Path], use_layers: bool = True) -> dict[str, Any]:
    """Contractors + CRMP contracts from ``crmp_packages.geojson`` (with geometry) or agencies.yaml (without)."""
    from shapely.geometry import shape

    cfg = _read_yaml(CONFIGS_DIR / "agencies.yaml")
    sla = cfg.get("demo_sla_rules", {})
    agencies = _agency_ids(db)
    rows: list[dict[str, Any]] = []
    path = find_layer("crmp_packages.geojson", geo_dir) if use_layers else None
    fc = _read_geojson(path)
    if fc is not None:
        for f in fc["features"]:
            p = dict(f.get("properties") or {})
            try:
                p["geom_wkt"] = shape(f["geometry"]).wkt if f.get("geometry") else None
            except Exception:
                p["geom_wkt"] = None
            rows.append(p)
    yaml_pk = {p["code"]: p for p in cfg.get("crmp_packages", [])}
    for code, p in yaml_pk.items():  # packages missing from the layer still exist (without geometry)
        if not any(r.get("code") == code for r in rows):
            rows.append(dict(p))
    n_new = 0
    for p in rows:
        if not p.get("code"):
            continue
        cname = p.get("contractor") or yaml_pk.get(p["code"], {}).get("contractor") or f"{p['code']} contractor"
        agency_id = agencies.get(str(p.get("agency") or "GHMC").upper())
        ctr = db.scalars(select(m.Contractor).where(m.Contractor.name == cname)).first()
        if ctr is None:
            ctr = m.Contractor(name=cname, agency_id=agency_id, gstin=None,
                               phone=None)
            db.add(ctr)
            db.flush()
        row = db.scalars(select(m.Contract).where(m.Contract.code == p["code"])).first()
        if row is None:
            row = m.Contract(code=p["code"])
            db.add(row)
            n_new += 1
        row.package = f"{p['code']} - GHMC CRMP (demo, approximate; zone hint: {p.get('zone_hint', 'n/a')})"
        row.agency_id, row.contractor_id = agency_id, ctr.id
        row.dlp_months = int(p.get("dlp_months") or 60)
        row.sla_rules = sla
        row.start_date, row.end_date = date(2026, 4, 1), date(2031, 3, 31)
        if p.get("geom_wkt"):
            row.geom_wkt = p["geom_wkt"]
            row.refresh_bbox()
    db.flush()
    return {"file": str(path) if path else None, "created": n_new, "total": len(rows)}


def seed_configs(db: Session) -> dict[str, Any]:
    pc = configstore.load_yaml(configstore.DEFAULT_PRIORITY_FILE)
    rc = configstore.load_yaml(configstore.DEFAULT_RATE_CARD_FILE)
    out = {"priority_config": 0, "rate_card": 0}
    ag = _agency_ids(db)
    PC, RC = m.PriorityConfig, m.RateCard
    for agency_id in (None, ag.get("GHMC")):
        own = PC.agency_id == agency_id if agency_id is not None else PC.agency_id.is_(None)
        if db.scalar(select(PC.id).where(own).limit(1)) is None:
            configstore.save_priority_config(db, agency_id, pc, actor=SEED_ACTOR, reason="seed from configs/")
            out["priority_config"] += 1
    for agency_id in (None, ag.get(str(rc.get("agency", "GHMC")))):
        own = RC.agency_id == agency_id if agency_id is not None else RC.agency_id.is_(None)
        if db.scalar(select(RC.id).where(own, RC.fy == rc.get("fy")).limit(1)) is None:
            configstore.save_rate_card(db, agency_id, rc, actor=SEED_ACTOR, reason="seed from configs/")
            out["rate_card"] += 1
    return out


def seed_users(db: Session, reset_passwords: bool = False) -> dict[str, Any]:
    ag = _agency_ids(db)
    created, warnings = 0, []
    for u in DEMO_USERS:
        row = db.scalars(select(m.User).where(m.User.username == u["username"])).first()
        if row is None:
            row = m.User(username=u["username"], active=True, created_at=now_utc())
            db.add(row)
            created += 1
        row.role = u["role"]
        row.phone, row.display_name, row.lang = u.get("phone"), u.get("display_name"), u.get("lang")
        row.agency_id = ag.get(u["agency"]) if u.get("agency") else None
        row.admin_unit_id = None
        if u.get("admin_unit"):
            au = db.scalars(select(m.AdminUnit).where(m.AdminUnit.code == u["admin_unit"],
                                                      m.AdminUnit.agency_id == row.agency_id)
                            .order_by(m.AdminUnit.id.desc())).first()
            if au is None:
                warnings.append(f"{u['username']}: admin unit {u['admin_unit']} missing -> agency-wide scope")
            else:
                row.admin_unit_id = au.id
        row.contractor_id = None
        if u.get("contract"):
            c = db.scalars(select(m.Contract).where(m.Contract.code == u["contract"])).first()
            row.contractor_id = c.contractor_id if c else None
            if c is None:
                warnings.append(f"{u['username']}: contract {u['contract']} missing")
        if row.password_hash is None or reset_passwords:
            row.password_hash = hash_password(DEMO_PASSWORD)
    db.flush()
    return {"created": created, "total": len(DEMO_USERS), "password": DEMO_PASSWORD, "warnings": warnings}


def seed_devices(db: Session) -> dict[str, Any]:
    certs = 0
    by_calib: dict[str, m.CalibCertRow] = {}
    for path in sorted((CONFIGS_DIR / "cameras").glob("*.json")):
        cert = CalibCert.load(path)
        row = db.scalars(select(m.CalibCertRow).where(m.CalibCertRow.calib_id == cert.calib_id)).first()
        if row is None:
            row = m.CalibCertRow(calib_id=cert.calib_id, issued_at=now_utc())
            db.add(row)
            certs += 1
        row.json, row.certified = cert.to_dict(), cert.certified
        by_calib[cert.calib_id] = row
    db.flush()
    ghmc = _agency_ids(db).get("GHMC")
    devs = 0
    for d in DEMO_DEVICES:
        row = db.scalars(select(m.Device).where(m.Device.device_id == d["device_id"])).first()
        if row is None:
            row = m.Device(device_id=d["device_id"], kind=d["kind"], agency_id=ghmc, vehicle_reg=d["vehicle_reg"],
                           created_at=now_utc())
            db.add(row)
            devs += 1
        cert = by_calib.get(d["calib_id"])
        row.calib_cert_id = cert.id if cert else None
    db.flush()
    return {"calib_certs": len(by_calib), "calib_certs_created": certs, "devices_created": devs,
            "devices": [d["device_id"] for d in DEMO_DEVICES]}


def seed_model_registry(db: Session) -> int:
    n = 0
    for name, rel, task in (("pothole_seg11n", "models/base_model/pothole_seg11n.pt", "seg"),):  # D50: the model in use
        path = PROJECT_ROOT / rel
        if not path.is_file():
            continue
        if db.scalars(select(m.ModelRegistry).where(m.ModelRegistry.name == name)).first() is not None:
            continue
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        db.add(m.ModelRegistry(name=name, version="2026-11-19", task=task, format="pt", sha256=h.hexdigest(),
                               metrics={"mean_box_ap50_17_sets": 0.529, "eyantra_false_alarm_rate": 0.002,
                                        "phone_outlines_box_ap50": 0.274,
                                        "note": "model A, YOLO11n-seg, merged data + hard aug (D48), in use (D50)"},
                               path=rel,
                               active=False, created_at=now_utc()))
        n += 1
    db.flush()
    return n


# =========================================================================== roads & POIs
def _lookup_admin_units(db: Session) -> dict[str, int]:
    out: dict[str, int] = {}
    for au in db.scalars(select(m.AdminUnit).order_by(m.AdminUnit.id)):
        out[au.code] = au.id
    return out


def seed_roads(db: Session, geo_dir: Path, chunk: int = 5000) -> dict[str, Any]:
    import shapely

    fc = _read_geojson(geo_dir / "road_segments.geojson")
    if fc is None:
        return {"skipped": "no road_segments.geojson"}
    agencies = _agency_ids(db)
    units = _lookup_admin_units(db)
    contracts = {c.code: c.id for c in db.scalars(select(m.Contract))}
    have = {(int(w), int(i)) for w, i in db.execute(select(m.RoadSegment.osm_way_id, m.RoadSegment.seg_idx))}
    rows: list[dict[str, Any]] = []
    bad = 0
    for f in fc["features"]:
        p = f.get("properties") or {}
        g = f.get("geometry") or {}
        try:
            key = (int(p["osm_way_id"]), int(p.get("seg_idx") or 0))
            if key in have:
                continue
            if g.get("type") != "LineString" or len(g.get("coordinates") or []) < 2:
                raise ValueError("LineString with >= 2 points expected")
            coords = g["coordinates"]
            xs, ys = [c[0] for c in coords], [c[1] for c in coords]
            geom = shapely.LineString(coords)
            surface = p.get("surface") if p.get("surface") in ("BT", "CC", "unknown") else "unknown"
            lit = p.get("lit") if p.get("lit") in ("yes", "no") else None
            rows.append({
                "osm_way_id": key[0], "seg_idx": key[1], "name": p.get("name"), "name_te": p.get("name_te"),
                "highway": str(p.get("highway") or "unclassified")[:32], "ref": p.get("ref"),
                "lanes": int(p["lanes"]) if p.get("lanes") not in (None, "") else None,
                "oneway": bool(p.get("oneway", False)), "lit": lit, "surface": surface,
                "geom_wkt": shapely.to_wkt(geom, rounding_precision=7),
                "min_lon": min(xs), "min_lat": min(ys), "max_lon": max(xs), "max_lat": max(ys),
                "length_m": float(p.get("length_m") or 0.0), "bearing_deg": p.get("bearing_deg"),
                "owner_agency_id": agencies.get(p.get("owner_agency")) if p.get("owner_agency") else None,
                "contract_id": contracts.get(p.get("contract_code")) if p.get("contract_code") else None,
                "admin_unit_id": units.get(p.get("admin_unit_code")) if p.get("admin_unit_code") else None,
                "gtfs_trips_per_day": float(p.get("gtfs_trips_per_day") or 0.0),
                "junction_start": bool(p.get("junction_start", False)),
                "junction_end": bool(p.get("junction_end", False)),
            })
            have.add(key)
        except (KeyError, TypeError, ValueError) as e:
            bad += 1
            if bad <= 5:
                log.warning("road segment skipped: %s (%s)", e, p.get("osm_way_id"))
    for i in range(0, len(rows), chunk):
        db.execute(insert(m.RoadSegment), rows[i:i + chunk])
    db.flush()
    # every segment has an owner or a jurisdiction-queue item (WP10 acceptance)
    queued = set(db.scalars(select(m.JurisdictionQueue.road_segment_id).where(
        m.JurisdictionQueue.road_segment_id.is_not(None), m.JurisdictionQueue.pothole_id.is_(None))))
    orphan_ids = [sid for sid in db.scalars(select(m.RoadSegment.id).where(m.RoadSegment.owner_agency_id.is_(None)))
                  if sid not in queued]
    now = now_utc()
    jq = [{"road_segment_id": sid, "reason": "road owner unknown: no ownership rule or override matched",
           "candidates": [], "created_at": now} for sid in orphan_ids]
    for i in range(0, len(jq), chunk):
        db.execute(insert(m.JurisdictionQueue), jq[i:i + chunk])
    db.flush()
    return {"inserted": len(rows), "skipped_invalid": bad, "queued_unowned": len(jq),
            "total": db.scalar(select(func.count()).select_from(m.RoadSegment))}


def seed_pois(db: Session, geo_dir: Path) -> dict[str, Any]:
    fc = _read_geojson(geo_dir / "pois.geojson")
    if fc is None:
        return {"skipped": "no pois.geojson"}
    rows = []
    for f in fc["features"]:
        p, g = f.get("properties") or {}, f.get("geometry") or {}
        if p.get("kind") not in m.POI_KINDS or g.get("type") != "Point":
            continue
        lon, lat = g["coordinates"][:2]
        src = p.get("source") if p.get("source") in m.POI_SOURCES else "osm"
        rows.append({"kind": p["kind"], "name": (p.get("name") or None) and str(p["name"])[:200],
                     "lat": float(lat), "lon": float(lon), "source": src})
    db.execute(delete(m.Poi).where(m.Poi.source.in_(("osm", "gtfs"))))  # file is the source of truth
    for i in range(0, len(rows), 5000):
        db.execute(insert(m.Poi), rows[i:i + 5000])
    db.flush()
    return {"inserted": len(rows)}


# =========================================================================== optional sample potholes
def seed_sample_potholes(db: Session, n: int = 12, seed: int = 7) -> int:
    """Synthetic potholes around Ameerpet/Punjagutta for UI work before the simulator runs (clearly labelled)."""
    import h3

    rng = random.Random(seed)
    existing = db.scalars(select(m.PotholeEvent).where(m.PotholeEvent.reason.like("demo seed%"))).first()
    if existing is not None:
        return 0
    sevs = [("small", 18.0, 0.15), ("medium", 38.0, 0.35), ("large", 72.0, 0.70), ("unknown_water", None, 0.45)]
    statuses = ["confirmed", "confirmed", "candidate", "planned", "work_ordered", "verified_closed"]
    now = now_utc()
    made = 0
    for i in range(n):
        lat, lon = 17.42 + rng.random() * 0.02, 78.44 + rng.random() * 0.02
        sev, d95, w = sevs[i % len(sevs)]
        water = sev == "unknown_water"
        area = round(w * w * 0.8, 3)
        st = statuses[i % len(statuses)]
        p = m.Pothole(lat=lat, lon=lon, h3_r12=h3.latlng_to_cell(lat, lon, 12), status=st, severity=sev,
                      depth_p95_mm=d95, depth_sigma_mm=None if water else 8.0, area_m2=area, area_sigma=0.1 * area,
                      volume_l=None if water else round(area * (d95 or 0) * 0.55, 2), width_m=w, length_m=w * 1.2,
                      water_filled=water, water_filled_ratio=1.0 if water else 0.0, quality=0.7,
                      method="synthetic", best_method="synthetic", priority=round(30 + rng.random() * 60, 1),
                      passes_seen=2, passes_total=3, trips_seen=2, first_seen=now - timedelta(days=3 + i),
                      last_seen=now - timedelta(hours=i), confirmed_at=None if st == "candidate" else now - timedelta(days=2))
        db.add(p)
        db.flush()
        db.add(m.PotholeEvent(pothole_id=p.id, ts=now, from_status=None, to_status=st,
                              reason="demo seed (synthetic sample, not a real detection)"))
        made += 1
    audit.record(db, "pothole", None, "seed_samples", actor=SEED_ACTOR, after={"n": made})
    db.flush()
    return made


# =========================================================================== driver
def seed_all(db: Session, *, fixture: bool = False, geo_dir: Optional[Path] = None, with_geo: bool = True,
             sample_potholes: int = 0, reset_passwords: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {"at": now_utc().isoformat()}
    gdir = resolve_geo_dir(fixture, geo_dir) if with_geo else None
    summary["geo_dir"] = str(gdir) if gdir else None
    summary["agencies_created"] = seed_agencies(db)
    summary["admin_units"] = seed_admin_units(db, gdir, use_layers=with_geo)
    summary["contracts"] = seed_contracts(db, gdir, use_layers=with_geo)
    summary["configs"] = seed_configs(db)
    summary["users"] = seed_users(db, reset_passwords)
    summary["devices"] = seed_devices(db)
    summary["model_registry_created"] = seed_model_registry(db)
    if gdir is not None:
        summary["roads"] = seed_roads(db, gdir)
        summary["pois"] = seed_pois(db, gdir)
    else:
        summary["roads"] = summary["pois"] = {
            "skipped": "no geo data yet (datasets/extracted/geo/ and tests/fixtures/geo_small/ missing; WP10)"}
    if sample_potholes:
        summary["sample_potholes"] = seed_sample_potholes(db, sample_potholes)
    db.commit()
    return summary
