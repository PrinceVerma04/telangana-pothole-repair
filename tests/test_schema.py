"""WP0 acceptance: edge payloads validate against JSON Schema AND pydantic DTOs; contracts load."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from measure.types import CalibCert, FrameMeasurement, PassMeasurement, PotholeFused, PotholeStatus, Severity
from repair.types import RepairPlan, SpecCode
from server.app.schemas import ObservationIn, TripFixBatch, pothole_can, work_order_can
from server.app.schemas.states import POTHOLE_TRANSITIONS, WORK_ORDER_TRANSITIONS, WorkOrderStatus

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "edge" / "schema"
EX = SCHEMA / "examples"


def _schema(name: str) -> dict:
    s = json.loads((SCHEMA / name).read_text())
    jsonschema.Draft202012Validator.check_schema(s)
    return s


OBS = _schema("observation.schema.json")
FIX = _schema("trip_fix.schema.json")


@pytest.mark.parametrize("name", ["observation_edge.json", "observation_jolt.json"])
def test_observation_examples(name: str) -> None:
    d = json.loads((EX / name).read_text())
    jsonschema.validate(d, OBS, format_checker=jsonschema.FormatChecker())
    ObservationIn.from_dict(d)


def test_trip_fix_example() -> None:
    d = json.loads((EX / "trip_fix.json").read_text())
    jsonschema.validate(d, FIX, format_checker=jsonschema.FormatChecker())
    TripFixBatch.from_dict(d)


def test_schema_and_dto_agree_on_rejects() -> None:
    good = json.loads((EX / "observation_edge.json").read_text())
    bad_cases = [
        {**good, "source": "drone"},
        {**good, "gps": {**good["gps"], "lat": 51.5}},  # London is not in India
        {**good, "detections": [{**good["detections"][0], "cls": "car"}]},
        {**good, "unexpected": 1},
    ]
    for bad in bad_cases:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bad, OBS)
        with pytest.raises(ValueError):
            ObservationIn.from_dict(bad)


def test_jolt_cannot_carry_detections() -> None:
    d = json.loads((EX / "observation_jolt.json").read_text())
    d["detections"] = json.loads((EX / "observation_edge.json").read_text())["detections"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(d, OBS)


def test_edge_obs_requires_frame() -> None:
    d = json.loads((EX / "observation_edge.json").read_text())
    d["frame"] = None
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(d, OBS)


@pytest.mark.parametrize("p", sorted((ROOT / "configs" / "cameras").glob("*.json")))
def test_camera_certs(p: Path) -> None:
    c = CalibCert.load(p)
    assert not c.certified, "demo certs must never be billing-grade"
    assert c.fx > 0 and c.img_w > 0 and 0.5 < c.h_true_m < 4.0


def test_configs_parse_and_classes_consistent() -> None:
    cfgs = {p.name: yaml.safe_load(p.read_text()) for p in (ROOT / "configs").rglob("*.yaml")}
    names = cfgs["classes.yaml"]["names"]
    canon = [names[i] for i in sorted(names)]
    # training datasets keep an ordered subset of the vocabulary (D15); the original 7-class build keeps all of it
    assert cfgs["india_pothole_seg7.yaml"]["names"] == names
    train_names = [cfgs["india_pothole_seg.yaml"]["names"][i] for i in sorted(cfgs["india_pothole_seg.yaml"]["names"])]
    assert train_names and set(train_names) <= set(canon)
    assert [canon.index(n) for n in train_names] == sorted(canon.index(n) for n in train_names)
    enum = OBS["properties"]["detections"]["items"]["properties"]["cls"]["enum"]
    assert enum == [names[i] for i in sorted(names)]
    w = cfgs["priority.default.yaml"]["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_measurement_roundtrip() -> None:
    fm = FrameMeasurement(depth_p95_mm=42.0, area_m2=0.2, volume_l=5.1, ground_xz=(0.3, 6.2), quality=0.8)
    assert FrameMeasurement.from_json(fm.to_json()) == fm
    pm = PassMeasurement(depth_p95_mm=40.0, n_frames=3, n_frames_total=3)
    assert PassMeasurement.from_dict(pm.to_dict()) == pm
    pf = PotholeFused(water_filled=True, severity=Severity.medium)
    assert pf.depth_quality.value == "unknown_water"


def test_repair_plan_contract() -> None:
    rp = RepairPlan(l_cut_m=0.75, w_cut_m=0.65, a_cut_m2=0.4875, d_design_mm=60, layered=False, d_patch_mm=60,
                    v_patch_l=29.25, mix_kg=72.1, mix_kg_sigma=9.0, bag_kg=25, bags=3, bags_low=3, bags_high=4,
                    emulsion_kg=0.146, sand_kg=0.49, labour_h=2.2, method="Cold mix", spec_code=SpecCode.IRC116_COLDMIX,
                    est_cost_inr=1406)
    assert RepairPlan.from_json(rp.to_json()) == rp


def test_state_machines_closed_and_sane() -> None:
    for frm, tos in POTHOLE_TRANSITIONS.items():
        assert frm in PotholeStatus and all(t in PotholeStatus for t in tos)
    assert pothole_can(PotholeStatus.candidate, PotholeStatus.confirmed)
    assert not pothole_can(PotholeStatus.candidate, PotholeStatus.verified_closed)
    assert pothole_can(PotholeStatus.in_progress, PotholeStatus.rejected)
    # every pothole state reachable from candidate
    seen, todo = {PotholeStatus.candidate}, [PotholeStatus.candidate]
    while todo:
        for t in POTHOLE_TRANSITIONS.get(todo.pop(), ()):
            if t not in seen:
                seen.add(t)
                todo.append(t)
    assert seen == set(PotholeStatus)
    assert work_order_can(WorkOrderStatus.issued, WorkOrderStatus.acknowledged)
    assert not work_order_can(WorkOrderStatus.issued, WorkOrderStatus.verified_closed)
    assert set(WORK_ORDER_TRANSITIONS) == set(WorkOrderStatus)
