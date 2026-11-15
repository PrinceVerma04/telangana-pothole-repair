from shapely.geometry import box

from geo.jurisdiction import resolve
from geo.tests.synth import ll, seg

GHMC, NHAI, RNB = 1, 2, 3


def _poly(x0, y0, x1, y1):
    (la0, lo0), (la1, lo1) = ll(x0, y0), ll(x1, y1)
    return box(lo0, la0, lo1, la1)


ADMIN = [
    (10, GHMC, _poly(-1000, -1000, 1000, 1000)),  # zone
    (11, GHMC, _poly(-500, -500, 500, 500)),  # circle
    (12, GHMC, _poly(-100, -100, 100, 100)),  # ward (most specific)
    (20, RNB, _poly(-2000, -2000, 2000, 2000)),  # R&B division (bigger)
]
CONTRACTS = [
    (100, GHMC, _poly(-300, -300, 300, 300)),  # CRMP package
    (101, GHMC, _poly(-3000, -3000, 3000, 3000)),
    (200, RNB, _poly(-50, -50, 50, 50)),
]


def test_owned_segment_with_contract_and_unit_uses_segment_fields():
    s = seg(1, 1, [(0, 0), (10, 0)], owner=GHMC, contract=555, unit=11)
    r = resolve(s, *ll(5, 0), ADMIN, CONTRACTS)
    assert (r.agency_id, r.contract_id, r.admin_unit_id, r.queued) == (GHMC, 555, 11, False)


def test_owned_segment_resolves_contract_and_most_specific_unit_from_polygons():
    s = seg(1, 1, [(0, 0), (10, 0)], owner=GHMC)
    r = resolve(s, *ll(5, 0), ADMIN, CONTRACTS)
    assert (r.agency_id, r.contract_id, r.admin_unit_id, r.queued) == (GHMC, 100, 12, False)
    assert "contract polygon" in r.reason
    # outside the CRMP package -> the wider contract; outside the ward -> the circle
    r2 = resolve(s, *ll(400, 0), ADMIN, CONTRACTS)
    assert (r2.contract_id, r2.admin_unit_id) == (101, 11)


def test_owner_agency_units_and_contracts_preferred():
    s = seg(1, 1, [(0, 0), (10, 0)], owner=RNB)
    r = resolve(s, *ll(5, 0), ADMIN, CONTRACTS)
    assert (r.agency_id, r.contract_id, r.admin_unit_id) == (RNB, 200, 20)
    nh = seg(2, 2, [(0, 0), (10, 0)], owner=NHAI)
    r = resolve(nh, *ll(5, 0), ADMIN, CONTRACTS)
    assert (r.agency_id, r.contract_id, r.admin_unit_id, r.queued) == (NHAI, None, 12, False)
    assert "departmental" in r.reason


def test_unowned_segment_is_queued_with_admin_unit_for_routing():
    s = seg(3, 3, [(0, 0), (10, 0)], owner=None)
    r = resolve(s, *ll(5, 0), ADMIN, CONTRACTS)
    assert r.queued and r.agency_id is None and r.contract_id is None
    assert r.admin_unit_id == 12
    assert "no owner" in r.reason
    fb = resolve(s, *ll(5, 0), ADMIN, CONTRACTS, fallback_admin_agency=True)
    assert (fb.agency_id, fb.queued, fb.admin_unit_id) == (GHMC, False, 12)


def test_no_segment_and_outside_everything_is_queued():
    r = resolve(None, *ll(5, 0), ADMIN, CONTRACTS)
    assert r.queued and r.admin_unit_id == 12 and "no matched road segment" in r.reason
    r = resolve(None, *ll(9000, 9000), ADMIN, CONTRACTS)
    assert r.queued and r.admin_unit_id is None and "outside every admin unit" in r.reason
