"""tools/dataset_review.py workflow (D22): split work, second look, locked test, exports; derive --exclude."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dataset_review", ROOT / "tools" / "dataset_review.py")
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)


def _idx(n_train=50, n_test=60):
    idx = {f"bh_{i:03d}": {"split": "train", "source": "bh", "file": f"bh_{i:03d}.jpg", "n": 1,
                           "classes": {"pothole": 1}} for i in range(n_train)}
    idx.update({f"rddin_{i:03d}": {"split": "test", "source": "rddin", "file": f"rddin_{i:03d}.jpg", "n": 0,
                                   "classes": {}} for i in range(n_test)})
    return idx


def test_reviewers_get_disjoint_batches_and_filters_apply(tmp_path):
    idx = _idx()
    with dr.db(tmp_path / "r.sqlite") as con:
        a = dr.next_item(con, idx, "asha", {}, now=1000.0)
        b = dr.next_item(con, idx, "ravi", {}, now=1000.0)
        held = lambda who: {r["image"] for r in con.execute("SELECT image FROM claims WHERE reviewer=?", (who,))}
        assert a and b and not (held("asha") & held("ravi"))
        assert len(held("asha")) == dr.BATCH
        t = dr.next_item(con, idx, "mina", {"split": "test"}, now=1000.0)
        assert idx[t["image"]]["split"] == "test"
        neg = dr.next_item(con, idx, "joe", {"cls": "__none__"}, now=1000.0)
        assert idx[neg["image"]]["n"] == 0


def test_reject_needs_a_second_reviewer(tmp_path):
    idx = _idx()
    with dr.db(tmp_path / "r.sqlite") as con:
        with pytest.raises(ValueError):
            dr.decide(con, idx, "asha", "bh_001", "reject", [], "")  # reason required
        r = dr.decide(con, idx, "asha", "bh_001", "reject", ["bad_mask"], "")
        assert r["status"] == "second_look"
        assert dr.decide(con, idx, "asha", "bh_001", "reject", ["not_road"], "")["round"] == 1  # own edit, still r1
        nxt = dr.next_item(con, idx, "ravi", {}, now=1000.0)
        assert nxt["image"] == "bh_001" and nxt["round"] == 2  # second looks are served first
        assert dr.next_item(con, idx, "asha", {}, now=1000.0)["image"] != "bh_001"  # never your own reject
        assert dr.decide(con, idx, "ravi", "bh_001", "reject", ["bad_mask"], "")["status"] == "rejected"
        with pytest.raises(PermissionError):
            dr.decide(con, idx, "mina", "bh_001", "keep", [], "")
        dr.decide(con, idx, "asha", "bh_002", "reject", ["blurry_dark"], "")
        assert dr.decide(con, idx, "ravi", "bh_002", "keep", [], "")["status"] == "keep"  # overruled: data stays
        assert dr.decide(con, idx, "asha", "bh_003", "fix", ["missing_labels"], "")["status"] == "fix"


def test_export_keeps_test_rejects_pending(tmp_path):
    idx = _idx()
    with dr.db(tmp_path / "r.sqlite") as con:
        for img in ("bh_005", "rddin_001"):
            dr.decide(con, idx, "asha", img, "reject", ["not_road"], "")
            dr.decide(con, idx, "ravi", img, "reject", ["not_road"], "")
        dr.decide(con, idx, "asha", "bh_006", "fix", ["bad_mask"], "")
        res = dr.export(con, idx, tmp_path / "out")
    assert res["rejected_trainval"] == 1 and res["rejected_test_pending"] == 1 and res["needs_fix"] == 1
    tv = (tmp_path / "out" / "rejected_trainval.txt").read_text()
    assert "bh_005" in tv and "rddin_001" not in tv
    assert res["decisions"] == 5


def test_session_cookie_is_signed():
    tok = dr.sign("asha", "k")
    assert dr.verify(tok, "k") == "asha"
    assert dr.verify("ravi|" + tok.split("|")[1], "k") is None and dr.verify(tok, "other") is None
    with pytest.raises(ValueError):
        dr.norm_name("<script>")


def test_derive_exclusion_never_touches_test(tmp_path):
    from dataset_tools.derive_dataset import Img, apply_exclusions, read_stem_list

    f = tmp_path / "rej.txt"
    f.write_text("# header\nbh_1  # bad_mask\nrddin_9.jpg\nmissing_x\n")
    stems = read_stem_list([f])
    assert stems == {"bh_1", "rddin_9", "missing_x"}
    imgs = [Img("bh_1", Path("a"), Path("a"), "train"), Img("bh_2", Path("b"), Path("b"), "val"),
            Img("rddin_9", Path("c"), Path("c"), "test")]
    kept, info = apply_exclusions(imgs, stems)
    assert [i.stem for i in kept] == ["bh_2", "rddin_9"]
    assert info == {"excluded": 1, "test_kept": 1, "not_found": 1}
