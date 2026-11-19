"""Keep test images OUT of the training dataset folders (D43): every test split lives under test_datasets/<name>/.

A dataset build (build_base_dataset.py, derive_dataset.py) still writes images/test + labels/test first; this module
then moves them to test_datasets/<name>/images + labels and points the dataset's data.yaml `test:` there (absolute
paths; Ultralytics accepts a list). Training folders therefore hold only train + val.

    python dataset_tools/test_split.py datasets/base_nagar_netra_pothole            # base layout (one test folder)
    python dataset_tools/test_split.py datasets/earlier_india_pothole_v2 --earlier  # 4 folders by file prefix
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST_ROOT = REPO / "test_datasets"
BASE_TEST = "nagar_netra_test"
# earlier v2 test split: file prefix -> test folder (the four sources of that split)
EARLIER_TESTS = {"rddin_": "rdd_india_locked", "bh_": "bharatpothole_dashcam", "ws_": "web_pothole_photos",
                 "ey_": "eyantra_clean_roads"}


def _set_yaml_test(data_yaml: Path, dirs: list[Path]) -> None:
    """Replace the `test:` line of a data.yaml (kept as text, so comments survive) with absolute image folders."""
    val = dirs[0].as_posix() if len(dirs) == 1 else "[" + ", ".join(d.as_posix() for d in dirs) + "]"
    text = data_yaml.read_text(encoding="utf-8")
    line = f"test: {val}  # D43: test images live in test_datasets/, never inside the training folder"
    text, n = re.subn(r"(?m)^test:.*$", line, text)
    if not n:
        text = text.rstrip("\n") + "\n" + line + "\n"
    data_yaml.write_text(text, encoding="utf-8")


def separate_test_split(dataset: Path, groups: dict[str, str] | str, test_root: Path = TEST_ROOT,
                        overwrite: bool = True) -> dict[str, int]:
    """Move <dataset>/images/test (+ labels/test) to <test_root>/<name>/{images,labels}; rewrite data.yaml `test:`.

    ``groups``: one folder name for the whole split, or {file prefix: folder name} (every image must match a prefix).
    An existing target folder's images/ and labels/ are replaced (``overwrite``), so a rebuild does not mix versions.
    """
    dataset = Path(dataset)
    img_dir, lab_dir = dataset / "images" / "test", dataset / "labels" / "test"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"{img_dir} missing (already separated?)")
    mapping = {"": groups} if isinstance(groups, str) else dict(groups)
    dest: dict[Path, str] = {}
    for p in sorted(x for x in img_dir.iterdir() if x.is_file()):
        name = next((n for pre, n in mapping.items() if p.name.startswith(pre)), None)
        if name is None:
            raise ValueError(f"{p.name}: no test folder for this file prefix ({sorted(mapping)})")
        dest[p] = name
    targets = sorted(set(dest.values())) or sorted(set(mapping.values()))
    for n in targets:
        for sub in ("images", "labels"):
            d = test_root / n / sub
            if d.exists() and any(d.iterdir()):
                if not overwrite:
                    raise SystemExit(f"{d} is not empty (overwrite=False)")
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for p, n in dest.items():
        shutil.move(str(p), str(test_root / n / "images" / p.name))
        lab = lab_dir / f"{p.stem}.txt"
        if lab.is_file():
            shutil.move(str(lab), str(test_root / n / "labels" / lab.name))
        counts[n] += 1
    if lab_dir.is_dir():
        for f in lab_dir.glob("*.cache"):  # Ultralytics label cache with the old absolute paths
            f.unlink()
        if not any(lab_dir.iterdir()):
            lab_dir.rmdir()
    img_dir.rmdir()
    if (dataset / "data.yaml").is_file():
        _set_yaml_test(dataset / "data.yaml", [(test_root / n / "images").resolve() for n in targets])
    return dict(counts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--earlier", action="store_true", help="split by file prefix into the four earlier test sets")
    ap.add_argument("--name", default=BASE_TEST, help="test folder name (without --earlier)")
    ap.add_argument("--test-root", type=Path, default=TEST_ROOT)
    a = ap.parse_args(argv)
    print(json.dumps(separate_test_split(a.dataset, EARLIER_TESTS if a.earlier else a.name, a.test_root), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
