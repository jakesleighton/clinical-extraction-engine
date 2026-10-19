"""scripts.working_set: seeded random draw, hand picks, validation against staged reports and gold."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evalharness.runner import read_id_file
from scripts.join_gold_labels import GOLD_COLUMNS
from scripts.working_set import draw_random, main, normalize_id
from tests.conftest import PACKS

PACK = str(PACKS / "prostate_pathology")


def _row(case_id: str, **over: object) -> dict[str, object]:
    row: dict[str, object] = {"case_id": case_id, "gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7,
                              "grade_group": 2, "pT": "pT2c", "pN": "pN0", "grade_group_source": "gdc",
                              "grade_group_printed": False, "complete": True}
    row.update(over)
    return row


@pytest.fixture
def staged(tmp_path: Path) -> dict[str, Path]:
    """Six staged reports; gold for five of them, one of those incomplete (no pT)."""
    reports = tmp_path / "reports"
    reports.mkdir()
    ids = [f"TCGA-AA-{i:04d}" for i in range(1, 7)]
    for i, case_id in enumerate(ids):
        (reports / f"{case_id}.txt").write_text(f"report {case_id} " + "x" * (10 * i) + "\n", encoding="utf-8")
    gold = tmp_path / "gold.csv"
    with gold.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_COLUMNS)
        w.writeheader()
        for case_id in ids[:4]:
            w.writerow(_row(case_id))
        w.writerow(_row(ids[4], pT="", complete=False))          # incomplete gold
        # ids[5] has a report but no gold row at all
    return {"reports": reports, "gold": gold, "out": tmp_path / "working_set.txt"}


def _argv(s: dict[str, Path], *rest: str) -> list[str]:
    return ["--pack", PACK, "--reports-dir", str(s["reports"]), "--gold", str(s["gold"]), "--out", str(s["out"]), *rest]


def test_normalize_id():
    assert normalize_id(" TCGA-AA-0001 ") == "TCGA-AA-0001"
    assert normalize_id("TCGA-AA-0001.txt") == "TCGA-AA-0001"
    assert normalize_id("data/tcga_prad/reports/TCGA-AA-0001.txt") == "TCGA-AA-0001"


def test_draw_random_is_seeded_and_sorted():
    pool = [f"C{i:02d}" for i in range(30)]
    a, b = draw_random(pool, 5, seed=0), draw_random(pool, 5, seed=0)
    assert a == b and len(a) == 5 and a == sorted(a) and set(a) <= set(pool)
    assert draw_random(pool, 5, seed=1) != a
    assert draw_random(pool, 100, seed=0) == sorted(pool)


def test_random_draws_only_complete_gold_cases(staged, capsys):
    assert main(_argv(staged, "random", "--n", "3", "--seed", "0")) == 0
    chosen = read_id_file(staged["out"])
    assert len(chosen) == 3 and chosen == sorted(chosen)
    assert set(chosen) <= {f"TCGA-AA-{i:04d}" for i in range(1, 5)}      # not the incomplete or unlabelled ones
    assert "drawn from 4 with complete labels" in capsys.readouterr().err
    # a second draw refuses to clobber the set unless forced
    assert main(_argv(staged, "random", "--n", "2")) == 1
    assert read_id_file(staged["out"]) == chosen
    assert main(_argv(staged, "random", "--n", "2", "--force")) == 0
    assert len(read_id_file(staged["out"])) == 2


def test_pick_keeps_order_and_flags_gold_gaps(staged, capsys):
    assert main(_argv(staged, "pick", "TCGA-AA-0006", "TCGA-AA-0005.txt", "TCGA-AA-0001", "TCGA-AA-0001")) == 0
    assert read_id_file(staged["out"]) == ["TCGA-AA-0006", "TCGA-AA-0005", "TCGA-AA-0001"]
    out = capsys.readouterr()
    assert "no gold labels" in out.out and "gold incomplete: pT" in out.out
    assert "2 case(s) have missing or incomplete gold" in out.err


def test_pick_rejects_unstaged_ids_and_writes_nothing(staged, capsys):
    assert main(_argv(staged, "pick", "TCGA-AA-0001")) == 0
    assert main(_argv(staged, "pick", "TCGA-AA-0002", "TCGA-ZZ-9999")) == 1
    assert "no staged report for: TCGA-ZZ-9999" in capsys.readouterr().err
    assert read_id_file(staged["out"]) == ["TCGA-AA-0001"]


def test_add_remove_show(staged, capsys):
    assert main(_argv(staged, "pick", "TCGA-AA-0001")) == 0
    assert main(_argv(staged, "add", "TCGA-AA-0002", "TCGA-AA-0001", "TCGA-AA-0003")) == 0
    assert read_id_file(staged["out"]) == ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003"]
    assert main(_argv(staged, "remove", "TCGA-AA-0002", "TCGA-AA-0004")) == 0
    assert "not in the working set: TCGA-AA-0004" in capsys.readouterr().err
    assert read_id_file(staged["out"]) == ["TCGA-AA-0001", "TCGA-AA-0003"]
    assert main(_argv(staged, "show")) == 0
    table = capsys.readouterr().out
    assert table.splitlines()[0].split()[:3] == ["case_id", "chars", "gleason_primary"]
    assert "TCGA-AA-0001" in table and "TCGA-AA-0003" in table and "pT2c" in table


def test_show_without_a_set_and_without_reports(tmp_path, staged, capsys):
    assert main(_argv(staged, "show")) == 1
    assert "no working set yet" in capsys.readouterr().err
    empty = {"reports": tmp_path / "nowhere", "gold": staged["gold"], "out": staged["out"]}
    assert main(_argv(empty, "random")) == 1
    assert "no staged reports" in capsys.readouterr().err


def test_random_without_gold_draws_from_staged_reports(staged, capsys):
    staged["gold"] = staged["gold"].with_name("missing.csv")
    assert main(_argv(staged, "random", "--n", "2")) == 0
    assert "warning: gold file not found" in capsys.readouterr().err
    assert len(read_id_file(staged["out"])) == 2


def test_read_id_file_skips_comments(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("# weird cases\nTCGA-AA-0001   # two foci\n\nTCGA-AA-0002\n", encoding="utf-8")
    assert read_id_file(f) == ["TCGA-AA-0001", "TCGA-AA-0002"]


@pytest.fixture
def pack_with_exemplar(tmp_path: Path) -> str:
    """A copy of the real packs whose prostate pack lists TCGA-AA-0002 as an exemplar."""
    import json
    import shutil
    import yaml
    from engine.packloader import load_pack
    from tests.conftest import REPORT, model_output_for_report
    packs = tmp_path / "packs_ex"
    shutil.copytree(PACKS, packs)
    pack_dir = packs / "prostate_pathology"
    (pack_dir / "exemplars" / "TCGA-AA-0002.txt").write_text(REPORT, encoding="utf-8")
    (pack_dir / "exemplars" / "TCGA-AA-0002.json").write_text(
        json.dumps(model_output_for_report(load_pack(pack_dir))), encoding="utf-8")
    data = yaml.safe_load((pack_dir / "pack.yaml").read_text(encoding="utf-8"))
    data["exemplars"] = [{"case_id": "TCGA-AA-0002", "source": "exemplars/TCGA-AA-0002.txt",
                          "output": "exemplars/TCGA-AA-0002.json"}]
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(pack_dir)


def test_random_never_draws_an_exemplar_case_and_show_marks_one(staged, pack_with_exemplar, capsys):
    argv = _argv(staged, "random", "--n", "10")
    argv[1] = pack_with_exemplar
    assert main(argv) == 0
    chosen = read_id_file(staged["out"])
    assert "TCGA-AA-0002" not in chosen and chosen == ["TCGA-AA-0001", "TCGA-AA-0003", "TCGA-AA-0004"]
    assert "exemplar case(s) left out" in capsys.readouterr().err
    argv = _argv(staged, "add", "TCGA-AA-0002")
    argv[1] = pack_with_exemplar
    assert main(argv) == 0
    out = capsys.readouterr()
    assert "pack exemplar: held out of every eval" in out.out
    assert "will be held out of every eval run: TCGA-AA-0002" in out.err
