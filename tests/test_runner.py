"""evalharness.runner: case selection and the exemplar hold-out.

A case the pack lists as an exemplar sits in every prompt, so scoring it
would be leakage. The runner holds exemplar case ids out of every run, even
when ``--ids`` / ``--cases`` names them explicitly.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest
import yaml

from engine.packloader import load_pack
from evalharness.runner import run, select_cases
from scripts.join_gold_labels import GOLD_COLUMNS
from tests.conftest import PACKS, REPORT, FakeOllama, model_output_for_report

IDS = ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003"]
EXEMPLAR = IDS[1]


def _gold_row(case_id: str) -> dict[str, object]:
    return {"case_id": case_id, "gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7, "grade_group": 2,
            "pT": "pT2c", "pN": "pN0", "grade_group_source": "derived_from_gleason", "grade_group_printed": False,
            "complete": True}


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A copy of the real packs with one exemplar wired into the prostate pack
    (taken from case TCGA-AA-0002), three staged reports and gold for all three."""
    packs = tmp_path / "packs"
    shutil.copytree(PACKS, packs)
    pack_dir = packs / "prostate_pathology"
    ex = pack_dir / "exemplars"
    (ex / f"{EXEMPLAR}.txt").write_text(REPORT, encoding="utf-8")
    (ex / f"{EXEMPLAR}.json").write_text(json.dumps(model_output_for_report(load_pack(pack_dir))), encoding="utf-8")
    data = yaml.safe_load((pack_dir / "pack.yaml").read_text(encoding="utf-8"))
    data["exemplars"] = [{"case_id": EXEMPLAR, "source": f"exemplars/{EXEMPLAR}.txt",
                          "output": f"exemplars/{EXEMPLAR}.json"}]
    (pack_dir / "pack.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    reports = tmp_path / "reports"
    reports.mkdir()
    for case_id in IDS:
        (reports / f"{case_id}.txt").write_text(REPORT.replace("SURGICAL PATHOLOGY REPORT", f"REPORT {case_id}"),
                                                 encoding="utf-8")
    gold = tmp_path / "gold.csv"
    with gold.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_COLUMNS)
        w.writeheader()
        w.writerows(_gold_row(c) for c in IDS)
    monkeypatch.setenv("CEE_RUNS_DIR", str(tmp_path / "runs"))
    return {"pack": pack_dir, "reports": reports, "gold": gold}


def test_pack_exposes_exemplar_case_ids(world):
    pack = load_pack(world["pack"])
    assert pack.exemplar_ids == [EXEMPLAR]
    assert load_pack(PACKS / "prostate_pathology").exemplar_ids == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-ZG-A8QW"]


def test_select_cases_drops_excluded_ids(world):
    gold = {c: {} for c in IDS}
    assert select_cases(gold, world["reports"], None, None) == IDS
    assert select_cases(gold, world["reports"], None, None, exclude={EXEMPLAR}) == [IDS[0], IDS[2]]
    assert select_cases(gold, world["reports"], [EXEMPLAR, IDS[2]], None, exclude={EXEMPLAR}) == [IDS[2]]
    assert select_cases(gold, world["reports"], None, 1, exclude={EXEMPLAR}) == [IDS[0]]


@pytest.mark.parametrize("ids, expected", [
    (None, [IDS[0], IDS[2]]),                       # the whole gold set: exemplar silently held out
    ([EXEMPLAR, IDS[0], IDS[2]], [IDS[0], IDS[2]]),  # --ids / --cases naming it explicitly: still held out
    ([IDS[2]], [IDS[2]]),
])
def test_exemplar_case_is_never_run_or_scored(world, monkeypatch, capsys, ids, expected):
    pack = load_pack(world["pack"])
    fake = FakeOllama(model_output_for_report(pack)).install(monkeypatch)
    report, out_dir = run(pack, model="m", reports_dir=world["reports"], ids=ids, gold_path=str(world["gold"]))

    ran = [json.loads(line)["case_id"] for line in (out_dir / "predictions.jsonl").read_text().splitlines()]
    assert ran == expected
    assert report.n_cases == len(expected)
    assert all(f"REPORT {EXEMPLAR}" not in req["prompt"].split("## Document")[-1] for req in fake.requests)
    assert all(f"REPORT {EXEMPLAR}" not in req["prompt"] for req in fake.requests)   # never sent as the document
    assert any(f"held out of the run (1): {EXEMPLAR}" in n for n in report.notes)
    err = capsys.readouterr().err
    if ids is None or EXEMPLAR in ids:
        assert f"holding out 1 exemplar case(s)" in err and EXEMPLAR in err
    else:
        assert "holding out" not in err
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["n_cases"] == len(expected)


def test_run_with_only_exemplar_cases_requested_stops(world, monkeypatch):
    pack = load_pack(world["pack"])
    FakeOllama(model_output_for_report(pack)).install(monkeypatch)
    with pytest.raises(SystemExit, match="1 requested case\\(s\\) are pack exemplars"):
        run(pack, model="m", reports_dir=world["reports"], ids=[EXEMPLAR], gold_path=str(world["gold"]))
