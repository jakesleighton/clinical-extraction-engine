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

from engine.packloader import PackError, load_pack
from evalharness.runner import main, run, select_cases
from scripts.join_gold_labels import GOLD_COLUMNS
from tests.conftest import PACKS, REPORT, FakeOllama, model_output_for_report

IDS = ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0003"]
EXEMPLAR = IDS[1]
VARIANT_EXEMPLAR = "TCGA-AA-0009"      # the opt-in arm's case: no report, no gold, still held out


def _gold_row(case_id: str) -> dict[str, object]:
    return {"case_id": case_id, "gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7, "grade_group": 2,
            "pT": "pT2c", "pN": "pN0", "grade_group_source": "derived_from_gleason", "grade_group_printed": False,
            "pT_printed": True, "pN_printed": True, "complete": True}


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
                          "output": f"exemplars/{EXEMPLAR}.json"},
                         {"case_id": VARIANT_EXEMPLAR, "source": f"exemplars/{EXEMPLAR}.txt",
                          "output": f"exemplars/{EXEMPLAR}.json", "variant": "hedging"}]
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
    assert pack.exemplar_ids == [EXEMPLAR, VARIANT_EXEMPLAR]
    assert load_pack(PACKS / "prostate_pathology").exemplar_ids == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-ZG-A8QW",
                                                                     "TCGA-V1-A9Z7"]


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
    assert any(f"held out of the run (2): {EXEMPLAR}, {VARIANT_EXEMPLAR}" in n for n in report.notes)
    assert all(req["prompt"].count("## Example") == 1 for req in fake.requests)      # the variant stays out
    err = capsys.readouterr().err
    if ids is None or EXEMPLAR in ids:
        assert f"holding out 1 exemplar case(s)" in err and EXEMPLAR in err
    else:
        assert "holding out" not in err
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["n_cases"] == len(expected)
    text = (out_dir / "report.md").read_text()
    for section in ("## Scores", "## Validator", "## Quotes", "## Cost"):
        assert section in text
    assert "## By group" not in text and "## Without the audited gold cases" not in text


def test_run_with_only_exemplar_cases_requested_stops(world, monkeypatch):
    pack = load_pack(world["pack"])
    FakeOllama(model_output_for_report(pack)).install(monkeypatch)
    with pytest.raises(SystemExit, match="1 requested case\\(s\\) are pack exemplars"):
        run(pack, model="m", reports_dir=world["reports"], ids=[EXEMPLAR], gold_path=str(world["gold"]))


def test_groups_gold_notes_and_variant_land_in_the_report(world, monkeypatch, tmp_path):
    pack = load_pack(world["pack"])
    fake = FakeOllama(model_output_for_report(pack)).install(monkeypatch)
    notes_md = tmp_path / "gold_notes.md"
    notes_md.write_text("# notes\n\n| case | field(s) | kind | gold | report says | note |\n|---|---|---|---|---|---|\n"
                        f"| {IDS[2]} | pT, pN | gold_finer | pT2b | \"Stage: T2\" | text prints no substage |\n"
                        "| TCGA-ZZ-9999 | pT | gold_stale | pT2c | x | not in this run |\n", encoding="utf-8")
    groups = {"first": [IDS[0]], "last": [IDS[2]], "both": [IDS[0], IDS[2]]}
    from evalharness.report import read_gold_notes
    report, out_dir = run(pack, model="m", reports_dir=world["reports"], gold_path=str(world["gold"]),
                          groups=groups, gold_notes=read_gold_notes(notes_md), exemplar_variants=["hedging"],
                          run_id="arm")
    assert all(req["prompt"].count("## Example") == 2 for req in fake.requests)      # the hedging arm is in
    assert any("variants on: hedging" in n for n in report.notes)
    assert list(report.groups) == ["first", "last", "both"]
    assert [report.groups[g].n_cases for g in report.groups] == [1, 1, 2]
    assert report.without_audited.n_cases == 1 and report.audited_in_run == [IDS[2]]
    assert report.metrics.overall.micro_f1 == pytest.approx(1.0) == report.without_audited.overall.micro_f1
    d = json.loads((out_dir / "metrics.json").read_text())
    assert d["n_cases"] == 2 and set(d["groups"]) == set(groups) and d["without_audited"]["n_cases"] == 1
    assert d["cost"]["calls"] == 2 and d["cost"]["tokens_in"] == 2 * 1234 and d["cost"]["tokens_out"] == 2 * 567
    assert d["cost"]["num_ctx"] == fake.requests[0]["options"]["num_ctx"] >= 8192 and d["validator"]["issues"] == []
    assert d["quotes"]["located"] == {"exact": sum(1 for _ in range(2) for env in model_output_for_report(pack).values()
                                                   for _q in env["evidence"])}
    text = (out_dir / "report.md").read_text()
    assert "## By group" in text and "| both | 2 |" in text
    assert "## Without the audited gold cases" in text and f"| {IDS[2]} | pT, pN | gold_finer |" in text
    assert "TCGA-ZZ-9999" not in text                       # an audited case outside the run is not listed
    assert "## Cost" in text and "| 2 | 0 | 2,468 / 1234 |" in text
    with pytest.raises(PackError, match="no exemplar variant"):
        run(pack, model="m", reports_dir=world["reports"], gold_path=str(world["gold"]), exemplar_variants=["nope"])


def test_one_context_window_per_run_sized_for_the_longest_prompt(world, monkeypatch, capsys):
    from engine.extract import default_num_ctx
    pack = load_pack(world["pack"])
    long_report = world["reports"] / f"{IDS[2]}.txt"
    long_report.write_text(long_report.read_text(encoding="utf-8") + "\nAddendum. " * 2500, encoding="utf-8")
    fake = FakeOllama(model_output_for_report(pack)).install(monkeypatch)
    report, out_dir = run(pack, model="m", reports_dir=world["reports"], gold_path=str(world["gold"]), run_id="win")
    windows = [req["options"]["num_ctx"] for req in fake.requests]
    per_call = [default_num_ctx(req["prompt"]) for req in fake.requests]
    assert len(windows) == 2 and per_call[0] < per_call[1]             # the long report alone would need more
    assert windows == [max(per_call)] * 2                               # every call gets the run's largest window
    assert f"context window {max(per_call)} for every call (sized for the longest prompt in the run)" in capsys.readouterr().err
    assert any(n.startswith(f"context window {max(per_call)} for every call") for n in report.notes)
    d = json.loads((out_dir / "metrics.json").read_text())
    assert d["cost"]["windows"] == {str(max(per_call)): 2} and d["cost"]["num_ctx"] == max(per_call)   # JSON keys
    fake.requests.clear()
    run(pack, model="m", reports_dir=world["reports"], gold_path=str(world["gold"]), run_id="win2", num_ctx=4096)
    assert [req["options"]["num_ctx"] for req in fake.requests] == [4096, 4096]       # an explicit value always wins
    assert "context window 4096 for every call (--num-ctx)" in capsys.readouterr().err


def test_cli_flags(world, monkeypatch, tmp_path, capsys):
    pack = load_pack(world["pack"])
    FakeOllama(model_output_for_report(pack)).install(monkeypatch)
    groups_csv = tmp_path / "groups.csv"
    groups_csv.write_text(f"case_id,group\n{IDS[0]},a\n{IDS[2]},b\n", encoding="utf-8")
    notes_md = tmp_path / "gold_notes.md"
    notes_md.write_text("| case | field(s) | kind | gold | report says | note |\n|---|---|---|---|---|---|\n"
                        f"| {IDS[0]} | pT | gold_finer | pT2b | x | y |\n", encoding="utf-8")
    code = main(["--pack", str(world["pack"]), "--model", "m", "--reports", str(world["reports"]),
                 "--gold", str(world["gold"]), "--groups", str(groups_csv), "--gold-notes", str(notes_md),
                 "--exemplar-variant", "hedging", "--run-id", "cli", "--think", "low"])
    assert code == 0
    out = capsys.readouterr().out
    assert "## By group" in out and "| a | 1 |" in out and "| without audited | 1 |" in out
