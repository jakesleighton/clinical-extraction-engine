"""evalharness.rescore: a saved run's model replies, re-validated with the
current pack, give a new run folder and never touch the old one."""

from __future__ import annotations

import json

from engine.packloader import load_pack
from evalharness.rescore import main, rescore, saved_calls
from evalharness.runner import run
from tests.conftest import FakeOllama, model_output_for_report
from tests.test_runner import IDS, world  # noqa: F401  (fixture)


def test_rescore_reproduces_a_run_from_its_ledger(world, monkeypatch, tmp_path, capsys):  # noqa: F811
    pack = load_pack(world["pack"])
    FakeOllama(model_output_for_report(pack)).install(monkeypatch)
    report, out_dir = run(pack, model="m", reports_dir=world["reports"], gold_path=str(world["gold"]), run_id="orig")
    before = (out_dir / "predictions.jsonl").read_text(encoding="utf-8")
    pairs = saved_calls("orig")
    assert [val["case_id"] for _, val in pairs] == [IDS[0], IDS[2]]

    re_report, re_dir = rescore("orig", world["pack"], world["reports"], gold_path=str(world["gold"]))
    assert re_dir.name == "orig-rescored" and re_dir != out_dir
    assert (out_dir / "predictions.jsonl").read_text(encoding="utf-8") == before      # the original is untouched
    recs = [json.loads(l) for l in (re_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    orig = [json.loads(l) for l in before.splitlines()]
    for a, b in zip(recs, orig):
        assert a["rescored_from"] == "orig" and a["run_id"] == "orig-rescored" and a["call_id"] == b["call_id"]
        assert a["fields"] == b["fields"] and a["issues"] == b["issues"]
    assert re_report.metrics.to_dict() == report.metrics.to_dict()
    assert re_report.notes[0].startswith("RE-SCORED from run orig")
    assert re_report.cost["calls"] == 2                                              # the original ledger's calls
    text = (re_dir / "report.md").read_text(encoding="utf-8")
    assert "RE-SCORED" in text and "## Scores" in text

    assert main(["--run-id", "orig", "--pack", str(world["pack"]), "--reports", str(world["reports"]),
                 "--gold", str(world["gold"]), "--as", "again"]) == 0
    assert "RE-SCORED from run orig" in capsys.readouterr().out
    assert (re_dir.parent / "again" / "metrics.json").exists()
