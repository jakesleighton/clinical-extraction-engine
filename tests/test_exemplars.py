"""The prostate pack's exemplars must be honest examples: every quote is found
verbatim in its own source, the output passes the validator with no issues,
every field is present, a wired exemplar names its case id, and that case is
not in the working set the eval runs on. Every (txt, json) pair in the pack's
exemplars/ directory is checked, wired or staged (the "+hedging" ablation arm
is staged there without being in the prompt)."""

from __future__ import annotations

import json

import pytest

from engine.packloader import load_pack
from engine.validate import apply_derivations, validate_output
from evalharness.runner import read_id_file
from tests.conftest import PACKS, ROOT

PROSTATE = load_pack(PACKS / "prostate_pathology")
WORKING_SET = ROOT / "data" / "tcga_prad" / "working_set.txt"
PAIRS = sorted(p.stem for p in (PACKS / "prostate_pathology" / "exemplars").glob("*.json"))


def test_the_approved_exemplars_are_wired_and_the_hedging_arm_is_staged():
    assert PROSTATE.exemplar_ids == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-ZG-A8QW"]
    assert PAIRS == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-V1-A9Z7", "TCGA-ZG-A8QW"]
    for ex in PROSTATE.exemplars:
        assert ex.case_id, "every wired exemplar must name the case it came from so the eval can hold it out"
        assert ex.source == f"exemplars/{ex.case_id}.txt" and ex.output == f"exemplars/{ex.case_id}.json"


@pytest.mark.parametrize("case_id", PAIRS)
def test_exemplar_output_is_clean_against_its_source(case_id):
    source = PROSTATE.resolve(f"exemplars/{case_id}.txt").read_text(encoding="utf-8")
    output = json.loads(PROSTATE.resolve(f"exemplars/{case_id}.json").read_text(encoding="utf-8"))
    assert list(output) == PROSTATE.field_ids, "exemplar output must cover every field, in pack order"
    result = validate_output(PROSTATE, output, source)
    apply_derivations(PROSTATE, result.record, result.issues)
    assert result.issues == [], [i.message for i in result.issues]
    for fid, env in output.items():
        if env["status"] == "not_mentioned":
            assert env["evidence"] == [] and env["value"] is None, fid
        else:
            assert env["evidence"], f"{fid}: a stated or negated field needs a quote"
    assert output["grade_group"]["status"] == "not_mentioned", "no TCGA-PRAD report prints a Grade Group"


def test_exemplar_cases_are_outside_the_working_set():
    if not WORKING_SET.exists():
        pytest.skip("no working set file")
    overlap = set(PAIRS) & set(read_id_file(WORKING_SET))
    assert not overlap, f"exemplar cases in the working set would be held out of the eval: {sorted(overlap)}"


def test_hedging_arm_demonstrates_a_hedged_finding():
    output = json.loads(PROSTATE.resolve("exemplars/TCGA-V1-A9Z7.json").read_text(encoding="utf-8"))
    assert output["margin_positive"]["hedging"] == "possible" and output["margin_positive"]["status"] == "value"
    wired = [json.loads(PROSTATE.resolve(ex.output).read_text(encoding="utf-8")) for ex in PROSTATE.exemplars]
    assert all(env["hedging"] == "definite" for out in wired for env in out.values())
    assert sum(env["conflict"] for out in wired for env in out.values()) == 1    # ZG-A8QW lymphovascular_invasion
