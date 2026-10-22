"""evalharness.report: the parts of report.md beyond the scores, computed from
saved records, never from a model."""

from __future__ import annotations

import json

import pytest

from evalharness.report import (GoldNote, audited_case_ids, build_run_report, classify_quote, cost_summary,
                                field_status, flag_summary, quote_summary, read_gold_notes, read_groups)
from tests.conftest import REPORT, envelope, model_output_for_report

TEXT = ("FINAL DIAGNOSIS. Poorly differentiated prostate. adenocarcinoma, acinar type. "
        "Extra-\nprostatic extension: present. Lymph nodes (0/12). Margins negative.")


def test_read_groups(tmp_path):
    f = tmp_path / "g.csv"
    f.write_text("case_id,group,extra\nA,one,x\nB,one,\nB,two,\nA,one,\n\n", encoding="utf-8")
    assert read_groups(f) == {"one": ["A", "B"], "two": ["B"]}
    bad = tmp_path / "bad.csv"
    bad.write_text("id,grp\nA,one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="case_id,group"):
        read_groups(bad)


def test_read_gold_notes_takes_table_rows_only(tmp_path):
    f = tmp_path / "gold_notes.md"
    f.write_text("# Gold notes\n\nProse with | a pipe.\n\n"
                 "| case | field(s) | kind | gold | report says | note |\n"
                 "|---|---|---|---|---|---|\n"
                 "| TCGA-AA-0001 | pT, pN | gold_finer | pT2b | \"Stage: T2\" | no substage |\n"
                 "| TCGA-AA-0002 | gleason_primary | gold_disagrees | 4 | \"5+3=8\" | biopsy score | extra cell |\n"
                 "| TCGA-AA-0001 | pN | gold_disagrees | pN0 | \"pN1\" | second row for a case |\n"
                 "| short | row |\n", encoding="utf-8")
    notes = read_gold_notes(f)
    assert [n.case_id for n in notes] == ["TCGA-AA-0001", "TCGA-AA-0002", "TCGA-AA-0001"]
    assert notes[0] == GoldNote("TCGA-AA-0001", ("pT", "pN"), "gold_finer", "pT2b", '"Stage: T2"', "no substage")
    assert audited_case_ids(notes) == ["TCGA-AA-0001", "TCGA-AA-0002"]


def test_the_tracked_gold_notes_parse():
    from tests.conftest import ROOT
    notes = read_gold_notes(ROOT / "data" / "gold_notes.md")
    assert audited_case_ids(notes) == ["TCGA-HC-A9TE", "TCGA-V1-A9OY", "TCGA-V1-A9OH", "TCGA-J4-A67T",
                                       "TCGA-SU-A7E7", "TCGA-HC-7749", "TCGA-CH-5741"]
    assert {n.kind for n in notes} == {"gold_disagrees", "gold_stale", "gold_finer", "amendment_rule"}
    groups = read_groups(ROOT / "data" / "tcga_prad" / "case_groups.csv")
    assert len(groups["random20"]) == 20 and len(groups["weird5"]) == 5
    assert "TCGA-CH-5753" in groups["random20"] and "TCGA-CH-5753" in groups["weird5"]


@pytest.mark.parametrize("quote, expected", [
    ("Poorly differentiated prostate adenocarcinoma, acinar type", "normalized"),   # punctuation only
    ("Extraprostatic extension: present", "normalized"),                           # hyphenated line break
    ("Poorly differentiated ... acinar type", "stitched"),
    ("Poorly differentiated adenocarcinoma, acinar type", "one_token"),             # one word dropped
    ("Poorly differentiated prostate large adenocarcinoma", "one_token"),           # one word inserted
    ("Poorly differentiated prostate adenocarcinoma of the left lobe", "partial"),  # starts on real text
    ("Tumor involves both lobes", "other"),
    ("...", "empty"),
])
def test_classify_quote(quote, expected):
    assert classify_quote(quote, TEXT) == expected


def _record(case_id, fields, issues=(), ok=True):
    return {"case_id": case_id, "ok": ok, "fields": fields, "issues": list(issues)}


def test_flag_summary_counts_by_code_with_cases_and_fields():
    recs = [_record("A", {"x": {}}, [{"severity": "flag", "code": "evidence_not_found", "field": "pT"},
                                    {"severity": "flag", "code": "evidence_not_found", "field": "pN"},
                                    {"severity": "flag", "code": "rule_failed", "rule_id": "gleason_sum"}]),
            _record("B", {"x": {}}, [{"severity": "flag", "code": "evidence_not_found", "field": "pT"}]),
            _record("C", None, [], ok=False)]
    fs = flag_summary(recs)
    assert fs["records"] == 3 and fs["records_missing"] == ["C"]
    assert fs["issues"] == [{"severity": "flag", "code": "evidence_not_found", "count": 3, "cases": 2,
                             "fields": {"pT": 2, "pN": 1}},
                            {"severity": "flag", "code": "rule_failed", "count": 1, "cases": 1,
                             "fields": {"gleason_sum": 1}}]


def test_field_status_quotes_and_located_passes(prostate):
    out = model_output_for_report(prostate)
    ev = lambda q, how: {"quote": q, "start": 0 if how else None, "end": 1 if how else None, "match": how}
    fields = {fid: {**env, "derivation": "stated" if env["status"] == "value" else None,
                    "evidence": [ev(q, "exact") for q in env["evidence"]]} for fid, env in out.items()}
    fields["pT"]["evidence"] = [ev("Pathologic stage pT2c", "punctuation"), ev("made up", None)]
    fields["pT"]["hedging"], fields["pT"]["conflict"] = "possible", True
    fields["grade_group"] = {**fields["grade_group"], "status": "value", "value": 2, "derivation": "computed_from_gleason",
                             "evidence": [ev("Gleason score 3+4=7", "exact")]}
    recs = [_record("A", fields), _record("B", None, ok=False)]
    rows = {r["field"]: r for r in field_status(prostate, recs)}
    assert rows["pT"] == {"field": "pT", "tier": "deterministic", "value": 1, "explicitly_absent": 0,
                          "not_mentioned": 0, "no_record": 1, "derived": 0, "hedged": 1, "conflict": 1,
                          "quotes": 2, "quotes_not_found": 1}
    assert rows["grade_group"]["derived"] == 1 and rows["grade_group"]["quotes"] == 0     # inherited evidence skipped
    assert rows["margin_positive"]["explicitly_absent"] == 1 and rows["treatment_effect"]["not_mentioned"] == 1
    q = quote_summary(recs, {"A": REPORT})
    assert q["located"]["punctuation"] == 1 and q["not_found"] == 1 and q["classes"] == {"other": 1}
    assert q["not_found_by_field"] == {"pT": 1} and q["examples"]["other"][0]["quote"] == "made up"
    assert q["located"]["exact"] == sum(len(e["evidence"]) for f, e in fields.items() if f not in ("pT", "grade_group"))
    # records written before the match kind was stored count as located ("legacy") when they have an offset
    legacy = _record("L", {"pT": {"status": "value", "value": "pT2", "derivation": "stated",
                                  "evidence": [{"quote": "x", "start": 3, "end": 4}]}})
    assert quote_summary([legacy], {})["located"] == {"legacy": 1}


def test_cost_summary_from_ledger_calls():
    calls = [{"latency_s": 10.0, "tokens_in": 1000, "tokens_out": 100, "params": {"num_ctx": 2000, "seed": 0},
              "meta": {"think": False, "done_reason": "stop"}},
             {"latency_s": 30.0, "tokens_in": 1200, "tokens_out": 300, "params": {"num_ctx": 2000, "seed": 0},
              "meta": {"think": None, "think_rejected": "400", "done_reason": "length"}, "error": "cut"}]
    c = cost_summary(calls, wall_s=41.6)
    assert (c["calls"], c["errors"], c["tokens_in"], c["tokens_out"]) == (2, 1, 2200, 400)
    assert (c["tokens_in_mean"], c["latency_median_s"], c["latency_max_s"], c["latency_sum_s"], c["wall_s"]) == (1100, 20.0, 30.0, 40, 42)
    assert c["think"] == {"False": 1, "None (rejected by the server)": 1} and c["done_reason"] == {"stop": 1, "length": 1}
    assert c["peak_tokens"] == 1500 and c["context_pressure"] == 0.75 and c["num_ctx"] == 2000
    empty = cost_summary([])
    assert empty["calls"] == 0 and empty["latency_median_s"] is None and empty["context_pressure"] is None


def test_build_run_report_scores_groups_and_audited_subsets(prostate):
    out = model_output_for_report(prostate)
    def rec(case_id, pt):
        fields = {fid: {**env, "derivation": "stated" if env["status"] == "value" else None,
                        "evidence": [{"quote": q, "start": 0, "end": 1, "match": "exact"} for q in env["evidence"]]}
                  for fid, env in out.items()}
        fields["pT"] = {**fields["pT"], "value": pt}
        fields["grade_group"] = {**fields["grade_group"], "status": "value", "value": 2, "derivation": "computed_from_gleason"}
        return _record(case_id, fields)
    records = [rec("A", "pT2c"), rec("B", "pT3"), rec("C", "pT2c")]
    gold = {c: {"gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7, "grade_group": 2, "pT": "pT3a" if c == "B" else "pT2c",
                "pN": "pN0", "grade_group_printed": False, "pT_printed": True, "pN_printed": True} for c in "ABC"}
    notes = [GoldNote("B", ("pT",), "gold_finer", "pT3a", '"pT3"', "no substage"), GoldNote("Z", ("pN",), "gold_stale", "", "", "")]
    rr = build_run_report(prostate, records, gold, texts={c: REPORT for c in "ABC"}, calls=[], wall_s=1.0,
                          groups={"ab": ["A", "B"], "c": ["C"], "none": ["Q"]}, gold_notes=notes, header=["h1"])
    assert rr.n_cases == 3 and rr.notes[0] == "h1"
    assert list(rr.groups) == ["ab", "c"] and rr.groups["ab"].n_cases == 2 and rr.groups["c"].overall.micro_f1 == 1.0
    assert rr.audited_in_run == ["B"] and rr.without_audited.n_cases == 2 and rr.without_audited.overall.micro_f1 == 1.0
    assert rr.metrics.overall.micro_f1 < 1.0 and rr.metrics.aggregate("overall (lenient)").micro_f1 == 1.0
    md = rr.to_markdown()
    assert "## By group" in md and "| ab | 2 |" in md and "| none" not in md
    assert "## Without the audited gold cases" in md and "| B | pT | gold_finer | pT3a |" in md and "| Z |" not in md
    assert "## Validator" in md and "No validator issues." in md and "## Quotes" in md and "## Cost" in md
    d = json.loads(json.dumps(rr.to_dict()))
    assert d["n_cases"] == 3 and set(d["groups"]) == {"ab", "c"} and d["without_audited"]["n_cases"] == 2
    assert d["audited_in_run"] == ["B"] and d["cost"]["calls"] == 0 and d["notes"][0] == "h1"
