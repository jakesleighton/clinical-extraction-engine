from engine.validate import apply_derivations, find_span, normalize_date, validate_output
from tests.conftest import REPORT, envelope, full_record, model_output_for_report


def _codes(result):
    return sorted(i.code for i in result.issues)


# ---------------------------------------------------------------------------
# evidence spans
# ---------------------------------------------------------------------------

def test_find_span_exact():
    start, end = find_span(REPORT, "Gleason score 3+4=7")
    assert REPORT[start:end] == "Gleason score 3+4=7"


def test_find_span_across_line_break_and_case():
    quote = "gleason score 3+4=7, involving both lobes"
    span = find_span(REPORT, quote)
    assert span is not None
    start, end = span
    assert " ".join(REPORT[start:end].split()).casefold() == quote.casefold()
    assert REPORT[start:end].startswith("Gleason") and REPORT[start:end].endswith("lobes")


def test_find_span_strips_quotes_and_handles_missing():
    assert find_span(REPORT, '"Perineural invasion: Present."') == find_span(REPORT, "Perineural invasion: Present.")
    assert find_span(REPORT, "no such text anywhere") is None
    assert find_span(REPORT, "   ") is None


def test_normalize_date():
    assert normalize_date("10/15/2010") == "2010-10-15"
    assert normalize_date("2010-10-15") == "2010-10-15"
    assert normalize_date("October 15, 2010") == "2010-10-15"
    assert normalize_date("15 Oct 2010") == "2010-10-15"
    assert normalize_date("2010-10") == "2010-10" and normalize_date("2010") == "2010"
    assert normalize_date("10/2010") == "2010-10"
    assert normalize_date("sometime in 2010") is None
    assert normalize_date("2010-13-40") is None


# ---------------------------------------------------------------------------
# validate_output
# ---------------------------------------------------------------------------

def test_good_output_has_no_issues_and_spans(prostate):
    result = validate_output(prostate, model_output_for_report(prostate), REPORT)
    assert result.issues == [], [i.message for i in result.issues]
    assert result.ok
    rec = result.record
    assert rec["gleason_primary"]["value"] == 3 and rec["gleason_primary"]["derivation"] == "stated"
    assert rec["report_date"]["value"] == "2010-10-15"             # normalized
    assert rec["tumor_percent_of_gland"]["unit"] == "percent"
    ev = rec["pT"]["evidence"][0]
    assert REPORT[ev["start"]:ev["end"]] == "Pathologic stage: pT2c pN0."
    assert rec["margin_positive"] == {"status": "explicitly_absent", "value": None, "unit": None,
                                      "hedging": "definite", "conflict": False, "derivation": None,
                                      "evidence": [{"quote": "Surgical margins: Negative for carcinoma.",
                                                    "start": REPORT.index("Surgical margins"),
                                                    "end": REPORT.index("Surgical margins") + len("Surgical margins: Negative for carcinoma.")}]}
    assert rec["grade_group"]["status"] == "not_mentioned" and rec["grade_group"]["derivation"] is None
    assert set(rec) == set(prostate.field_ids)


def test_cross_field_rule_failure_is_flagged(prostate):
    out = model_output_for_report(prostate)
    out["gleason_total"] = envelope("value", 8, ["Gleason score 3+4=7"])
    result = validate_output(prostate, out, REPORT)
    assert result.ok                                            # a flag, not an error
    [issue] = result.issues
    assert (issue.code, issue.rule_id, issue.on_fail) == ("rule_failed", "gleason_sum", "retry_then_flag")
    assert "gleason_total=8" in issue.message


def test_rules_skip_when_inputs_missing(prostate):
    out = model_output_for_report(prostate)
    out["gleason_total"] = envelope()                          # not_mentioned
    result = validate_output(prostate, out, REPORT)
    assert result.issues == []


def test_rule_uses_function_library(prostate):
    out = model_output_for_report(prostate)
    out["grade_group"] = envelope("value", 4, ["Grade Group 4"])   # inconsistent with 3+4, quote not in report
    result = validate_output(prostate, out, REPORT)
    assert _codes(result) == ["evidence_not_found", "rule_failed"]
    assert [i.rule_id for i in result.issues if i.rule_id] == ["grade_group_matches_gleason"]


def test_status_value_inconsistencies(prostate):
    out = full_record(prostate, {
        "gleason_primary": envelope("value", None, ["Gleason score 3+4=7"]),           # value missing
        "gleason_secondary": envelope("not_mentioned", 4, []),                       # value without status
        "margin_positive": envelope("explicitly_absent", None, []),                  # evidence missing
        "nodes_examined": envelope("value", 12, ["ninety-nine nodes"]),              # quote not found
        "report_date": envelope("value", "last Tuesday", ["Reported: 10/15/2010"]),  # bad date
        "tumor_dimension": envelope("value", 1.5, ["approximately 20% of the gland"], unit=None, measurement=True),
    })
    result = validate_output(prostate, out, REPORT)
    assert _codes(result) == ["bad_date", "evidence_missing", "evidence_not_found", "missing_unit",
                              "missing_value", "value_without_status"]
    assert not result.ok and [i.code for i in result.errors] == ["missing_value"]
    rec = result.record
    assert rec["gleason_secondary"]["value"] is None
    assert rec["nodes_examined"]["evidence"] == [{"quote": "ninety-nine nodes", "start": None, "end": None}]
    assert rec["report_date"]["value"] == "last Tuesday"
    assert rec["tumor_dimension"]["value"] == 1.5 and rec["tumor_dimension"]["unit"] is None


def test_schema_violations_are_errors_but_contained(prostate):
    out = model_output_for_report(prostate)
    out["pT"] = envelope("value", "T3a", ["Pathologic stage: pT2c pN0."])    # not in allowed
    out["pN"]["status"] = "maybe"
    del out["nodes_positive"]
    out["bogus"] = envelope()
    result = validate_output(prostate, out, REPORT)
    assert _codes(result) == ["missing_field", "schema", "schema", "unknown_field"]
    assert not result.ok
    assert result.record["pT"]["status"] == "not_mentioned" and result.record["pT"]["value"] is None
    assert result.record["gleason_primary"]["value"] == 3            # the rest survived


# ---------------------------------------------------------------------------
# derived fields
# ---------------------------------------------------------------------------

def test_derivation_fills_grade_group_with_provenance(prostate):
    result = validate_output(prostate, model_output_for_report(prostate), REPORT)
    apply_derivations(prostate, result.record, result.issues)
    gg = result.record["grade_group"]
    assert gg["status"] == "value" and gg["value"] == 2
    assert gg["derivation"] == "computed_from_gleason"
    assert gg["hedging"] == "definite" and gg["conflict"] is False
    assert [e["quote"] for e in gg["evidence"]] == ["Gleason score 3+4=7"]   # inherited, de-duplicated
    assert gg["evidence"][0]["start"] == REPORT.index("Gleason score 3+4=7")
    assert result.issues == []


def test_derivation_leaves_stated_value_alone(prostate):
    out = model_output_for_report(prostate)
    out["grade_group"] = envelope("value", 2, ["Gleason score 3+4=7"])
    result = validate_output(prostate, out, REPORT)
    apply_derivations(prostate, result.record, result.issues)
    assert result.record["grade_group"]["derivation"] == "stated"


def test_derivation_needs_all_inputs(prostate):
    out = model_output_for_report(prostate)
    out["gleason_secondary"] = envelope()
    result = validate_output(prostate, out, REPORT)
    apply_derivations(prostate, result.record, result.issues)
    assert result.record["grade_group"]["status"] == "not_mentioned"
    assert result.record["grade_group"]["derivation"] is None


def test_derivation_inherits_weakest_hedging_and_conflict(prostate):
    out = model_output_for_report(prostate)
    out["gleason_primary"]["hedging"] = "probable"
    out["gleason_secondary"]["conflict"] = True
    result = validate_output(prostate, out, REPORT)
    apply_derivations(prostate, result.record, result.issues)
    gg = result.record["grade_group"]
    assert gg["hedging"] == "probable" and gg["conflict"] is True


def test_derivation_always_flags_disagreement(prostate):
    pack = prostate.model_copy(deep=True)
    pack.derived_fields[0].when = "always"
    out = model_output_for_report(pack)
    out["grade_group"] = envelope("value", 5, ["Gleason score 3+4=7"])
    result = validate_output(pack, out, REPORT)
    assert [i.rule_id for i in result.issues] == ["grade_group_matches_gleason"]
    apply_derivations(pack, result.record, result.issues)
    assert "derivation_disagrees" in _codes(result)
    assert result.record["grade_group"]["value"] == 2
    assert result.record["grade_group"]["derivation"] == "computed_from_gleason"


def test_derivation_outside_allowed_is_flagged(prostate):
    pack = prostate.model_copy(deep=True)
    pack.derived_fields[0].rule = "gleason_primary + gleason_secondary"     # 7, not a grade group
    out = model_output_for_report(pack)
    result = validate_output(pack, out, REPORT)
    apply_derivations(pack, result.record, result.issues)
    assert _codes(result) == ["derivation_failed"]
    assert result.record["grade_group"]["status"] == "not_mentioned"
