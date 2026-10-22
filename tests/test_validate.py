import pytest

from engine.validate import apply_derivations, find_span, locate, normalize_date, validate_output
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


OCR = ("FINAL DIAGNOSIS. Poorly differentiated prostate. adenocarcinoma, acinar type. "
       "Tumor measures 2.0 cm; a second nodule measures 20 cm. Extra-\nprostatic extension: present. "
       "Lymph nodes (0/12). PSA 8,6 ng/ml. Collected 2010-10-12.")


def test_locate_reports_which_pass_found_the_quote():
    assert locate(REPORT, "Gleason score 3+4=7")[2] == "exact"
    assert locate(REPORT, "gleason score 3+4=7,  involving both lobes")[2] == "whitespace"
    start, end, how = locate(OCR, "Poorly differentiated prostate adenocarcinoma, acinar type")
    assert how == "punctuation"
    assert OCR[start:end] == "Poorly differentiated prostate. adenocarcinoma, acinar type"


@pytest.mark.parametrize("quote, found", [
    ("Poorly differentiated prostate adenocarcinoma acinar type", True),   # OCR line-break period, comma
    ("Extraprostatic extension: present", True),                           # hyphenated line break
    ("Lymph nodes 0/12", True),                                            # parentheses ignored, slash kept
    ("2.0 cm; a second", True),                                            # decimal point kept and present
    ("20 cm; a second", False),                                            # "2.0" never matches "20"
    ("8,6 ng/ml", True),
    ("86 ng/ml", False),                                                   # digit comma kept
    ("Collected 2010 10 12", False),                                       # digit hyphens kept
    ("Poorly differentiated adenocarcinoma, acinar type", False),          # a dropped word is still a miss
    ("Poorly differentiated ... acinar type", False),                      # a stitched quote is still a miss
    ("...", False),                                                        # nothing left to match
])
def test_find_span_ignores_punctuation_but_not_words_or_digit_separators(quote, found):
    assert (find_span(OCR, quote) is not None) is found


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
                                                    "end": REPORT.index("Surgical margins") + len("Surgical margins: Negative for carcinoma."),
                                                    "match": "exact"}]}
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
    assert rec["nodes_examined"]["evidence"] == [{"quote": "ninety-nine nodes", "start": None, "end": None, "match": None}]
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
    out["grade_group"] = envelope("value", 2, ["Grade Group 2"])
    result = validate_output(prostate, out, REPORT + "Grade Group 2.\n")     # a report that prints it
    apply_derivations(prostate, result.record, result.issues)
    assert result.issues == [] and result.record["grade_group"]["derivation"] == "stated"


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
    out["grade_group"] = envelope("value", 5, ["Grade Group 5"])
    result = validate_output(pack, out, REPORT + "Grade Group 5.\n")
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


# ---------------------------------------------------------------------------
# derived pT / pN: input modes, `when: check`, `on_none: skip`
# ---------------------------------------------------------------------------

def _derive(pack, overrides):
    """Validate REPORT output with some envelopes replaced, run the derivations,
    return (record, issue codes)."""
    output = model_output_for_report(pack)
    output.update(overrides)
    result = validate_output(pack, output, REPORT)
    apply_derivations(pack, result.record, result.issues)
    return result.record, _codes(result)


def test_printed_stage_that_agrees_with_the_findings_stays_stated(prostate):
    record, codes = _derive(prostate, {})
    assert codes == []
    assert (record["pT"]["value"], record["pT"]["derivation"]) == ("pT2c", "stated")
    assert (record["pN"]["value"], record["pN"]["derivation"]) == ("pN0", "stated")
    assert record["pT"]["evidence"] == [{"quote": "Pathologic stage: pT2c pN0.", "start": REPORT.index("Pathologic"),
                                         "end": REPORT.index("pN0.") + 4, "match": "exact"}]


def test_unprinted_stage_is_derived_from_findings_and_nodes_with_inherited_evidence(prostate):
    record, codes = _derive(prostate, {"pT": envelope(), "pN": envelope()})
    assert codes == []
    pt, pn = record["pT"], record["pN"]
    assert (pt["status"], pt["value"], pt["derivation"]) == ("value", "pT2c", "computed_from_findings")
    assert (pn["status"], pn["value"], pn["derivation"]) == ("value", "pN0", "computed_from_nodes")
    # the evidence is the contributing inputs' quotes, the negated findings included
    assert [e["quote"] for e in pt["evidence"]] == ["radical prostatectomy", "Extraprostatic extension: Not identified.",
                                                     "Seminal vesicle invasion: Negative.", "involving both lobes"]
    assert [e["quote"] for e in pn["evidence"]] == ["Twelve lymph nodes", "(0/12)"]
    assert all(e["start"] is not None and e["match"] == "exact" for e in pt["evidence"] + pn["evidence"])
    assert pt["hedging"] == "definite" and pt["conflict"] is False


def test_seminal_vesicle_invasion_gives_pt3b_and_a_disagreement_is_flagged_not_overwritten(prostate):
    svi = envelope("value", "bilateral", ["Seminal vesicle invasion"], hedging="probable")
    record, codes = _derive(prostate, {"pT": envelope(), "seminal_vesicle_invasion": svi})
    assert codes == [] and record["pT"]["value"] == "pT3b"
    assert record["pT"]["hedging"] == "probable"                                      # weakest input hedging
    # a printed pT2c against findings that say pT3b: the stated value stands, the disagreement is flagged
    record, codes = _derive(prostate, {"seminal_vesicle_invasion": svi})
    assert "derivation_disagrees" in codes
    assert (record["pT"]["value"], record["pT"]["derivation"]) == ("pT2c", "stated")


def test_coarser_printed_stage_agrees_with_the_computed_one(prostate):
    epe = envelope("value", "focal", ["Extraprostatic extension"])
    pt = envelope("value", "pT3", ["Pathologic stage: pT2c pN0."])     # the guard wants a printed stage, not this one
    record, codes = _derive(prostate, {"extraprostatic_extension": epe, "pT": pt})
    assert "derivation_disagrees" not in codes
    assert (record["pT"]["value"], record["pT"]["derivation"]) == ("pT3", "stated")


def test_silent_finding_blocks_the_derivation_and_optional_laterality_does_not(prostate):
    record, codes = _derive(prostate, {"pT": envelope(), "extraprostatic_extension": envelope()})
    assert codes == [] and record["pT"]["status"] == "not_mentioned" and record["pT"]["derivation"] is None
    record, codes = _derive(prostate, {"pT": envelope(), "tumor_laterality": envelope()})
    assert codes == [] and (record["pT"]["value"], record["pT"]["derivation"]) == ("pT2", "computed_from_findings")


def test_count_input_reads_an_explicitly_absent_count_as_zero(prostate):
    none_positive = envelope("explicitly_absent", None, ["negative for metastatic carcinoma"])
    record, codes = _derive(prostate, {"pN": envelope(), "nodes_positive": none_positive})
    assert codes == []
    pn = record["pN"]
    assert (pn["status"], pn["value"], pn["derivation"]) == ("value", "pN0", "computed_from_nodes")
    assert [e["quote"] for e in pn["evidence"]] == ["Twelve lymph nodes", "(0/12)", "negative for metastatic carcinoma"]
    # the same envelope on a `required` input blocks the derivation (the v3 behaviour)
    strict = prostate.model_copy(deep=True)
    strict.derived_for("pN").inputs = ["nodes_examined", "nodes_positive"]
    record, codes = _derive(strict, {"pN": envelope(), "nodes_positive": none_positive})
    assert codes == [] and record["pN"]["status"] == "not_mentioned"


def test_on_none_skip_leaves_the_field_without_a_flag(prostate):
    biopsy = envelope("value", "needle_biopsy", ["radical prostatectomy"])
    record, codes = _derive(prostate, {"pT": envelope(), "specimen_procedure": biopsy})
    assert codes == [] and record["pT"]["status"] == "not_mentioned"
    no_nodes = envelope("explicitly_absent", None, ["Twelve lymph nodes"])
    record, codes = _derive(prostate, {"pN": envelope(), "nodes_examined": no_nodes, "nodes_positive": envelope()})
    assert codes == [] and record["pN"]["status"] == "not_mentioned"
    # under the strict default (on_none: flag) the same None is a derivation_failed flag
    strict = prostate.model_copy(deep=True)
    strict.derived_for("pN").on_none = "flag"
    zero = envelope("value", 0, ["Twelve lymph nodes"])
    record, codes = _derive(strict, {"pN": envelope(), "nodes_examined": zero})
    assert "derivation_failed" in codes and record["pN"]["status"] == "not_mentioned"


# ---------------------------------------------------------------------------
# stated-value guard (`stated_pattern`): a stated stage must quote a printed one
# ---------------------------------------------------------------------------

def test_stated_stage_quoted_from_a_node_finding_is_set_aside_and_derived(prostate):
    pn = envelope("value", "pN0", ["(0/12)"])                      # right value, quoted from the node count
    record, codes = _derive(prostate, {"pN": pn})
    assert codes == ["stated_without_printed_value"]
    env = record["pN"]
    assert (env["status"], env["value"], env["derivation"]) == ("value", "pN0", "computed_from_nodes")
    assert [e["quote"] for e in env["evidence"]] == ["Twelve lymph nodes", "(0/12)"]      # the inputs' evidence
    # pT the same way, and a Grade Group quoted from the Gleason score
    pt = envelope("value", "pT2c", ["Extraprostatic extension: Not identified."])
    gg = envelope("value", 2, ["Gleason score 3+4=7"])
    record, codes = _derive(prostate, {"pT": pt, "grade_group": gg})
    assert sorted(codes) == ["stated_without_printed_value", "stated_without_printed_value"]
    assert (record["pT"]["value"], record["pT"]["derivation"]) == ("pT2c", "computed_from_findings")
    assert (record["grade_group"]["value"], record["grade_group"]["derivation"]) == (2, "computed_from_gleason")


def test_guard_message_names_the_quotes_and_an_underivable_field_stays_empty(prostate):
    biopsy = envelope("value", "needle_biopsy", ["radical prostatectomy"])
    pt = envelope("value", "pT3a", ["Perineural invasion: Present."])
    output = model_output_for_report(prostate)
    output.update({"specimen_procedure": biopsy, "pT": pt})
    result = validate_output(prostate, output, REPORT)
    [issue] = result.issues
    assert issue.code == "stated_without_printed_value" and issue.field == "pT" and issue.severity == "flag"
    assert "'pT3a' returned as stated" in issue.message and "'Perineural invasion: Present.'" in issue.message
    apply_derivations(prostate, result.record, result.issues)
    assert result.record["pT"] == {"status": "not_mentioned", "value": None, "unit": None, "hedging": None,
                                   "conflict": False, "evidence": [], "derivation": None}


def test_guard_matches_the_located_source_span_not_the_model_s_spelling(prostate):
    # the quote drops the colon and the period: located by the punctuation pass, and the span it lands on
    # ("Pathologic stage: pT2c pN0") prints the stage, so the value stands as stated
    pt = envelope("value", "pT2c", ["pathologic stage pT2c pN0"])
    record, codes = _derive(prostate, {"pT": pt})
    assert codes == []
    assert (record["pT"]["value"], record["pT"]["derivation"]) == ("pT2c", "stated")
    assert record["pT"]["evidence"][0]["match"] == "punctuation"
    # a quote the locator cannot find is judged as written (evidence_not_found is its own flag)
    pt = envelope("value", "pT2c", ["Final stage pT2c, per the addendum"])
    record, codes = _derive(prostate, {"pT": pt})
    assert codes == ["evidence_not_found"]
    assert (record["pT"]["value"], record["pT"]["derivation"]) == ("pT2c", "stated")


def test_guard_asks_for_a_printed_value_of_the_kind_not_the_same_value(prostate):
    pt = envelope("value", "pT3a", ["Pathologic stage: pT2c pN0."])    # a value error, not a provenance error
    record, codes = _derive(prostate, {"pT": pt})
    assert codes == ["derivation_disagrees"]
    assert (record["pT"]["value"], record["pT"]["derivation"]) == ("pT3a", "stated")


def test_guard_only_applies_where_the_pack_declares_a_pattern(prostate):
    loose = prostate.model_copy(deep=True)
    loose.derived_for("pN").stated_pattern = None
    pn = envelope("value", "pN0", ["(0/12)"])
    record, codes = _derive(loose, {"pN": pn})
    assert codes == [] and (record["pN"]["value"], record["pN"]["derivation"]) == ("pN0", "stated")

