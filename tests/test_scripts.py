import csv
import json

from scripts.join_gold_labels import (GOLD_COLUMNS, SCORED, gold_columns, label_row, main, parse_group, parse_pattern,
                                      parse_stage, printed_phrase, stated_columns, stated_patterns)
from tests.conftest import PACKS, REPORT


def test_parsers():
    assert parse_pattern("Pattern 3") == 3 and parse_pattern("4") == 4 and parse_pattern(None) is None
    assert parse_pattern("Not Reported") is None and parse_pattern("Pattern 7") is None
    assert parse_group("Group 2") == 2 and parse_group("") is None
    assert parse_stage("T3a", "T") == "pT3a" and parse_stage("pT2c", "T") == "pT2c" and parse_stage("TX", "T") == "pTX"
    assert parse_stage("N0", "N") == "pN0" and parse_stage("NX", "N") == "pNX" and parse_stage("N1", "N") == "pN1"
    assert parse_stage("Not Reported", "T") is None and parse_stage("N0", "T") is None


def test_gold_columns_follow_the_pack(prostate):
    assert gold_columns(prostate) == GOLD_COLUMNS
    assert stated_columns(prostate) == {"grade_group_printed": "grade_group", "pT_printed": "pT", "pN_printed": "pN"}
    assert list(stated_patterns(prostate)) == ["grade_group", "pT", "pN"]
    # the join compiles exactly what the validator's guard compiles: one regex per field, from the pack
    assert all(stated_patterns(prostate)[d.id].pattern == d.stated_regex.pattern for d in prostate.derived_fields)


def test_printed_grade_group_pattern(prostate):
    gg = stated_patterns(prostate)["grade_group"]
    yes = ["Grade Group 2 (Gleason 3+4=7)", "GLEASON GRADE GROUP: 3", "ISUP grade 4", "ISUP 5", "(GG 1)",
           "Grade Group (ISUP) 2", "WHO grade group 3", "grade group of 4", "Grade group 2/5", "Grade Groups 1"]
    no = ["Gleason score 3+4=7", "Gleason 4+5=9 (Gleason 5: 20%)", "graded per ISUP 2005 consensus",
          "ISUP 2014 modified Gleason", "lymph node group 2", "an egg 3", "Grade group: not applicable",
          "TNM HISTOLOGIC GRADE = G3", "HISTOLOGIC GRADE: G3-4, Poorly differentiated"]
    for s in yes:
        assert printed_phrase(gg, s) is not None, s
    for s in no:
        assert printed_phrase(gg, s) is None, s
    assert printed_phrase(gg, "... Prostate.  Grade   Group 2 ...") == "Grade Group 2"


def test_printed_stage_patterns(prostate):
    pt, pn = stated_patterns(prostate)["pT"], stated_patterns(prostate)["pN"]
    t_yes = ["Pathologic stage: pT2c pN0.", "pT3a NO MX", "ypT2 N0", "TNM STAGE: AT LEAST pT2c, pNO, pMX",
             "AJCC/UICC stage: pT3bNOMX", "Stage: T2 NO M0, II B", "stage is T1", "STAGE = T3", "PT2",
             "T 3a N 0", "T2c, NO",
             # one site's template as the OCR renders it: a period at every field break
             "Pathological Staging (pTNM): p. T 3a. M X.", "Staging (pTNM): T 2c. N x. M x.", "T 3b. N 1. Mx"]
    t_no = ["Gleason score 3+4=7", "pt. is a 65-year-old", "DEPT 3", "the stage of disease", "Tumor T1-weighted",
            "Extraprostatic extension: Not identified.", "hip. T 2 weighted images", "AJCC Cancer Staging Manual"]
    n_yes = ["Pathologic stage: pT2c pN0.", "pNO (0/3)", "pNI (1/5)", "pNo, pMX", "pNX", "T2 NO M0",
             "AJCC/UICC stage: pT3bNOMX", "pT3a, N1", "pT2c,No", "T3a NO", "pT3aN1R1", "pT3b N0 M0 R0",
             "Pathologic Stage: pT3apN0pMX.", "Staging (pTNM): T 2c. N 0. M x.", "pT3b. pNO", "N X. M X"]
    n_no = ["no lymph nodes submitted", "pT2c Nodes negative", "No metastasis", "N 0 nodes", "Nx.",
            "1 of 7 lymph nodes positive for metastatic adenocarcinoma", "pT3a pMX", "(0/12)",
            "Pathological Staging (pTNM): p. T 3a. M X.", "No. Metastasis identified", "Lymph Node Status: Negative (0/3)"]
    for s in t_yes:
        assert printed_phrase(pt, s) is not None, s
    for s in t_no:
        assert printed_phrase(pt, s) is None, s
    for s in n_yes:
        assert printed_phrase(pn, s) is not None, s
    for s in n_no:
        assert printed_phrase(pn, s) is None, s
    assert printed_phrase(pt, "summary:  pT3b, max. diameter") == "pT3"


def test_join_fills_the_printed_columns_from_the_pack_patterns(tmp_path, capsys):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "TCGA-AA-0001.txt").write_text(REPORT, encoding="utf-8")                   # prints pT2c pN0
    (reports / "TCGA-AA-0002.txt").write_text(REPORT.replace("Pathologic stage: pT2c pN0.", "Grade Group 2."),
                                              encoding="utf-8")                           # prints a Grade Group only
    diag = {"primary_gleason_grade": "Pattern 3", "secondary_gleason_grade": "Pattern 4", "gleason_grade_group": None,
            "ajcc_pathologic_t": "T2c", "ajcc_pathologic_n": "N0"}
    cache = tmp_path / "gdc.json"
    cache.write_text(json.dumps([{"submitter_id": f"TCGA-AA-000{i}", "diagnoses": [diag]} for i in (1, 2, 3)]),
                     encoding="utf-8")
    out, report = tmp_path / "gold.csv", tmp_path / "mismatch.md"
    assert main(["--pack", str(PACKS / "prostate_pathology"), "--reports-dir", str(reports), "--cache", str(cache),
                 "--out", str(out), "--report", str(report)]) == 0
    with out.open(encoding="utf-8") as fh:
        rows = {r["case_id"]: r for r in csv.DictReader(fh)}
    assert list(rows["TCGA-AA-0001"]) == GOLD_COLUMNS
    assert (rows["TCGA-AA-0001"]["grade_group_printed"], rows["TCGA-AA-0001"]["pT_printed"],
            rows["TCGA-AA-0001"]["pN_printed"]) == ("False", "True", "True")
    assert (rows["TCGA-AA-0002"]["grade_group_printed"], rows["TCGA-AA-0002"]["pT_printed"],
            rows["TCGA-AA-0002"]["pN_printed"]) == ("True", "False", "False")
    assert (rows["TCGA-AA-0003"]["pT_printed"], rows["TCGA-AA-0003"]["complete"]) == ("", "True")   # no report: blank
    text = report.read_text(encoding="utf-8")
    assert "## pT printed in the report text (the pack's stated_pattern; gold column pT_printed)" in text
    assert "'pT2'=1" in text and "'Grade Group 2'=1" in text
    err = capsys.readouterr().err
    assert "reports that print a pN value (pN_printed, by the pack's stated_pattern): 1 of 2 staged" in err


ALLOWED = {"gleason_primary": [1, 2, 3, 4, 5], "gleason_secondary": [1, 2, 3, 4, 5],
           "gleason_total": [2, 3, 4, 5, 6, 7, 8, 9, 10], "grade_group": [1, 2, 3, 4, 5],
           "pT": ["pT2", "pT2a", "pT2b", "pT2c", "pT2+", "pT3", "pT3a", "pT3b", "pT4", "pTX"], "pN": ["pN0", "pN1", "pNX"]}


def test_label_row_derives_grade_group_and_maps_stage():
    hit = {"submitter_id": "TCGA-AA-0001", "diagnoses": [{"primary_gleason_grade": "Pattern 4",
                                                          "secondary_gleason_grade": "Pattern 3",
                                                          "gleason_grade_group": None,
                                                          "ajcc_pathologic_t": "T3a", "ajcc_pathologic_n": "N0"}]}
    row, notes = label_row(hit, ALLOWED)
    assert notes == []
    assert row == {"case_id": "TCGA-AA-0001", "gleason_primary": 4, "gleason_secondary": 3, "gleason_total": 7,
                   "grade_group": 3, "pT": "pT3a", "pN": "pN0", "grade_group_source": "derived_from_gleason",
                   "grade_group_printed": None, "pT_printed": None, "pN_printed": None, "complete": True}


def test_label_row_prefers_gdc_grade_group_and_flags_unmapped():
    hit = {"submitter_id": "TCGA-AA-0002", "diagnoses": [{"primary_gleason_grade": "Pattern 3",
                                                          "secondary_gleason_grade": "Pattern 4",
                                                          "gleason_grade_group": "Group 2",
                                                          "ajcc_pathologic_t": "T1c", "ajcc_pathologic_n": "Not Reported"}]}
    row, notes = label_row(hit, ALLOWED)
    assert row["grade_group"] == 2 and row["grade_group_source"] == "gdc"
    assert row["pT"] is None and row["pN"] is None and row["complete"] is False
    assert len(notes) == 1 and "pT='pT1c'" in notes[0]


def test_label_row_handles_missing_diagnosis():
    row, notes = label_row({"submitter_id": "TCGA-AA-0003"}, ALLOWED)
    assert all(row[c] is None for c in SCORED) and row["complete"] is False and notes == []

