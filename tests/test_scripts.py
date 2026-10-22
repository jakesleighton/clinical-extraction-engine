from scripts.join_gold_labels import (SCORED, label_row, parse_group, parse_pattern, parse_stage, printed_grade_group,
                                      printed_stage)


def test_parsers():
    assert parse_pattern("Pattern 3") == 3 and parse_pattern("4") == 4 and parse_pattern(None) is None
    assert parse_pattern("Not Reported") is None and parse_pattern("Pattern 7") is None
    assert parse_group("Group 2") == 2 and parse_group("") is None
    assert parse_stage("T3a", "T") == "pT3a" and parse_stage("pT2c", "T") == "pT2c" and parse_stage("TX", "T") == "pTX"
    assert parse_stage("N0", "N") == "pN0" and parse_stage("NX", "N") == "pNX" and parse_stage("N1", "N") == "pN1"
    assert parse_stage("Not Reported", "T") is None and parse_stage("N0", "T") is None


def test_printed_grade_group_regex():
    yes = ["Grade Group 2 (Gleason 3+4=7)", "GLEASON GRADE GROUP: 3", "ISUP grade 4", "ISUP 5", "(GG 1)",
           "Grade Group (ISUP) 2", "WHO grade group 3", "grade group of 4", "Grade group 2/5", "Grade Groups 1"]
    no = ["Gleason score 3+4=7", "Gleason 4+5=9 (Gleason 5: 20%)", "graded per ISUP 2005 consensus",
          "ISUP 2014 modified Gleason", "lymph node group 2", "an egg 3", "Grade group: not applicable"]
    for s in yes:
        assert printed_grade_group(s) is not None, s
    for s in no:
        assert printed_grade_group(s) is None, s
    assert printed_grade_group("... Prostate.  Grade   Group 2 ...") == "Grade Group 2"


def test_printed_stage_regex():
    t_yes = ["Pathologic stage: pT2c pN0.", "pT3a NO MX", "ypT2 N0", "TNM STAGE: AT LEAST pT2c, pNO, pMX",
             "AJCC/UICC stage: pT3bNOMX", "Stage: T2 NO M0, II B", "stage is T1", "STAGE = T3", "PT2",
             "T 3a N 0", "T2c, NO"]
    t_no = ["Gleason score 3+4=7", "pt. is a 65-year-old", "DEPT 3", "the stage of disease", "Tumor T1-weighted"]
    n_yes = ["Pathologic stage: pT2c pN0.", "pNO (0/3)", "pNI (1/5)", "pNo, pMX", "pNX", "T2 NO M0",
             "AJCC/UICC stage: pT3bNOMX", "pT3a, N1", "pT2c,No", "T3a NO", "pT3aN1R1", "pT3b N0 M0 R0"]
    n_no = ["no lymph nodes submitted", "pT2c Nodes negative", "No metastasis", "N 0 nodes", "Nx."]
    for s in t_yes:
        assert printed_stage(s, "T") is not None, s
    for s in t_no:
        assert printed_stage(s, "T") is None, s
    for s in n_yes:
        assert printed_stage(s, "N") is not None, s
    for s in n_no:
        assert printed_stage(s, "N") is None, s
    assert printed_stage("summary:  pT3b, max. diameter", "T") == "pT3"


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

