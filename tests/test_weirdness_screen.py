"""scripts.weirdness_screen: the heuristics on small texts, the site guards, and one end-to-end run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.join_gold_labels import GOLD_COLUMNS
from scripts.weirdness_screen import (ADDENDUM, TREATMENT, Flag, discount_site_templates, gdc_conflicts,
                                      gleason_pairs, hedging_density, main, negated, ocr_score, part_count, rank,
                                      site_of)
from tests.conftest import PACKS

PACK = str(PACKS / "prostate_pathology")


def test_addendum_pattern_skips_clia_boilerplate():
    assert not ADDENDUM.search("certified under the Clinical Laboratory Improvement Amendments of 1988 (CLIA)")
    assert not ADDENDUM.search("Clinical Laboratory Improvement. Amendments of 1988")      # OCR period
    assert ADDENDUM.search("ADDENDUM: The positive margin was reviewed")
    assert ADDENDUM.search("Amendments. Reason: Clerical Error.")
    assert ADDENDUM.search("will be presented in a follow-up report. Follow-up report.")
    assert ADDENDUM.search("COPY ONLY DO NOT FILE. Correction to results. Correction.")
    assert ADDENDUM.search("The TNM stage of the tumor is revised as: TNM stage: pT3a")
    assert ADDENDUM.search("SURGICAL PATHOLOGY REVISED REPORT") and ADDENDUM.search("Signed Supplemental.")
    assert not ADDENDUM.search("CAP Protocol (revised June 2012)")


def test_amended_value_flag_needs_addendum_language_and_two_values():
    from scripts.weirdness_screen import screen
    corrected = ("FINAL DIAGNOSIS: adenocarcinoma, Gleason 3+4=7. TNM STAGE: AT LEAST pT2c, pNO, pMX. "
                 "Signed Supplemental. After review the tumor penetrates the capsule. "
                 "The TNM stage of the tumor is revised as: pT3a, pNo, pMX.")
    two_stages_no_addendum = "Gleason 3+4=7. Stage pT2c on the right, pT3a on the left. pN0."
    addendum_same_values = "Gleason 3+4=7, pT2c pN0. ADDENDUM: immunostains confirm the diagnosis. pT2c pN0."
    reports = {"TCGA-AA-0001": corrected, "TCGA-AA-0002": two_stages_no_addendum, "TCGA-AA-0003": addendum_same_values}
    flags = screen(reports, {}, {}, tails=0)
    names = {c: {fl.name for fl in fs} for c, fs in flags.items()}
    assert {"addendum", "pT_multi", "amended_value"} <= names["TCGA-AA-0001"]
    assert "pT_multi" in names["TCGA-AA-0002"] and "amended_value" not in names["TCGA-AA-0002"]
    assert "addendum" in names["TCGA-AA-0003"] and "amended_value" not in names["TCGA-AA-0003"]


def test_gleason_pairs_forms_and_history():
    text = ("History of biopsy: Gleason score 3+4=7 in two cores. FINAL: Prostatic adenocarcinoma, "
            "Gleason score 7 (4+3). Gleason score (primary + secondary): 4 + 3 = 7.")
    assert gleason_pairs(text) == [(3, 4, True), (4, 3, False)]
    synoptic = "PRIMARY GLEASON GRADE: 3. SECONDARY GLEASON GRADE: 4. GLEASON SUM SCORE: 7."
    assert gleason_pairs(synoptic) == [(3, 4, False)]
    assert gleason_pairs("Gleason score 7. No patterns given.") == []


def test_negation_and_treatment():
    t = "9. Treatment effect: Not identified. 10. Margins: negative."
    m = TREATMENT.search(t)
    assert negated(t, m)
    t2 = "atrophic glands suggestive of preoperative hormone deprivation therapy, tumor shows"
    assert not negated(t2, TREATMENT.search(t2))
    t3 = "Evidence of neo-adjuvant treatment: Not specified."
    assert negated(t3, TREATMENT.search(t3))


def test_stage_values_including_compact_tnm():
    from scripts.weirdness_screen import PN_TEXT, PT_TEXT, stage_values
    assert stage_values("Pathologic stage: pT3a pN0. Final: pT 3b, pN1 (2/10).", PT_TEXT, "T") == ["pT3a", "pT3b"]
    assert stage_values("Pathologic stage: pT3a pN0. Final: pT 3b, pN1 (2/10).", PN_TEXT, "N") == ["pN0", "pN1"]
    assert stage_values("Conclusion: Prostate adenocarcinoma, Gleson score 7. pT2N1. ICD-O", PT_TEXT, "T") == ["pT2"]
    assert stage_values("Conclusion: Prostate adenocarcinoma, Gleson score 7. pT2N1. ICD-O", PN_TEXT, "N") == ["pN1"]
    assert stage_values("pT2c pNX; no pT4 here but pt3 lower-case", PT_TEXT, "T") == ["pT2c", "pT4", "pT3"]
    assert stage_values("Stage: PT3b N1; earlier PT 2 and p T3b, pT2C", PT_TEXT, "T") == ["pT3b", "pT2", "pT2c"]
    assert stage_values("pT2c pNX; also pT3bNx and PT3b N1", PN_TEXT, "N") == ["pNX", "pN1"]
    assert stage_values("Dept 3 reviewed. pTNM: pT2 N0; mptotal 3", PT_TEXT, "T") == ["pT2"]
    # a letter O typed for zero: 139 reports say "pNO", 100+ say "pT3a NO MX" or "pT3bNOMX"
    assert stage_values("TNM STAGE: AT LEAST pT2c, pNO, pMX. revised as: pT3a, pNo, pMX", PN_TEXT, "N") == ["pN0"]
    assert stage_values("AJCC/UICC stage: pT3bNOMX. Other: pT2c NO Mx. pT3bNOR1 and pT2cNOL1", PN_TEXT, "N") == ["pN0"]
    assert stage_values("pT2 NO TUMOR SEEN. pT3a No residual tumor. pT2NO PASSED", PN_TEXT, "N") == []


def test_part_count_letters_and_numbers():
    assert part_count("A. PROSTATE: tumor. B. LYMPH NODES: benign. C. FAT: benign.") == 3
    assert part_count("1. Left node. 2. Right node. 3. Prostate. 4. Bladder neck.") == 4
    assert part_count("Only prose, no parts.") == 0


def test_ocr_score_rises_with_fragmented_words():
    clean = "The prostate shows adenocarcinoma, Gleason score 3+4=7, confined to the gland, margins negative."
    damaged = "Sur gi cal Pat hol ogy Repor t. Speci men Dat e. I nqui ry. Pr ost at e wi th ade noc arc ino ma."
    assert ocr_score(clean) < ocr_score(damaged)
    assert ocr_score("I n q u i r y") >= 3                      # spaced-out letters


def test_hedging_density_ignores_gross_description_phrases():
    gross = "Dissection reveals twelve possible lymph nodes. Nineteen possible unremarkable lymph nodes."
    assert hedging_density(gross)[0] == 0
    diag = "Findings suspicious for extraprostatic extension; margin status is questionable and equivocal."
    assert hedging_density(diag)[0] == 3


def test_site_template_discount_and_per_site_cap():
    flags = {f"TCGA-KK-{i:04d}": [Flag("deferred", 3, "x"), Flag("short", 2, "y")] for i in range(6)}
    flags["TCGA-CH-0001"] = [Flag("deferred", 3, "x")]
    flags["TCGA-KK-0000"] = [Flag("deferred", 3, "x")]            # only one KK case lacks `short`
    discount_site_templates(flags)
    kk = flags["TCGA-KK-0001"]
    assert [f.weight for f in kk] == [1, 1] and all("site template" in f.detail for f in kk)
    assert flags["TCGA-CH-0001"][0].weight == 3                   # CH has one report: no discount
    reports = {c: "x" * 100 for c in flags}
    assert site_of("TCGA-KK-0001") == "KK"
    top = rank(flags, reports, top=10, per_site=2)
    assert top[0] == "TCGA-CH-0001" and sum(c.startswith("TCGA-KK") for c in top) == 2


def test_gdc_conflicts(tmp_path):
    cache = tmp_path / "gdc.json"
    cache.write_text(json.dumps([
        {"submitter_id": "TCGA-CH-5764", "diagnoses": [{"ajcc_pathologic_t": "T3b", "ajcc_pathologic_n": "N0"},
                                                       {"ajcc_pathologic_t": "T3", "ajcc_pathologic_n": "N1"}]},
        {"submitter_id": "TCGA-CH-5790", "diagnoses": [{"ajcc_pathologic_t": "T2c", "ajcc_pathologic_n": "N0"},
                                                       {"ajcc_pathologic_t": "T1", "ajcc_pathologic_n": None}]},
        {"submitter_id": "TCGA-AA-0001", "diagnoses": [{"ajcc_pathologic_t": "T2c", "ajcc_pathologic_n": "N0"}]},
    ]), encoding="utf-8")
    conflicts = gdc_conflicts(cache)
    assert conflicts["TCGA-CH-5764"].startswith("pN:") and "pT3b pN0 vs pT3 pN1" in conflicts["TCGA-CH-5764"]
    assert conflicts["TCGA-CH-5790"].startswith("pT:") and "TCGA-AA-0001" not in conflicts
    assert gdc_conflicts(tmp_path / "missing.json") == {}


def test_end_to_end(tmp_path, capsys):
    reports = tmp_path / "reports"
    reports.mkdir()
    plain = ("FINAL DIAGNOSIS: Prostate, radical prostatectomy: prostatic adenocarcinoma, Gleason score 3+4=7, "
             "confined to the prostate, margins negative. pT2c pN0. " * 3)
    for i in range(1, 6):
        (reports / f"TCGA-AA-{i:04d}.txt").write_text(plain + f"case {i}\n", encoding="utf-8")
    weird = ("Diagnosis: adenocarcinoma of the prostate, Gleason 4+3=7. amended after consultation. "
             "Immunohistochemistry pending; results will be presented in a follow-up report. "
             "Treatment effect: present, consistent with hormone therapy.\n")
    (reports / "TCGA-BB-0001.txt").write_text(weird, encoding="utf-8")
    gold = tmp_path / "gold.csv"
    with gold.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_COLUMNS)
        w.writeheader()
        for i in range(1, 6):
            w.writerow({"case_id": f"TCGA-AA-{i:04d}", "gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7,
                        "grade_group": 2, "pT": "pT2c", "pN": "pN0", "complete": True})
        w.writerow({"case_id": "TCGA-BB-0001", "gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7,
                    "grade_group": 2, "pT": "", "pN": "pN0", "complete": False})
    out = tmp_path / "candidates.md"
    assert main(["--pack", PACK, "--reports-dir", str(reports), "--gold", str(gold),
                 "--gdc-cache", str(tmp_path / "none.json"), "--out", str(out), "--top", "3", "--tails", "0"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "### 1. TCGA-BB-0001" in text
    assert "text Gleason 4+3 but GDC says 3+4" in text and "deferred" in text and "no pT in GDC" in text
    assert "treatment-related finding" in text and "amended" in text
    assert "6 reports screened" in text
    assert "warning: GDC cache not found" in capsys.readouterr().err
