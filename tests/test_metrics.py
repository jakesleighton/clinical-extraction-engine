import math

import pytest

from engine.packloader import PackError
from evalharness.metrics import Aggregate, canon, compute_metrics, score_field
from evalharness.runner import load_gold


def pred(value=None, derivation="stated"):
    """A canonical record with one scored field set (or none)."""
    if value is None:
        env = {"status": "not_mentioned", "value": None, "derivation": None}
    else:
        env = {"status": "value", "value": value, "derivation": derivation}
    return {"fields": {"gleason_primary": env, "grade_group": dict(env)}}


def test_canon():
    assert canon("pT3a") == canon(" PT3A ") == "pt3a"
    assert canon(3) == canon("3") == canon(3.0) == canon("3.0") == 3
    assert canon(2.5) == canon("2.5") == 2.5
    assert canon(None) is None and canon("") is None and canon("  ") is None
    assert canon(True) == "true"


def test_score_field_hand_computed():
    gold = {"c1": {"gleason_primary": 3}, "c2": {"gleason_primary": 3}, "c3": {"gleason_primary": 3},
            "c4": {"gleason_primary": None}, "c5": {"gleason_primary": ""}}
    preds = {"c1": pred(3),          # TP
             "c2": pred(4),          # wrong: FP + FN
             "c3": pred(None),       # missing: FN
             "c4": pred(3),          # gold unknown: unscored
             "c5": pred(5)}          # gold blank: unscored
    s = score_field("gleason_primary", "standard", preds, gold)
    assert (s.n, s.tp, s.fp, s.fn, s.tn, s.predicted) == (3, 1, 1, 2, 0, 2)
    assert s.precision == 0.5 and s.recall == pytest.approx(1 / 3) and s.f1 == pytest.approx(0.4)
    assert s.accuracy == pytest.approx(1 / 3) and s.coverage == pytest.approx(2 / 3)


def test_score_field_absent_token_and_missing_record():
    gold = {"a": {"f": "__absent__"}, "b": {"f": "__absent__"}, "c": {"f": "x"}}
    preds = {"a": {"fields": {"f": {"status": "explicitly_absent", "value": None}}},     # TN
             "b": {"fields": {"f": {"status": "value", "value": "x"}}},                  # FP
             "c": None}                                                                  # no record: FN
    s = score_field("f", "hard", preds, gold)
    assert (s.n, s.tp, s.fp, s.fn, s.tn) == (3, 0, 1, 1, 1)
    assert s.precision == 0 and s.recall == 0 and s.f1 is None and s.accuracy == pytest.approx(1 / 3)


def test_empty_scores_are_none_not_zero_division():
    s = score_field("f", "standard", {}, {})
    assert s.n == 0 and s.precision is None and s.recall is None and s.f1 is None and s.coverage is None
    agg = Aggregate("x", [s])
    assert agg.micro_f1 is None and agg.macro_f1 is None


def test_compute_metrics_tiers_subsets_and_notes(prostate):
    gold = {
        "c1": {"gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7, "grade_group": 2, "pT": "pT2c", "pN": "pN0"},
        "c2": {"gleason_primary": 4, "gleason_secondary": 3, "gleason_total": 7, "grade_group": 3, "pT": "pT3a", "pN": "pN1"},
        "c3": {"gleason_primary": 3, "gleason_secondary": 3, "gleason_total": 6, "grade_group": 1, "pT": "pT2", "pN": ""},
    }

    def rec(gp, gs, gt, gg, gg_src, pt, pn):
        def env(v, src="stated"):
            return ({"status": "value", "value": v, "derivation": src} if v is not None
                    else {"status": "not_mentioned", "value": None, "derivation": None})
        return {"fields": {"gleason_primary": env(gp), "gleason_secondary": env(gs), "gleason_total": env(gt),
                           "grade_group": env(gg, gg_src), "pT": env(pt), "pN": env(pn)}}

    preds = {
        "c1": rec(3, 4, 7, 2, "computed_from_gleason", "pT2c", "pN0"),   # all right; GG derived
        "c2": rec(4, 3, 7, 2, "stated", "pT3a", "pN0"),                  # GG wrong (stated), pN wrong
        "c3": rec(3, 3, None, 1, "computed_from_gleason", "pT2", None),  # total missing; pN unscored
    }
    report = compute_metrics(prostate, preds, gold)
    by = {(s.field, s.subset): s for s in report.fields}

    assert [(s.field, s.subset) for s in report.fields] == [
        ("gleason_primary", "all"), ("gleason_secondary", "all"), ("gleason_total", "all"),
        ("grade_group", "all"), ("grade_group", "stated"), ("grade_group", "computed_from_gleason"),
        ("pT", "all"), ("pN", "all")]
    assert (by["gleason_primary", "all"].tp, by["gleason_primary", "all"].n) == (3, 3)
    assert (by["gleason_total", "all"].tp, by["gleason_total", "all"].fn) == (2, 1)
    assert (by["grade_group", "all"].tp, by["grade_group", "all"].fp, by["grade_group", "all"].fn) == (2, 1, 1)
    assert (by["grade_group", "stated"].n, by["grade_group", "stated"].tp) == (1, 0)
    assert (by["grade_group", "computed_from_gleason"].n, by["grade_group", "computed_from_gleason"].tp) == (2, 2)
    assert (by["pN", "all"].n, by["pN", "all"].tp, by["pN", "all"].fp) == (2, 1, 1)
    assert by["pT", "all"].tier == "deterministic" and by["gleason_primary", "all"].tier == "standard"

    tiers = {t.name: t for t in report.tiers}
    assert set(tiers) == {"deterministic", "standard"}
    assert [f.field for f in tiers["deterministic"].fields] == ["pT", "pN"]
    det = tiers["deterministic"]
    assert (det.tp, det.fp, det.fn) == (4, 1, 1)
    assert det.micro_precision == pytest.approx(0.8) and det.micro_recall == pytest.approx(0.8)
    assert det.micro_f1 == pytest.approx(0.8)
    assert det.macro_f1 == pytest.approx((1.0 + 0.5) / 2)
    # only the `all` rows feed the aggregates
    assert [f.field for f in report.overall.fields] == ["gleason_primary", "gleason_secondary", "gleason_total",
                                                        "grade_group", "pT", "pN"]
    assert report.n_cases == 3 and report.n_missing_predictions == 0
    assert any("counted twice" in n for n in report.notes)
    assert any("derivation label" in n for n in report.notes)

    md = report.to_markdown()
    assert "| grade_group | stated |" in md and "| deterministic |" in md and "Notes:" in md
    d = report.to_dict()
    assert d["overall"]["tp"] == report.overall.tp and math.isclose(d["fields"][0]["f1"], 1.0)


def test_load_gold_rejects_values_outside_vocabulary(prostate, tmp_path):
    good = tmp_path / "good.csv"
    good.write_text("case_id,gleason_primary,gleason_secondary,gleason_total,grade_group,pT,pN,grade_group_printed,extra\n"
                    "A,3,4,7,2,pT2c,pN0,True,ignored\n"
                    "B,,,,,pT3a,,,\n", encoding="utf-8")
    gold = load_gold(prostate, good)
    assert gold == {"A": {"gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7, "grade_group": 2,
                          "pT": "pT2c", "pN": "pN0", "grade_group_printed": True},
                    "B": {"gleason_primary": None, "gleason_secondary": None, "gleason_total": None,
                          "grade_group": None, "pT": "pT3a", "pN": None, "grade_group_printed": None}}

    bad = tmp_path / "bad.csv"
    bad.write_text("case_id,gleason_primary,gleason_secondary,gleason_total,grade_group,pT,pN,grade_group_printed\n"
                   "A,3,4,7,2,T2c,N0,False\n", encoding="utf-8")
    with pytest.raises(PackError, match="T2c"):
        load_gold(prostate, bad)

    badbool = tmp_path / "badbool.csv"
    badbool.write_text("case_id,gleason_primary,gleason_secondary,gleason_total,grade_group,pT,pN,grade_group_printed\n"
                       "A,3,4,7,2,pT2c,pN0,maybe\n", encoding="utf-8")
    with pytest.raises(PackError, match="true/false/blank"):
        load_gold(prostate, badbool)

    nocol = tmp_path / "nocol.csv"
    nocol.write_text("case_id,gleason_primary,gleason_secondary,gleason_total,grade_group,pT,pN\n"
                     "A,3,4,7,2,pT2c,pN0\n", encoding="utf-8")
    with pytest.raises(PackError, match="grade_group_printed"):
        load_gold(prostate, nocol)

    short = tmp_path / "short.csv"
    short.write_text("case_id,gleason_primary\nA,3\n", encoding="utf-8")
    with pytest.raises(PackError, match="missing columns"):
        load_gold(prostate, short)


def test_gold_anchored_subsets_and_provenance_note(prostate):
    def rec(gg, src):
        return {"fields": {"grade_group": {"status": "value", "value": gg, "derivation": src}}}
    gold = {"p1": {"grade_group": 2, "grade_group_printed": True},    # printed; model stated 2: TP
            "p2": {"grade_group": 3, "grade_group_printed": True},    # printed; model derived 3: TP, but missed-stated
            "n1": {"grade_group": 5, "grade_group_printed": False},   # not printed; model claims stated 4: hallucinated
            "n2": {"grade_group": 1, "grade_group_printed": False},   # not printed; derived 1: TP
            "u1": {"grade_group": 2, "grade_group_printed": None}}    # unknown: scored in `all` only
    preds = {"p1": rec(2, "stated"), "p2": rec(3, "computed_from_gleason"), "n1": rec(4, "stated"),
             "n2": rec(1, "computed_from_gleason"), "u1": rec(2, "stated")}
    report = compute_metrics(prostate, preds, gold, fields=["grade_group"])
    by = {s.subset: s for s in report.fields}
    assert (by["all"].n, by["all"].tp) == (5, 4)
    assert (by["stated"].n, by["stated"].tp) == (2, 2)
    assert (by["computed_from_gleason"].n, by["computed_from_gleason"].tp, by["computed_from_gleason"].fp) == (2, 1, 1)
    notes = " ".join(report.notes)
    assert "gold column `grade_group_printed`" in notes and "1 unknown" in notes
    assert "labelled 3 case(s) stated; 1 print the value, 1 do not (false stated claims: n1)" in notes
    assert "Of the 2 printing case(s), the model found 1 stated" in notes
    assert "Gleason-derived value" in notes                      # the pack's eval_note

    # without the column (in-memory gold from an older CSV) the subsets fall back to the model's label
    plain = {c: {"grade_group": g["grade_group"]} for c, g in gold.items()}
    report = compute_metrics(prostate, preds, plain, fields=["grade_group"])
    by = {s.subset: s for s in report.fields}
    assert by["stated"].n == 3 and by["computed_from_gleason"].n == 2
    assert any("derivation label" in n and "has no values" in n for n in report.notes)
