import pytest

from engine.rules import RuleError, compile_rule, grade_group_from, pn_from_nodes, pt_from_findings


@pytest.mark.parametrize("p, s, expected", [
    (3, 3, 1), (2, 3, 1), (3, 2, 1), (1, 1, 1),
    (3, 4, 2),
    (4, 3, 3), (5, 2, 3), (2, 5, 3),
    (4, 4, 4), (3, 5, 4), (5, 3, 4),
    (4, 5, 5), (5, 4, 5), (5, 5, 5),
    (0, 3, None), (6, 3, None), (3, None, None),
])
def test_grade_group_from(p, s, expected):
    assert grade_group_from(p, s) == expected


@pytest.mark.parametrize("procedure, epe, svi, laterality, expected", [
    ("radical_prostatectomy", None, "bilateral", "bilateral", "pT3b"),       # SVI wins over EPE
    ("radical_prostatectomy", "focal", "left", None, "pT3b"),
    ("radical_prostatectomy", "focal", None, "bilateral", "pT3a"),           # EPE without SVI
    ("radical_prostatectomy", "present_unspecified", None, None, "pT3a"),
    ("radical_prostatectomy", None, None, "bilateral", "pT2c"),              # organ-confined, both lobes
    ("radical_prostatectomy", None, None, "left", "pT2"),                    # unilateral: a or b unknown
    ("radical_prostatectomy", None, None, None, "pT2"),                      # laterality not given
    ("needle_biopsy", None, None, "bilateral", None),                        # no pathologic stage
    (None, None, None, None, None),
])
def test_pt_from_findings(procedure, epe, svi, laterality, expected):
    assert pt_from_findings(procedure, epe, svi, laterality) == expected


@pytest.mark.parametrize("examined, positive, expected", [
    (12, 0, "pN0"), (7, 1, "pN1"), (1, 0, "pN0"), (5, 5, "pN1"),
    (0, 0, None), (None, 0, None), (12, None, None), (-1, 0, None), (12, -1, None), (True, 0, None),
])
def test_pn_from_nodes(examined, positive, expected):
    assert pn_from_nodes(examined, positive) == expected


def test_stage_functions_are_whitelisted_for_packs():
    r = compile_rule("pt_from_findings(specimen_procedure, extraprostatic_extension, seminal_vesicle_invasion, tumor_laterality)")
    assert r.evaluate({"specimen_procedure": "radical_prostatectomy", "extraprostatic_extension": None,
                       "seminal_vesicle_invasion": None, "tumor_laterality": "bilateral"}) == "pT2c"
    assert compile_rule("pn_from_nodes(nodes_examined, nodes_positive)").evaluate(
        {"nodes_examined": 3, "nodes_positive": 1}) == "pN1"


def test_arithmetic_and_comparison():
    r = compile_rule("gleason_primary + gleason_secondary == gleason_total")
    assert r.names == {"gleason_primary", "gleason_secondary", "gleason_total"}
    assert r.evaluate({"gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 7}) is True
    assert r.evaluate({"gleason_primary": 3, "gleason_secondary": 4, "gleason_total": 8}) is False


def test_functions_boolean_and_membership():
    r = compile_rule("grade_group == grade_group_from(gleason_primary, gleason_secondary)")
    assert r.functions == {"grade_group_from"}
    assert r.evaluate({"grade_group": 2, "gleason_primary": 3, "gleason_secondary": 4})
    assert compile_rule("(pN == 'pN1') == (nodes_positive > 0)").evaluate({"pN": "pN0", "nodes_positive": 0})
    assert compile_rule("pT in ('pT3a', 'pT3b') and not margin").evaluate({"pT": "pT3a", "margin": False})
    assert compile_rule("1 <= a <= 5").evaluate({"a": 5})
    assert not compile_rule("1 <= a <= 5").evaluate({"a": 6})
    assert compile_rule("max(a, b) - abs(-c)").evaluate({"a": 1, "b": 9, "c": 4}) == 5


@pytest.mark.parametrize("expr", [
    "__import__('os').system('rm -rf /')",
    "a.__class__",
    "a[0]",
    "(lambda: 1)()",
    "a if b else c",
    "[x for x in a]",
    "open('f')",
    "grade_group_from(a, b=1)",
    "a = 1",
    "a +",
    "a ** 2",
])
def test_rejects_unsafe_or_unsupported(expr):
    with pytest.raises(RuleError):
        compile_rule(expr)


def test_evaluate_errors_are_rule_errors():
    with pytest.raises(RuleError, match="needs values"):
        compile_rule("a + b").evaluate({"a": 1})
    with pytest.raises(RuleError, match="failed"):
        compile_rule("a / b").evaluate({"a": 1, "b": 0})
    with pytest.raises(RuleError, match="failed"):
        compile_rule("a < b").evaluate({"a": None, "b": 1})


def test_compile_is_cached():
    assert compile_rule("a + b") is compile_rule("a + b")
