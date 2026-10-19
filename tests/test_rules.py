import pytest

from engine.rules import RuleError, compile_rule, grade_group_from


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
