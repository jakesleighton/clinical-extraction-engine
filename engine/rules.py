"""Pack rule expressions, evaluated without ``eval()``.

Packs carry small expressions such as
``gleason_primary + gleason_secondary == gleason_total`` (validation) and
``grade_group_from(gleason_primary, gleason_secondary)`` (derivation).

Why not ``eval``: a pack is data, and public packs will be written by other
people. ``eval`` would run anything they typed. Instead we parse the expression
with the standard ``ast`` module and walk the tree ourselves, allowing only
arithmetic, comparisons, boolean logic, field names, constants and calls to
the functions whitelisted in ``FUNCTIONS``. Anything else is a load-time
error, so a bad rule fails when the pack is loaded, not mid-run.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from functools import lru_cache
from typing import Any


class RuleError(Exception):
    """A rule failed to parse, references something unknown, or blew up."""


# --------------------------------------------------------------------------
# Whitelisted function library. Packs may call these by name. Add a function
# here (and a test) rather than special-casing anything in engine code.
# --------------------------------------------------------------------------

def grade_group_from(primary: int, secondary: int) -> int | None:
    """ISUP/WHO Grade Group from the two Gleason patterns.

    Gleason <= 6 -> 1; 3+4 -> 2; 4+3 -> 3; 8 -> 4; 9-10 -> 5. Returns None for
    patterns outside 1-5. A score of 7 with an unusual pattern pair (e.g. 5+2)
    follows the primary pattern: primary 3 -> 2, otherwise 3.
    """
    if primary not in (1, 2, 3, 4, 5) or secondary not in (1, 2, 3, 4, 5):
        return None
    total = primary + secondary
    if total <= 6:
        return 1
    if total == 7:
        return 2 if primary == 3 else 3
    if total == 8:
        return 4
    return 5


def pt_from_findings(procedure: str | None, extraprostatic_extension: Any, seminal_vesicle_invasion: Any,
                     laterality: str | None = None) -> str | None:
    """Pathologic T category of a radical prostatectomy from the extent
    findings (AJCC 7th edition, the system the TCGA gold uses).

    A finding argument is its extracted value when present and ``None`` when
    the report says it is absent (the engine passes ``None`` for
    ``explicitly_absent``; a finding the report never mentions blocks the
    derivation before this runs). Seminal vesicle invasion -> pT3b;
    extraprostatic extension without it -> pT3a; neither -> organ-confined:
    pT2c when the tumor is bilateral, otherwise pT2 (the findings cannot
    separate pT2a from pT2b). ``None`` for any other procedure (a biopsy has
    no pathologic stage), and never pT4, which needs the adjacent-organ
    findings the pack does not extract.
    """
    if procedure != "radical_prostatectomy":
        return None
    if seminal_vesicle_invasion is not None:
        return "pT3b"
    if extraprostatic_extension is not None:
        return "pT3a"
    return "pT2c" if laterality == "bilateral" else "pT2"


def pn_from_nodes(nodes_examined: Any, nodes_positive: Any) -> str | None:
    """pN1 when any examined regional node carries metastasis, pN0 when at
    least one node was examined and none does, ``None`` when no node was
    examined (no nodes, no pathologic N)."""
    if isinstance(nodes_examined, bool) or isinstance(nodes_positive, bool):
        return None
    if not isinstance(nodes_examined, int) or not isinstance(nodes_positive, int):
        return None
    if nodes_examined < 1 or nodes_positive < 0:
        return None
    return "pN1" if nodes_positive > 0 else "pN0"


FUNCTIONS: dict[str, Any] = {
    "grade_group_from": grade_group_from,
    "pt_from_findings": pt_from_findings,
    "pn_from_nodes": pn_from_nodes,
    "abs": abs,
    "min": min,
    "max": max,
}

# --------------------------------------------------------------------------
# The evaluator
# --------------------------------------------------------------------------

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_UNARY = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class Rule:
    """A parsed expression. ``names`` are the field ids it reads."""

    def __init__(self, expr: str):
        self.expr = expr
        try:
            self._tree = ast.parse(expr.strip(), mode="eval")
        except SyntaxError as e:
            raise RuleError(f"cannot parse rule {expr!r}: {e.msg}") from None
        names: set[str] = set()
        functions: set[str] = set()
        self._check(self._tree.body, names, functions)
        self.names = frozenset(names)
        self.functions = frozenset(functions)

    # -- static check at parse time ------------------------------------------
    def _check(self, node: ast.AST, names: set[str], functions: set[str]) -> None:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, str, bool, type(None))):
                raise RuleError(f"unsupported constant in rule {self.expr!r}")
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise RuleError(f"unsupported operator in rule {self.expr!r}")
            self._check(node.left, names, functions)
            self._check(node.right, names, functions)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                raise RuleError(f"unsupported operator in rule {self.expr!r}")
            self._check(node.operand, names, functions)
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                self._check(v, names, functions)
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                if type(op) not in _COMPARE:
                    raise RuleError(f"unsupported comparison in rule {self.expr!r}")
            self._check(node.left, names, functions)
            for c in node.comparators:
                self._check(c, names, functions)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for e in node.elts:
                self._check(e, names, functions)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise RuleError(
                    f"rule {self.expr!r} calls a function that is not whitelisted; "
                    f"allowed: {sorted(FUNCTIONS)}"
                )
            if node.keywords:
                raise RuleError(f"keyword arguments are not supported in rule {self.expr!r}")
            functions.add(node.func.id)
            for a in node.args:
                self._check(a, names, functions)
        else:
            raise RuleError(
                f"unsupported syntax ({type(node).__name__}) in rule {self.expr!r}; "
                "rules may use arithmetic, comparisons, and/or/not, field names and "
                "whitelisted functions"
            )

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, values: Mapping[str, Any]) -> Any:
        missing = self.names - set(values)
        if missing:
            raise RuleError(f"rule {self.expr!r} needs values for {sorted(missing)}")
        try:
            return self._eval(self._tree.body, values)
        except RuleError:
            raise
        except Exception as e:  # TypeError on None comparisons, ZeroDivisionError, ...
            raise RuleError(f"rule {self.expr!r} failed: {type(e).__name__}: {e}") from None

    def _eval(self, node: ast.AST, values: Mapping[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](self._eval(node.left, values), self._eval(node.right, values))
        if isinstance(node, ast.UnaryOp):
            return _UNARY[type(node.op)](self._eval(node.operand, values))
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._eval(v, values) for v in node.values)
            return any(self._eval(v, values) for v in node.values)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, values)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, values)
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self._eval(e, values) for e in node.elts)
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            return FUNCTIONS[node.func.id](*(self._eval(a, values) for a in node.args))
        raise RuleError(f"unsupported syntax in rule {self.expr!r}")

    def __repr__(self) -> str:
        return f"Rule({self.expr!r})"


@lru_cache(maxsize=256)
def compile_rule(expr: str) -> Rule:
    """Parse once, reuse everywhere (the loader, the validator, the deriver)."""
    return Rule(expr)
