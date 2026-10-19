"""Validate and normalize what the model returned; apply derived fields.

Three layers, in order:

1. **Shape** — the pydantic record model from ``engine/schema.py``. With
   constrained decoding this should always pass; if it does not, the server
   did not honor the grammar and we want to know.
2. **Semantics per field** — status/value agreement, required evidence,
   evidence quotes located in the source (character offsets), units on
   measurements, date formats. These are things a grammar cannot express.
3. **Cross-field rules** — the pack's ``validation`` expressions, run only
   when every field they mention has a value.

Then :func:`apply_derivations` fills ``derived_fields`` (e.g. Grade Group
from the Gleason patterns), recording provenance in ``derivation``.

Issues are never raised; they are collected with a ``severity`` (``error``:
the field's output is unusable; ``flag``: usable but worth a human look or,
in Phase 2, a retry) so one bad field never sinks a record.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import ValidationError

from engine.packloader import DerivedField, FieldSpec, Pack
from engine.rules import RuleError, compile_rule
from engine.schema import build_record_model

Severity = Literal["error", "flag"]


@dataclass
class Issue:
    code: str
    message: str
    field: str | None = None
    severity: Severity = "flag"
    rule_id: str | None = None
    on_fail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    record: dict[str, dict[str, Any]]      # field id -> normalized envelope
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def flags(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "flag"]


# ---------------------------------------------------------------------------
# Evidence spans
# ---------------------------------------------------------------------------

def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """Locate ``quote`` in ``text`` and return ``(start, end)`` offsets.

    Exact match first. Failing that, a whitespace- and case-insensitive match:
    OCR'd reports break lines in odd places and models tidy them up when they
    copy. The returned offsets always index the original text.
    """
    quote = quote.strip().strip('"“”').strip()
    if not quote:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote)

    norm_chars: list[str] = []
    index_map: list[int] = []
    prev_space = False
    for pos, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                norm_chars.append(" ")
                index_map.append(pos)
            prev_space = True
        else:
            norm_chars.append(ch.casefold())
            index_map.append(pos)
            prev_space = False
    norm_text = "".join(norm_chars)
    norm_quote = " ".join(quote.casefold().split())
    j = norm_text.find(norm_quote)
    if j < 0:
        return None
    start = index_map[j]
    end = index_map[j + len(norm_quote) - 1] + 1
    return start, end


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y",
                 "%d %B %Y", "%d %b %Y", "%m-%d-%Y", "%Y%m%d")


def normalize_date(value: str) -> str | None:
    """ISO ``YYYY-MM-DD`` (or ``YYYY-MM`` / ``YYYY`` when that is all there is);
    ``None`` when the string is not a recognizable date."""
    s = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            datetime.strptime(s, "%Y-%m-%d")
            return s
        except ValueError:
            return None
    if re.fullmatch(r"\d{4}-\d{2}", s) or re.fullmatch(r"\d{4}", s):
        return s
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    m = re.fullmatch(r"(\d{1,2})/(\d{4})", s)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    return None


def _empty_envelope() -> dict[str, Any]:
    return {"status": "not_mentioned", "value": None, "unit": None, "hedging": None,
            "conflict": False, "evidence": [], "derivation": None}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_output(pack: Pack, parsed: dict[str, Any], text: str) -> ValidationResult:
    """Check ``parsed`` (the model's JSON) against the pack and the source
    ``text``; return the normalized record plus issues."""
    issues: list[Issue] = []
    record: dict[str, dict[str, Any]] = {}
    Record = build_record_model(pack)

    for extra in set(parsed) - set(pack.field_ids):
        issues.append(Issue("unknown_field", f"model returned a field the pack does not declare: {extra}",
                            field=extra))

    for f in pack.fields:
        out = _empty_envelope()
        raw_env = parsed.get(f.id)
        if raw_env is None:
            issues.append(Issue("missing_field", f"{f.id}: missing from the model output", f.id, "error"))
            record[f.id] = out
            continue
        env_model = Record.model_fields[f.id].annotation
        try:
            env = env_model.model_validate(raw_env)      # type: ignore[union-attr]
        except ValidationError as e:
            msgs = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
            issues.append(Issue("schema", f"{f.id}: output does not match the schema ({msgs})", f.id, "error"))
            record[f.id] = out
            continue

        out["status"] = env.status
        out["hedging"] = env.hedging
        out["conflict"] = bool(env.conflict)
        value = env.value
        unit = getattr(env, "unit", None)

        if env.status == "value":
            if value is None:
                issues.append(Issue("missing_value", f"{f.id}: status is value but value is null", f.id, "error"))
            else:
                value, unit = _normalize_value(f, value, unit, issues)
                out["value"], out["unit"], out["derivation"] = value, unit, "stated"
        else:
            if value is not None:
                issues.append(Issue("value_without_status",
                                    f"{f.id}: value {value!r} given with status {env.status}; value dropped", f.id))

        quotes = [q.strip() for q in env.evidence if isinstance(q, str) and q.strip()]
        if env.status != "not_mentioned" and not quotes and f.evidence == "required":
            issues.append(Issue("evidence_missing", f"{f.id}: status {env.status} needs an evidence quote", f.id))
        for q in quotes:
            span = find_span(text, q)
            if span is None:
                issues.append(Issue("evidence_not_found", f"{f.id}: quote not found in source: {q[:80]!r}", f.id))
                out["evidence"].append({"quote": q, "start": None, "end": None})
            else:
                out["evidence"].append({"quote": q, "start": span[0], "end": span[1]})
        record[f.id] = out

    _run_rules(pack, record, issues)
    return ValidationResult(record=record, issues=issues)


def _normalize_value(f: FieldSpec, value: Any, unit: Any, issues: list[Issue]) -> tuple[Any, Any]:
    if f.type == "string":
        return " ".join(str(value).split()), None
    if f.type == "date":
        iso = normalize_date(str(value))
        if iso is None:
            issues.append(Issue("bad_date", f"{f.id}: {value!r} is not a recognizable date", f.id))
            return str(value).strip(), None
        return iso, None
    if f.type == "measurement":
        if unit is None:
            issues.append(Issue("missing_unit", f"{f.id}: measurement has no unit (allowed: {f.units})", f.id))
        return _tidy_number(value), unit
    if f.type == "number":
        return _tidy_number(value), f.unit
    return value, None


def _tidy_number(v: Any) -> Any:
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def values_of(record: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Field id -> value, for fields that have one."""
    return {fid: env["value"] for fid, env in record.items()
            if env["status"] == "value" and env["value"] is not None}


def _run_rules(pack: Pack, record: dict[str, dict[str, Any]], issues: list[Issue]) -> None:
    values = values_of(record)
    for rule in pack.validation:
        compiled = compile_rule(rule.rule)
        if not compiled.names <= set(values):
            continue                                        # a referenced field has no value: rule does not apply
        try:
            ok = bool(compiled.evaluate(values))
        except RuleError as e:
            issues.append(Issue("rule_error", f"{rule.id}: {e}", rule_id=rule.id, on_fail=rule.on_fail))
            continue
        if not ok:
            bound = ", ".join(f"{n}={values[n]!r}" for n in sorted(compiled.names))
            issues.append(Issue("rule_failed", f"{rule.id}: `{rule.rule}` is false ({bound})",
                                rule_id=rule.id, on_fail=rule.on_fail))


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def _weakest_hedging(pack: Pack, levels: list[str | None]) -> str | None:
    order = list(pack.hedging_levels)
    present = [h for h in levels if h in order]
    return max(present, key=order.index) if present else None


def apply_derivations(pack: Pack, record: dict[str, dict[str, Any]], issues: list[Issue]) -> None:
    """Fill ``derived_fields`` in place. The model never calculates; this does,
    deterministically, and records provenance in ``derivation``."""
    for d in pack.derived_fields:
        env = record[d.id]
        stated = env["status"] == "value" and env["value"] is not None
        if stated and d.when == "not_stated":
            continue
        inputs = {i: record[i] for i in d.inputs}
        if not all(e["status"] == "value" and e["value"] is not None for e in inputs.values()):
            continue                                        # cannot derive; leave the field as extracted
        try:
            computed = compile_rule(d.rule).evaluate({i: e["value"] for i, e in inputs.items()})
        except RuleError as e:
            issues.append(Issue("derivation_error", f"{d.id}: {e}", d.id))
            continue
        spec = pack.field(d.id)
        if computed is None or (spec.allowed is not None and computed not in spec.allowed):
            issues.append(Issue("derivation_failed",
                                f"{d.id}: rule produced {computed!r}, not an allowed value", d.id))
            continue
        if stated and env["value"] != computed:
            issues.append(Issue("derivation_disagrees",
                                f"{d.id}: stated {env['value']!r} but {d.derivation} gives {computed!r}", d.id))
        env["status"], env["value"], env["derivation"] = "value", computed, d.derivation
        env["hedging"] = _weakest_hedging(pack, [e["hedging"] for e in inputs.values()])
        env["conflict"] = any(e["conflict"] for e in inputs.values())
        if d.evidence == "inherit":
            seen: set[tuple[Any, ...]] = set()
            merged: list[dict[str, Any]] = []
            for e in inputs.values():
                for ev in e["evidence"]:
                    key = (ev["quote"], ev["start"], ev["end"])
                    if key not in seen:
                        seen.add(key)
                        merged.append(dict(ev))
            env["evidence"] = merged


def _unused(_: DerivedField) -> None:   # keeps the import meaningful for type checkers
    return None
