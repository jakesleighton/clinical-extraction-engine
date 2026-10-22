"""Validate and normalize what the model returned; apply derived fields.

Three layers, in order:

1. **Shape** — the pydantic record model from ``engine/schema.py``. With
   constrained decoding this should always pass; if it does not, the server
   did not honor the grammar and we want to know.
2. **Semantics per field** — status/value agreement, required evidence,
   evidence quotes located in the source (character offsets, and how strict
   a match it took), units on measurements, date formats. These are things a
   grammar cannot express.
3. **Stated-value guard** — for a field the pack derives with a
   ``stated_pattern``, a value the model calls stated must be quoted from
   text that prints one; otherwise it is set aside for the derivation
   (:func:`apply_stated_guards`).
4. **Cross-field rules** — the pack's ``validation`` expressions, run only
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

# Characters the third pass may ignore: sentence punctuation, quotes, brackets,
# dashes and bullets. Symbols that carry a value stay significant (+ = % / < > &),
# and a period, comma or hyphen between two digits (2.0 cm, 8,6 ng/ml, 3-4 mm,
# 2010-10-12) stays too, so "2.0 cm" can never match "20 cm".
_IGNORABLE = set(".,;:!?'\"`“”‘’()[]{}-–—_*|•·")
_DIGIT_JOINERS = ".,-"

MatchKind = Literal["exact", "whitespace", "punctuation"]


def _kept(text: str, pos: int) -> bool:
    ch = text[pos]
    if ch.isspace():
        return False
    if ch not in _IGNORABLE:
        return True
    return (ch in _DIGIT_JOINERS and 0 < pos < len(text) - 1
            and text[pos - 1].isdigit() and text[pos + 1].isdigit())


def _squeeze(text: str) -> tuple[str, list[int]]:
    """``text`` with ignorable characters removed and the rest lower-cased, plus
    a map from each kept character back to its offset in ``text``."""
    chars: list[str] = []
    index_map: list[int] = []
    for pos, ch in enumerate(text):
        if _kept(text, pos):
            for c in ch.lower():            # a few letters lower-case to two characters
                chars.append(c)
                index_map.append(pos)
    return "".join(chars), index_map


def locate(text: str, quote: str) -> tuple[int, int, MatchKind] | None:
    """Locate ``quote`` in ``text``: ``(start, end, how)`` or ``None``.

    Three passes, each only tried when the previous one fails, so a quote
    that matched before this policy still gets the same offsets:

    1. ``exact``: the quote as given.
    2. ``whitespace``: whitespace runs and letter case ignored. OCR'd reports
       break lines in odd places and models tidy them up when they copy.
    3. ``punctuation``: punctuation, dashes and line-break characters ignored
       as well (decision of 2026-10-22, a narrow amendment to the exact-quote
       policy). The TCGA-Reports OCR put a period at every original line
       break ("prostate. adenocarcinoma"), and a model that quotes the sentence
       without it should not be flagged. Value-bearing symbols and digit
       separators are still significant, and a dropped or added word is still
       a miss: the retry loop owns those.

    The returned offsets always index the original text and land on the real
    span, whichever pass found it.
    """
    quote = quote.strip().strip('"“”').strip()
    if not quote:
        return None
    i = text.find(quote)
    if i >= 0:
        return i, i + len(quote), "exact"

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
            for c in ch.casefold():
                norm_chars.append(c)
                index_map.append(pos)
            prev_space = False
    norm_text = "".join(norm_chars)
    norm_quote = " ".join(quote.casefold().split())
    j = norm_text.find(norm_quote)
    if j >= 0:
        return index_map[j], index_map[j + len(norm_quote) - 1] + 1, "whitespace"

    squeezed, squeeze_map = _squeeze(text)
    squeezed_quote, _ = _squeeze(quote)
    if not squeezed_quote:
        return None
    k = squeezed.find(squeezed_quote)
    if k < 0:
        return None
    return squeeze_map[k], squeeze_map[k + len(squeezed_quote) - 1] + 1, "punctuation"


def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """``(start, end)`` offsets of ``quote`` in ``text`` (see :func:`locate`), or ``None``."""
    hit = locate(text, quote)
    return None if hit is None else (hit[0], hit[1])


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
            hit = locate(text, q)
            if hit is None:
                issues.append(Issue("evidence_not_found", f"{f.id}: quote not found in source: {q[:80]!r}", f.id))
                out["evidence"].append({"quote": q, "start": None, "end": None, "match": None})
            else:
                out["evidence"].append({"quote": q, "start": hit[0], "end": hit[1], "match": hit[2]})
        record[f.id] = out

    apply_stated_guards(pack, record, issues, text)
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
# Stated-value guard
# ---------------------------------------------------------------------------

def apply_stated_guards(pack: Pack, record: dict[str, dict[str, Any]], issues: list[Issue], text: str) -> None:
    """For every derived field that declares a ``stated_pattern``: a value the
    model returned (provenance ``stated``) stands only when one of its evidence
    quotes prints a value of that kind, i.e. the regex matches the quoted span
    of the source (the quote as given, when it was not located). Otherwise the
    envelope is emptied (``not_mentioned``, no evidence) and the record flagged
    ``stated_without_printed_value``, so :func:`apply_derivations` can fill
    the field from the findings with the provenance it deserves.

    This is the mechanical half of "only a printed stage counts". A model that
    returns the node finding "1 of 7 nodes positive" as pN1 has the right
    value and the wrong provenance, and in the Phase 1 re-baseline no
    instruction wording stopped that for every model. The gold join
    (scripts/join_gold_labels.py) runs the same regex over the whole report
    to fill the field's ``stated_column``, so the eval's stated/derived split
    and this guard agree by construction. The guard asks whether a value of
    the kind is printed, not which: a stated pT3 quoted from "pT2c" is a
    value error for the metric, not a provenance error for the guard.
    """
    for d in pack.derived_fields:
        regex = d.stated_regex
        if regex is None:
            continue
        env = record[d.id]
        if env["status"] != "value" or env["value"] is None:
            continue
        spans = [text[ev["start"]:ev["end"]] if ev.get("start") is not None else ev["quote"]
                 for ev in env["evidence"]]
        if any(regex.search(s) for s in spans):
            continue
        quoted = "; ".join(repr(s[:60]) for s in spans) or "no evidence"
        issues.append(Issue("stated_without_printed_value",
                            f"{d.id}: {env['value']!r} returned as stated but no evidence quote prints a {d.id} "
                            f"value ({quoted}); set aside for the derivation", d.id))
        record[d.id] = _empty_envelope()


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------

def _weakest_hedging(pack: Pack, levels: list[str | None]) -> str | None:
    order = list(pack.hedging_levels)
    present = [h for h in levels if h in order]
    return max(present, key=order.index) if present else None


def _derivation_inputs(d: DerivedField, record: dict[str, dict[str, Any]]) \
        -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Values the rule runs on, per input mode, and the envelopes that
    contributed (for inherited evidence, hedging and conflict). ``None`` when
    an input the rule needs is not there, which means: cannot derive.

    required  must carry a value
    finding   a value when present; ``None`` when the document says the finding
              is absent (``explicitly_absent``: that is information); silence blocks
    optional  a value when present, else ``None``; never blocks
    count     a value when present; ``0`` when the document says there are none
              (``explicitly_absent`` on a count is a zero); silence blocks
    """
    values: dict[str, Any] = {}
    used: list[dict[str, Any]] = []
    for fid, mode in d.input_modes.items():
        e = record[fid]
        if e["status"] == "value" and e["value"] is not None:
            values[fid] = e["value"]
            used.append(e)
        elif mode == "finding" and e["status"] == "explicitly_absent":
            values[fid] = None
            used.append(e)
        elif mode == "count" and e["status"] == "explicitly_absent":
            values[fid] = 0
            used.append(e)
        elif mode == "optional":
            values[fid] = None
        else:
            return None, []
    return values, used


def _agrees(stated: Any, computed: Any) -> bool:
    """A stated value agrees with the computed one when equal, or when it is a
    coarser form of it (a printed ``pT3`` against a computed ``pT3a``)."""
    if stated == computed:
        return True
    if isinstance(stated, str) and isinstance(computed, str):
        a, b = stated.strip().casefold(), computed.strip().casefold()
        return a.startswith(b) or b.startswith(a)
    return False


def apply_derivations(pack: Pack, record: dict[str, dict[str, Any]], issues: list[Issue]) -> None:
    """Fill ``derived_fields`` in place. The model never calculates; this does,
    deterministically, and records provenance in ``derivation``.

    ``when`` decides what happens to a value the model stated: ``not_stated``
    leaves it alone and never computes; ``check`` computes, flags a
    disagreement (``derivation_disagrees``) and keeps the stated value; ``always``
    computes, flags a disagreement and overwrites.
    """
    for d in pack.derived_fields:
        env = record[d.id]
        stated = env["status"] == "value" and env["value"] is not None
        if stated and d.when == "not_stated":
            continue
        values, inputs = _derivation_inputs(d, record)
        if values is None:
            continue                                        # cannot derive; leave the field as extracted
        try:
            computed = compile_rule(d.rule).evaluate(values)
        except RuleError as e:
            issues.append(Issue("derivation_error", f"{d.id}: {e}", d.id))
            continue
        spec = pack.field(d.id)
        if computed is None:
            if d.on_none == "flag":
                issues.append(Issue("derivation_failed", f"{d.id}: rule produced None, not an allowed value", d.id))
            continue                                        # on_none: skip -> the inputs do not pin a value
        if spec.allowed is not None and computed not in spec.allowed:
            issues.append(Issue("derivation_failed",
                                f"{d.id}: rule produced {computed!r}, not an allowed value", d.id))
            continue
        if stated and not _agrees(env["value"], computed):
            issues.append(Issue("derivation_disagrees",
                                f"{d.id}: stated {env['value']!r} but {d.derivation} gives {computed!r}", d.id))
        if stated and d.when == "check":
            continue                                        # the document's own statement stands, flagged above
        env["status"], env["value"], env["derivation"] = "value", computed, d.derivation
        env["hedging"] = _weakest_hedging(pack, [e["hedging"] for e in inputs])
        env["conflict"] = any(e["conflict"] for e in inputs)
        if d.evidence == "inherit":
            seen: set[tuple[Any, ...]] = set()
            merged: list[dict[str, Any]] = []
            for e in inputs:
                for ev in e["evidence"]:
                    key = (ev["quote"], ev["start"], ev["end"])
                    if key not in seen:
                        seen.add(key)
                        merged.append(dict(ev))
            env["evidence"] = merged


def _unused(_: DerivedField) -> None:   # keeps the import meaningful for type checkers
    return None
