"""Per-field precision / recall / F1, grouped by tier.

Scoring rules (per field, per case):

* gold has a value
    * prediction has the same value         -> TP
    * prediction has a different value      -> FP and FN (wrong is both)
    * prediction has no value               -> FN
* gold explicitly says absent (``absent_token``, used by packs whose gold
  encodes "not determinable from the note")
    * prediction has no value               -> TN
    * prediction has a value                -> FP
* gold is empty/unknown                     -> the case is not scored for that field

A case with no prediction record at all (the model call failed) counts as
"no value" for every field. Values are compared after light canonicalization
(numbers numerically, strings case/whitespace-insensitively), so gold must be
written in the pack's vocabulary (``pT3a``, not ``T3a``).

Fields with a ``derived_fields`` entry are additionally reported on two
subsets, ``stated`` and ``<derivation>``. When the derived field declares a
``stated_column`` and the gold carries it, the split is **gold-anchored**:
``stated`` is the cases whose report prints the value, the other subset the
cases whose report does not, and a provenance note counts how often the
model's own ``derivation`` label disagrees with the gold (a "stated" claim on
a report that prints nothing is a hallucinated value). Without such a column
the subsets fall back to the model's own label, and the note says so.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from engine.packloader import Pack

DEFAULT_ABSENT_TOKEN = "__absent__"


def canon(v: Any) -> Any:
    """Comparable form of a value: ints for integral numbers, numeric strings
    parsed, other strings case-folded with whitespace collapsed."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return int(v) if float(v).is_integer() else float(v)
    s = " ".join(str(v).split()).casefold()
    if s == "":
        return None
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return s


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


@dataclass
class FieldScore:
    field: str
    tier: str
    subset: str = "all"
    n: int = 0           # scored cases
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    predicted: int = 0   # cases where the prediction had a value
    note: str | None = None

    @property
    def precision(self) -> float | None:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float | None:
        return _safe_div(self.tp + self.tn, self.n)

    @property
    def coverage(self) -> float | None:
        return _safe_div(self.predicted, self.n)

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "tier": self.tier, "subset": self.subset, "n": self.n,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "precision": self.precision, "recall": self.recall, "f1": self.f1,
                "accuracy": self.accuracy, "coverage": self.coverage, "note": self.note}


@dataclass
class Aggregate:
    name: str
    fields: list[FieldScore] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return sum(f.tp for f in self.fields)

    @property
    def fp(self) -> int:
        return sum(f.fp for f in self.fields)

    @property
    def fn(self) -> int:
        return sum(f.fn for f in self.fields)

    @property
    def micro_precision(self) -> float | None:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def micro_recall(self) -> float | None:
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def micro_f1(self) -> float | None:
        p, r = self.micro_precision, self.micro_recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def macro_f1(self) -> float | None:
        f1s = [f.f1 for f in self.fields if f.n > 0 and f.f1 is not None]
        return sum(f1s) / len(f1s) if f1s else None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "fields": [f.field for f in self.fields], "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "micro_precision": self.micro_precision, "micro_recall": self.micro_recall,
                "micro_f1": self.micro_f1, "macro_f1": self.macro_f1}


@dataclass
class MetricsReport:
    fields: list[FieldScore]
    tiers: list[Aggregate]
    overall: Aggregate
    n_cases: int
    n_missing_predictions: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_cases": self.n_cases, "n_missing_predictions": self.n_missing_predictions,
                "fields": [f.to_dict() for f in self.fields], "tiers": [t.to_dict() for t in self.tiers],
                "overall": self.overall.to_dict(), "notes": self.notes}

    def to_markdown(self) -> str:
        def pct(x: float | None) -> str:
            return "-" if x is None else f"{100 * x:.1f}"
        lines = [f"Cases scored: {self.n_cases} (no prediction for {self.n_missing_predictions})", "",
                 "| field | subset | tier | n | TP | FP | FN | P | R | F1 | acc | cov |",
                 "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for f in self.fields:
            lines.append(f"| {f.field} | {f.subset} | {f.tier} | {f.n} | {f.tp} | {f.fp} | {f.fn} | "
                         f"{pct(f.precision)} | {pct(f.recall)} | {pct(f.f1)} | {pct(f.accuracy)} | {pct(f.coverage)} |")
        lines += ["", "| group | fields | TP | FP | FN | micro-P | micro-R | micro-F1 | macro-F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for agg in [*self.tiers, self.overall]:
            lines.append(f"| {agg.name} | {len(agg.fields)} | {agg.tp} | {agg.fp} | {agg.fn} | "
                         f"{pct(agg.micro_precision)} | {pct(agg.micro_recall)} | {pct(agg.micro_f1)} | {pct(agg.macro_f1)} |")
        if self.notes:
            lines += ["", "Notes:"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"


def _prediction(record: dict[str, Any] | None, field_id: str) -> tuple[bool, Any, str | None]:
    """(has_value, value, derivation) for one field of one canonical record."""
    if not record:
        return False, None, None
    env = (record.get("fields") or {}).get(field_id)
    if not env or env.get("status") != "value" or env.get("value") is None:
        return False, None, None
    return True, env.get("value"), env.get("derivation")


def score_field(field_id: str, tier: str, predictions: dict[str, dict[str, Any] | None],
                gold: dict[str, dict[str, Any]], *, subset: str = "all",
                case_filter: set[str] | None = None, absent_token: str = DEFAULT_ABSENT_TOKEN) -> FieldScore:
    score = FieldScore(field=field_id, tier=tier, subset=subset)
    for case_id, gold_row in gold.items():
        if case_filter is not None and case_id not in case_filter:
            continue
        g = gold_row.get(field_id)
        if g is None or (isinstance(g, str) and g.strip() == ""):
            continue
        has_value, value, _ = _prediction(predictions.get(case_id), field_id)
        score.n += 1
        score.predicted += has_value
        if isinstance(g, str) and g.strip() == absent_token:
            if has_value:
                score.fp += 1
            else:
                score.tn += 1
        elif has_value and canon(value) == canon(g):
            score.tp += 1
        elif has_value:
            score.fp += 1
            score.fn += 1
        else:
            score.fn += 1
    return score


def compute_metrics(pack: Pack, predictions: dict[str, dict[str, Any] | None], gold: dict[str, dict[str, Any]],
                    *, fields: list[str] | None = None, absent_token: str = DEFAULT_ABSENT_TOKEN) -> MetricsReport:
    """Score ``predictions`` (case id -> canonical record) against ``gold``
    (case id -> {field id: value}) for the pack's scored fields."""
    field_ids = fields if fields is not None else (pack.scored_fields or pack.field_ids)
    scores: list[FieldScore] = []
    notes: list[str] = []
    for fid in field_ids:
        spec = pack.field(fid)
        main = score_field(fid, spec.tier or "standard", predictions, gold, absent_token=absent_token)
        scores.append(main)
        derived = pack.derived_for(fid)
        if derived is not None:
            tier = spec.tier or "standard"
            claimed = {c for c in gold if _prediction(predictions.get(c), fid)[2] == "stated"}
            col = derived.stated_column
            known = {c: gold[c].get(col) for c in gold if col and isinstance(gold[c].get(col), bool)}
            if known:
                stated = {c for c, v in known.items() if v}
                not_stated = {c for c, v in known.items() if not v}
                unknown = len(gold) - len(known)
                false_claims = sorted(claimed & not_stated)
                scores.append(score_field(fid, tier, predictions, gold, subset="stated",
                                          case_filter=stated, absent_token=absent_token))
                scores.append(score_field(fid, tier, predictions, gold, subset=derived.derivation,
                                          case_filter=not_stated, absent_token=absent_token))
                notes.append(f"{fid}: `all` is case-level (stated + derived). Subsets are gold-anchored on "
                             f"gold column `{col}`: `stated` = reports that print the value ({len(stated)}), "
                             f"`{derived.derivation}` = reports that do not ({len(not_stated)})"
                             + (f"; {unknown} unknown, scored in `all` only" if unknown else "") + ".")
                notes.append(f"{fid} provenance: the model labelled {len(claimed)} case(s) stated; "
                             f"{len(claimed & stated)} print the value, {len(false_claims)} do not"
                             + (f" (false stated claims: {', '.join(false_claims[:10])})" if false_claims else "")
                             + (f", {len(claimed - set(known))} unknown" if claimed - set(known) else "")
                             + f". Of the {len(stated)} printing case(s), the model found {len(stated & claimed)} stated.")
            else:
                computed = {c for c in gold if _prediction(predictions.get(c), fid)[2] == derived.derivation}
                scores.append(score_field(fid, tier, predictions, gold, subset="stated",
                                          case_filter=claimed, absent_token=absent_token))
                scores.append(score_field(fid, tier, predictions, gold, subset=derived.derivation,
                                          case_filter=computed, absent_token=absent_token))
                notes.append(f"{fid}: `all` is case-level (stated + derived). `stated` and `{derived.derivation}` "
                             f"subsets are defined by the extractor's own derivation label"
                             + (f" because gold column `{col}` has no values" if col else
                                " because the pack declares no `stated_column`")
                             + ", i.e. whether the model claims it found the value stated.")
            if derived.eval_note:
                notes.append(f"{fid}: {' '.join(derived.eval_note.split())}")

    by_tier: dict[str, list[FieldScore]] = defaultdict(list)
    for s in scores:
        if s.subset == "all":
            by_tier[s.tier].append(s)
    tier_order = ["deterministic", "standard", "hard"]
    tiers = [Aggregate(name=t, fields=by_tier[t]) for t in tier_order if by_tier.get(t)]
    overall = Aggregate(name="overall", fields=[s for s in scores if s.subset == "all"])
    missing = sum(1 for c in gold if not predictions.get(c))
    return MetricsReport(fields=scores, tiers=tiers, overall=overall, n_cases=len(gold),
                         n_missing_predictions=missing, notes=notes)
