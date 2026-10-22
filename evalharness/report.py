"""Everything ``report.md`` carries beyond the headline scores, computed from
what a run saved (the canonical records, the ledger, the report texts) and
never from a model call: the group split (random-20 against weird-5), the
scores without the audited gold cases, validator flag counts, per-field
status and quote behaviour, a classification of the quotes the matcher could
not find, and cost. The runner calls :func:`build_run_report` at the end of a
live run; ``evalharness.rescore`` calls it again on a saved run, so a report
can always be recomputed from the ledger without touching a model.

Inputs that live in the repository:

* a groups CSV (``case_id,group``; a case may belong to several groups), for
  example ``data/tcga_prad/case_groups.csv``;
* ``data/gold_notes.md``, whose table lists the audited cases (gold that the
  report text does not support, or that is finer than the text); every row's
  first cell is a case id and the whole case is left out of the "without
  audited cases" scores.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.packloader import Pack
from evalharness.metrics import MetricsReport, compute_metrics, pct

# ---------------------------------------------------------------------------
# Inputs: groups and gold notes
# ---------------------------------------------------------------------------


def read_groups(path: str | Path) -> dict[str, list[str]]:
    """``case_id,group`` CSV (header required) -> {group: [case ids]} in file
    order. A case listed under two groups is in both."""
    groups: dict[str, list[str]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"case_id", "group"} <= set(reader.fieldnames):
            raise ValueError(f"{path}: expected columns case_id,group; has {reader.fieldnames}")
        for row in reader:
            case_id, group = (row.get("case_id") or "").strip(), (row.get("group") or "").strip()
            if case_id and group and case_id not in groups.setdefault(group, []):
                groups[group].append(case_id)
    return groups


@dataclass(frozen=True)
class GoldNote:
    case_id: str
    fields: tuple[str, ...]
    kind: str
    gold: str
    report: str
    note: str


_TABLE_ROW = re.compile(r"^\|(.*)\|\s*$")


def read_gold_notes(path: str | Path) -> list[GoldNote]:
    """The audited cases in a gold-notes file: every row of a Markdown table
    whose columns are ``case | field(s) | kind | gold | report says | note``
    (further columns are ignored). Header and separator rows are skipped, so
    the file can carry prose around the table."""
    notes: list[GoldNote] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _TABLE_ROW.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 6:
            continue
        head = cells[0]
        if not head or head.lower() in ("case", "case_id", "case id") or set(head) <= set("-: "):
            continue
        fields = tuple(f.strip() for f in cells[1].split(",") if f.strip())
        notes.append(GoldNote(case_id=head, fields=fields, kind=cells[2], gold=cells[3],
                              report=cells[4], note=cells[5]))
    return notes


def audited_case_ids(notes: list[GoldNote]) -> list[str]:
    seen: list[str] = []
    for n in notes:
        if n.case_id not in seen:
            seen.append(n.case_id)
    return seen


# ---------------------------------------------------------------------------
# Validator flags and per-field status
# ---------------------------------------------------------------------------


def flag_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Issue counts by severity and code over the run's records, with the
    cases and fields each code touched."""
    count: Counter[tuple[str, str]] = Counter()
    cases: dict[tuple[str, str], set[str]] = {}
    fields: dict[tuple[str, str], Counter[str]] = {}
    missing: list[str] = []
    for r in records:
        if not r.get("fields"):
            missing.append(r.get("case_id") or "?")
        for i in r.get("issues") or []:
            key = (i.get("severity") or "?", i.get("code") or "?")
            count[key] += 1
            cases.setdefault(key, set()).add(r.get("case_id") or "?")
            where = i.get("field") or i.get("rule_id")
            if where:
                fields.setdefault(key, Counter())[where] += 1
    rows = []
    for key in sorted(count):
        rows.append({"severity": key[0], "code": key[1], "count": count[key], "cases": len(cases[key]),
                     "fields": dict(fields.get(key, Counter()).most_common(8))})
    return {"records": len(records), "records_missing": missing, "issues": rows}


def _located(ev: dict[str, Any]) -> str | None:
    """Which locator pass found a quote: the stored ``match`` kind; ``legacy``
    for records written before the kind was stored; None when not found."""
    if ev.get("match"):
        return str(ev["match"])
    return "legacy" if ev.get("start") is not None else None


def field_status(pack: Pack, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per field: how the run's records are distributed over the three
    statuses, how often the engine filled the value (a non-stated
    ``derivation``), hedging and conflict counts, and quote behaviour. The
    hard tier has no gold, so this is the only view of it."""
    definite = next(iter(pack.hedging_levels), "definite")
    rows = []
    for f in pack.fields:
        st: Counter[str] = Counter()
        hedged = conflict = derived = quotes = not_found = 0
        for r in records:
            env = (r.get("fields") or {}).get(f.id)
            if not env:
                st["no_record"] += 1
                continue
            st[env.get("status") or "?"] += 1
            hedged += env.get("hedging") not in (None, definite)
            conflict += bool(env.get("conflict"))
            own = env.get("derivation") in (None, "stated")
            derived += not own
            if own:                                   # inherited evidence is a copy of the inputs' quotes
                for ev in env.get("evidence") or []:
                    quotes += 1
                    not_found += _located(ev) is None
        rows.append({"field": f.id, "tier": f.tier or "standard", "value": st["value"],
                     "explicitly_absent": st["explicitly_absent"], "not_mentioned": st["not_mentioned"],
                     "no_record": st["no_record"], "derived": derived, "hedged": hedged, "conflict": conflict,
                     "quotes": quotes, "quotes_not_found": not_found})
    return rows


# ---------------------------------------------------------------------------
# Quotes the matcher could not find: how far off are they?
# ---------------------------------------------------------------------------

QUOTE_CLASSES = ("normalized", "stitched", "one_token", "partial", "other", "empty")


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"-\s*\n\s*", "", s)                  # a word hyphenated across a line break is one word
    s = re.sub(r"\.(?!\d)", " ", s)                  # sentence and OCR line-break periods go; decimals stay
    s = re.sub(r"[^a-z0-9%./+<>=]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify_quote(quote: str, text: str) -> str:
    """How a quote the locator could not find relates to the text:

    normalized  the same words, only punctuation or line breaks differ (the
                locator should have found it; a count here is a matcher bug)
    stitched    fragments joined with "..." that are each real text
    one_token   one dropped or inserted word
    partial     starts or ends on real text (first or last four words found)
                but the rest departs from it
    other       paraphrase, reordering or fabrication: nothing anchors
    """
    nq, nt = _norm(quote), _norm(text)
    if not nq:
        return "empty"
    if nq in nt:
        return "normalized"
    if re.search(r"\.\.\.|…", quote):
        frags = [f for f in (_norm(x) for x in re.split(r"\.\.\.|…", quote)) if f]
        if frags and all(f in nt for f in frags):
            return "stitched"
    qt, tt, n = nq.split(), nt.split(), len(nq.split())
    for i in range(n):
        cand = " ".join(qt[:i] + qt[i + 1:])
        if n > 2 and cand and (" " + cand + " ") in (" " + nt + " "):
            return "one_token"
    for start in range(0, max(0, len(tt) - n)):
        window = tt[start:start + n + 1]
        for j in range(len(window)):
            if window[:j] + window[j + 1:] == qt:
                return "one_token"
    if n >= 4 and (" ".join(qt[:4]) in nt or " ".join(qt[-4:]) in nt):
        return "partial"
    return "other"


def quote_summary(records: list[dict[str, Any]], texts: dict[str, str]) -> dict[str, Any]:
    """Which locator pass found each quote, and a class for every quote no
    pass found. Derived fields are skipped (their evidence is copied from
    the inputs). ``texts`` maps case id to the report text."""
    located: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    by_field: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str]]] = {}
    for r in records:
        case_id = r.get("case_id") or "?"
        for fid, env in (r.get("fields") or {}).items():
            if env.get("derivation") not in (None, "stated"):
                continue
            for ev in env.get("evidence") or []:
                how = _located(ev)
                if how:
                    located[how] += 1
                    continue
                cls = classify_quote(str(ev.get("quote") or ""), texts.get(case_id, ""))
                classes[cls] += 1
                by_field[fid] += 1
                ex = examples.setdefault(cls, [])
                if len(ex) < 3:
                    ex.append({"case_id": case_id, "field": fid, "quote": str(ev.get("quote") or "")[:100]})
    return {"located": dict(located), "not_found": sum(classes.values()), "classes": dict(classes),
            "not_found_by_field": dict(by_field.most_common(8)), "examples": examples}


# ---------------------------------------------------------------------------
# Cost, from the ledger's llm_call lines
# ---------------------------------------------------------------------------


# Above this share of the context window the largest call is one longer report
# or thinking trace away from being cut off, which Ollama does without an error.
CONTEXT_PRESSURE_LIMIT = 0.9


def cost_summary(calls: list[dict[str, Any]], wall_s: float | None = None) -> dict[str, Any]:
    """Token, latency and window figures from the run's ``llm_call`` lines.
    ``context_pressure`` is the largest call's tokens (prompt plus output) over
    ``num_ctx``; ``context_pressure_high`` says it passed
    ``CONTEXT_PRESSURE_LIMIT``, and the report then carries a warning."""
    lat = [c["latency_s"] for c in calls if isinstance(c.get("latency_s"), (int, float))]
    tin = [c["tokens_in"] for c in calls if isinstance(c.get("tokens_in"), int)]
    tout = [c["tokens_out"] for c in calls if isinstance(c.get("tokens_out"), int)]
    think: Counter[str] = Counter()
    done: Counter[str] = Counter()
    for c in calls:
        meta = c.get("meta") or {}
        think[f"{meta.get('think')!r}" + (" (rejected by the server)" if meta.get("think_rejected") else "")] += 1
        done[str(meta.get("done_reason"))] += 1
    params = dict(calls[0].get("params") or {}) if calls else {}
    num_ctx = params.get("num_ctx")
    peak = max((c.get("tokens_in") or 0) + (c.get("tokens_out") or 0) for c in calls) if calls else None
    pressure = peak / num_ctx if peak and num_ctx else None
    return {"calls": len(calls), "errors": sum(1 for c in calls if c.get("error")),
            "tokens_in": sum(tin), "tokens_in_mean": round(statistics.mean(tin)) if tin else None,
            "tokens_out": sum(tout), "tokens_out_mean": round(statistics.mean(tout)) if tout else None,
            "latency_median_s": round(statistics.median(lat), 1) if lat else None,
            "latency_max_s": round(max(lat), 1) if lat else None,
            "latency_sum_s": round(sum(lat)) if lat else None,
            "wall_s": None if wall_s is None else round(wall_s),
            "think": dict(think), "done_reason": dict(done), "params": params,
            "num_ctx": num_ctx, "peak_tokens": peak,
            "context_pressure": (round(pressure, 2) if pressure is not None else None),
            "context_pressure_high": (pressure is not None and pressure > CONTEXT_PRESSURE_LIMIT)}


def context_warning(cost: dict[str, Any]) -> str | None:
    """The note a report carries when the largest call passed the limit."""
    if not cost.get("context_pressure_high"):
        return None
    return (f"context pressure: the largest call used {cost['peak_tokens']:,} of {cost['num_ctx']:,} tokens "
            f"({100 * cost['peak_tokens'] / cost['num_ctx']:.1f}% of the window, above {pct(CONTEXT_PRESSURE_LIMIT, 0)}%). A reply or "
            f"thinking trace that outgrows the window is cut off or shifted without an error: raise --num-ctx, or "
            f"let the auto-sizer reserve its think-aware headroom (4096 tokens with thinking off, 8192 on).")


# ---------------------------------------------------------------------------
# One run's report
# ---------------------------------------------------------------------------


@dataclass
class RunReport:
    metrics: MetricsReport
    header: list[str]
    groups: dict[str, MetricsReport] = field(default_factory=dict)
    without_audited: MetricsReport | None = None
    audited_in_run: list[str] = field(default_factory=list)
    audited_notes: list[GoldNote] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    status: list[dict[str, Any]] = field(default_factory=list)
    quotes: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)       # e.g. context pressure; rendered under Notes

    # the fields older callers read straight off the metrics
    @property
    def n_cases(self) -> int:
        return self.metrics.n_cases

    @property
    def notes(self) -> list[str]:
        return [*self.header, *self.metrics.notes]

    def to_dict(self) -> dict[str, Any]:
        return {**self.metrics.to_dict(), "notes": self.notes,
                "groups": {g: m.to_dict() for g, m in self.groups.items()},
                "without_audited": self.without_audited.to_dict() if self.without_audited else None,
                "audited_in_run": self.audited_in_run,
                "validator": self.flags, "field_status": self.status, "quotes": self.quotes, "cost": self.cost,
                "warnings": self.warnings}

    def to_markdown(self) -> str:
        return render_markdown(self)


def build_run_report(pack: Pack, records: list[dict[str, Any]], gold: dict[str, dict[str, Any]], *,
                     texts: dict[str, str], calls: list[dict[str, Any]], wall_s: float | None = None,
                     groups: dict[str, list[str]] | None = None, gold_notes: list[GoldNote] | None = None,
                     header: list[str] | None = None) -> RunReport:
    """Score ``records`` against ``gold`` (only the cases in ``gold`` are
    scored) and gather everything else the report shows."""
    predictions = {r["case_id"]: (r if r.get("fields") else None) for r in records}
    cases = [c for c in gold if c in predictions]
    scored_gold = {c: gold[c] for c in cases}
    rr = RunReport(metrics=compute_metrics(pack, predictions, scored_gold), header=list(header or []))

    for name, members in (groups or {}).items():
        sub = {c: gold[c] for c in cases if c in set(members)}
        if sub:
            rr.groups[name] = compute_metrics(pack, {c: predictions[c] for c in sub}, sub)
    if gold_notes:
        audited = set(audited_case_ids(gold_notes))
        rr.audited_in_run = [c for c in cases if c in audited]
        rr.audited_notes = [n for n in gold_notes if n.case_id in set(rr.audited_in_run)]
        kept = {c: gold[c] for c in cases if c not in audited}
        rr.without_audited = compute_metrics(pack, {c: predictions[c] for c in kept}, kept)
    rr.flags = flag_summary(records)
    rr.status = field_status(pack, records)
    rr.quotes = quote_summary(records, texts)
    rr.cost = cost_summary(calls, wall_s)
    warning = context_warning(rr.cost)
    if warning:
        rr.warnings.append(warning)
    return rr


def _agg_line(name: str, m: MetricsReport) -> str:
    cells = []
    for agg_name in ("deterministic", "standard", "hard", "overall", "overall (lenient)"):
        agg = m.aggregate(agg_name)
        cells.append(pct(agg.micro_f1) if agg else "-")
    return f"| {name} | {m.n_cases} | " + " | ".join(cells) + " |"


def render_markdown(rr: RunReport) -> str:
    out: list[str] = []
    if rr.header:
        out += [f"- {h}" for h in rr.header] + [""]
    out += ["## Scores (exact match is the headline; `lenient_*` rows and `(lenient)` aggregates are labelled)", "",
            rr.metrics.tables_markdown()]

    if rr.groups:
        out += ["## By group", "",
                "Micro-F1 per tier. A case can belong to more than one group.", "",
                "| group | n | deterministic | standard | hard | overall | overall (lenient) |",
                "|---|---:|---:|---:|---:|---:|---:|",
                _agg_line("all", rr.metrics)]
        out += [_agg_line(g, m) for g, m in rr.groups.items()]
        names = list(rr.groups)
        out += ["", "F1 per field (`all` subset rows):", "",
                "| field | all | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
        for f in rr.metrics.fields:
            if f.subset != "all":
                continue
            cells = []
            for g in names:
                row = next((x for x in rr.groups[g].fields if x.field == f.field and x.subset == "all"), None)
                cells.append(pct(row.f1) if row else "-")
            out.append(f"| {f.field} | {pct(f.f1)} | " + " | ".join(cells) + " |")
        out.append("")

    if rr.without_audited is not None:
        n_aud = len(rr.audited_in_run)
        out += ["## Without the audited gold cases", ""]
        if n_aud:
            out += [f"{n_aud} audited case(s) in this run, left out below as whole cases: "
                    + ", ".join(rr.audited_in_run) + ".", ""]
            out += ["| case | field(s) | kind | gold | report says |", "|---|---|---|---|---|"]
            out += [f"| {n.case_id} | {', '.join(n.fields)} | {n.kind} | {n.gold} | {n.report} |" for n in rr.audited_notes]
            out.append("")
        else:
            out += ["None of the audited cases is in this run; the scores below equal the headline.", ""]
        out += ["| set | n | deterministic | standard | hard | overall | overall (lenient) |",
                "|---|---:|---:|---:|---:|---:|---:|",
                _agg_line("all", rr.metrics), _agg_line("without audited", rr.without_audited), ""]
        out += ["| field | all | without audited |", "|---|---:|---:|"]
        for f in rr.metrics.fields:
            if f.subset != "all":
                continue
            row = next((x for x in rr.without_audited.fields if x.field == f.field and x.subset == "all"), None)
            out.append(f"| {f.field} | {pct(f.f1)} | {pct(row.f1) if row else '-'} |")
        out.append("")

    fl = rr.flags
    out += ["## Validator", ""]
    if fl:
        out += [f"Records: {fl['records']}, without a usable record (bad JSON or server error): "
                f"{len(fl['records_missing'])}" + (" (" + ", ".join(fl["records_missing"]) + ")" if fl["records_missing"] else "") + ".", ""]
        if fl["issues"]:
            out += ["| severity:code | count | cases | fields / rules (top) |", "|---|---:|---:|---|"]
            for i in fl["issues"]:
                where = ", ".join(f"{k} {v}" for k, v in i["fields"].items()) or "-"
                out.append(f"| {i['severity']}:{i['code']} | {i['count']} | {i['cases']} | {where} |")
        else:
            out.append("No validator issues.")
        out.append("")
    if rr.status:
        out += ["Per field (every field, the hard tier included; `derived` = the engine filled the value; "
                "quotes exclude inherited evidence):", "",
                "| field | tier | value | explicitly_absent | not_mentioned | derived | hedged | conflict | quotes | not found |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in rr.status:
            out.append(f"| {s['field']} | {s['tier']} | {s['value']} | {s['explicitly_absent']} | {s['not_mentioned']} | "
                       f"{s['derived']} | {s['hedged']} | {s['conflict']} | {s['quotes']} | {s['quotes_not_found']} |")
        out.append("")

    q = rr.quotes
    if q:
        loc = q["located"]
        out += ["## Quotes", "",
                "Located by pass: " + (", ".join(f"{k} {v}" for k, v in loc.items()) or "none")
                + f". Not found: {q['not_found']}" + (" (" + ", ".join(f"{k} {q['classes'].get(k, 0)}" for k in QUOTE_CLASSES
                                                                     if q["classes"].get(k)) + ")" if q["not_found"] else "") + ".",
                "", "`normalized` = same words, only punctuation or line breaks differ (should be 0: the locator "
                "covers that); `stitched` = real fragments joined with \"...\"; `one_token` = one dropped or inserted "
                "word; `partial` = starts or ends on real text but the rest departs; `other` = paraphrase or "
                "fabrication. Stitched and one-token quotes are what a retry loop fixes."]
        if q["not_found_by_field"]:
            out.append("Not found by field: " + ", ".join(f"{k} {v}" for k, v in q["not_found_by_field"].items()) + ".")
        for cls in QUOTE_CLASSES:
            for ex in q["examples"].get(cls, []):
                out.append(f"- {cls}: {ex['case_id']} {ex['field']}: \"{ex['quote']}\"")
        out.append("")

    c = rr.cost
    if c:
        out += ["## Cost", "",
                "| calls | errors | tokens in (sum / mean) | tokens out (sum / mean) | median s | max s | sum s | wall s | "
                "num_ctx | peak in+out | think | done_reason |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
                f"| {c['calls']} | {c['errors']} | {c['tokens_in']:,} / {c['tokens_in_mean'] or '-'} | "
                f"{c['tokens_out']:,} / {c['tokens_out_mean'] or '-'} | {c['latency_median_s'] or '-'} | "
                f"{c['latency_max_s'] or '-'} | {c['latency_sum_s'] or '-'} | {c['wall_s'] or '-'} | "
                f"{c['num_ctx'] or '-'} | {c['peak_tokens'] or '-'}"
                + (f" ({pct(c['context_pressure'], 0)}% of the window)" if c["context_pressure"] else "") + " | "
                + (", ".join(f"{k} x{v}" for k, v in c["think"].items()) or "-") + " | "
                + (", ".join(f"{k} x{v}" for k, v in c["done_reason"].items()) or "-") + " |", ""]
        if c["params"]:
            out.append("Params recorded on every call: `" + ", ".join(f"{k}={v}" for k, v in c["params"].items()) + "`.")
            out.append("")

    if rr.metrics.notes or rr.warnings:
        out += ["Notes:"] + [f"- {n}" for n in [*rr.metrics.notes, *rr.warnings]]
    return "\n".join(out).rstrip() + "\n"
