"""Build the TCGA-PRAD gold label CSV for the prostate pack.

    python -m scripts.join_gold_labels

Pulls Gleason primary/secondary, Grade Group, pT and pN for every TCGA-PRAD
case from the GDC API (``diagnoses.*`` fields), maps them into the pack's
vocabulary, derives Gleason total and (when GDC has no Grade Group) the
Grade Group from the patterns, aligns to the staged reports, and writes:

* ``packs/prostate_pathology/labels/tcga_prad_gold.csv`` — one row per case,
  including three "printed" columns found by regex over the staged report
  text: ``grade_group_printed`` (the report literally prints a Grade Group:
  "Grade Group 2", "ISUP 3", "GG 4"), ``pT_printed`` and ``pN_printed`` (the
  report prints a pathologic T / N category: "pT3a", "T2c N0", "Stage: T2";
  the letter O stands for zero in 139+ reports). The eval anchors its
  stated/derived subsets on these columns, never on the model's own claim.
  Blank when the case has no staged report.
* ``data/tcga_prad/label_mismatch_report.md`` — reports without labels,
  labels without reports, incomplete labels, raw values seen and unmapped

The working set (which cases the smoke test and the eval run) is a separate
step: ``python -m scripts.working_set random --n 20`` for a seeded draw, or
``python -m scripts.working_set pick ID ...`` for hand-picked cases.

The raw GDC response is cached at ``data/tcga_prad/gdc_cases.json`` so
re-runs are offline and the source data stays inspectable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from engine.packloader import load_pack
from engine.rules import grade_group_from
from scripts._gdc import fetch_project_cases

PROJECT = "TCGA-PRAD"
GDC_FIELDS = [
    "submitter_id",
    "diagnoses.primary_gleason_grade",
    "diagnoses.secondary_gleason_grade",
    "diagnoses.gleason_grade_group",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
]
GOLD_COLUMNS = ["case_id", "gleason_primary", "gleason_secondary", "gleason_total", "grade_group",
                "pT", "pN", "grade_group_source", "grade_group_printed", "pT_printed", "pN_printed", "complete"]
PRINTED_COLUMNS = {"grade_group_printed": "grade_group", "pT_printed": "pT", "pN_printed": "pN"}
SCORED = ["gleason_primary", "gleason_secondary", "gleason_total", "grade_group", "pT", "pN"]
_MISSING = {"", "not reported", "unknown", "not available", "not applicable", "--", "none"}


def _clean(s: Any) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    return None if s.lower() in _MISSING else s


def parse_pattern(raw: Any) -> int | None:
    """``"Pattern 3"`` / ``"3"`` -> 3."""
    s = _clean(raw)
    if s is None:
        return None
    m = re.search(r"([1-5])\b", s)
    return int(m.group(1)) if m else None


def parse_group(raw: Any) -> int | None:
    """``"Group 2"`` / ``"2"`` -> 2."""
    s = _clean(raw)
    if s is None:
        return None
    m = re.search(r"([1-5])\b", s)
    return int(m.group(1)) if m else None


def parse_stage(raw: Any, prefix: str) -> str | None:
    """GDC ``ajcc_pathologic_t`` ``"T3a"`` -> ``"pT3a"``; ``"N0"`` -> ``"pN0"``.
    Returns None for missing values or values that are not that axis."""
    s = _clean(raw)
    if s is None:
        return None
    s = s.replace(" ", "")
    if s[:1].lower() == "p":
        s = s[1:]
    if not s[:1].upper() == prefix.upper():
        return None
    rest = s[1:]
    rest = "X" if rest.upper() == "X" else rest.lower()      # pT3a, pT2c, pTX
    return "p" + prefix.upper() + rest


_PRINTED_GG = re.compile(
    r"\b(?:grade\s*groups?|isup(?:\s*(?:grade|group))?|gg)(?:\s*\(isup\))?\s*(?:grade\s*)?[:=\-]?\s*(?:of\s+)?([1-5])\b",
    re.IGNORECASE)


def printed_grade_group(text: str) -> str | None:
    """The phrase where the report literally prints a Grade Group ("Grade Group 2",
    "ISUP grade 3", "GG 4"), or None. A Gleason score alone does not count, and
    neither does "ISUP 2005" (a consensus citation, not a grade)."""
    m = _PRINTED_GG.search(text)
    return " ".join(m.group(0).split()) if m else None


# A printed pathologic T category: "pT3a", "ypT2c", "PT2" (one all-caps site),
# "T2c N0" / "T 3a NO" (a TNM string without the p), "Stage: T2" / "stage is T1".
# "pt" in lower case is never a stage (it is the start of "pt." or a word), so
# the p-forms are case-sensitive; the stage-word form is not.
_PRINTED_PT = re.compile(
    r"(?<![A-Za-z])[ymr]?[pP]T\s?[1-4]"
    r"|\bT\s?[1-4][a-c]?\s*,?\s*p?N\s?[0-3XxOoI]"
    r"|(?i:stage)[^.\n]{0,20}\bT\s?[1-4]")
# A printed pathologic N category: "pN0", "pNO" / "pNo" / "pNI" (the OCR's
# letter O and I for 0 and 1), "pNX", "N0 M0", or the N part of a TNM string
# ("pT2c NO", "T3a, N1", "pT3bNOMX", "pT3aN1R1"); the category may run straight
# into a following M, R, L, V or G descriptor but not into any other letter
# ("T2c Nodes" is not a stage).
_PRINTED_PN = re.compile(
    r"(?<![A-Za-z])[ymr]?[pP]N\s?[0-3XxOoI](?=$|[^A-Za-z0-9]|[MmRrLlVvGg])"
    r"|\bN\s?[0-3XxOo]\s*,?\s*p?M\s?[01xX]"
    r"|\b[pP]?T\s?[1-4][a-c]?\s*,?\s*p?N\s?[0-3XxOoI](?=$|[^A-Za-z0-9]|[MmRrLlVvGg])")


def printed_stage(text: str, axis: str) -> str | None:
    """The phrase where the report prints a pathologic ``"T"`` or ``"N"``
    category, or None. A heuristic over OCR text (see the patterns above),
    good enough to anchor the eval's stated/derived subsets: a case the regex
    misses lands in the "not printed" subset, which the eval reports
    separately, never in the headline."""
    pattern = {"T": _PRINTED_PT, "N": _PRINTED_PN}[axis]
    m = pattern.search(text)
    return " ".join(m.group(0).split()) if m else None


def label_row(hit: dict[str, Any], allowed: dict[str, list[Any]]) -> tuple[dict[str, Any], list[str]]:
    """One gold row from one GDC case hit, plus notes on anything unmapped."""
    notes: list[str] = []
    case_id = hit.get("submitter_id", "")
    diagnoses = hit.get("diagnoses") or [{}]
    # TCGA cases carry one diagnosis; if several, prefer the one with Gleason data.
    diag = next((d for d in diagnoses if parse_pattern(d.get("primary_gleason_grade")) is not None), diagnoses[0])

    p = parse_pattern(diag.get("primary_gleason_grade"))
    s = parse_pattern(diag.get("secondary_gleason_grade"))
    total = p + s if p is not None and s is not None else None
    gg = parse_group(diag.get("gleason_grade_group"))
    source = "gdc" if gg is not None else ""
    if gg is None and p is not None and s is not None:
        gg = grade_group_from(p, s)
        source = "derived_from_gleason"
    pt = parse_stage(diag.get("ajcc_pathologic_t"), "T")
    pn = parse_stage(diag.get("ajcc_pathologic_n"), "N")

    row: dict[str, Any] = {"case_id": case_id, "gleason_primary": p, "gleason_secondary": s,
                           "gleason_total": total, "grade_group": gg, "pT": pt, "pN": pn,
                           "grade_group_source": source, "grade_group_printed": None,
                           "pT_printed": None, "pN_printed": None}
    for col in SCORED:
        v = row[col]
        if v is not None and col in allowed and v not in allowed[col]:
            notes.append(f"{case_id}: {col}={v!r} (raw {diag.get(_RAW_KEY[col])!r}) is not in the pack vocabulary; blanked")
            row[col] = None
    row["complete"] = all(row[c] is not None for c in SCORED)
    return row, notes


_RAW_KEY = {"gleason_primary": "primary_gleason_grade", "gleason_secondary": "secondary_gleason_grade",
            "gleason_total": "primary_gleason_grade", "grade_group": "gleason_grade_group",
            "pT": "ajcc_pathologic_t", "pN": "ajcc_pathologic_n"}


def raw_value_counts(hits: list[dict[str, Any]]) -> dict[str, Counter]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for hit in hits:
        for diag in hit.get("diagnoses") or [{}]:
            for key in ("primary_gleason_grade", "secondary_gleason_grade", "gleason_grade_group",
                        "ajcc_pathologic_t", "ajcc_pathologic_n"):
                counts[key][str(diag.get(key))] += 1
    return counts


def write_gold(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_COLUMNS)
        w.writeheader()
        for row in sorted(rows, key=lambda r: r["case_id"]):
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in GOLD_COLUMNS})


def write_mismatch_report(path: Path, rows: list[dict[str, Any]], report_ids: set[str],
                          counts: dict[str, Counter], notes: list[str],
                          printed: dict[str, str | None] | None = None,
                          printed_stages: dict[str, dict[str, str | None]] | None = None) -> None:
    label_ids = {r["case_id"] for r in rows}
    both = label_ids & report_ids
    complete = {r["case_id"] for r in rows if r["complete"]}
    missing_by_field = Counter(c for r in rows for c in SCORED if r[c] is None)
    lines = ["# TCGA-PRAD label / report alignment", "",
             f"- GDC cases with labels: {len(label_ids)}",
             f"- staged reports: {len(report_ids)}",
             f"- cases with both: {len(both)}",
             f"- cases with both and complete labels: {len(both & complete)}",
             f"- reports without labels: {len(report_ids - label_ids)}",
             f"- labels without reports: {len(label_ids - report_ids)}", "",
             "## Incomplete labels (count of cases missing each field)", ""]
    lines += [f"- {c}: {n}" for c, n in missing_by_field.most_common()] or ["- none"]
    lines += ["", "## Grade Group source", ""]
    lines += [f"- {k or '(none)'}: {n}" for k, n in Counter(r['grade_group_source'] for r in rows).most_common()]
    if printed is not None:
        hits = {c: ph for c, ph in printed.items() if ph}
        lines += ["", "## Grade Group printed in the report text (regex; gold column grade_group_printed)", "",
                  f"- reports that print one: {len(hits)} of {len(printed)} staged",
                  "- phrases matched: " + (", ".join(f"{ph!r}={n}" for ph, n in
                                                     Counter(ph.lower() for ph in hits.values()).most_common(20))
                                            or "none"),
                  "- cases (first 50): " + (", ".join(sorted(hits)[:50]) or "none")]
    for axis, col in (("T", "pT_printed"), ("N", "pN_printed")):
        found = {c: v.get(axis) for c, v in (printed_stages or {}).items()}
        if not found:
            continue
        hits = {c: ph for c, ph in found.items() if ph}
        missing = sorted(c for c, ph in found.items() if not ph)
        by_site = Counter(c.split("-")[1] if c.count("-") >= 2 else "?" for c in missing)
        lines += ["", f"## p{axis} printed in the report text (regex; gold column {col})", "",
                  f"- reports that print one: {len(hits)} of {len(found)} staged; {len(missing)} do not",
                  "- phrases matched (top 20): " + (", ".join(f"{ph!r}={n}" for ph, n in
                                                              Counter(hits.values()).most_common(20)) or "none"),
                  "- reports without one, by site: " + (", ".join(f"{s}={n}" for s, n in by_site.most_common()) or "none"),
                  "- reports without one (first 60): " + (", ".join(missing[:60]) or "none")]
    lines += ["", "## Raw GDC values seen", ""]
    for key, counter in counts.items():
        lines.append(f"- {key}: " + ", ".join(f"{v}={n}" for v, n in counter.most_common()))
    lines += ["", "## Values outside the pack vocabulary (blanked)", ""]
    lines += [f"- {n}" for n in notes] or ["- none"]
    lines += ["", "## Reports without labels", ""]
    lines += [f"- {c}" for c in sorted(report_ids - label_ids)] or ["- none"]
    lines += ["", "## Labels without reports (first 50)", ""]
    lines += [f"- {c}" for c in sorted(label_ids - report_ids)[:50]] or ["- none"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="packs/prostate_pathology")
    ap.add_argument("--reports-dir", default="data/tcga_prad/reports")
    ap.add_argument("--out", default=None, help="gold CSV (default: the pack's eval.gold path)")
    ap.add_argument("--report", default="data/tcga_prad/label_mismatch_report.md")
    ap.add_argument("--cache", default="data/tcga_prad/gdc_cases.json", help="raw GDC response cache")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and query GDC again")
    args = ap.parse_args(argv)

    pack = load_pack(args.pack)
    allowed = {f.id: f.allowed for f in pack.fields if f.allowed is not None}
    out = Path(args.out) if args.out else pack.resolve(pack.eval.gold)

    cache = Path(args.cache)
    if cache.exists() and not args.refresh:
        hits = json.loads(cache.read_text(encoding="utf-8"))
        print(f"using cached GDC response: {cache} ({len(hits)} cases)", file=sys.stderr)
    else:
        print(f"querying GDC for {PROJECT} clinical fields ...", file=sys.stderr)
        hits = fetch_project_cases(PROJECT, GDC_FIELDS)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(hits, indent=1), encoding="utf-8")

    rows, notes = [], []
    for hit in hits:
        row, row_notes = label_row(hit, allowed)
        rows.append(row)
        notes.extend(row_notes)

    reports_dir = Path(args.reports_dir)
    report_ids = {p.stem for p in reports_dir.glob("*.txt")} if reports_dir.exists() else set()
    printed: dict[str, str | None] = {}
    printed_stages: dict[str, dict[str, str | None]] = {}
    for row in rows:
        if row["case_id"] in report_ids:
            text = (reports_dir / f"{row['case_id']}.txt").read_text(encoding="utf-8")
            printed[row["case_id"]] = printed_grade_group(text)
            row["grade_group_printed"] = printed[row["case_id"]] is not None
            printed_stages[row["case_id"]] = {"T": printed_stage(text, "T"), "N": printed_stage(text, "N")}
            row["pT_printed"] = printed_stages[row["case_id"]]["T"] is not None
            row["pN_printed"] = printed_stages[row["case_id"]]["N"] is not None
    write_gold(rows, out)
    write_mismatch_report(Path(args.report), rows, report_ids, raw_value_counts(hits), notes, printed, printed_stages)
    n_complete = sum(r["complete"] for r in rows)
    eligible = sum(1 for r in rows if r["complete"] and r["case_id"] in report_ids)
    print(f"gold: {len(rows)} cases ({n_complete} complete) -> {out}", file=sys.stderr)
    print(f"mismatch report -> {args.report}", file=sys.stderr)
    print(f"cases with complete labels and a staged report: {eligible}", file=sys.stderr)
    print(f"reports that print a Grade Group: {sum(1 for v in printed.values() if v)} of {len(printed)} staged",
          file=sys.stderr)
    for axis in ("T", "N"):
        n = sum(1 for v in printed_stages.values() if v[axis])
        print(f"reports that print a p{axis} category: {n} of {len(printed_stages)} staged "
              f"({len(printed_stages) - n} do not; the eval scores those as derived)", file=sys.stderr)
    print("next: python -m scripts.working_set random --n 20   (seeded smoke-test draw)", file=sys.stderr)
    print("  or: python -m scripts.working_set pick ID ...     (hand-picked cases)", file=sys.stderr)
    if not report_ids:
        print("note: no staged reports found; run scripts.fetch_tcga_reports first, then re-run this",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
