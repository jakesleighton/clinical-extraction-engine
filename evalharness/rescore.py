"""Re-score a saved run with the current pack and engine, without a model.

    python -m evalharness.rescore --run-id p1-baseline-gemma4-12b-q4km \\
        --pack packs/prostate_pathology --reports data/tcga_prad/reports \\
        [--as p1-baseline-gemma4-12b-q4km-rescored] [--groups CSV] [--gold-notes MD] [--gold CSV]

Every model reply is kept verbatim in the run ledger (``runs/<run_id>.jsonl``,
the ``raw`` field of each ``llm_call`` line). This takes those replies, runs
today's validator and derivations over them, and writes a fresh
``predictions.jsonl``, ``metrics.json`` and ``report.md`` under a new run id
(``<run_id>-rescored`` by default). The model outputs do not change, so the
result measures the scoring machinery, not the model: it is how a matcher
change, a new derived field, a lenient column or a gold audit can be shown
on finished runs before anyone spends model time. The report says so in its
header, and the original run folder is never touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from adapters import jsonl
from engine import ledger
from engine.packloader import load_pack
from engine.validate import apply_derivations, validate_output
from evalharness.report import build_run_report
from evalharness.runner import ledger_calls, load_gold, read_gold_notes, read_groups, read_texts, write_outputs


def saved_calls(run_id: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(llm_call, validation)`` line pairs of a saved run, one per case (the
    last pair when a case was run twice under the same id), in ledger order."""
    path = ledger.ledger_path(run_id)
    if not path.exists():
        raise SystemExit(f"no ledger for run {run_id!r} at {path}")
    calls = {line["call_id"]: line for line in ledger.read(path) if line.get("kind") == "llm_call"}
    by_case: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for line in ledger.read(path):
        if line.get("kind") == "validation" and line.get("call_id") in calls and line.get("case_id"):
            by_case[line["case_id"]] = (calls[line["call_id"]], line)
    return list(by_case.values())


def rescore(run_id: str, pack_path: str | Path, reports_dir: Path, *, new_run_id: str | None = None,
            gold_path: str | None = None, groups_path: str | None = None, gold_notes_path: str | None = None):
    pack = load_pack(pack_path)
    gold = load_gold(pack, gold_path)
    pairs = saved_calls(run_id)
    if not pairs:
        raise SystemExit(f"run {run_id!r} has no (call, validation) pairs to re-score")
    new_run_id = new_run_id or f"{run_id}-rescored"
    out_dir = ledger.runs_dir() / new_run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [val["case_id"] for _, val in pairs]
    texts = read_texts(reports_dir, [c for c in cases if (reports_dir / f"{c}.txt").exists()])
    records: list[dict[str, Any]] = []
    old_versions: set[Any] = set()
    for call, val in pairs:
        case_id = val["case_id"]
        old_versions.add(val.get("pack_version"))
        parsed: dict[str, Any] | None = None
        error = call.get("error")
        if error is None and call.get("raw"):
            try:
                obj = json.loads(call["raw"])
                parsed = obj if isinstance(obj, dict) else None
            except json.JSONDecodeError as e:
                error = f"response is not valid JSON: {e}"
        fields, issues = None, []
        if parsed is not None and case_id in texts:
            validated = validate_output(pack, parsed, texts[case_id])
            fields, issues = validated.record, validated.issues
            apply_derivations(pack, fields, issues)
        ok = fields is not None and not any(i.severity == "error" for i in issues)
        records.append({"case_id": case_id, "pack": pack.name, "pack_version": pack.version,
                        "document_type": pack.document.type if pack.document else None,
                        "model": call.get("model"), "run_id": new_run_id, "call_id": call.get("call_id"),
                        "prompt_hash": call.get("prompt_hash"), "ok": ok, "error": error,
                        "fields": fields, "issues": [i.to_dict() for i in issues],
                        "rescored_from": run_id})
    jsonl.write_records(records, out_dir / "predictions.jsonl")

    model = next((c.get("model") for c, _ in pairs if c.get("model")), "?")
    header = [f"RE-SCORED from run {run_id}: the model replies are unchanged (recorded under pack "
              f"v{', v'.join(str(v) for v in sorted(old_versions, key=str))}); validated and derived now with pack "
              f"{pack.name} v{pack.version}. This measures the scoring machinery, not the model.",
              f"model={model} run={new_run_id} (no model calls; the cost line repeats the original run's ledger)"]
    if pack.exemplar_ids:
        header.append(f"pack exemplars held out ({len(pack.exemplar_ids)}): " + ", ".join(sorted(pack.exemplar_ids)))
    groups = read_groups(groups_path) if groups_path else None
    gold_notes = read_gold_notes(gold_notes_path) if gold_notes_path else None
    if groups:
        header.append("groups: " + ", ".join(f"{g} ({len(m)})" for g, m in groups.items()))
    if gold_notes:
        header.append(f"gold notes: {len(gold_notes)} audited entries on {len({n.case_id for n in gold_notes})} case(s)")
    scored_gold = {c: gold[c] for c in cases if c in gold and c not in set(pack.exemplar_ids)}
    calls = ledger_calls(run_id, [c.get("call_id") for c, _ in pairs])
    report = build_run_report(pack, records, scored_gold, texts=texts, calls=calls, wall_s=None,
                              groups=groups, gold_notes=gold_notes, header=header)
    write_outputs(out_dir, report)
    return report, out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-score a saved run with the current pack and engine (no model calls).")
    ap.add_argument("--run-id", required=True, help="the saved run (its ledger runs/<run_id>.jsonl is the input)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--reports", required=True, help="directory of <case_id>.txt files")
    ap.add_argument("--as", dest="new_run_id", default=None, help="run id for the outputs (default <run_id>-rescored)")
    ap.add_argument("--gold", default=None, help="override the pack's gold CSV path")
    ap.add_argument("--groups", default=None, metavar="CSV")
    ap.add_argument("--gold-notes", default=None, metavar="MD")
    args = ap.parse_args(argv)
    report, out_dir = rescore(args.run_id, args.pack, Path(args.reports), new_run_id=args.new_run_id,
                              gold_path=args.gold, groups_path=args.groups, gold_notes_path=args.gold_notes)
    print(report.to_markdown())
    print(f"written: {out_dir}/report.md, metrics.json, predictions.jsonl", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
