"""Run a pack's eval set end to end and write a report.

    python -m evalharness.runner --pack packs/prostate_pathology --model qwen2.5:7b \\
        --reports data/tcga_prad/reports --ids data/tcga_prad/working_set.txt
    python -m evalharness.runner ... --cases TCGA-XX-1234 TCGA-YY-5678

For every case in the pack's gold CSV that has a report file
(``<reports>/<case_id>.txt``), extract, then score the pack's ``eval.scored``
fields. ``--ids FILE`` (one id per line, ``#`` comments) and/or ``--cases``
restrict the run; without either, every gold case with a report runs. Cases
the pack lists as exemplars (``exemplars[].case_id``) are held out of every
run, even when ``--ids``/``--cases`` names them: they are in the prompt, so
scoring them would be leakage. Outputs under ``runs/<run_id>/``:
``predictions.jsonl`` (canonical records), ``metrics.json`` and ``report.md``.
The per-call ledger is ``runs/<run_id>.jsonl`` as always.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pandas as pd

from adapters import jsonl
from engine import ledger
from engine.extract import EXEMPLAR_NUM_CTX, THINK_CHOICES, extract_document, parse_think
from engine.packloader import Pack, PackError, load_pack
from evalharness.metrics import DEFAULT_ABSENT_TOKEN, MetricsReport, compute_metrics


_TRUE, _FALSE = {"true", "t", "1", "yes", "y"}, {"false", "f", "0", "no", "n"}


def parse_bool(raw: str) -> bool | None:
    """``True``/``False``/blank gold columns -> bool or None; anything else raises ValueError."""
    s = raw.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    if s == "":
        return None
    raise ValueError(raw)


def load_gold(pack: Pack, path: str | Path | None = None,
              absent_token: str = DEFAULT_ABSENT_TOKEN) -> dict[str, dict[str, Any]]:
    """Gold CSV -> {case_id: {field: value}} for the pack's scored fields,
    plus every ``stated_column`` a derived field declares (parsed to
    True/False/None: does the document print the value?).

    Fails loudly when a column is missing or an enum column holds a value the
    pack does not allow, because a vocabulary mismatch would otherwise show
    up as a mysteriously low score.
    """
    if pack.eval is None:
        raise PackError(f"pack {pack.name!r} has no `eval` section")
    gold_path = Path(path) if path else pack.resolve(pack.eval.gold)
    if not gold_path.exists():
        raise PackError(f"gold file not found: {gold_path} (run scripts/join_gold_labels.py first)")
    df = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    id_col = pack.eval.id_column
    aux = [d.stated_column for d in pack.derived_fields if d.stated_column]
    missing = [c for c in [id_col, *pack.eval.scored, *aux] if c not in df.columns]
    if missing:
        raise PackError(f"{gold_path}: missing columns {missing}; has {list(df.columns)}")
    gold: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for _, row in df.iterrows():
        case_id = row[id_col].strip()
        values: dict[str, Any] = {}
        for fid in pack.eval.scored:
            spec = pack.field(fid)
            raw = row[fid].strip()
            if raw == "" or raw == absent_token:
                values[fid] = raw or None
                continue
            if spec.type == "integer":
                try:
                    values[fid] = int(float(raw))
                except ValueError:
                    problems.append(f"{case_id}.{fid}={raw!r} is not an integer")
                    continue
            elif spec.type == "number":
                try:
                    values[fid] = float(raw)
                except ValueError:
                    problems.append(f"{case_id}.{fid}={raw!r} is not a number")
                    continue
            else:
                values[fid] = raw
            if spec.allowed is not None and values[fid] not in spec.allowed:
                problems.append(f"{case_id}.{fid}={raw!r} is not in allowed {spec.allowed}")
        for col in aux:
            try:
                values[col] = parse_bool(row[col])
            except ValueError:
                problems.append(f"{case_id}.{col}={row[col]!r} is not true/false/blank")
        gold[case_id] = values
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise PackError(f"{gold_path}: gold values outside the pack vocabulary:\n  {shown}{more}")
    return gold


def read_id_file(path: str | Path) -> list[str]:
    """One case id per line; blank lines and ``#`` comments are ignored."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def select_cases(gold: dict[str, dict[str, Any]], reports_dir: Path, ids: list[str] | None,
                 limit: int | None, exclude: Collection[str] = ()) -> list[str]:
    """The requested ids (or every gold case) that have gold and a report,
    minus ``exclude`` (the pack's exemplar cases), then ``limit``."""
    cases = ids if ids is not None else sorted(gold)
    cases = [c for c in cases if c in gold and (reports_dir / f"{c}.txt").exists() and c not in exclude]
    return cases[:limit] if limit else cases


def run(pack: Pack, *, model: str, reports_dir: Path, backend: str = "ollama", ids: list[str] | None = None,
        limit: int | None = None, run_id: str | None = None, gold_path: str | None = None,
        **params: Any) -> tuple[MetricsReport, Path]:
    run_id = run_id or ledger.new_run_id()
    out_dir = ledger.runs_dir() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = load_gold(pack, gold_path)
    exemplars = set(pack.exemplar_ids)
    requested = ids if ids is not None else sorted(gold)
    held_out = sorted({c for c in requested if c in exemplars})
    if held_out:
        print(f"  holding out {len(held_out)} exemplar case(s) listed in the pack (in the prompt, never scored): "
              + ", ".join(held_out), file=sys.stderr)
    cases = select_cases(gold, reports_dir, ids, limit, exclude=exemplars)
    if ids is not None:
        usable = set(select_cases(gold, reports_dir, ids, None, exclude=exemplars))
        dropped = [c for c in ids if c not in usable and c not in exemplars]
        if dropped:
            print(f"  skipping {len(dropped)} requested case(s) with no gold row or no report: "
                  + ", ".join(dropped[:10]) + (" ..." if len(dropped) > 10 else ""), file=sys.stderr)
    if not cases:
        raise SystemExit(f"no cases to run: gold has {len(gold)} ids, reports in {reports_dir}"
                         + (f"; {len(held_out)} requested case(s) are pack exemplars" if held_out else ""))
    print(f"run {run_id}: {len(cases)} case(s), model {model}, backend {backend}", file=sys.stderr)

    predictions: dict[str, dict[str, Any] | None] = {}
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, case_id in enumerate(cases, 1):
        text = (reports_dir / f"{case_id}.txt").read_text(encoding="utf-8")
        result = extract_document(pack, text, model=model, backend=backend, case_id=case_id,
                                  run_id=run_id, **params)
        record = result.to_record()
        records.append(record)
        predictions[case_id] = record if result.record is not None else None
        status = "ok" if result.ok else "FAIL"
        print(f"  [{i}/{len(cases)}] {case_id} {status} {result.llm.latency_s:.1f}s "
              f"{len(result.issues)} issue(s)" + (f" error: {result.llm.error}" if result.llm.error else ""),
              file=sys.stderr)
    jsonl.write_records(records, out_dir / "predictions.jsonl")

    scored_gold = {c: gold[c] for c in cases}
    report = compute_metrics(pack, predictions, scored_gold)
    report.notes.insert(0, f"model={model} backend={backend} pack={pack.name} v{pack.version} "
                           f"run={run_id} wall={time.perf_counter() - t0:.0f}s")
    if exemplars:
        report.notes.insert(1, f"pack exemplars held out of the run ({len(exemplars)}): "
                               + ", ".join(sorted(exemplars)))
    (out_dir / "metrics.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    return report, out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a pack's eval set and score it.")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reports", required=True, help="directory of <case_id>.txt files")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--ids", default=None, help="file with one case id per line (the working set)")
    ap.add_argument("--cases", nargs="+", default=None, metavar="ID", help="run exactly these case ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--gold", default=None, help="override the pack's gold CSV path")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-ctx", type=int, default=None,
                    help=f"context window (default {EXEMPLAR_NUM_CTX} when the pack has exemplars, else the backend's)")
    ap.add_argument("--think", default="false", choices=THINK_CHOICES,
                    help="thinking mode for models that have one; false (default) or a level for gpt-oss")
    args = ap.parse_args(argv)

    pack = load_pack(args.pack)
    ids = None
    if args.ids or args.cases:
        ids = (read_id_file(args.ids) if args.ids else []) + list(args.cases or [])
    params: dict[str, Any] = {"temperature": args.temperature, "seed": args.seed, "think": parse_think(args.think)}
    if args.num_ctx:
        params["num_ctx"] = args.num_ctx
    report, out_dir = run(pack, model=args.model, reports_dir=Path(args.reports), backend=args.backend,
                          ids=ids, limit=args.limit, run_id=args.run_id, gold_path=args.gold, **params)
    print(report.to_markdown())
    print(f"written: {out_dir}/report.md, metrics.json, predictions.jsonl", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
