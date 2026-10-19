"""Per-document pipeline: prompt -> complete -> validate -> derive -> record.

    python -m engine.extract data/tcga_prad/reports/TCGA-XX-0000.txt \\
        --pack packs/prostate_pathology --model qwen2.5:7b

prints the canonical record as JSON and the ledger path. Quality is not
judged here; that is the eval harness's job.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine import ledger
from engine.llm import LLMResult, complete
from engine.packloader import Pack, load_pack
from engine.prompt_builder import build_prompt, build_schema
from engine.validate import Issue, apply_derivations, validate_output

# The prompt with the pack's three exemplars is roughly 9-12k tokens; the backend's
# default context (8192) would silently truncate it from the front, dropping the
# instructions. Runs that include exemplars get this unless the caller sets num_ctx.
EXEMPLAR_NUM_CTX = 16384

THINK_CHOICES = ("false", "true", "low", "medium", "high")


def parse_think(raw: str) -> bool | str:
    """``--think`` value -> what the backend sends: ``false``/``true`` become
    booleans, a level (``low``/``medium``/``high``, for gpt-oss) stays a string."""
    s = raw.strip().lower()
    if s not in THINK_CHOICES:
        raise ValueError(f"--think must be one of {', '.join(THINK_CHOICES)}; got {raw!r}")
    return {"false": False, "true": True}.get(s, s)


@dataclass
class ExtractionResult:
    case_id: str | None
    pack: Pack
    llm: LLMResult
    prompt_hash: str
    run_id: str
    record: dict[str, dict[str, Any]] | None       # field id -> envelope; None when the call failed
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.record is not None and not any(i.severity == "error" for i in self.issues)

    def to_record(self) -> dict[str, Any]:
        """The canonical, provenance-preserving record one adapter or another
        will later flatten into tables."""
        return {
            "case_id": self.case_id,
            "pack": self.pack.name,
            "pack_version": self.pack.version,
            "document_type": self.pack.document.type if self.pack.document else None,
            "model": self.llm.model,
            "run_id": self.run_id,
            "call_id": self.llm.call_id,
            "prompt_hash": self.prompt_hash,
            "ok": self.ok,
            "error": self.llm.error,
            "fields": self.record,
            "issues": [i.to_dict() for i in self.issues],
        }


def extract_document(
    pack: Pack,
    text: str,
    *,
    model: str,
    backend: str = "ollama",
    case_id: str | None = None,
    run_id: str | None = None,
    include_exemplars: bool = True,
    **params: Any,
) -> ExtractionResult:
    run_id = run_id or ledger.default_run_id()
    if include_exemplars and pack.exemplars:
        params.setdefault("num_ctx", EXEMPLAR_NUM_CTX)
    prompt = build_prompt(pack, text, include_exemplars=include_exemplars)
    schema = build_schema(pack)
    prompt_hash = ledger.sha256_text(prompt)

    llm = complete(prompt, schema, model, backend=backend, run_id=run_id, **params)

    record: dict[str, dict[str, Any]] | None = None
    issues: list[Issue] = []
    if llm.parsed is not None:
        validated = validate_output(pack, llm.parsed, text)
        record, issues = validated.record, validated.issues
        apply_derivations(pack, record, issues)

    ledger.append(run_id, {
        "kind": "validation",
        "run_id": run_id,
        "call_id": llm.call_id,
        "case_id": case_id,
        "pack": pack.name,
        "pack_version": pack.version,
        "prompt_hash": prompt_hash,
        "llm_error": llm.error,
        "ok": record is not None and not any(i.severity == "error" for i in issues),
        "n_errors": sum(i.severity == "error" for i in issues),
        "n_flags": sum(i.severity == "flag" for i in issues),
        "issues": [i.to_dict() for i in issues],
    })
    return ExtractionResult(case_id=case_id, pack=pack, llm=llm, prompt_hash=prompt_hash,
                            run_id=run_id, record=record, issues=issues)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract one document with a task pack.")
    ap.add_argument("document", help="path to a .txt report")
    ap.add_argument("--pack", required=True, help="pack directory, e.g. packs/prostate_pathology")
    ap.add_argument("--model", required=True, help="model name on the backend, e.g. qwen2.5:7b")
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--case-id", default=None, help="defaults to the document's file stem")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-ctx", type=int, default=None,
                    help=f"context window (Ollama option; default {EXEMPLAR_NUM_CTX} when the pack has exemplars)")
    ap.add_argument("--think", default="false", choices=THINK_CHOICES,
                    help="thinking mode for models that have one; false (default) or a level for gpt-oss")
    ap.add_argument("--out", default=None, help="write the record JSON here instead of stdout")
    ap.add_argument("--show-prompt", action="store_true", help="print the prompt and exit")
    args = ap.parse_args(argv)

    pack = load_pack(args.pack)
    path = Path(args.document)
    text = path.read_text(encoding="utf-8")
    if args.show_prompt:
        print(build_prompt(pack, text))
        return 0
    params: dict[str, Any] = {"think": parse_think(args.think)}
    if args.num_ctx:
        params["num_ctx"] = args.num_ctx
    run_id = args.run_id or ledger.new_run_id()
    result = extract_document(pack, text, model=args.model, backend=args.backend,
                              case_id=args.case_id or path.stem, run_id=run_id,
                              temperature=args.temperature, seed=args.seed, **params)
    record = result.to_record()
    out = json.dumps(record, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(out + "\n", encoding="utf-8")
    else:
        print(out)
    status = "ok" if result.ok else "FAILED"
    print(f"[{status}] {len(result.issues)} issue(s); {result.llm.latency_s:.1f}s; "
          f"tokens in/out {result.llm.tokens_in}/{result.llm.tokens_out}; "
          f"ledger: {ledger.ledger_path(run_id)}", file=sys.stderr)
    if result.llm.error:
        print(f"error: {result.llm.error}", file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
