"""``complete()``: the one door every model call goes through.

Why one function: portability. Ollama today, vLLM on Seadragon for batch
runs, a commercial endpoint for the Phase 4 benchmark, the institutional AIP
if Phase 5 needs it. Each is a backend module behind this signature, and the
rest of the engine never learns which one is in use. It also means the run
ledger sees every call, which is what makes the provenance record complete.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from engine import ledger
from engine.backends import BackendError, get_backend


@dataclass
class LLMResult:
    parsed: dict | None        # schema-constrained JSON, if parseable
    raw: str
    model: str
    latency_s: float
    tokens_in: int | None
    tokens_out: int | None
    error: str | None
    call_id: str | None = None  # cross-reference into the run ledger


def complete(
    prompt: str,
    schema: dict,
    model: str,
    *,
    temperature: float = 0.0,
    seed: int | None = None,
    backend: str = "ollama",
    run_id: str | None = None,
    **params: Any,
) -> LLMResult:
    """Send ``prompt`` to ``model`` on ``backend`` under grammar constraints
    derived from ``schema``; parse the JSON; write one ledger line; return.

    Never raises for a model/server problem: the error lands in
    ``LLMResult.error`` (and the ledger) so a batch run keeps going.
    """
    run_id = run_id or ledger.default_run_id()
    call_id = uuid.uuid4().hex[:12]
    raw, tokens_in, tokens_out, error, meta = "", None, None, None, {}

    t0 = time.perf_counter()
    try:
        be = get_backend(backend)
        resp = be.generate(prompt, schema, model, temperature=temperature, seed=seed, **params)
        raw, tokens_in, tokens_out, meta = resp.raw, resp.tokens_in, resp.tokens_out, resp.meta
    except BackendError as e:
        error = str(e)
    except Exception as e:  # noqa: BLE001 - a batch run must survive one bad call
        error = f"{type(e).__name__}: {e}"
    latency_s = time.perf_counter() - t0

    parsed: dict | None = None
    if error is None:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                parsed = obj
            else:
                error = f"model returned JSON that is not an object ({type(obj).__name__})"
        except json.JSONDecodeError as e:
            error = f"response is not valid JSON: {e}"
            if meta.get("done_reason") == "length":
                error += " (output was cut off: raise num_predict)"

    ledger.append(run_id, {
        "kind": "llm_call",
        "run_id": run_id,
        "call_id": call_id,
        "backend": backend,
        "model": model,
        "prompt_hash": ledger.sha256_text(prompt),
        "prompt_chars": len(prompt),
        "schema_hash": ledger.sha256_json(schema),
        "params": {"temperature": temperature, "seed": seed, **params},
        "latency_s": round(latency_s, 3),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "parse_ok": parsed is not None,
        "error": error,
        "raw": raw,
        "meta": meta,
    })
    return LLMResult(parsed=parsed, raw=raw, model=model, latency_s=latency_s,
                     tokens_in=tokens_in, tokens_out=tokens_out, error=error, call_id=call_id)
