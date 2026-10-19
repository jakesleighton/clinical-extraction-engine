"""Ollama backend: the only module in the codebase that imports an HTTP client.

Ollama runs as a local server (``ollama serve``, default ``localhost:11434``).
We POST to ``/api/generate`` with ``format`` set to the JSON schema, which
turns on grammar-constrained decoding on the server side.

Things worth knowing about Ollama's defaults:

* ``num_ctx`` (context window) defaults to a few thousand tokens. A prompt
  longer than that is silently truncated from the front, which drops our
  instructions first. We set 8192 by default; raise it for long reports.
* ``temperature`` 0 plus a fixed ``seed`` makes runs repeatable for a given
  model build, which the ablation evals rely on.
* The response carries ``prompt_eval_count`` / ``eval_count`` (tokens in/out)
  and ``done_reason``; ``"length"`` means the output hit ``num_predict`` and
  the JSON is almost certainly cut off.
* ``think`` is a top-level request field (not an option). Current open-weight
  families (Qwen 3.5, Gemma 4, gpt-oss) reason before answering unless told
  not to; for grammar-constrained extraction that only costs time, so we send
  ``think: false`` by default. gpt-oss cannot switch reasoning off and takes
  a level instead (``"low"``). A model with no thinking mode may reject the
  field with HTTP 400; then we retry once without it and record that in
  ``meta``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from engine.backends import BackendError, BackendResponse

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 600.0
DEFAULT_NUM_CTX = 8192
DEFAULT_THINK: bool | str = False      # False | True | "low" | "medium" | "high" (levels are for gpt-oss)


def resolve_host(host: str | None = None) -> str:
    h = host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST
    if "://" not in h:
        h = "http://" + h
    return h.rstrip("/")


def _make_client(timeout: float) -> httpx.Client:
    # Tests monkeypatch this to return a client with a mock transport.
    return httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0))


def generate(
    prompt: str,
    schema: dict[str, Any],
    model: str,
    *,
    temperature: float = 0.0,
    seed: int | None = None,
    host: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    num_ctx: int = DEFAULT_NUM_CTX,
    think: bool | str | None = DEFAULT_THINK,
    **options: Any,
) -> BackendResponse:
    """One generation. Extra keyword arguments go into Ollama's ``options``
    (``num_predict``, ``top_p``, ...). ``think=None`` leaves the field out and
    lets the model do whatever it does by default."""
    opts: dict[str, Any] = {"temperature": temperature, "num_ctx": num_ctx, **options}
    if seed is not None:
        opts["seed"] = seed
    body: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False, "format": schema, "options": opts}
    if think is not None:
        body["think"] = think
    think_meta: dict[str, Any] = {"think": think}
    url = f"{resolve_host(host)}/api/generate"
    try:
        with _make_client(timeout) as client:
            resp = client.post(url, json=body)
            if resp.status_code == 400 and "think" in body and "think" in resp.text.lower():
                # no thinking mode to switch (or the level is not accepted): retry without the field
                body.pop("think")
                think_meta = {"think": None, "think_requested": think, "think_rejected": resp.text[:200]}
                resp = client.post(url, json=body)
    except httpx.ConnectError as e:
        raise BackendError(f"cannot reach Ollama at {url} (is `ollama serve` running?): {e}") from None
    except httpx.TimeoutException as e:
        raise BackendError(f"Ollama timed out after {timeout:.0f}s at {url}: {e}") from None
    if resp.status_code != 200:
        detail = resp.text[:500]
        hint = f" (pull it first: `ollama pull {model}`)" if "not found" in detail.lower() else ""
        raise BackendError(f"Ollama returned HTTP {resp.status_code}{hint}: {detail}")
    data = resp.json()
    return BackendResponse(
        raw=data.get("response", ""),
        tokens_in=data.get("prompt_eval_count"),
        tokens_out=data.get("eval_count"),
        meta={**{k: data.get(k) for k in ("done", "done_reason", "total_duration", "load_duration",
                                           "prompt_eval_duration", "eval_duration")}, **think_meta},
    )
