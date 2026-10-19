"""Model-server backends behind ``engine.llm.complete``.

A backend is a module exposing::

    generate(prompt: str, schema: dict, model: str, *, temperature: float,
             seed: int | None, **params) -> BackendResponse

It talks to one kind of server (Ollama now; vLLM, commercial endpoints and
the institutional AIP later) and nothing else in the codebase speaks HTTP.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any


class BackendError(Exception):
    """The server could not be reached or returned an error."""


@dataclass
class BackendResponse:
    raw: str                                   # the model's text output
    tokens_in: int | None = None
    tokens_out: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)   # server-specific extras (timings, done_reason)


def get_backend(name: str) -> ModuleType:
    try:
        module = importlib.import_module(f"engine.backends.{name}")
    except ModuleNotFoundError as e:
        if e.name == f"engine.backends.{name}":
            raise BackendError(f"unknown backend {name!r} (no module engine/backends/{name}.py)") from None
        raise
    if not hasattr(module, "generate"):
        raise BackendError(f"backend {name!r} has no generate() function")
    return module
