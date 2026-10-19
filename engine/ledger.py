"""JSONL run ledger.

Every ``complete()`` call appends one line to ``runs/<run_id>.jsonl``; the
extraction pipeline appends a second line with the validation outcome, linked
by ``call_id``. The ledger replaces the observability an agent framework would
have given us, and doubles as the provenance record (Aim 3): for any value in
any output table you can walk back to the model, prompt hash, raw response
and validation status that produced it.

Lines are appended one at a time with the file opened in append mode, so a
crash mid-run loses at most the line being written.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_RUNS_DIR = "runs"
_default_run_id: str | None = None


def runs_dir(override: str | Path | None = None) -> Path:
    """``runs/`` under the current directory unless ``CEE_RUNS_DIR`` says otherwise."""
    return Path(override or os.environ.get("CEE_RUNS_DIR") or DEFAULT_RUNS_DIR)


def new_run_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)


def default_run_id() -> str:
    """One run id per process, created on first use."""
    global _default_run_id
    if _default_run_id is None:
        _default_run_id = new_run_id()
    return _default_run_id


def ledger_path(run_id: str, directory: str | Path | None = None) -> Path:
    return runs_dir(directory) / f"{run_id}.jsonl"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str))


def append(run_id: str, record: dict[str, Any], directory: str | Path | None = None) -> Path:
    """Append one record (a ``ts`` is added if missing). Returns the file path."""
    path = ledger_path(run_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": now_iso(), **record}, ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


def read(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
