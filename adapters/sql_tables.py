"""Stub: canonical records -> SQL-ready tables (Aim 3, months 8-12).

Intended shape (pandas DataFrames, one per table, ready for ``to_sql``):

* ``cases``   one row per case: case_id, pack, pack_version, model, run_id, ok
* ``values``  one row per case x field: status, value, unit, hedging, conflict,
              derivation, evidence_start, evidence_end, evidence_quote
* ``issues``  one row per validation issue: case_id, field, code, severity, message

Keeping ``values`` long (not wide) means a new pack adds rows, not columns,
and the provenance columns travel with every value.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def to_tables(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError("sql_tables adapter is an Aim 3 deliverable (not Phase 0)")
