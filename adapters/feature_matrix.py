"""Stub: canonical records -> model-ready feature matrix (Aim 3).

Intended shape: one row per case, one column per field value plus
``<field>__status`` and ``<field>__hedging`` columns, enums one-hot or
categorical as the consumer asks; evidence dropped (it lives in the SQL
tables). Fed to the downstream modeling groups, never used for training here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def to_feature_matrix(records: Iterable[dict[str, Any]]) -> Any:
    raise NotImplementedError("feature_matrix adapter is an Aim 3 deliverable (not Phase 0)")
