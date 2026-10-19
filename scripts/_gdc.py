"""Minimal GDC API client (public data, no token needed).

Uses the standard library on purpose: the one-HTTP-client rule in the brief
is about model calls, but there is no reason for a staging script to pull
in anything beyond ``urllib``.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from typing import Any

GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"
PAGE_SIZE = 500


def fetch_project_cases(project_id: str, fields: list[str], *, page_size: int = PAGE_SIZE,
                        timeout: float = 120.0) -> list[dict[str, Any]]:
    """All case records of a GDC project with the requested fields
    (dotted names such as ``diagnoses.primary_gleason_grade`` are allowed)."""
    filters = {"op": "in", "content": {"field": "project.project_id", "value": [project_id]}}
    hits: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {"filters": json.dumps(filters), "fields": ",".join(fields),
                  "size": str(page_size), "from": str(offset), "format": "JSON"}
        url = GDC_CASES_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        page = data["data"]["hits"]
        hits.extend(page)
        pagination = data["data"].get("pagination", {})
        total = int(pagination.get("total", len(hits)))
        print(f"  GDC {project_id}: {len(hits)}/{total} cases", file=sys.stderr)
        if not page or len(hits) >= total:
            break
        offset += len(page)
    return hits
