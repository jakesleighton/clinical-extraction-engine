"""Pack + document -> prompt text + JSON schema.

The prompt is rendered from the pack alone (instructions, field catalog,
output format, exemplars) plus the document. It is deterministic: the same
pack version and document always give the same string, so the prompt hash in
the ledger identifies the prompt exactly.

Nothing here is model-specific. Chat templating (system/user roles) is the
backend's business; we hand it one string.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from engine.packloader import FieldSpec, Pack
from engine.schema import build_json_schema

DOC_START = "<<<DOCUMENT"
DOC_END = "DOCUMENT>>>"


def build_schema(pack: Pack) -> dict[str, Any]:
    return build_json_schema(pack)


def _field_line(f: FieldSpec) -> str:
    bits = [f.type]
    if f.allowed:
        bits.append("allowed: " + " | ".join(str(a) for a in f.allowed))
    if f.units:
        bits.append("units: " + ", ".join(f.units))
    if f.unit:
        bits.append(f"unit: {f.unit}")
    if f.kind == "finding":
        bits.append("finding")
    head = f"- {f.id} ({'; '.join(bits)})"
    body = " ".join(f.instructions.split())
    return f"{head}\n  {body}" if body else head


def field_catalog(pack: Pack) -> str:
    return "\n".join(_field_line(f) for f in pack.fields)


def output_format(pack: Pack) -> str:
    lines = ["Return one JSON object with one key per field id listed above.",
             "Each field is an object with these keys, in this order:"]
    for prop in pack.envelope:
        desc = " ".join(prop.description.split())
        if prop.name == "status":
            lines.append(f"- {prop.name}: one of " + " | ".join(pack.status_values))
            for k, v in pack.status_values.items():
                lines.append(f"    {k}: {' '.join(v.split())}")
        elif prop.name == "hedging":
            lines.append(f"- {prop.name}: one of " + " | ".join(pack.hedging_levels))
            for k, v in pack.hedging_levels.items():
                lines.append(f"    {k}: {' '.join(v.split())}")
        else:
            lines.append(f"- {prop.name}: {desc}")
    if pack.kinds:
        lines.append("")
        lines.append("Field kinds:")
        for k, v in pack.kinds.items():
            lines.append(f"- {k}: {' '.join(v.split())}")
    used_types = sorted({f.type for f in pack.fields})
    if pack.field_types and used_types:
        lines.append("")
        lines.append("Value types:")
        for t in used_types:
            if t in pack.field_types:
                lines.append(f"- {t}: {' '.join(pack.field_types[t].split())}")
    return "\n".join(lines)


def exemplar_blocks(pack: Pack, variants: Collection[str] = ()) -> list[str]:
    """One "## Example" block per exemplar the prompt shows: the pack's
    untagged exemplars plus any whose ``variant`` is switched on."""
    blocks = []
    for ex in pack.prompt_exemplars(variants):
        source = pack.resolve(ex.source).read_text(encoding="utf-8").strip()
        output = json.loads(pack.resolve(ex.output).read_text(encoding="utf-8"))
        blocks.append(
            "## Example\n\n"
            f"{DOC_START}\n{source}\n{DOC_END}\n\n"
            "Output:\n```json\n" + json.dumps(output, indent=1, ensure_ascii=False) + "\n```"
        )
    return blocks


def build_prompt(pack: Pack, text: str, *, include_exemplars: bool = True,
                 exemplar_variants: Collection[str] = ()) -> str:
    parts = [
        pack.instructions_text,
        "## Fields to extract\n\n" + field_catalog(pack),
        "## Output format\n\n" + output_format(pack),
    ]
    if include_exemplars:
        parts.extend(exemplar_blocks(pack, exemplar_variants))
    parts.append(f"## Document\n\n{DOC_START}\n{text.strip()}\n{DOC_END}")
    parts.append("Return the JSON object for this document now.")
    return "\n\n".join(parts) + "\n"
