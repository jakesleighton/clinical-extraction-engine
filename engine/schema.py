"""One pack -> one pydantic model -> both the decoding grammar and the validator.

Schema-first means the pack is the only place a field is defined. From it we
generate a pydantic model of the record the LLM must return. That model gives
us two things that must never drift apart:

* the JSON schema sent to the model server for constrained decoding (Ollama's
  ``format`` parameter; vLLM's guided decoding later), and
* the parser that checks what came back.

If those were written separately they would disagree eventually, and the
disagreement would surface as mysterious validation failures.

Constrained decoding, in one paragraph: the server compiles the JSON schema
into a grammar. At every generation step the model proposes a probability
over all tokens, and the grammar masks out every token that could not
continue a valid document. The model therefore cannot emit a key we did not
ask for, a string where an integer belongs, or an enum value outside the
list. What it guarantees is *shape*; whether the value is *true* is the job
of the validator (``engine/validate.py``) and the eval harness.
"""

from __future__ import annotations

import copy
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, create_model

from engine.packloader import FieldSpec, Pack


def _value_type(f: FieldSpec) -> Any:
    if f.type == "integer":
        return Literal[tuple(f.allowed)] if f.allowed else int   # type: ignore[misc]
    if f.type == "enum":
        return Literal[tuple(f.allowed)]                         # type: ignore[misc]
    if f.type in ("number", "measurement"):
        return float
    return str                                                   # string, date


def build_envelope_model(pack: Pack, f: FieldSpec) -> type[BaseModel]:
    """The object the model returns for one field, properties in envelope order."""
    statuses = tuple(pack.status_values)
    hedging = tuple(pack.hedging_levels)
    definitions: dict[str, Any] = {}
    for prop in pack.envelope:
        if prop.name == "evidence":
            definitions["evidence"] = (list[str], ...)
        elif prop.name == "status":
            definitions["status"] = (Literal[statuses], ...)          # type: ignore[valid-type]
        elif prop.name == "value":
            definitions["value"] = (Optional[_value_type(f)], ...)
        elif prop.name == "unit":
            if f.type == "measurement":
                definitions["unit"] = (Optional[Literal[tuple(f.units)]], ...)   # type: ignore[misc]
        elif prop.name == "hedging":
            definitions["hedging"] = (Literal[hedging], ...)            # type: ignore[valid-type]
        elif prop.name == "conflict":
            definitions["conflict"] = (bool, ...)
        else:
            definitions[prop.name] = (Optional[str], ...)
    return create_model(
        f"{f.id}_envelope",
        __config__=ConfigDict(extra="forbid"),
        **definitions,
    )


def build_record_model(pack: Pack) -> type[BaseModel]:
    """The whole record: one envelope per field, in pack order."""
    if pack._record_model is not None:
        return pack._record_model
    definitions = {f.id: (build_envelope_model(pack, f), ...) for f in pack.fields}
    model = create_model("Record", __config__=ConfigDict(extra="forbid"), **definitions)
    pack._record_model = model
    return model


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``$ref``/``$defs`` and drop ``title`` so the schema is one plain
    tree. Grammar compilers and commercial "strict" modes each support a
    different subset of JSON Schema; this keeps us on the common denominator."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return walk(copy.deepcopy(defs[name]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def build_json_schema(pack: Pack) -> dict[str, Any]:
    """JSON schema for the record: every property required, no extras,
    ``value`` nullable. This is what goes into Ollama's ``format``."""
    return _inline_refs(build_record_model(pack).model_json_schema())
