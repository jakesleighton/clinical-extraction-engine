"""Load, validate and merge task packs.

A pack is a folder holding ``pack.yaml`` (plus optional ``exemplars/`` and
``labels/``). ``extends: base`` pulls in the base pack's machinery; this module
merges the two and hands back one typed ``Pack`` object that the prompt
builder, validator and eval harness all read from. Nothing else in the engine
reads YAML.

Why typed (pydantic) instead of a dict: a malformed pack must fail at load
time with a message naming the file and the key, not three layers deep inside
the prompt builder at run time. ``extra="forbid"`` on every section turns a
typo like ``instuctions:`` into a load error instead of a silently ignored key.
"""

from __future__ import annotations

import copy
import keyword
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from engine.rules import FUNCTIONS, RuleError, compile_rule

PACK_FILENAME = "pack.yaml"

# The engine keys off these exact strings everywhere (prompt, validator, eval).
STATUS_VALUES = ("value", "explicitly_absent", "not_mentioned")
REQUIRED_ENVELOPE = ("evidence", "status", "value", "hedging", "conflict")

FieldType = Literal["integer", "number", "string", "enum", "date", "measurement"]
Tier = Literal["deterministic", "standard", "hard"]
Kind = Literal["attribute", "finding"]
EvidencePolicy = Literal["required", "optional"]


class PackError(Exception):
    """A pack file is missing, malformed, or inconsistent."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

class PackMeta(_Strict):
    name: str
    version: int
    extends: str | None = None


class DocumentSpec(_Strict):
    type: str
    unit: str = "report"
    case_grouping: str = "patient_id"


class FieldDefaults(_Strict):
    status: Literal["three_state"] = "three_state"
    evidence: EvidencePolicy = "required"
    tier: Tier = "standard"
    kind: Kind = "attribute"


class Defaults(_Strict):
    field: FieldDefaults = FieldDefaults()


class FieldSpec(_Strict):
    id: str
    type: FieldType
    instructions: str = ""
    allowed: list[int | float | str] | None = None
    units: list[str] | None = None      # measurement fields
    unit: str | None = None             # number fields with a fixed unit
    # The four below are filled from ``defaults.field`` when a pack omits them.
    status: Literal["three_state"] | None = None
    evidence: EvidencePolicy | None = None
    tier: Tier | None = None
    kind: Kind | None = None
    _defaulted: list[str] = PrivateAttr(default_factory=list)   # which of the four came from defaults

    @model_validator(mode="after")
    def _consistent(self) -> "FieldSpec":
        if not self.id.isidentifier() or keyword.iskeyword(self.id) or self.id.startswith("_"):
            raise ValueError(f"field id {self.id!r} must be a plain identifier (letters, digits, _)")
        if self.id.startswith("model_"):
            raise ValueError(f"field id {self.id!r} may not start with 'model_'")
        if self.type == "enum" and not self.allowed:
            raise ValueError(f"enum field {self.id!r} needs a non-empty `allowed` list")
        if self.type == "measurement" and not self.units:
            raise ValueError(f"measurement field {self.id!r} needs a non-empty `units` list")
        if self.type != "measurement" and self.units:
            raise ValueError(f"field {self.id!r}: `units` is only for measurement fields")
        if self.type == "integer" and self.allowed is not None:
            bad = [a for a in self.allowed if isinstance(a, bool) or not isinstance(a, int)]
            if bad:
                raise ValueError(f"integer field {self.id!r}: `allowed` must be integers, got {bad}")
        if self.type in ("date", "measurement", "number") and self.allowed:
            raise ValueError(f"field {self.id!r}: `allowed` is not meaningful for type {self.type}")
        if self.allowed is not None and len(set(map(str, self.allowed))) != len(self.allowed):
            raise ValueError(f"field {self.id!r}: duplicate entries in `allowed`")
        return self


class ValidationRule(_Strict):
    id: str
    rule: str
    on_fail: Literal["flag", "retry_then_flag"] = "flag"


class DerivedField(_Strict):
    id: str
    derivation: str
    inputs: list[str]
    rule: str
    when: Literal["not_stated", "always"] = "not_stated"
    evidence: Literal["inherit", "none"] = "inherit"
    stated_column: str | None = None    # gold column (true/false/blank): does the document print the value?
    eval_note: str | None = None


class EnvelopeProperty(_Strict):
    name: str
    description: str = ""


class Instructions(_Strict):
    # ``global`` is a Python keyword, hence the alias.
    global_: str = Field(default="", alias="global")
    domain: str = ""


class SelfConsistency(_Strict):
    enabled: bool = False
    k: int = 5


class Features(_Strict):
    evidence_verification: bool = False
    validation_retry: bool = False
    self_consistency: SelfConsistency = SelfConsistency()
    verifier_agent: bool = False
    reconciliation: bool = False


class Reconciliation(_Strict):
    precedence: list[str] = []
    on_unresolved: Literal["flag"] = "flag"


class Exemplar(_Strict):
    source: str
    output: str
    case_id: str | None = None   # the eval case the exemplar was taken from; the eval runner never scores it


class EvalSpec(_Strict):
    gold: str
    id_column: str = "case_id"
    scored: list[str] = []


# ---------------------------------------------------------------------------
# The merged pack
# ---------------------------------------------------------------------------

class Pack(_Strict):
    """A fully merged, validated pack. Build one with :func:`load_pack`."""

    pack: PackMeta
    document: DocumentSpec | None = None      # None only for a bare base pack
    defaults: Defaults = Defaults()
    status_values: dict[str, str]
    hedging_levels: dict[str, str]
    kinds: dict[str, str] = {}
    field_types: dict[str, str] = {}
    envelope: list[EnvelopeProperty]
    instructions: Instructions = Instructions()
    fields: list[FieldSpec]
    derived_fields: list[DerivedField] = []
    validation: list[ValidationRule] = []
    reconciliation: Reconciliation = Reconciliation()
    features: Features = Features()
    exemplars: list[Exemplar] = []
    eval: EvalSpec | None = None
    root: Path | None = Field(default=None, exclude=True)   # pack directory; not part of the YAML
    _record_model: Any = PrivateAttr(default=None)           # cache for engine.schema

    # -- convenience ---------------------------------------------------------
    @property
    def name(self) -> str:
        return self.pack.name

    @property
    def version(self) -> int:
        return self.pack.version

    @property
    def field_ids(self) -> list[str]:
        return [f.id for f in self.fields]

    def field(self, field_id: str) -> FieldSpec:
        for f in self.fields:
            if f.id == field_id:
                return f
        raise KeyError(field_id)

    @property
    def instructions_text(self) -> str:
        parts = [self.instructions.global_.strip(), self.instructions.domain.strip()]
        return "\n\n".join(p for p in parts if p)

    @property
    def scored_fields(self) -> list[str]:
        return list(self.eval.scored) if self.eval else []

    def derived_for(self, field_id: str) -> DerivedField | None:
        for d in self.derived_fields:
            if d.id == field_id:
                return d
        return None

    @property
    def exemplar_ids(self) -> list[str]:
        """Case ids the exemplars were taken from. They sit in the prompt, so
        scoring them would be leakage: the eval runner holds them out of every
        run, whatever ``--ids`` or ``--cases`` asks for."""
        return [ex.case_id for ex in self.exemplars if ex.case_id]

    def resolve(self, relative: str) -> Path:
        """A path written in the pack, resolved against the pack directory."""
        if self.root is None:
            raise PackError("pack has no root directory; load it from disk to resolve paths")
        return (self.root / relative).resolve()

    def to_yaml(self) -> str:
        data = self.model_dump(by_alias=True, mode="json")
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    # -- consistency checks --------------------------------------------------
    @model_validator(mode="after")
    def _consistent(self) -> "Pack":
        if tuple(self.status_values) != STATUS_VALUES:
            raise ValueError(
                f"status_values must be exactly {list(STATUS_VALUES)} in that order; "
                f"got {list(self.status_values)}"
            )
        if not self.hedging_levels:
            raise ValueError("hedging_levels must not be empty")
        env_names = [e.name for e in self.envelope]
        missing = [n for n in REQUIRED_ENVELOPE if n not in env_names]
        if missing:
            raise ValueError(f"envelope is missing properties {missing}")

        ids = [f.id for f in self.fields]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate field ids: {dupes}")
        known = set(ids)
        if self.fields and self.document is None:
            raise ValueError("a pack that declares fields must declare `document` (type, unit, case_grouping)")

        # apply field defaults, remembering which keys the pack left to them
        d = self.defaults.field
        for f in self.fields:
            for key in ("status", "evidence", "tier", "kind"):
                if getattr(f, key) is None:
                    setattr(f, key, getattr(d, key))
                    if key not in f._defaulted:
                        f._defaulted.append(key)

        for rule in self.validation:
            refs = _rule_refs(rule.rule, f"validation rule {rule.id!r}")
            unknown = refs - known
            if unknown:
                raise ValueError(f"validation rule {rule.id!r} references unknown fields {sorted(unknown)}")

        seen_derived: set[str] = set()
        for der in self.derived_fields:
            if der.id in seen_derived:
                raise ValueError(f"derived_fields: duplicate entry for {der.id!r}")
            seen_derived.add(der.id)
            if der.id not in known:
                raise ValueError(f"derived_fields: {der.id!r} is not a declared field")
            unknown = set(der.inputs) - known
            if unknown:
                raise ValueError(f"derived field {der.id!r}: unknown inputs {sorted(unknown)}")
            if der.id in der.inputs:
                raise ValueError(f"derived field {der.id!r} cannot be its own input")
            refs = _rule_refs(der.rule, f"derived field {der.id!r}")
            outside = refs - set(der.inputs)
            if outside:
                raise ValueError(
                    f"derived field {der.id!r}: rule references {sorted(outside)} "
                    f"which are not listed in `inputs`"
                )

        if self.eval:
            unknown = set(self.eval.scored) - known
            if unknown:
                raise ValueError(f"eval.scored references unknown fields {sorted(unknown)}")

        ex_ids = self.exemplar_ids
        dupes = sorted({i for i in ex_ids if ex_ids.count(i) > 1})
        if dupes:
            raise ValueError(f"exemplars: duplicate case_id {dupes}")
        if self.root is not None:
            for ex in self.exemplars:
                for p in (ex.source, ex.output):
                    if not (self.root / p).exists():
                        raise ValueError(f"exemplar file not found: {self.root / p}")
        return self


def _rule_refs(expr: str, where: str) -> set[str]:
    try:
        rule = compile_rule(expr)
    except RuleError as e:
        raise ValueError(f"{where}: {e}") from None
    return set(rule.names) - set(FUNCTIONS)


# ---------------------------------------------------------------------------
# Raw files: validated per file so errors name the file, then merged as dicts
# ---------------------------------------------------------------------------

class RawPack(_Strict):
    """One pack.yaml as written. Everything optional: base lacks `document`,
    extensions lack `envelope`; the merged :class:`Pack` enforces completeness."""

    pack: PackMeta
    document: DocumentSpec | None = None
    defaults: Defaults | None = None
    status_values: dict[str, str] | None = None
    hedging_levels: dict[str, str] | None = None
    kinds: dict[str, str] | None = None
    field_types: dict[str, str] | None = None
    envelope: list[EnvelopeProperty] | None = None
    instructions: Instructions | None = None
    fields: list[FieldSpec] | None = None
    derived_fields: list[DerivedField] | None = None
    validation: list[ValidationRule] | None = None
    reconciliation: Reconciliation | None = None
    features: Features | None = None
    exemplars: list[Exemplar] | None = None
    eval: EvalSpec | None = None


def _format_errors(err: ValidationError) -> str:
    lines = []
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        lines.append(f"  {loc}: {e['msg']}")
    return "\n".join(lines)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PackError(f"pack file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise PackError(f"{path}: YAML syntax error: {e}") from None
    if not isinstance(data, dict):
        raise PackError(f"{path}: top level must be a mapping")
    try:
        RawPack.model_validate(data)
    except ValidationError as e:
        raise PackError(f"{path}: invalid pack\n{_format_errors(e)}") from None
    for section in ("fields", "derived_fields"):
        ids = [item["id"] for item in (data.get(section) or [])]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise PackError(f"{path}: duplicate field ids in `{section}`: {dupes}")
    return data


def _merge_by_id(base: list[dict], ext: list[dict]) -> list[dict]:
    out = [copy.deepcopy(x) for x in base]
    index = {x["id"]: i for i, x in enumerate(out)}
    for item in ext:
        if item["id"] in index:
            out[index[item["id"]]] = copy.deepcopy(item)
        else:
            index[item["id"]] = len(out)
            out.append(copy.deepcopy(item))
    return out


def _deep_merge(base: dict, ext: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in ext.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merge_pack_dicts(base: dict[str, Any], ext: dict[str, Any]) -> dict[str, Any]:
    """Apply the merge rules documented at the top of packs/base/pack.yaml."""
    out = copy.deepcopy(base)
    for key, val in ext.items():
        if val is None:
            continue
        if key == "pack":
            out["pack"] = copy.deepcopy(val)             # the leaf's identity wins
        elif key in ("fields", "derived_fields"):
            out[key] = _merge_by_id(out.get(key) or [], val)
        elif key == "validation":
            out[key] = (out.get(key) or []) + copy.deepcopy(val)
        elif isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)         # instructions, features, defaults, ...
        else:
            out[key] = copy.deepcopy(val)                 # document, exemplars, eval, lists
    return out


def _resolve_pack_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.parent
    return p


def _load_chain(pack_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """[(dir, raw), ...] from the root ancestor down to the requested pack."""
    chain: list[tuple[Path, dict[str, Any]]] = []
    seen: set[Path] = set()
    current = pack_dir.resolve()
    while True:
        if current in seen:
            raise PackError(f"circular `extends` chain involving {current}")
        seen.add(current)
        raw = _read_yaml(current / PACK_FILENAME)
        chain.append((current, raw))
        parent = raw["pack"].get("extends")
        if not parent:
            break
        current = (current.parent / parent).resolve()
        if not (current / PACK_FILENAME).exists():
            raise PackError(
                f"{pack_dir / PACK_FILENAME}: extends {parent!r} but {current / PACK_FILENAME} does not exist"
            )
    chain.reverse()
    return chain


def load_pack(path: str | Path) -> Pack:
    """Load ``packs/<name>/`` (or its pack.yaml), merging the ``extends`` chain.

    Raises :class:`PackError` with a readable message on any problem.
    """
    pack_dir = _resolve_pack_dir(path)
    chain = _load_chain(pack_dir)
    merged: dict[str, Any] = {}
    for _, raw in chain:
        merged = merge_pack_dicts(merged, raw)
    merged["root"] = pack_dir.resolve()
    try:
        return Pack.model_validate(merged)
    except ValidationError as e:
        raise PackError(f"{pack_dir / PACK_FILENAME}: invalid after merging its `extends` chain\n{_format_errors(e)}") from None


def load_merged_yaml(text: str, root: Path | None = None) -> Pack:
    """Rebuild a :class:`Pack` from :meth:`Pack.to_yaml` output (round-trip)."""
    data = yaml.safe_load(text)
    if root is not None:
        data["root"] = root
    try:
        return Pack.model_validate(data)
    except ValidationError as e:
        raise PackError(f"invalid merged pack\n{_format_errors(e)}") from None


def list_packs(packs_dir: str | Path) -> list[str]:
    return sorted(p.name for p in Path(packs_dir).iterdir() if (p / PACK_FILENAME).is_file())


# ---------------------------------------------------------------------------
# `python -m engine.packloader packs/prostate_pathology`: show the merged pack
# ---------------------------------------------------------------------------

def describe(pack: Pack) -> str:
    """Human-readable summary of a merged pack. A ``*`` after a field key
    means the value came from ``defaults.field`` rather than the pack."""
    doc = pack.document
    lines = [
        f"pack: {pack.name} v{pack.version}" + (f"  (extends: {pack.pack.extends})" if pack.pack.extends else ""),
        f"root: {pack.root}",
        "document: " + (f"{doc.type}  unit={doc.unit}  case_grouping={doc.case_grouping}" if doc else "-"),
        "envelope: " + " -> ".join(e.name for e in pack.envelope),
        "status values: " + " | ".join(pack.status_values),
        "hedging levels: " + " | ".join(pack.hedging_levels),
        f"instructions: global {len(pack.instructions.global_)} chars (base), domain {len(pack.instructions.domain)} chars (pack)",
        f"field defaults: " + "  ".join(f"{k}={v}" for k, v in pack.defaults.field.model_dump().items()),
        f"fields: {len(pack.fields)}",
    ]
    for f in pack.fields:
        extras = []
        if f.allowed is not None:
            extras.append("allowed=" + ",".join(str(a) for a in f.allowed))
        if f.units:
            extras.append("units=" + ",".join(f.units))
        if f.unit:
            extras.append(f"unit={f.unit}")
        keys = "  ".join(f"{k}={getattr(f, k)}{'*' if k in f._defaulted else ''}"
                         for k in ("tier", "kind", "evidence", "status"))
        lines.append(f"  {f.id:<28} {f.type:<12} {keys}" + (("  " + " ".join(extras)) if extras else ""))
    lines.append(f"derived_fields: {len(pack.derived_fields)}")
    for der in pack.derived_fields:
        lines.append(f"  {der.id} <- {der.rule}   inputs={der.inputs}  when={der.when}  "
                     f"evidence={der.evidence}  derivation={der.derivation!r}")
    lines.append(f"validation rules: {len(pack.validation)}")
    for rule in pack.validation:
        lines.append(f"  {rule.id}: {rule.rule}   on_fail={rule.on_fail}")
    lines.append("reconciliation precedence: " + (", ".join(pack.reconciliation.precedence) or "-"))
    feats = pack.features.model_dump()
    sc = feats.pop("self_consistency")
    on = [k for k, v in feats.items() if v] + (["self_consistency"] if sc["enabled"] else [])
    lines.append(f"features on: {', '.join(on) or 'none'}  (switches: {', '.join(feats)}, self_consistency k={sc['k']})")
    lines.append(f"exemplars: {len(pack.exemplars)}"
                 + (f"  (held out of every eval: {', '.join(pack.exemplar_ids)})" if pack.exemplar_ids else ""))
    if pack.eval:
        lines.append(f"eval: gold={pack.eval.gold}  id_column={pack.eval.id_column}  scored={pack.eval.scored}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(description="Load a pack (merging its `extends` chain) and describe it.")
    ap.add_argument("pack", help="pack directory, e.g. packs/prostate_pathology")
    ap.add_argument("--yaml", action="store_true", help="print the merged pack as YAML instead")
    ap.add_argument("--schema", action="store_true", help="print the JSON schema sent to the model server")
    args = ap.parse_args(argv)
    try:
        pack = load_pack(args.pack)
    except PackError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.yaml:
        print(pack.to_yaml(), end="")
    elif args.schema:
        from engine.schema import build_json_schema
        print(json.dumps(build_json_schema(pack), indent=1))
    else:
        print(describe(pack))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
