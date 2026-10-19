import shutil
from pathlib import Path

import pytest
import yaml

from engine.packloader import PackError, load_merged_yaml, load_pack, merge_pack_dicts
from tests.conftest import PACKS


def test_base_loads_standalone(base):
    assert base.name == "base" and base.version == 1
    assert base.fields == [] and base.document is None
    assert [e.name for e in base.envelope] == ["evidence", "status", "value", "unit", "hedging", "conflict"]
    assert list(base.status_values) == ["value", "explicitly_absent", "not_mentioned"]
    assert list(base.hedging_levels) == ["definite", "probable", "possible", "indeterminate"]


def test_prostate_merges_base(prostate):
    assert prostate.name == "prostate_pathology" and prostate.pack.extends == "base"
    assert prostate.document.type == "surgical_pathology_report"
    assert len(prostate.fields) == 23
    assert "tumor_laterality" in prostate.field_ids and "specimen_laterality" not in prostate.field_ids
    # machinery inherited from base
    assert prostate.status_values == {**prostate.status_values}  # present
    assert [e.name for e in prostate.envelope][0] == "evidence"
    assert prostate.instructions_text.startswith("You extract structured data")
    assert "prostate surgical pathology report" in prostate.instructions_text
    # defaults applied where the pack was silent, respected where it was not
    gp = prostate.field("gleason_primary")
    assert (gp.status, gp.evidence, gp.tier, gp.kind) == ("three_state", "required", "standard", "attribute")
    assert prostate.field("pT").tier == "deterministic"
    assert prostate.field("margin_positive").kind == "finding"
    assert prostate.field("tumor_dimension").units == ["mm", "cm"]
    # vocabulary sections
    assert [r.id for r in prostate.validation] == ["gleason_sum", "grade_group_matches_gleason",
                                                   "nodes_positive_le_examined", "pN_matches_nodes"]
    assert [d.id for d in prostate.derived_fields] == ["grade_group"]
    assert prostate.derived_for("grade_group").derivation == "computed_from_gleason"
    assert prostate.derived_for("gleason_total") is None
    assert prostate.scored_fields == ["gleason_primary", "gleason_secondary", "gleason_total", "grade_group", "pT", "pN"]
    assert prostate.reconciliation.precedence == ["prostatectomy_over_biopsy", "highest_grade_among_cores"]
    assert not prostate.features.self_consistency.enabled and prostate.features.self_consistency.k == 5
    assert prostate.root == (PACKS / "prostate_pathology").resolve()


@pytest.mark.parametrize("name", ["base", "prostate_pathology"])
def test_round_trip_through_yaml(name):
    pack = load_pack(PACKS / name)
    again = load_merged_yaml(pack.to_yaml(), root=pack.root)
    assert again.model_dump(exclude={"root"}) == pack.model_dump(exclude={"root"})
    assert again.to_yaml() == pack.to_yaml()


# ---------------------------------------------------------------------------
# malformed packs fail loudly, naming the file and the problem
# ---------------------------------------------------------------------------

def _packs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "packs"
    d.mkdir()
    shutil.copytree(PACKS / "base", d / "base")
    return d


def _write_pack(packs_dir: Path, name: str, body: dict) -> Path:
    (packs_dir / name).mkdir()
    path = packs_dir / name / "pack.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


GOOD = {
    "pack": {"name": "mini", "version": 1, "extends": "base"},
    "document": {"type": "note", "unit": "report", "case_grouping": "patient_id"},
    "fields": [
        {"id": "a", "type": "integer", "allowed": [1, 2, 3], "instructions": "a"},
        {"id": "b", "type": "integer", "instructions": "b"},
        {"id": "grade", "type": "enum", "allowed": ["x", "y"], "instructions": "g"},
    ],
    "validation": [{"id": "ab", "rule": "a <= b", "on_fail": "flag"}],
}


def test_minimal_extension_loads(tmp_path):
    d = _packs_dir(tmp_path)
    _write_pack(d, "mini", GOOD)
    pack = load_pack(d / "mini")
    assert pack.field_ids == ["a", "b", "grade"]
    assert pack.field("a").tier == "standard"


@pytest.mark.parametrize("mutate, needle", [
    (lambda p: p.update({"feilds": []}), "feilds"),
    (lambda p: p["fields"][2].pop("allowed"), "allowed"),
    (lambda p: p["fields"][0].update({"instuctions": "typo"}), "instuctions"),
    (lambda p: p["fields"][0].update({"type": "float"}), "type"),
    (lambda p: p["fields"][0].update({"allowed": [1, "two"]}), "integers"),
    (lambda p: p["fields"].append(dict(p["fields"][0])), "duplicate field ids"),
    (lambda p: p["validation"].append({"id": "r", "rule": "a + zzz == b"}), "zzz"),
    (lambda p: p["validation"].append({"id": "r", "rule": "__import__('os').system('x')"}), "not whitelisted"),
    (lambda p: p["validation"].append({"id": "r", "rule": "a.__class__"}), "unsupported syntax"),
    (lambda p: p["validation"].append({"id": "r", "rule": "a +"}), "cannot parse"),
    (lambda p: p.update({"derived_fields": [{"id": "nope", "derivation": "d", "inputs": ["a"], "rule": "a"}]}), "nope"),
    (lambda p: p.update({"derived_fields": [{"id": "b", "derivation": "d", "inputs": ["a"], "rule": "a + grade"}]}), "inputs"),
    (lambda p: p.update({"eval": {"gold": "g.csv", "scored": ["a", "ghost"]}}), "ghost"),
    (lambda p: p.update({"exemplars": [{"source": "exemplars/x.txt", "output": "exemplars/x.json"}]}), "exemplar file not found"),
    (lambda p: p["pack"].update({"extends": "nonexistent"}), "nonexistent"),
    (lambda p: p.update({"features": {"evidence_verification": "maybe"}}), "evidence_verification"),
])
def test_malformed_pack_fails_loudly(tmp_path, mutate, needle):
    import copy
    d = _packs_dir(tmp_path)
    body = copy.deepcopy(GOOD)
    mutate(body)
    path = _write_pack(d, "mini", body)
    with pytest.raises(PackError) as err:
        load_pack(d / "mini")
    msg = str(err.value)
    assert needle in msg, msg
    assert "pack.yaml" in msg


def test_yaml_syntax_error_names_file(tmp_path):
    d = _packs_dir(tmp_path)
    (d / "broken").mkdir()
    (d / "broken" / "pack.yaml").write_text("pack: {name: x\n", encoding="utf-8")
    with pytest.raises(PackError, match="YAML syntax error"):
        load_pack(d / "broken")


def test_missing_pack_dir():
    with pytest.raises(PackError, match="not found"):
        load_pack(PACKS / "does_not_exist")


def test_circular_extends(tmp_path):
    d = _packs_dir(tmp_path)
    _write_pack(d, "p1", {"pack": {"name": "p1", "version": 1, "extends": "p2"}})
    _write_pack(d, "p2", {"pack": {"name": "p2", "version": 1, "extends": "p1"}})
    with pytest.raises(PackError, match="circular"):
        load_pack(d / "p1")


def test_extension_overrides_field_and_concatenates_rules():
    base = {"pack": {"name": "b", "version": 1}, "fields": [{"id": "x", "type": "integer", "allowed": [1, 2]}],
            "validation": [{"id": "r1", "rule": "x > 0"}], "instructions": {"global": "G"},
            "features": {"validation_retry": False, "self_consistency": {"enabled": False, "k": 5}}}
    ext = {"pack": {"name": "e", "version": 2, "extends": "b"},
           "fields": [{"id": "x", "type": "integer", "allowed": [1, 2, 3]}, {"id": "y", "type": "string"}],
           "validation": [{"id": "r2", "rule": "x < 10"}], "instructions": {"domain": "D"},
           "features": {"self_consistency": {"k": 3}}}
    merged = merge_pack_dicts(base, ext)
    assert merged["pack"] == ext["pack"]
    assert [f["id"] for f in merged["fields"]] == ["x", "y"]
    assert merged["fields"][0]["allowed"] == [1, 2, 3]
    assert [r["id"] for r in merged["validation"]] == ["r1", "r2"]
    assert merged["instructions"] == {"global": "G", "domain": "D"}
    assert merged["features"] == {"validation_retry": False, "self_consistency": {"enabled": False, "k": 3}}
