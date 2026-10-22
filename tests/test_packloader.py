import shutil
from pathlib import Path

import pytest
import yaml

from engine.packloader import PackError, describe, load_merged_yaml, load_pack, merge_pack_dicts
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
    assert [d.id for d in prostate.derived_fields] == ["grade_group", "pT", "pN"]
    gg = prostate.derived_for("grade_group")
    assert gg.derivation == "computed_from_gleason" and gg.when == "not_stated" and gg.on_none == "flag"
    assert gg.input_ids == ["gleason_primary", "gleason_secondary"]
    assert gg.input_modes == {"gleason_primary": "required", "gleason_secondary": "required"}   # list form
    pt = prostate.derived_for("pT")
    assert pt.derivation == "computed_from_findings" and pt.when == "check" and pt.on_none == "skip"
    assert pt.input_modes == {"specimen_procedure": "required", "extraprostatic_extension": "finding",
                              "seminal_vesicle_invasion": "finding", "tumor_laterality": "optional"}
    assert pt.stated_column == "pT_printed" and prostate.derived_for("pN").stated_column == "pN_printed"
    assert prostate.derived_for("pN").derivation == "computed_from_nodes"
    assert prostate.derived_for("gleason_total") is None
    assert prostate.scored_fields == ["gleason_primary", "gleason_secondary", "gleason_total", "grade_group", "pT", "pN"]
    assert prostate.eval.lenient == {"pT": "prefix"}
    # exemplars: three in every prompt, the hedging arm only on request, all four held out
    assert prostate.exemplar_ids == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-ZG-A8QW", "TCGA-V1-A9Z7"]
    assert prostate.exemplar_variants == ["hedging"]
    assert [ex.case_id for ex in prostate.prompt_exemplars()] == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-ZG-A8QW"]
    assert [ex.case_id for ex in prostate.prompt_exemplars(["hedging"])][-1] == "TCGA-V1-A9Z7"
    with pytest.raises(PackError, match="no exemplar variant \\['hedgng'\\]"):
        prostate.prompt_exemplars(["hedgng"])
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
    (lambda p: p.update({"derived_fields": [{"id": "b", "derivation": "d", "inputs": {"a": "maybe"}, "rule": "a"}]}), "optional"),
    (lambda p: p.update({"derived_fields": [{"id": "b", "derivation": "d", "inputs": {"a": "finding"}, "rule": "a", "when": "sometimes"}]}), "check"),
    (lambda p: p.update({"derived_fields": [{"id": "b", "derivation": "d", "inputs": ["a"], "rule": "a", "on_none": "ignore"}]}), "skip"),
    (lambda p: p.update({"eval": {"gold": "g.csv", "scored": ["a"], "lenient": {"b": "prefix"}}}), "not in eval.scored"),
    (lambda p: p.update({"eval": {"gold": "g.csv", "scored": ["a"], "lenient": {"a": "fuzzy"}}}), "prefix"),
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


def test_exemplar_case_ids_are_optional_unique_and_listed(tmp_path):
    d = _packs_dir(tmp_path)
    ex = d / "mini" / "exemplars"
    ex.mkdir(parents=True)
    for name in ("one", "two"):
        (ex / f"{name}.txt").write_text("x", encoding="utf-8")
        (ex / f"{name}.json").write_text("{}", encoding="utf-8")
    body = dict(GOOD, exemplars=[
        {"case_id": "TCGA-AA-0001", "source": "exemplars/one.txt", "output": "exemplars/one.json"},
        {"source": "exemplars/two.txt", "output": "exemplars/two.json"},            # case_id is optional
    ])
    (d / "mini" / "pack.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    pack = load_pack(d / "mini")
    assert pack.exemplar_ids == ["TCGA-AA-0001"]
    assert "held out of every eval: TCGA-AA-0001" in describe(pack)

    body["exemplars"][1]["case_id"] = "TCGA-AA-0001"
    (d / "mini" / "pack.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    with pytest.raises(PackError, match="duplicate case_id"):
        load_pack(d / "mini")


def test_exemplar_variants_are_opt_in_but_always_held_out(tmp_path):
    d = _packs_dir(tmp_path)
    ex = d / "mini" / "exemplars"
    ex.mkdir(parents=True)
    for name in ("one", "two"):
        (ex / f"{name}.txt").write_text("x", encoding="utf-8")
        (ex / f"{name}.json").write_text("{}", encoding="utf-8")
    body = dict(GOOD, exemplars=[
        {"case_id": "C-1", "source": "exemplars/one.txt", "output": "exemplars/one.json"},
        {"case_id": "C-2", "source": "exemplars/two.txt", "output": "exemplars/two.json", "variant": "hedging"},
    ])
    (d / "mini" / "pack.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    pack = load_pack(d / "mini")
    assert pack.exemplar_ids == ["C-1", "C-2"] and pack.exemplar_variants == ["hedging"]
    assert [e.case_id for e in pack.prompt_exemplars()] == ["C-1"]
    assert [e.case_id for e in pack.prompt_exemplars(("hedging",))] == ["C-1", "C-2"]
    with pytest.raises(PackError, match="no exemplar variant"):
        pack.prompt_exemplars(["typo"])
    assert "variant=hedging" in describe(pack)
    # the inputs mapping and the new keys survive a YAML round trip
    body["derived_fields"] = [{"id": "b", "derivation": "d", "inputs": {"a": "optional"}, "rule": "a",
                               "when": "check", "on_none": "skip"}]
    (d / "mini" / "pack.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    pack = load_pack(d / "mini")
    again = load_merged_yaml(pack.to_yaml(), root=pack.root)
    assert again.derived_for("b").input_modes == {"a": "optional"} and again.derived_for("b").when == "check"
