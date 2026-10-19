import json
import shutil

import yaml

from engine.packloader import load_pack
from engine.prompt_builder import DOC_END, DOC_START, build_prompt, field_catalog, output_format
from tests.conftest import PACKS, REPORT


def test_prompt_structure(prostate):
    prompt = build_prompt(prostate, REPORT)
    assert prompt == build_prompt(prostate, REPORT)                      # deterministic
    assert prompt.index("You extract structured data") < prompt.index("## Fields to extract")
    assert prompt.index("## Fields to extract") < prompt.index("## Output format") < prompt.index("## Document")
    assert f"{DOC_START}\n{REPORT.strip()}\n{DOC_END}" in prompt
    assert prompt.rstrip().endswith("Return the JSON object for this document now.")
    assert "Silence is not absence" in prompt and "prostate surgical pathology report" in prompt
    for fid in prostate.field_ids:
        assert f"- {fid} (" in prompt
    assert "- pT (enum; allowed: pT2 | pT2a | pT2b | pT2c | pT2+ | pT3 | pT3a | pT3b | pT4 | pTX)" in prompt
    assert "- tumor_dimension (measurement; units: mm, cm)" in prompt
    assert "- margin_positive (enum; allowed: focal | extensive | present_unspecified; finding)" in prompt
    assert "- tumor_percent_of_gland (number; unit: percent)" in prompt


def test_output_format_lists_vocabularies(prostate):
    text = output_format(prostate)
    assert "- status: one of value | explicitly_absent | not_mentioned" in text
    assert "- hedging: one of definite | probable | possible | indeterminate" in text
    assert "Field kinds:" in text and "- finding:" in text
    assert "Value types:" in text and "- measurement:" in text
    assert "- evidence:" in text and "- conflict:" in text


def test_field_catalog_flattens_instructions(prostate):
    catalog = field_catalog(prostate)
    assert "\n  Primary (most prevalent) Gleason pattern, the first number in the Gleason score as written." in catalog


def test_exemplars_are_rendered(tmp_path):
    packs = tmp_path / "packs"
    shutil.copytree(PACKS / "base", packs / "base")
    (packs / "mini" / "exemplars").mkdir(parents=True)
    (packs / "mini" / "exemplars" / "ex01.txt").write_text("Gleason 4+4=8.", encoding="utf-8")
    example = {"g": {"evidence": ["Gleason 4+4=8"], "status": "value", "value": 8, "hedging": "definite",
                     "conflict": False}}
    (packs / "mini" / "exemplars" / "ex01.json").write_text(json.dumps(example), encoding="utf-8")
    (packs / "mini" / "pack.yaml").write_text(yaml.safe_dump({
        "pack": {"name": "mini", "version": 1, "extends": "base"},
        "document": {"type": "note"},
        "fields": [{"id": "g", "type": "integer", "instructions": "Gleason sum."}],
        "exemplars": [{"source": "exemplars/ex01.txt", "output": "exemplars/ex01.json"}],
    }, sort_keys=False), encoding="utf-8")
    pack = load_pack(packs / "mini")
    prompt = build_prompt(pack, "Gleason 3+3=6.")
    assert prompt.count("## Example") == 1
    assert prompt.index("## Example") < prompt.index("## Document")
    assert "Gleason 4+4=8." in prompt and '"value": 8' in prompt
    assert "## Example" not in build_prompt(pack, "x", include_exemplars=False)
