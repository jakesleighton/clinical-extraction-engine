import json

import pytest

from engine import ledger
from engine.backends.ollama import DEFAULT_NUM_CTX
from engine.extract import default_num_ctx, extract_document, main, parse_think
from engine.packloader import PackError
from tests.conftest import PACKS, REPORT, FakeOllama


def test_end_to_end_with_mocked_server(prostate, fake_ollama, runs_dir):
    result = extract_document(prostate, REPORT, model="m", case_id="CASE-1", run_id="e2e", seed=0)
    assert result.ok and result.issues == []
    rec = result.to_record()
    assert rec["case_id"] == "CASE-1" and rec["pack"] == "prostate_pathology" and rec["run_id"] == "e2e"
    assert rec["prompt_hash"] == result.prompt_hash and len(rec["prompt_hash"]) == 64
    assert rec["fields"]["gleason_total"]["value"] == 7
    gg = rec["fields"]["grade_group"]
    assert gg["value"] == 2 and gg["derivation"] == "computed_from_gleason"
    assert all(e["start"] is not None for f in rec["fields"].values() for e in f["evidence"])

    # the prompt the server saw contains the document, the field catalog and the pack's exemplars
    [req] = fake_ollama.requests
    assert REPORT.strip() in req["prompt"] and "- gleason_primary (integer; allowed: 1 | 2 | 3 | 4 | 5)" in req["prompt"]
    assert req["format"]["required"] == prostate.field_ids
    assert req["prompt"].count("## Example") == len(prostate.prompt_exemplars()) == 3
    assert req["options"]["num_ctx"] == default_num_ctx(req["prompt"]) == 16384   # three exemplars: the wider context

    lines = list(ledger.read(runs_dir / "e2e.jsonl"))
    assert [l["kind"] for l in lines] == ["llm_call", "validation"]
    assert lines[0]["call_id"] == lines[1]["call_id"] == result.llm.call_id
    assert lines[1]["case_id"] == "CASE-1" and lines[1]["ok"] is True and lines[1]["n_flags"] == 0
    assert lines[1]["exemplars"] == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-ZG-A8QW"]
    assert lines[1]["exemplar_variants"] == []
    # the stage was printed and the engine's derivation from the findings agrees: provenance stays "stated"
    assert rec["fields"]["pT"]["derivation"] == "stated" and rec["fields"]["pN"]["derivation"] == "stated"


def test_default_num_ctx_is_sized_from_the_prompt():
    assert default_num_ctx("") == DEFAULT_NUM_CTX == 8192
    assert default_num_ctx("x" * 30000) == 14336                    # 10000 tokens + 4096 headroom -> next 2048 step
    assert default_num_ctx("x" * (3 * (16384 - 4096))) == 16384     # exactly on the boundary
    assert default_num_ctx("x" * (3 * (16384 - 4096) + 3)) == 18432


def test_num_ctx_default_follows_the_prompt(prostate, fake_ollama, runs_dir):
    extract_document(prostate, REPORT, model="m", run_id="ctx", include_exemplars=False)
    assert fake_ollama.requests[-1]["options"]["num_ctx"] == DEFAULT_NUM_CTX
    extract_document(prostate, REPORT, model="m", run_id="ctx")
    assert fake_ollama.requests[-1]["options"]["num_ctx"] == 16384
    extract_document(prostate, REPORT, model="m", run_id="ctx", exemplar_variants=("hedging",))
    assert fake_ollama.requests[-1]["options"]["num_ctx"] == 20480       # four exemplars need the next size up
    extract_document(prostate, REPORT, model="m", run_id="ctx", num_ctx=32768)
    assert fake_ollama.requests[-1]["options"]["num_ctx"] == 32768       # an explicit value always wins
    assert fake_ollama.requests[-1]["think"] is False


def test_exemplar_variant_adds_the_hedging_arm(prostate, fake_ollama, runs_dir):
    extract_document(prostate, REPORT, model="m", case_id="C", run_id="arm", exemplar_variants=["hedging"])
    req = fake_ollama.requests[-1]
    assert req["prompt"].count("## Example") == 4 and "TCGA-V1-A9Z7" not in req["prompt"].split("## Document")[-1]
    line = list(ledger.read(runs_dir / "arm.jsonl"))[1]
    assert line["exemplars"] == ["TCGA-EJ-7315", "TCGA-HC-A6AN", "TCGA-ZG-A8QW", "TCGA-V1-A9Z7"]
    assert line["exemplar_variants"] == ["hedging"]
    with pytest.raises(PackError, match="no exemplar variant"):
        extract_document(prostate, REPORT, model="m", run_id="arm", exemplar_variants=["hedgng"])


def test_parse_think():
    assert parse_think("false") is False and parse_think("True") is True and parse_think("low") == "low"
    with pytest.raises(ValueError, match="--think must be one of"):
        parse_think("maybe")


def test_failed_call_still_returns_and_logs(monkeypatch, prostate, runs_dir):
    FakeOllama(raw_text="not json").install(monkeypatch)
    result = extract_document(prostate, REPORT, model="m", case_id="C", run_id="bad")
    assert not result.ok and result.record is None and result.issues == []
    rec = result.to_record()
    assert rec["ok"] is False and rec["fields"] is None and "not valid JSON" in rec["error"]
    lines = list(ledger.read(runs_dir / "bad.jsonl"))
    assert lines[1]["ok"] is False and lines[1]["llm_error"] == result.llm.error


def test_cli_writes_record(tmp_path, fake_ollama, runs_dir, capsys):
    doc = tmp_path / "CASE-9.txt"
    doc.write_text(REPORT, encoding="utf-8")
    out = tmp_path / "out.json"
    code = main([str(doc), "--pack", str(PACKS / "prostate_pathology"), "--model", "m",
                 "--run-id", "cli", "--out", str(out), "--num-ctx", "4096"])
    assert code == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["case_id"] == "CASE-9" and rec["fields"]["pT"]["value"] == "pT2c"
    assert fake_ollama.requests[0]["options"]["num_ctx"] == 4096
    assert fake_ollama.requests[0]["think"] is False
    assert "[ok]" in capsys.readouterr().err
    assert (runs_dir / "cli.jsonl").exists()
    code = main([str(doc), "--pack", str(PACKS / "prostate_pathology"), "--model", "gpt-oss:20b",
                 "--run-id", "cli2", "--out", str(out), "--think", "low"])
    assert code == 0 and fake_ollama.requests[1]["think"] == "low"
    assert fake_ollama.requests[1]["options"]["num_ctx"] == 16384
    code = main([str(doc), "--pack", str(PACKS / "prostate_pathology"), "--model", "m",
                 "--run-id", "cli3", "--out", str(out), "--exemplar-variant", "hedging"])
    assert code == 0 and fake_ollama.requests[2]["prompt"].count("## Example") == 4
    assert fake_ollama.requests[2]["options"]["num_ctx"] == 20480


def test_cli_show_prompt(tmp_path, capsys):
    doc = tmp_path / "x.txt"
    doc.write_text(REPORT, encoding="utf-8")
    assert main([str(doc), "--pack", str(PACKS / "prostate_pathology"), "--model", "m", "--show-prompt"]) == 0
    out = capsys.readouterr().out
    assert "## Fields to extract" in out and "<<<DOCUMENT" in out and out.count("## Example") == 3
    assert main([str(doc), "--pack", str(PACKS / "prostate_pathology"), "--model", "m", "--show-prompt",
                 "--exemplar-variant", "hedging"]) == 0
    assert capsys.readouterr().out.count("## Example") == 4
