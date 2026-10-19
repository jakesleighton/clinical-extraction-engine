import json

import httpx

from engine import ledger
from engine.backends.ollama import resolve_host
from engine.llm import complete
from tests.conftest import FakeOllama

SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"],
          "additionalProperties": False}


def test_complete_round_trip_and_ledger(monkeypatch, runs_dir):
    fake = FakeOllama({"x": 1}).install(monkeypatch)
    result = complete("say x", SCHEMA, "test-model", seed=7, run_id="run1", num_predict=50)

    assert result.parsed == {"x": 1} and result.error is None
    assert result.raw == '{"x": 1}' and result.model == "test-model"
    assert (result.tokens_in, result.tokens_out) == (1234, 567)
    assert result.latency_s >= 0 and len(result.call_id) == 12

    [req] = fake.requests
    assert req["model"] == "test-model" and req["prompt"] == "say x" and req["stream"] is False
    assert req["format"] == SCHEMA                              # the grammar goes with every call
    assert req["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 50, "seed": 7}
    assert req["think"] is False                                # thinking off unless asked (top-level field)

    [line] = list(ledger.read(runs_dir / "run1.jsonl"))
    assert line["kind"] == "llm_call" and line["call_id"] == result.call_id
    assert line["prompt_hash"] == ledger.sha256_text("say x")
    assert line["parse_ok"] is True and line["error"] is None and line["raw"] == '{"x": 1}'
    assert line["params"] == {"temperature": 0.0, "seed": 7, "num_predict": 50}
    assert line["meta"]["done_reason"] == "stop" and line["meta"]["think"] is False and "ts" in line


def test_think_level_passes_through_and_is_logged(monkeypatch, runs_dir):
    fake = FakeOllama({"x": 1}).install(monkeypatch)
    complete("p", SCHEMA, "gpt-oss:20b", run_id="run-think", think="low")
    assert fake.requests[0]["think"] == "low"
    [line] = list(ledger.read(runs_dir / "run-think.jsonl"))
    assert line["params"]["think"] == "low" and line["meta"]["think"] == "low"
    fake = FakeOllama({"x": 1}).install(monkeypatch)
    complete("p", SCHEMA, "m", run_id="run-nothink", think=None)
    assert "think" not in fake.requests[0]                      # None leaves the model's default alone


def test_think_rejected_by_model_is_retried_without_it(monkeypatch, runs_dir):
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": 'registry.ollama.ai/library/m:latest does not support thinking'})
        return httpx.Response(200, json={"model": body["model"], "response": '{"x": 2}', "done": True,
                                         "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 5})

    from engine.backends import ollama
    monkeypatch.setattr(ollama, "_make_client", lambda timeout: httpx.Client(transport=httpx.MockTransport(handler)))
    result = complete("p", SCHEMA, "m", run_id="run-retry")
    assert result.parsed == {"x": 2} and result.error is None
    assert len(seen) == 2 and "think" in seen[0] and "think" not in seen[1]
    [line] = list(ledger.read(runs_dir / "run-retry.jsonl"))
    assert line["meta"]["think"] is None and "does not support thinking" in line["meta"]["think_rejected"]
    assert line["meta"]["think_requested"] is False             # what was asked for stays on record
    assert "think" not in line["params"]                        # params = the caller's explicit arguments only


def test_invalid_json_is_an_error_not_an_exception(monkeypatch, runs_dir):
    FakeOllama(raw_text='{"x": ', done_reason="length").install(monkeypatch)
    result = complete("p", SCHEMA, "m", run_id="run2")
    assert result.parsed is None and "not valid JSON" in result.error and "num_predict" in result.error
    [line] = list(ledger.read(runs_dir / "run2.jsonl"))
    assert line["parse_ok"] is False and line["error"] == result.error


def test_non_object_json(monkeypatch, runs_dir):
    FakeOllama([1, 2]).install(monkeypatch)
    result = complete("p", SCHEMA, "m", run_id="run3")
    assert result.parsed is None and "not an object" in result.error


def test_server_unreachable(monkeypatch, runs_dir):
    FakeOllama(exc=httpx.ConnectError("connection refused")).install(monkeypatch)
    result = complete("p", SCHEMA, "m", run_id="run4")
    assert result.parsed is None and "cannot reach Ollama" in result.error and "ollama serve" in result.error


def test_model_not_found_hint(monkeypatch, runs_dir):
    FakeOllama(status_code=404).install(monkeypatch)
    result = complete("p", SCHEMA, "missing:latest", run_id="run5")
    assert "HTTP 404" in result.error and "ollama pull missing:latest" in result.error


def test_unknown_backend(runs_dir):
    result = complete("p", SCHEMA, "m", backend="nope", run_id="run6")
    assert "unknown backend" in result.error
    [line] = list(ledger.read(runs_dir / "run6.jsonl"))
    assert line["backend"] == "nope"


def test_default_run_id_is_shared(monkeypatch, runs_dir):
    FakeOllama({"x": 1}).install(monkeypatch)
    a = complete("p", SCHEMA, "m")
    b = complete("p", SCHEMA, "m")
    assert a.call_id != b.call_id
    [path] = list(runs_dir.glob("*.jsonl"))
    assert len(list(ledger.read(path))) == 2


def test_resolve_host(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert resolve_host() == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    assert resolve_host() == "http://0.0.0.0:11434"
    assert resolve_host("https://gpu-node:8080/") == "https://gpu-node:8080"
