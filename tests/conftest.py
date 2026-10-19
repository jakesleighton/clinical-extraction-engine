"""Shared fixtures: the real packs, a synthetic report, a mocked Ollama."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))           # run the tests without `pip install -e .`

from engine.packloader import Pack, load_pack  # noqa: E402
from engine.backends import ollama             # noqa: E402

PACKS = ROOT / "packs"

# A small synthetic prostatectomy report. Public-style text, no real patient.
REPORT = """SURGICAL PATHOLOGY REPORT
Collected: 10/12/2010    Reported: 10/15/2010
SPECIMEN: Prostate and seminal vesicles, radical prostatectomy; bilateral pelvic lymph nodes.
FINAL DIAGNOSIS:
Prostatic adenocarcinoma, acinar type, Gleason score 3+4=7,
involving both lobes, approximately 20% of the gland.
Extraprostatic extension: Not identified.
Seminal vesicle invasion: Negative.
Surgical margins: Negative for carcinoma.
Perineural invasion: Present.
Lymphovascular invasion: Not identified.
Lymph nodes: Twelve lymph nodes, negative for metastatic carcinoma (0/12).
Pathologic stage: pT2c pN0.
"""


def envelope(status: str = "not_mentioned", value: Any = None, evidence: tuple[str, ...] | list[str] = (),
             hedging: str = "definite", conflict: bool = False, unit: str | None = None,
             measurement: bool = False) -> dict[str, Any]:
    env: dict[str, Any] = {"evidence": list(evidence), "status": status, "value": value}
    if measurement:
        env["unit"] = unit
    env.update({"hedging": hedging, "conflict": conflict})
    return env


def full_record(pack: Pack, overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """A complete model output: every field not_mentioned unless overridden."""
    overrides = overrides or {}
    return {f.id: overrides.get(f.id, envelope(measurement=(f.type == "measurement"))) for f in pack.fields}


def model_output_for_report(pack: Pack) -> dict[str, Any]:
    """What a good model should return for REPORT (grade_group left unstated)."""
    v = "value"
    absent = "explicitly_absent"
    return full_record(pack, {
        "specimen_procedure": envelope(v, "radical_prostatectomy", ["radical prostatectomy"]),
        "tumor_laterality": envelope(v, "bilateral", ["involving both lobes"]),
        "report_date": envelope(v, "10/15/2010", ["Reported: 10/15/2010"]),
        "collection_date": envelope(v, "10/12/2010", ["Collected: 10/12/2010"]),
        "histologic_type": envelope(v, "acinar_adenocarcinoma", ["Prostatic adenocarcinoma, acinar type"]),
        "gleason_primary": envelope(v, 3, ["Gleason score 3+4=7"]),
        "gleason_secondary": envelope(v, 4, ["Gleason score 3+4=7"]),
        "gleason_total": envelope(v, 7, ["Gleason score 3+4=7"]),
        "tumor_percent_of_gland": envelope(v, 20, ["approximately 20% of the gland"]),
        "extraprostatic_extension": envelope(absent, None, ["Extraprostatic extension: Not identified."]),
        "seminal_vesicle_invasion": envelope(absent, None, ["Seminal vesicle invasion: Negative."]),
        "margin_positive": envelope(absent, None, ["Surgical margins: Negative for carcinoma."]),
        "lymphovascular_invasion": envelope(absent, None, ["Lymphovascular invasion: Not identified."]),
        "perineural_invasion": envelope(v, "present", ["Perineural invasion: Present."]),
        "nodes_examined": envelope(v, 12, ["Twelve lymph nodes", "(0/12)"]),
        "nodes_positive": envelope(v, 0, ["(0/12)"]),
        "pT": envelope(v, "pT2c", ["Pathologic stage: pT2c pN0."]),
        "pN": envelope(v, "pN0", ["Pathologic stage: pT2c pN0."]),
    })


@pytest.fixture(scope="session")
def base() -> Pack:
    return load_pack(PACKS / "base")


@pytest.fixture(scope="session")
def prostate() -> Pack:
    return load_pack(PACKS / "prostate_pathology")


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "runs"
    monkeypatch.setenv("CEE_RUNS_DIR", str(d))
    return d


class FakeOllama:
    """Stands in for the Ollama server: records requests, returns canned output."""

    def __init__(self, response_obj: Any = None, *, status_code: int = 200, raw_text: str | None = None,
                 done_reason: str = "stop", exc: Exception | None = None):
        self.response_obj = response_obj
        self.status_code = status_code
        self.raw_text = raw_text
        self.done_reason = done_reason
        self.exc = exc
        self.requests: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.exc is not None:
            raise self.exc
        self.requests.append(json.loads(request.content))
        if self.status_code != 200:
            return httpx.Response(self.status_code, json={"error": f"model '{self.requests[-1]['model']}' not found"})
        text = self.raw_text if self.raw_text is not None else json.dumps(self.response_obj)
        return httpx.Response(200, json={"model": self.requests[-1]["model"], "response": text, "done": True,
                                         "done_reason": self.done_reason, "prompt_eval_count": 1234,
                                         "eval_count": 567, "total_duration": 1_000_000})

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeOllama":
        monkeypatch.setattr(ollama, "_make_client",
                            lambda timeout: httpx.Client(transport=httpx.MockTransport(self.handler)))
        return self


@pytest.fixture
def fake_ollama(monkeypatch: pytest.MonkeyPatch, prostate: Pack) -> FakeOllama:
    return FakeOllama(model_output_for_report(prostate)).install(monkeypatch)
