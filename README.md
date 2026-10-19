# clinical-extraction-engine

A config-driven engine that turns clinical free text (pathology reports,
transfusion reaction narratives) into structured, validated, analysis-ready
data, with every value tied to the characters in the source that support it.

Built and benchmarked on public data (TCGA pathology reports). Adapting it to
a new document type is a configuration folder (a **task pack**), not new
code.

**Status: Phase 0 complete (scaffold, staged data, one smoke run).** Nothing
here has produced a number yet. Phase 1 adds the first per-field F1 results
on TCGA-PRAD.

## Why it looks the way it does

- **Schema-first.** A pack declares the fields once. From that one
  declaration the engine generates the prompt, the JSON schema the model
  must fill, the validator, and the eval columns. Nothing can drift apart
  because nothing is written twice.
- **Constrained decoding.** The JSON schema is compiled to a grammar on the
  model server, which masks out every token that would break the schema. The
  model cannot invent a key, put a string where an integer belongs, or
  return an enum value we did not list. That guarantees *shape*. Whether the
  value is *true* is the job of the validator and the eval harness.
- **Three states, not two.** Every field answers `value`,
  `explicitly_absent`, or `not_mentioned`. "Margins negative" and a report
  that never mentions margins are different facts; a two-state design
  conflates them.
- **Evidence spans.** Every value carries verbatim quotes, which the engine
  locates in the source text as character offsets. One requirement, three
  payoffs: the verification loop, the provenance record, and the future
  reviewer UI.
- **Boring Python.** Orchestration is the standard library plus `pydantic`,
  `pyyaml`, `pandas` and `httpx`. No agent frameworks: every agentic pattern
  here is loops, conditionals and HTTP, visible and testable. A JSONL run
  ledger replaces framework observability and doubles as provenance.
- **One door for the model.** `engine.llm.complete(prompt, schema, model)` is
  the only function that talks to a model server. Backends (Ollama now;
  vLLM, commercial endpoints, institutional platforms later) are swappable
  modules behind it.
- **Everything behind a switch.** Each agentic feature (Phase 2) is an
  ablation switch in the pack, so its contribution is a measured delta, not
  a guess.

The vocabulary is deliberate: this engine is *developed and validated*. No
model is trained or fine-tuned anywhere in this repository.

## Architecture

```
                 packs/<name>/pack.yaml  (+ base)
                          |
                 engine/packloader.py  ->  typed Pack
                          |
        +-----------------+------------------+
        |                                    |
 engine/prompt_builder.py            engine/schema.py
 (instructions + field catalog       (pydantic record model ->
  + exemplars + document)             JSON schema for the grammar)
        |                                    |
        +---------------> engine/llm.complete() <------ engine/backends/ollama.py
                                   |                     (the only HTTP client)
                                   v
                          engine/validate.py
                 shape -> per-field semantics + evidence offsets
                       -> cross-field rules -> derived fields
                                   |
                                   v
                  canonical record  (engine/extract.py)
                                   |
             +---------------------+---------------------+
             v                     v                     v
     evalharness/            adapters/jsonl        adapters/sql_tables,
   per-field P/R/F1 by tier                        feature_matrix (Aim 3)

 every complete() call -> runs/<run_id>.jsonl   (engine/ledger.py)
```

## Quickstart

One command does all of the below on a Mac with miniforge and Homebrew
(`bash scripts/phase0_setup.sh`; add `--no-ollama` to stop before the model
server). The steps, by hand:

```bash
# miniforge / conda (pins Python 3.12 to match the containers):
mamba env create -f environment.yml && conda activate cee
# or plain venv:
#   python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

pytest                                   # pack loader round-trips, validator, metrics, mocked Ollama

# stage the public data (needs network; ~11 MB download, then GDC API queries)
python -m scripts.fetch_tcga_reports     # TCGA-Reports corpus -> data/tcga_prad/reports/<case_id>.txt
python -m scripts.join_gold_labels       # GDC clinical -> packs/prostate_pathology/labels/tcga_prad_gold.csv
python -m scripts.working_set random --n 20            # initial smoke-test set: a seeded draw ...
python -m scripts.working_set pick TCGA-XX-1234 ...    # ... or exactly the cases you choose (add/remove/show too)
python -m scripts.weirdness_screen                     # rank reports by cheap trouble heuristics -> data/weird_case_candidates.md
python -m engine.packloader packs/prostate_pathology   # show the merged pack (--yaml, --schema)

# one real end-to-end run (Ollama must be running: `ollama serve`, `ollama pull qwen2.5:7b`)
python -m engine.extract data/tcga_prad/reports/<case_id>.txt \
    --pack packs/prostate_pathology --model qwen2.5:7b

# the Phase 1 loop: the working set, scored (--cases ID ... runs specific cases instead)
python -m evalharness.runner --pack packs/prostate_pathology --model qwen2.5:7b \
    --reports data/tcga_prad/reports --ids data/tcga_prad/working_set.txt
```

`OLLAMA_HOST` points the Ollama backend somewhere other than
`localhost:11434`; `CEE_RUNS_DIR` moves the ledger directory.

## How a task pack works

```
packs/
  base/pack.yaml                  document-agnostic machinery: three-state status,
                                  hedging levels, evidence envelope, negation/amendment
                                  instructions, field defaults, ablation switches
  prostate_pathology/
    pack.yaml                     extends: base. Fields, enums, cross-field rules,
                                  derived fields, precedence, eval spec
    exemplars/                    (document, expected JSON) pairs shown in the prompt (none yet; Phase 1)
    labels/tcga_prad_gold.csv     gold labels for the scored fields
```

A field is a few lines of YAML:

```yaml
- id: gleason_primary
  type: integer                  # integer | number | string | enum | date | measurement
  allowed: [1, 2, 3, 4, 5]
  tier: standard                 # deterministic | standard | hard  (evals report by tier)
  instructions: Primary (most prevalent) Gleason pattern ...
```

For every field the model returns one object, generated in this order so it
finds the text before it commits to a value:

```json
{"evidence": ["Gleason score 3+4=7"], "status": "value", "value": 3,
 "hedging": "definite", "conflict": false}
```

The engine then adds what the model must never do: character offsets for
each quote, cross-field checks (`gleason_primary + gleason_secondary ==
gleason_total`), and **derived fields** computed deterministically from
extracted values (Grade Group from the Gleason patterns), each stamped with
its provenance (`derivation: stated | computed_from_gleason`).

Rules are plain expressions evaluated by a small whitelisting interpreter
(`engine/rules.py`), never `eval()`: packs are data, and public packs will be
written by other people.

## Repository layout

```
packs/            task packs (base + prostate_pathology)
engine/           packloader, schema, prompt_builder, llm (+ backends/), validate, extract, ledger, rules
evalharness/      metrics (per-field P/R/F1 by tier), runner (end-to-end eval of a pack)
adapters/         jsonl (real); sql_tables, feature_matrix (Aim 3 stubs)
scripts/          fetch_tcga_reports, join_gold_labels (data staging), working_set (which cases to run),
                  weirdness_screen (ranks reports by cheap trouble heuristics, no model calls)
containers/       Dockerfile (local), singularity.def (HPC batch)
tests/            pytest suite; the model server is mocked
data/, runs/      gitignored: staged reports and run ledgers never enter the repo
                  (exceptions: data/weird_cases.md and data/tcga_prad/working_set.txt, public case ids)
notes/            gitignored: local working notes and project-internal documents
```

## Phases

| Phase | What lands |
|---|---|
| 0 | pack format, repo skeleton, `complete()` + Ollama, TCGA-PRAD staged (446 reports, gold labels for 423), a 24-case working set: a seeded draw of 20 plus five hand-picked hard cases (`data/weird_cases.md`) |
| 1 | extraction core on the prostate pack; first per-field F1 on the scored core (Gleason primary/secondary/total, Grade Group, pT, pN) |
| 2 | agentic features one at a time, each with its ablation number: evidence verification, validator-feedback retry, self-consistency confidence, verifier agent, case reconciliation |
| 3 | second public pack (structurally different) as the generality proof |
| 4 | commercial-model benchmark sweep on public data; OCR path on scanned PDFs |
| 5 | transfusion pack and divisional validation with clinical reviewers |

## Data

TCGA-Reports (Kefeli & Tatonetti, *Patterns* 2024): 9,523 OCR-cleaned TCGA
pathology reports; the prostate subset (TCGA-PRAD) is the development
corpus. Gold labels come from the GDC clinical tables. All public, no PHI.
Nothing under `data/` or `runs/` is committed.

## License

MIT, see `LICENSE`.

## Acknowledgment

Supported by the DPLM Boot Walk Award (The University of Texas MD Anderson
Cancer Center, 2026–2027). Author: Jake Leighton.

## Prior art

arXiv:2511.10658 benchmarked 15 open-weight LLMs on pathology/radiology
structured extraction and found that task specification and reference
annotation quality matter more than model size. This project builds the
reusable task-specification and validation layer that result points at.
