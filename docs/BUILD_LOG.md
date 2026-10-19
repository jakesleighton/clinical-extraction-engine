# Build log

One entry per step, oldest first. Each entry says what changed, why, and how
it was checked, so that someone can rebuild this project from scratch in the
same order and understand each decision as they go. Code lives in git; this
file carries the reasoning that a commit message cannot. The README explains
the design as it stands today; this log explains how it got there.

Conventions:

- Every change to the engine, the packs, the scripts or the tests gets an
  entry here in the same commit. The commit message names the entry
  (for example `build log 1.1`).
- A baseline eval is frozen: no pack, exemplar or code change lands between
  the first and the last run of a baseline, so every number in a report
  traces to one commit hash. Scoring is derived from the saved predictions
  and ledger and may be recomputed later; what the models produced may not.
- Run ids are explicit (`p1-baseline-<model>`), never auto-generated, when
  the run will be cited.
- Working notes (session walkthroughs, the handoff document, build briefs)
  stay in the local `notes/` folder, which git ignores. Decisions made there
  are recorded here once they are settled.

## Rebuild from scratch

1. Clone the repository and create the environment
   (`mamba env create -f environment.yml && conda activate cee`, or a venv
   with `pip install -e ".[dev]"`). Run `pytest` first: everything in the
   suite runs against a mocked model server, so a green suite proves the
   code before any download.
2. Stage the public data: `python -m scripts.fetch_tcga_reports` (TCGA-Reports
   corpus, prostate subset) then `python -m scripts.join_gold_labels` (GDC
   clinical tables to the gold CSV). Expect 446 staged reports, a 500-row
   gold CSV with 423 complete rows, 374 cases with both.
3. Recreate the working set: `data/tcga_prad/working_set.txt` is tracked in
   git (24 ids: a seed-0 draw of 20 plus five hand-picked hard cases, one of
   which was already in the draw), so nothing to do unless you want a fresh
   draw (`python -m scripts.working_set random --n 20`).
4. Pull models with Ollama. On Apple Silicon pull the explicit GGUF
   quantisation tags (for example `qwen3.5:9b-q4_K_M`), not the bare tags:
   see entry 1.2.
5. Run one extraction (`python -m engine.extract <report> --pack
   packs/prostate_pathology --model <model>`), then the eval
   (`python -m evalharness.runner ...`, see the README quickstart).

## Phase 0

### 0.1 Scaffold (2026-10-16; committed as 9a326fb on 2026-10-19)

What: the repository skeleton. Two task packs (`packs/base/pack.yaml`, the
document-agnostic machinery; `packs/prostate_pathology/pack.yaml`, 23 fields,
four cross-field rules, one derived field, an eval spec). The engine:
`packloader` (YAML to typed objects, `extends` merge, loud errors), `rules`
(a whitelisting expression interpreter, no `eval`), `schema` (pack to
pydantic model to JSON schema, one source of truth), `prompt_builder`,
`llm.complete()` (the one door to a model server, with a JSONL ledger line
per call), `backends/ollama` (the only module that imports an HTTP client),
`validate` (shape, per-field semantics, evidence offsets, cross-field rules,
derived fields), `extract` (per-document pipeline and CLI), `ledger`. The
eval harness (`metrics`: per-field precision, recall and F1 by tier;
`runner`: gold plus reports to predictions, metrics and a report). Staging
scripts (`fetch_tcga_reports`, `join_gold_labels`, `working_set`), output
adapters (`jsonl` real, two stubs), container recipes, and the test suite.

Why: schema-first so the prompt, the grammar, the parser and the eval
columns come from one declaration and cannot drift; constrained decoding so
the model cannot break the shape, with truth left to the validator and the
eval; evidence quotes written before values so each value is anchored to
real text, with offsets computed by the engine because models copy well and
count badly; three states so "negative" and "not mentioned" stay distinct;
derived fields as pack configuration (Grade Group from the Gleason patterns,
with `derivation: stated | computed_from_gleason`) so the engine has no
field-specific code.

Checked: 106 tests against a mocked model server.

### 0.2 First end-to-end run (2026-10-16)

What: `scripts/phase0_setup.sh` ran the whole chain on a Mac with miniforge
and Homebrew: tests, data staging, a seed-0 working set of 20, Ollama with
`qwen2.5:7b`, one real extraction of TCGA-CH-5753 (55 s, 3 validator flags).

Found: the model invented a Grade Group of 4 for Gleason 4+5 and labelled it
"stated"; the cross-field rule `grade_group_matches_gleason` caught it. Two
evidence quotes were not in the source (one fabricated, one with a dropped
word). Dates were correctly `not_mentioned` but with the whole report pasted
as evidence, which cost most of the output tokens.

Changed in response (wording and robustness only): the base pack now
requires an empty evidence list when status is `not_mentioned`; the
prostate pack says only a literally printed Grade Group counts, never one
computed from the patterns; the setup script tolerates a non-Homebrew
Ollama. Decision: evidence matching stays exact plus whitespace and case
tolerance; a dropped word is a miss, and the remedy is the Phase 2 retry
loop asking for an exact quote, not a fuzzier matcher.

Checked: 114 tests; the smoke run repeated with the new wording.

### 0.3 Gold-anchored Grade Group subsets (2026-10-16)

What: `join_gold_labels` writes `grade_group_printed` (a regex over each
report for "Grade Group N", "ISUP N", "GG N"). The prostate pack's derived
field names it as `stated_column`. `compute_metrics` scores the `stated`
subset as the gold-printed cases and the derived subset as the rest, and
adds a provenance note counting how often the model claims "stated" where
nothing is printed.

Why: the smoke run showed the model will claim "stated" falsely, so eval
subsets defined by the model's own claim would be circular. Audit: none of
the 446 TCGA-PRAD reports prints a Grade Group (they predate the 2014 ISUP
system), so on this corpus the `stated` subset is empty and every "stated"
Grade Group is an invention, which makes it a countable hallucination class.

Checked: 116 tests; the gold CSV regenerated offline from the cached GDC
response matches the previous columns row for row.

### 0.4 Weird-case screen fix and the 24-case working set (2026-10-19)

What: `scripts/weirdness_screen.py` ranks reports by cheap heuristics
(addenda, hedging density, OCR artefacts, several Gleason scores, text
versus gold conflicts). Bug fixed: the pT and pN values printed in the text
were never parsed, and "NO" or "pNO" with a letter O (an OCR artefact in
139 or more reports) must read as N0. A new `amended_value` flag marks
addendum or correction language next to more than one pT, pN or final
Gleason value, discounted where that language is a site template.

Five hard cases chosen from the ranking and added to the working set with
`python -m scripts.working_set add`: TCGA-CH-5741 (a follow-up line
contradicts the described nodal metastasis; gold right), TCGA-HC-7749 (an
addendum changes pT2c to pT3 and the margin; gold follows the addendum),
TCGA-J4-A67T (a correction changes pT2c to pT3a; gold is stale),
TCGA-V1-A9OH (Gleason 5+3 versus 5+4 in the same report; gold matches
neither), TCGA-CH-5753 (the hedged smoke case, already in the draw).
Working set: 24 ids. Descriptions in `data/weird_cases.md`.

Open design question recorded: the base pack's amendment rule ("the latest
statement wins") is wrong for CH-5741 and right for HC-7749 and J4-A67T.

Checked: 127 tests.

### 0.5 Close-out and public push (2026-10-19)

What: MIT `LICENSE`; README status, phases table and layout; `.gitignore`
keeps `data/` (except the working set and weird-case notes), `runs/` and
`notes/` out of the repository; the Dockerfile recipe replicated in a fresh
environment with the same file set (127 tests). Initial commit 9a326fb
pushed to the public repository; `docker build` and the in-image test run
confirmed on the Mac.

Decision: session notes and internal documents stay local. The README tells
the public story; this log carries the build history.

## Phase 1

### 1.1 Exemplars, hold-out, thinking off, context default (2026-10-19; commit 8839fca)

What: three exemplar reports wired into the prostate pack (TCGA-EJ-7315,
TCGA-HC-A6AN, TCGA-ZG-A8QW), each as a (document, expected JSON) pair under
`packs/prostate_pathology/exemplars/` and rendered as "## Example" blocks
before the document. A fourth pair (TCGA-V1-A9Z7, a genuinely hedged margin)
is staged in the same folder as a commented-out entry: the "+hedging
exemplar" ablation arm, to be enabled only if the baseline shows hedging
failures. Pack version 1 to 2.

How the exemplars were chosen: a reproducible filter over the 374 cases
with a report and complete gold, using the weirdness screen inversely (clean,
no addenda, no conflicts), then one report from each of the three largest
report templates and three different stage pictures. Each expected JSON was
drafted by hand and validated by the engine itself: every quote found
verbatim, zero validator issues, gold values matched, Grade Group derived.
Two judgment calls became rules: a lymphovascular-invasion entry that is
"not identified" in the findings but "must have been present" in a comment
is `explicitly_absent` with `conflict: true` and both quotes (flag, do not
silently choose); when a report never names the procedure, classify it from
the specimen it describes and quote that description (now in the field's
instructions, not just the exemplar).

Hold-out: `Pack.exemplar_ids`; the runner drops those ids from every run,
even when requested explicitly, and says so in the report; `working_set
random` never draws them. Scoring an exemplar would be leakage.

Backend: Ollama's `think` field is sent as `false` by default because the
current open-weight families think unless told not to and the reasoning
would be wasted inside a grammar-constrained JSON reply; `--think low` (or
medium, high) for gpt-oss, which cannot switch thinking off; a model that
rejects the field with HTTP 400 is retried once without it and the
rejection is logged in the ledger meta. Context: with exemplars in the
prompt the prompt is about 34k characters (9 to 12k tokens), so `num_ctx`
defaults to 16384 whenever the pack has exemplars (the old 8192 default
would have truncated from the front, dropping the instructions first);
`--num-ctx` still overrides.

Checked: 146 tests, 0 skipped. Test suite run on the Mac before the commit.

### 1.2 Baseline eval on the working set (2026-10-19)

What: five models, one run each, on the 24-case working set, identical
settings (three exemplars, `num_ctx` 16384, thinking off, temperature 0,
seed 0, gpt-oss at `--think low`), run ids `p1-baseline-<model>`. Settings
are "deterministic-intent": recorded on every ledger line, not a promise of
bit-identical replays on GPU inference. Nothing in the repository changed
between the first and the last run, so every number traces to commit 8839fca.

Found during the runs: on Apple Silicon the bare tags of the newer model
families (qwen3.5, gemma4) pull Ollama's MLX builds, and the Homebrew build
of Ollama returns HTTP 501 "structured output is unavailable" for them
because the formula does not build the xgrammar library that constrained
decoding on the MLX engine needs. GGUF models run on the llama.cpp engine
with its own grammar sampler and are unaffected. Resolution: pull the
explicit GGUF quantisations (`qwen3.5:9b-q4_K_M`, `gemma4:12b-it-q4_K_M`,
`qwen3.5:27b-q4_K_M`) and run every model on the same engine; the failed
MLX run folders are kept under their original run ids as a record. The
lineup that ran: qwen2.5:7b, qwen3.5:9b-q4_K_M, gemma4:12b-it-q4_K_M,
gpt-oss:20b, qwen3.5:27b-q4_K_M.

Results (exact-match micro-F1 on the six gold fields, all 24 cases /
random-20 / weird-5; median seconds per case): qwen2.5:7b 90 / 92 / 82,
31 s; qwen3.5:9b 88 / 92 / 75, 39 s; gemma4:12b 85 / 88 / 75, 32 s;
gpt-oss:20b 87 / 90 / 75, 38 s; qwen3.5:27b 87 / 90 / 75, 159 s. The
standard tier (three Gleason fields and Grade Group) is 95 for every model
except gemma4 (93); the spread between models is entirely in pT and pN.
Reading each miss against the report text: six of the 24 cases carry a gold
value the report does not support (a biopsy Gleason score in the gold, a
stale stage before a correction, a pN0 where both synopses print pN1, and
two stages finer than the printed text); five cases from one site print no
stage line at all, where qwen2.5:7b infers a stage (wrongly three times) and
the newer models answer `not_mentioned`; one amended report is handled by
the base pack's amendment rule exactly as written, against the gold. After
that filter the genuine scored-field errors are qwen2.5:7b 5, gemma4 2 (both
flagged by cross-field rules or the conflict flag), the other three 0.

Validator behaviour: no run produced a bad record. The Grade Group
provenance claim ("stated" where nothing is printed) seen in the Phase 0
smoke run is gone on that case for the same model (the engine derived the
value, provenance `computed_from_gleason`) and occurs once in 120 model-cases
overall, all by qwen2.5:7b. Three rule failures across all runs, each a real
catch (a 4/5-notation Gleason misread, twice a follow-up line contradicting
the described nodal metastasis). Evidence flags: 10 to 86 per run, and most
are one corpus artefact: the OCR inserted a period at every original line
break and the models quote the sentence without it, which the matcher's
whitespace and case tolerance does not cover; fabricated or paraphrased
quotes were 0 to 5 per run. The newer models put `conflict: true` on exactly
the cases the weirdness screen had picked; hedging is under-used by all
(two models never hedged a value; three of five called the deliberately
hedged margin definite).

Cost: 10.8k to 11.4k prompt tokens per case (the exemplars), 1.5k to 1.7k
output tokens; wall 13 to 16 minutes per run except qwen3.5:27b at 59
minutes for the same scores as gpt-oss:20b.

Decisions put to the owner for the iterate step, none applied here: which
models go forward (recommended gemma4, qwen3.5:9b, gpt-oss:20b); a derived
stage with provenance for reports that print none (the Grade Group pattern,
`computed_from_findings` / `computed_from_nodes`) versus "as written"; keep
the amendment rule and record the one counter-example; a lenient
printed-granularity column for pT; a tracked gold audit list; the staged
hedging exemplar as the first ablation arm; punctuation tolerance in the
evidence matcher (touches a settled decision); folding the outside analysis
into the runner's report.

Checked: the analysis is computed from the saved predictions and ledgers by
two scripts kept outside the repository until the baseline is closed (they
move into `evalharness` as part of the next entry); the long-form report
lives in the local notes folder. Test suite unchanged at 146.
