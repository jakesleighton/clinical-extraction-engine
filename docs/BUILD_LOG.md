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

### 1.3 Evidence matcher: punctuation and line breaks (2026-10-22)

What: the quote locator in `engine/validate.py` gained a third pass. A
quote is looked for exactly, then with whitespace and case folded (the
Phase 0 behaviour), then with punctuation and line-break characters ignored
as well; a `.`, `,` or `-` between two digits is kept, so `2.0` never
matches `20` and `8,6` never matches `86`. Offsets always land on the real
span of the source text. Each evidence entry now records which pass found
it (`match: exact | whitespace | punctuation`, null when not found), so a
report can say how clean a model's quoting is rather than only whether it
failed.

Why: the baseline (entry 1.2) traced most evidence flags to one corpus
artefact, a period inserted at every original line break by the OCR, which
models quote without. That is a measurement error, not a model error. The
settled fuzzy-matching policy (whitespace and case only; a dropped word is a
miss; the retry loop asks for exact quotes) is kept and narrowly amended:
punctuation tolerance only. Dropped or inserted words still miss, and are
what the Phase 2 retry loop is for. Recorded in the decision log.

Checked: eleven locator cases including the digit-separator rules, a
stitched quote and a dropped word (both still misses). Re-scoring the five
saved baseline runs with the new locator (entry 1.4 explains how) moved
qwen3.5:9b from 86 evidence flags to 21, gpt-oss:20b from 29 to 16,
qwen3.5:27b from 42 to 7, qwen2.5:7b from 42 to 21 and gemma4 from 11 to
8; what remains is stitched, one-word-off, partial or invented quotes, the
retry loop's job.

### 1.4 Reports that carry their own analysis, and offline re-scoring (2026-10-22)

What: `evalharness/report.py` computes everything the baseline write-up had
to compute outside the repository, from the saved records and the ledger:
the group split (a tracked `case_id,group` file, `data/tcga_prad/case_groups.csv`,
holds the random-20 and weird-5 memberships; one case is in both), scores
without the audited gold cases, validator flag counts by code with the
fields they touched, a per-field table of status, derived, hedged and
conflict counts (the only view of the hard tier, which has no gold), which
locator pass found each quote, a class for every quote no pass found
(normalized, stitched, one_token, partial, other), and a cost table from the
ledger (tokens, latency, wall, think and done_reason, context pressure). The
runner takes `--groups` and `--gold-notes` and writes all of it into
`report.md` and `metrics.json`.

`evalharness/rescore.py` re-validates a finished run from its ledger with the
current pack and engine, without a model call, into a new run folder marked
"RE-SCORED". The ledger keeps every reply verbatim, so any change to the
scoring machinery can be shown on finished runs before model time is spent.
It is also how the numbers in entries 1.3 and 1.5 were produced.

Why: the step-2 analysis scripts lived outside the repository by design (a
baseline is frozen); once the baseline closed, the analysis belongs in the
tool so the full sweep needs nothing else. A report that can be rebuilt
from the ledger is also the provenance claim made concrete.

Checked: unit tests for each table, an end-to-end runner test with groups,
gold notes and the opt-in exemplar arm on the mocked server, and a rescore
test that reproduces a mocked run's scores from its ledger.

### 1.5 Derived pT and pN with provenance, lenient stage column, gold audit list (2026-10-22)

What: the Grade Group mechanism now covers the stage. The prostate pack
(version 3) declares `pT` as derived from the extent findings of a radical
prostatectomy (seminal vesicle invasion gives pT3b, extraprostatic
extension pT3a, neither is organ-confined: pT2c when bilateral, otherwise
pT2) and `pN` from the node counts, with provenance labels
`computed_from_findings` and `computed_from_nodes`; two functions in the
whitelisted rule library, no stage logic in engine code. The field
instructions now say that only a printed stage counts and that the engine
derives the rest. Three pack-level mechanisms were added to make that
expressible: input modes (`required`, `finding`: an explicitly absent
finding is information and is passed as absent while silence blocks the
derivation, `optional`), `when: check` (a printed value stands; a
disagreement with the findings is flagged as `derivation_disagrees`, a
coarser printed form such as pT3 against pT3a counts as agreement), and
`on_none: skip` (a rule that cannot pin a value, a biopsy or no nodes, is
not a failure). Inherited evidence carries the negated findings' quotes.

The gold CSV gained `pT_printed` and `pN_printed` (regex over the report
text, letter O read as zero; 89 of 446 staged reports print no T category,
117 no N category, almost all in two report templates), and the eval
scores each stage field on `all`, `stated` and the derived subset,
gold-anchored as for Grade Group, with the false-provenance note: a model
that writes a stage where none is printed is now visible as a false
"stated" claim.

`eval.lenient: {pT: prefix}` adds one labelled row per lenient field (a
printed pT3 scores against gold pT3a, either way round) and "(lenient)"
aggregates; exact match stays the headline.

`data/gold_notes.md` (tracked) lists the seven working-set cases whose gold
the report text does not support, with the quoted text, each quote checked
by the locator: three where the text says otherwise, one stale correction,
two where the gold is finer than the text, and the amendment-rule case
(TCGA-CH-5741), where the gold is right and the base pack's rule, kept as
written, gives the other value. The runner scores with and without these
cases as whole cases.

Why: entry 1.2's decisions 2, 4 and 5. The "as written" policy forfeits
about a fifth of the corpus on two fields; letting the model infer a stage
is the failure mode qwen2.5:7b demonstrated. Deriving with provenance keeps
the model honest and the number readable.

Checked: 212 tests. Re-scoring the saved baseline replies with the new
machinery (model outputs unchanged, so a measurement of the scoring, not of
the models): overall micro-F1 qwen3.5:9b 88 to 91 (92 lenient), gpt-oss:20b
87 to 90 (92), gemma4 85 to 87 (89), qwen3.5:27b 87 to 90 (92), qwen2.5:7b
unchanged at 90; on the five no-stage KK reports the engine derived pT for
5 of 5 cases from qwen3.5:9b's and gpt-oss's findings and 3 of 5 from
gemma4's, all correct; without the seven audited cases the three carried
models score 95 to 99. `derivation_disagrees` fired on the amended report
(the follow-up's pT3a/pN0 against the described pT3b/pN1 findings) and on
the report with two different printed stages, which is the flag doing its
job. The amendment-rule question is closed as "keep, record the
counter-example" (decision 3).

### 1.6 Hedging exemplar as an opt-in arm; context sized from the prompt (2026-10-22)

What: exemplars can carry a `variant` name. The prompt always shows the
untagged exemplars; `--exemplar-variant hedging` (runner and single-document
CLI) adds the tagged ones, and every exemplar case, tagged or not, is held
out of every eval, so an arm and its baseline score the same cases. The
staged fourth exemplar (TCGA-V1-A9Z7, a genuinely hedged margin) is now
wired that way. The ledger's validation line records the exemplar cases in
the prompt and the variants that were on.

The context window is now sized from the prompt instead of a constant
(about a token per three characters plus 4096 of headroom, rounded up to a
2048 step, never below the backend's 8192): the three-exemplar prompt still
gets 16384, so the re-baseline is comparable with entry 1.2, and the
four-exemplar prompt gets 20480, which the old constant would have
truncated from the front. `--num-ctx` still overrides, and the value used
is on every ledger line.

Why: entry 1.2's decision 6: two models never hedged and three of five
called the deliberately hedged margin definite, the condition the staged
exemplar was reserved for. An ablation arm must differ from its baseline in
exactly one thing, hence the variant switch rather than a second pack.

Checked: prompt tests count three and four "## Example" blocks, the
hold-out test covers the tagged case, the exemplar test validates all four
pairs against the current pack (every stage derivation agrees with the
printed stage). Test suite: 212, 0 skipped.


### 1.7 Re-baseline and hedging arm on the working set (2026-10-22; code at commit 8d7169f)

What: six runs on the 24-case working set with the entry 1.3 to 1.6
machinery (pack v3). Arm A, run ids `p1-rebase-<model>`: the three carried
models (gemma4:12b-it-q4_K_M, qwen3.5:9b-q4_K_M, gpt-oss:20b at `--think
low`) under the pack v3 prompt, with `--groups` and `--gold-notes`, context
window sized automatically (16384). Arm B, run ids `p1-hedge-<model>`: the
same with `--exemplar-variant hedging` (four exemplars, window 20480). Same
settings as entry 1.2 otherwise. Nothing in the repository changed between
the first and the last run; the step 3 commit was made after the first run
had started, with the working tree verified unchanged, so every number
traces to commit 8d7169f. Every run: 24 of 24 records, no errors, every call
ended with `stop`.

Results (exact-match micro-F1, all 24 / random-20 / weird-5; the same
replies re-scored with the new machinery in brackets, entry 1.5, for the
prompt-only comparison): gemma4:12b 87.5 / 91.1 / 75.0 (86.8); qwen3.5:9b
90.1 / 94.2 / 75.0 (90.8); gpt-oss:20b 90.8 / 95.0 / 75.0 (90.5). Without
the seven audited gold cases: 95.5 / 99.0 / 99.0. Standard tier unchanged
since entry 1.2 (92.7 / 94.8 / 94.8). The derived stage subsets are the
strongest part of the stage score (pT and pN derived: 100 for two models,
89 and 80 for gemma4, every derived value correct); no model missed a
printed stage; `derivation_disagrees` fired on the four cases where the
printed stage contradicts the findings or a second printed stage, as
designed. The v3 instructions removed the "stage where none is printed" pT
claims and halved the pN ones. Arm B moved the exact headline by +0.4 / +0.7
/ −0.7 (lenient +0.4 / 0 / −1.5): it produced the intended hedged margin for
qwen3.5:9b only (gemma4 already hedged it, gpt-oss still calls it definite),
pulled two models onto the synoptic stage line on the amended report (the
amendment rule lost), improved quote fidelity for the two non-thinking
models, and costs 28% more prompt tokens per case.

Found: (1) a model-independent failure mode: a lymph node finding ("no tumor
in 6 nodes", "metastatic in 2 of 6", a staging line with an empty N slot)
returned as a pN value where the report prints no N category, by all three
models in both arms (one to two cases each; values correct, provenance
wrong), plus one histologic grade returned as a Grade Group; the instruction
alone does not stop it. (2) gemma4 writes `nodes_positive` as
`explicitly_absent` for a count of zero, which blocks the pN derivation (two
cases). (3) gpt-oss:20b's longest call reached 99.3% of its 16384 window
(11,238 prompt + 5,024 output tokens, about four thousand of them the
thinking trace at `low`); nothing was truncated, but the headroom the
auto-sizer reserves (4096) is the size of that trace. (4) Wall times are not
comparable this session: per-call latency ran at about twice the entry 1.2
figure in stretches of four of the six runs with the same token counts
(another workload on the machine); token counts are the cost reference.

Decisions put to the owner, none applied here: a `stated_pattern` guard on
derived fields (a model-returned stage or Grade Group must quote a printed
stage line, else it is set aside and derived; the same regexes as the gold
`*_printed` columns) plus two instruction sentences; a count input mode
(`explicitly_absent` passed as 0) plus an instruction for `nodes_positive`;
park the hedging exemplar (variant stays wired); a think-aware context
headroom (4096 off, 8192 on) and a context-pressure note above 90% of the
window; iterate on those before the 374-case sweep; carry all three models
into the sweep; add the second two-printed-stages case to the gold notes.

Checked: all four columns (entry 1.2 baseline, its re-score, arm A, arm B)
are recomputed from the saved predictions, ledgers and gold by one script
kept with the long-form report in the local notes folder; the re-score
column reproduces entry 1.5's numbers. Test suite unchanged at 212.


### 1.8 Pack v4: stated-value guard, count inputs, think-aware context (2026-10-22)

What: the step 4 iteration landed as pack v4 plus engine support, before
any new model run, so the next runs (`p1-iter1-<model>`) measure the
instruction changes alone. Every part is in this commit:

- `stated_pattern` on a derived field (`engine/packloader.py`,
  `engine/validate.py`): a value the model calls stated stands only when one
  of its evidence quotes prints a value of that kind, judged on the located
  source span by a regex the pack declares; otherwise the envelope is
  emptied, flagged `stated_without_printed_value`, and the derivation fills
  it with the right provenance. The prostate pack declares one pattern each
  for pT, pN and Grade Group, and `scripts/join_gold_labels.py` now reads
  those same patterns from the pack to fill the gold's `*_printed` columns
  (its own copies of the regexes are gone), so the eval's stated/derived
  split and the guard agree by construction. The patterns are written in
  verbose form with comments, and were widened for one site's template as
  the OCR renders it ("p. T 3a", "Staging (pTNM): T 2c. N 0. M x",
  "pT3apN0pMX"): 28 reports gain a printed pT and 24 a printed pN (385 and
  353 of 446), none of them in the 24-case working set, so no earlier
  number moves; the gold CSV is regenerated.
- `count` input mode for derived fields: `explicitly_absent` on a count is
  passed to the rule as 0. `nodes_positive` feeds the pN derivation that way.
- Four instruction sentences: a lymph node finding or a stage line without
  an N category is not an N stage; a histologic grade is not a Grade Group;
  `nodes_positive` is a count (0 when nodes were examined and none has
  tumor, `explicitly_absent` only when none were examined); an unclear,
  equivocal or "cannot be excluded" margin is focal with hedging possible or
  probable, not `explicitly_absent`.
- Context sizing: the auto-sizer reserves 8192 tokens of headroom when
  thinking is on (any level) and 4096 when off, so gpt-oss:20b at `--think
  low` now gets 20480 with three exemplars. The report flags a run whose
  largest call used more than 90% of its window (`context_pressure_high`
  in the cost summary, plus a note).
- The second two-printed-stages case joins `data/gold_notes.md` (eight
  audited cases). The hedging exemplar is parked: the variant stays wired,
  no run uses it.

Results (the six entry 1.7 runs re-scored with pack v4, no model calls;
exact micro-F1 all 24 / random-20 / weird-5): gemma4:12b 87.5 / 91.1 / 75.0
to 88.3 / 92.1 / 75.0 in arm A and 87.9 / 91.6 / 75.0 to 88.3 / 92.1 / 75.0
in arm B, both from the count input (two reports where it wrote the zero
node count as `explicitly_absent` now derive pN0; its derived-pN subset goes
from 80 to 100); qwen3.5:9b and gpt-oss:20b unchanged (90.1 / 90.8 and
90.8 / 90.1). The guard fired 11 times across the six runs: 9 where the
gold says no value is printed (the node-finding and histologic-grade cases
of entry 1.7, every one now derived with the same value and the right
provenance; the provenance table's "false stated" column is empty for every
v4 arm) and 2 where the report prints a stage but the quote the model gave
does not show it (a node-status line; a two-letter quote). No scored value
changed because of a trip. The context note fires on three of the six
re-scored runs (99.3%, 90.7% and 90.2% of the window); under the new
headroom the 99.3% call would sit at 79%.

Checked: 222 tests (was 212), the regenerated gold CSV differs from the
previous one only in the 52 flipped `*_printed` cells, the four exemplar
outputs pass the guard, and the comparison script of entry 1.7 gained the
v4 and iter1 arms, a guard-trip table and a margin watch on the two hedged
cases; its output and the six re-scored run folders sit in the local notes
folder with the step summary.

