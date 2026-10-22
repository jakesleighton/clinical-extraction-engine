# Weird cases (hand-picked)

Five TCGA-PRAD reports chosen because they are hard: amendments that change
the diagnosis, a report that contradicts itself, hedged margins, deferred
results, gold labels that disagree with the text. Each entry notes what makes
it hard and what the correct answer should be, so Phase 1 can spot-check the
unscored fields against them and Phase 2 can regress against known failures.

This file is tracked in git (see .gitignore); report excerpts are public
TCGA text. "Gold" is the GDC row in
`packs/prostate_pathology/labels/tcga_prad_gold.csv`. The five are in
`data/tcga_prad/working_set.txt` alongside the seeded random draw.

Chosen 2026-10-19 from `data/weird_case_candidates.md`
(`python -m scripts.weirdness_screen`), after reading the candidates in full.

Gold labels that the report text does not support (found while reading the
Phase 1 baseline outputs) are recorded case by case in `data/gold_notes.md`,
which the eval runner reads (`--gold-notes`) to score with and without them.

## The five

### TCGA-CH-5741
- Why it is hard: the diagnosis describes "1.5 cm carcinoma metastasis
  without evidence of extranodal tumor growth in one of three lymph nodes"
  and "extensive bilateral infiltration of the seminal vesicles"; the summary
  says "pT3b ... pNI (1/5), RI" (letter I typed for 1). The follow-up report,
  issued after CD31/D2-40 stains for vessel invasion, then states "The final
  tumor classification is thus as follows: pT3a ... pNO (0/3), L0, V0, R1",
  which contradicts the findings above it. Also "Gleason 4: 50%, Gleason 4:
  30%", where the second should read Gleason 5.
- Field(s) affected: pT, pN, and the amendment rule itself.
- Expected answer: Gleason 4+5=9; pT3b; pN1; margins positive (R1). Gold
  agrees (4+5, pT3b, pN1).
- Notes: a "latest statement wins" rule gives pT3a and pN0 here. The
  follow-up line should raise `conflict` rather than override findings it
  contradicts. Same lab and translator template as TCGA-CH-5772, where the
  same follow-up wording carries consistent values; the 5741 line reads like
  a template carried over from another case.

### TCGA-HC-7749
- Why it is hard: an addendum at the top of the report re-reads the
  separately submitted right lateral margin as "clear of the true surgical
  margin", finds "a single gland representing prostatic adenocarcinoma ...
  within the fat" as focal extraprostatic extension, and says "The staging is
  therefore changed to a pT3". The synoptic below still reads "Extraprostatic
  extension: Not identified", "extends to the right lateral margin" and
  "pTNM stage: pT2cN0". Gleason is written as "Primary pattern: 4/5",
  "Secondary pattern: 3/5", "Tertiary pattern: 5/5", "Total Gleason score:
  7/10".
- Field(s) affected: pT (pT2c to pT3), margins (positive to clear),
  extraprostatic extension, tertiary pattern.
- Expected answer: Gleason 4+3=7 with tertiary pattern 5; pT3 (the text never
  writes a substage); margins negative per the addendum; pN0 (0/5 and 0/4).
- Notes: gold is pT3a, so a literal "pT3" scores as a miss on pT. Decide
  whether the eval treats pT3 against pT3a as a partial match or records it
  as an eval note.

### TCGA-J4-A67T
- Why it is hard: headed "COPY ONLY DO NOT FILE. Correction to results." The
  original all-caps diagnosis says "TNM STAGE: AT LEAST pT2c, pNO, pMX" and
  "THERE IS NO EXTRAPROSTATIC INVASION SEEN ELSEWHERE"; the signed
  supplemental says the tumor "focally penetrates through the capsule of the
  right posterior prostate" and "The TNM stage of the tumor is revised as:
  TNM stage: pT3a, pNo, pMX". "SEVERITY IN EXTENT OF TUMOR: 3/5" is a scale,
  not a Gleason pattern. Eight negative nodes.
- Field(s) affected: pT (pT2c to pT3a); pN written as pNO and pNo.
- Expected answer: Gleason 3+4=7; pT3a (the correction wins); pN0; margin
  positive at the right posterior prostate (iatrogenic capsular incision).
- Notes: gold keeps the superseded pT2c and has no pN, so a correct
  extraction scores wrong on pT. Record as a gold-quality note.

### TCGA-V1-A9OH
- Why it is hard: the final diagnosis says "Gleason score 5+3=8", the comment
  says "a mix of Gleason grades 5, 4, and 3 with predominately pattern 5", and
  the synoptic says "Gleason score: 5+4, primary pattern 5, secondary pattern
  3". The report contradicts itself on the secondary pattern. Stage is written
  "pT3bNOMX"; both seminal vesicles are involved; 16 nodes negative; right
  bladder margin positive.
- Field(s) affected: gleason_secondary, gleason_total, grade_group, pT, pN.
- Expected answer: primary 5; secondary 3 or 4 with `conflict` set (4 fits
  the comment better); pT3b; pN0; margins positive.
- Notes: gold says 4+4, pT2b and no pN, which matches nothing in the text.
  Three fields will score wrong however well the model reads. Gold-quality
  note.

### TCGA-CH-5753
- Why it is hard: hedged throughout ("the status of the resection margin in
  unclear in the region of the right apex", "difficult to make any assessment
  here", "questionable contact", "the R status must most likely be assumed to
  be R0"); the result is deferred to a follow-up report that then adds L1 and
  settles R0; a previously described mantle cell lymphoma is mentioned and
  excluded; "pN1 (2/10)" is split as 2/7 right and 0/3 left.
- Field(s) affected: margins, hedging, pN, evidence quotes.
- Expected answer: Gleason 4+5=9; pT3b; pN1; margins negative (R0) with
  hedging set.
- Notes: the Phase 0 smoke case. On the first run qwen2.5:7b invented a
  "stated" Grade Group and misquoted two spans, so it already has known
  failure modes to regress against. Gold agrees with the text.

## Not picked, but worth an eval note

- TCGA-CH-5765: "amended after consultation" and then "3+4=7" in the remark
  against "Gleason 4+3-7 (approx. 40% Gleason 4)" in the classification.
  Gold 3+4. Same stressor as TCGA-V1-A9OH, in 694 characters.
- TCGA-CH-5772 and TCGA-CH-5764: a second GDC diagnosis entry says pN1
  against the report's negative nodes. The gold row (pN0) is right.
- TCGA-V1-A9O7 and TCGA-V1-A9ZR: gold carries the biopsy Gleason (5+4, 4+4)
  while the prostatectomy says 5+5 and 5+4. GDC Gleason sometimes reflects
  the biopsy, not the prostatectomy.
- TCGA-TK-A8OK: 564 characters, "Gleason score 7=4+3" against gold 3+4,
  "pT2N1" in the text and no stage in gold.
- Across the corpus, 139 reports write "pNO" and 100 more write "pT3a NO MX"
  or "pT3bNOMX": the letter O stands for zero. The pack's pN vocabulary is
  pN0/pN1/pNX, so the model has to normalise this and the eval should expect it.

## Template for new entries

### <case_id>
- Why it is hard:
- Field(s) affected:
- Expected answer:
- Notes:
