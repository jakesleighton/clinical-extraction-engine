# Gold notes: audited cases

One row per case where the GDC gold label and the staged report text were
read against each other and found to disagree, or where the gold is finer
than anything the text prints. The eval runner reads this file
(`--gold-notes data/gold_notes.md`) and reports every score twice: on all
cases and without these cases as whole cases, so a sweep can show how much
of a gap is the labels rather than the extractor. The rows are also the
places where "the model is wrong" needs a second look before it is believed.

Columns: `case` (TCGA case id), `field(s)` (the scored fields the row is
about, comma separated), `kind` (gold_disagrees: the text supports another
value; gold_stale: a correction in the text supersedes the gold;
gold_finer: the gold carries a substage the text never prints;
amendment_rule: the gold is right and the pack's amendment rule, applied as
written, gives the other value), `gold`, `report says` (quoted text, as
printed, OCR periods included) and `note`. Only the first cell of a row is
read by the runner, so the other columns are free text. Report excerpts are
public TCGA text. Case-level background for the five hand-picked cases is in
`data/weird_cases.md`.

Audited 2026-10-19 (Phase 1 baseline, all 24 working-set cases read against
five models' outputs) and recorded 2026-10-22.

| case | field(s) | kind | gold | report says | note |
|---|---|---|---|---|---|
| TCGA-HC-A9TE | pN | gold_disagrees | pN0 | "Regional lymph nodes: pN1" in both synoptic summaries; "1 of 7 lymph. nodes positive for metastatic adenocarcinoma" | GDC records N0; the specimen has a positive node. All five baseline models answered pN1. (The same report prints pT3b in one synopsis and pT3a in the other; gold pT3b; not a gold error, a conflict.) |
| TCGA-V1-A9OY | gleason_primary, gleason_total, grade_group | gold_disagrees | 4+3=7, Grade Group 3 | "Prostatic adenocarcinoma, Gleason score 5+3=8, with extensive extraprostatic. extension"; synoptic "Gleason score: 5+3; primary pattern 5 (80%), secondary pattern 3" | The prostatectomy is graded 5+3=8 in the diagnosis and the synoptic; the gold 4+3=7 is not stated anywhere for this specimen (GDC Gleason sometimes carries the biopsy score). All five models answered 5+3=8. |
| TCGA-V1-A9OH | gleason_primary, gleason_secondary, pT | gold_disagrees | 4+4=8, pT2b (no pN) | "Gleason score 5+3=8, with extracapsular. invasion"; synoptic "Gleason score: 5+4, primary pattern 5, secondary pattern 3"; "AJCC/UICC stage: pT3bNOMX"; "Total number of lymph nodes examined: 16", "Lymph node status: Negative" | The report contradicts itself on the secondary pattern (3 in the diagnosis and the pattern line, 4 in the synoptic header) but never says 4+4; it prints pT3b with both seminal vesicles involved, not pT2b; it prints pN0 where the gold is blank. Expected: primary 5, secondary 3 or 4 with conflict set, pT3b, pN0. |
| TCGA-J4-A67T | pT | gold_stale | pT2c | original "TNM STAGE: AT LEAST pT2c, pNO, pMX"; correction "The TNM stage of the tumor is revised as: TNM stage: pT3a, pNo, pMX" | A signed correction supersedes the original stage; GDC kept pT2c. A model that follows the correction (all five did) scores wrong on pT. Gold has no pN although the text prints pN0 twice. |
| TCGA-SU-A7E7 | pT | gold_finer | pT2b | "Stage: T2 NO M0, II B" | The text prints no substage. Every model answered pT2 (faithful). The lenient column (prefix match) scores this as correct; the exact column does not. |
| TCGA-HC-7749 | pT | gold_finer | pT3a | addendum "The staging. is therefore changed to a pT3"; the synoptic below still prints "pTNM stage: pT2cN0" | The addendum prints no substage (the focal extraprostatic extension it describes is pT3a by definition, but the text never says so). A model that follows the addendum answers pT3: correct under the lenient column, wrong under exact. |
| TCGA-CH-5741 | pT, pN | amendment_rule | pT3b, pN1 | summary "pT3b, max. tumor diameter 5.5 cm, Gleason 4+5=9 (Gleason 5 = 20%), L1, V0, pNI (1/5), RI" and diagnosis "1.5 cm carcinoma metastasis without evidence of extranodal tumor growth in one of three. lymph nodes"; follow-up "The final tumor classification is thus as follows: pT3a. Gleason 4+5=9 (Gleason 4: 50%, Gleason 4: 30%), pNO (0/3), L0, V0, R1" | The gold matches the diagnosis. The follow-up, issued after vessel-invasion stains, restates the whole classification with pT3a and pN0 (0/3) although it only re-examined vessel invasion and contradicts the described nodal metastasis and seminal vesicle infiltration. The base pack's amendment rule ("an addendum changes an item: report the revised value, set conflict") gives pT3a / pN0 here and is kept as written (decision of 2026-10-22, step 3 item 3): this case is the recorded counter-example, not a special case in the rule. The pN_matches_nodes rule flags the contradiction when a model also extracts the node counts. |

Known gold quirks that are not errors (left in the scores): TCGA-CH-5764 and
TCGA-CH-5772 each carry a second GDC diagnosis entry saying pN1 against the
report's negative nodes; the gold row (pN0) is the right one. TCGA-V1-A9O7
and TCGA-V1-A9ZR carry the biopsy Gleason score in GDC while the
prostatectomy report grades 5+5 and 5+4; neither is in the working set, so
they are listed here for the full sweep, not as table rows, until they have
been read in full.
