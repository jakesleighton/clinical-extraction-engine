"""Rank the staged reports by how likely they are to break an extractor.

    python -m scripts.weirdness_screen                 # -> data/weird_case_candidates.md
    python -m scripts.weirdness_screen --top 30

No model calls: cheap text heuristics plus what the gold already knows.
Each flag has a weight; a report's score is the sum of its flags; the top N
go into a Markdown file with one line per flag and the matching excerpt, so
the five weird cases can be picked by reading, not by scrolling 446 reports.

Text heuristics (regex over the report, case-insensitive):

* addendum / amendment / follow-up report language (CLIA boilerplate
  excluded), with extra weight when the addendum revises or corrects
* a diagnosis deferred to a later report ("supplemental report to follow")
* hedging-term density (per 1,000 chars), top 5 %
* specimen part count (A. B. C. ... or 1. 2. 3. ...), top 5 %
* length outliers: the shortest and longest ``--tails`` reports
* translation and OCR damage markers (language names, non-ASCII, stray
  symbols, spaced-out letters, fragmented words), top 5 %
* several distinct Gleason scores (biopsy history counted separately), or a
  tertiary pattern with a grade
* several pT / pN values in the text
* treatment effect actually present (synoptic "not identified" ignored)
* "no residual tumor" / vanishing-cancer findings
* a biopsy report (cores) with no prostatectomy
* another malignancy mentioned (lymphoma, urothelial, ...)

Gold heuristics (from the gold CSV and the cached GDC response):

* the text's Gleason score or pT/pN disagrees with the GDC label
* GDC carries two diagnosis entries for the case that disagree on pT or pN
* pT or pN missing in GDC

Two guards keep one hospital's template from flooding the list: a flag that
half or more of a tissue source site's reports share is a template feature,
so its weight drops to 1 and the line says so; and the ranked list holds at
most ``--per-site`` cases from any one site.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.packloader import load_pack
from scripts.join_gold_labels import parse_pattern, parse_stage

# --------------------------------------------------------------------------- patterns
ADDENDUM = re.compile(r"\b(addend(?:um|a)|(?<!improvement )(?<!improvement\. )amend(?:ed|ment)s?|"
                      r"supplement(?:al|ary) (?:report|diagnosis)|signed supplemental|"
                      r"revised (?:report|diagnosis)|(?:is|was|were|has been|hereby) revised|"
                      r"correct(?:ed|ion)s? (?:to |of )?(?:the )?(?:results?|report|diagnosis)|follow-? ?up report|"
                      r"modified report|re-?reviewed?|retyped|replaces? (?:the )?(?:previous|prior))\b", re.I)
REVISES = re.compile(r"\b(revis\w*|correct\w*|typograph\w*|error|consultation|changed|amended|"
                     r"upgrad\w*|downgrad\w*|reclassif\w*|instead of|rather than|should read)\b", re.I)
DEFERRED = re.compile(r"\b(?:supplement(?:al|ary) report|report|results?|findings) (?:to|will) follow\b|"
                      r"\bwill be (?:presented|reported|issued) (?:in|as) (?:a |an )?(?:follow-? ?up|supplement\w*|"
                      r"addendum|separate)\b|\bpending\b", re.I)
HEDGE = re.compile(r"\b(suspicious|suspicion|consistent with|compatible with|cannot (?:be )?(?:excluded?|ruled? out|"
                   r"assess(?:ed)?)|can ?not (?:exclude|rule out)|difficult to|equivocal|questionable|possibl[ey]|"
                   r"probabl[ey]|likely|favou?r(?:s|ed|ing)?|atypical|indeterminate|uncertain|unclear|borderline|"
                   r"not definitive|at least|versus)\b|\?", re.I)
HEDGE_EXCLUDE = re.compile(r"\bpossibl[ey] (?:\w+ )?lymph nodes?\b|\bconsistent with (?:reactive|benign|normal)\b", re.I)
SHORT_OK = {"a", "an", "and", "the", "of", "in", "on", "at", "to", "is", "as", "by", "or", "for", "no", "not", "are",
            "was", "be", "it", "its", "has", "had", "one", "two", "six", "ten", "see", "per", "out", "up", "all", "any",
            "cm", "mm", "ml", "g", "kg", "cc", "hpf", "pt", "pn", "pm", "am", "dr", "md", "ca", "nos", "pin", "epe",
            "svi", "lvi", "pni", "psa", "r", "l", "x", "vs", "e", "g", "i", "ii", "iii", "iv", "v", "vi", "he", "h",
            "th", "st", "nd", "rd", "yes", "non", "ink", "fat", "apex", "mid", "end", "max", "min", "mg", "cut", "set",
            "due", "new", "old", "can", "may", "who", "ajcc", "tnm", "cap", "sv", "rt", "lt", "r0", "r1", "l0", "l1",
            "v0", "v1", "g1", "g2", "g3", "n0", "n1", "nx", "t2", "t3", "hgpin", "ihc", "pt2", "pt3", "ph", "ns"}
WORD = re.compile(r"[A-Za-z]+")
TRANSLATION = re.compile(r"\b(translat\w*|original (?:language|report)|german|french|italian|korean|russian|polish|"
                         r"portuguese|spanish|dutch|vietnamese|chinese)\b", re.I)
OCR_SYMBOLS = re.compile(r"[^\x00-\x7F]|[|¦~^{}\\]")
OCR_SPACED = re.compile(r"\b(?:[A-Za-z] ){3,}[A-Za-z]\b")
GLEASON_PAIR = re.compile(r"gleason[^.;]{0,45}?\b([1-5])\s*\+\s*([1-5])\b", re.I)
GLEASON_SYNOPTIC = re.compile(r"primary (?:gleason )?(?:grade|pattern)[^.:]{0,15}:?\s*([1-5])\b.{0,80}?"
                              r"secondary (?:gleason )?(?:grade|pattern)[^.:]{0,15}:?\s*([1-5])\b", re.I | re.S)
HISTORY = re.compile(r"\b(biops\w*|cores?|history|previous|prior|preoperative|pre-op\w*|clinical)\b", re.I)
TERTIARY = re.compile(r"\btertiary\b[^.:]{0,25}?\b([3-5])\b|\b([3-5])\b[^.]{0,12}?\bas (?:a )?tertiary\b", re.I)
PT_TEXT = re.compile(r"\bp\s?T\s*([1-4][a-c]?)(?![0-9a-mo-z])", re.I)              # pT3a, PT2c, p T3b, pT2N1 (an N may follow)
PN_TEXT = re.compile(r"\b(?:p\s?N|p\s?T\s*[1-4][a-c]?\s*N)\s*(?:([01x])(?![0-9a-z])"        # pN0, pNX, pT2N1, pT3bNx
                     r"|(o)(?=\s*[mrl]\s*[0-9x]|\s*[.,;:()]|\s*$))", re.I)    # pNO, pT3a NO MX, pT3bNOR1: letter O for zero
TREATMENT = re.compile(r"\b(treatment effect|hormon\w*|androgen deprivation|anti-?androgen|neo-?adjuvant|radiation|"
                       r"radiotherapy|brachytherapy|post-?therapy|post-?treatment|lupron|leuprolide|bicalutamide|"
                       r"casodex|zoladex|goserelin|chemotherapy|cryotherapy|hifu)\b", re.I)
NEGATED = re.compile(r"\b(no|not|none|absent|n/a|na|negative|without|unknown|unspecified|nil)\b", re.I)
NO_TUMOR = re.compile(r"\bno residual (?:invasive |viable )?(?:tumou?r|carcinoma|adenocarcinoma|malignancy)\b|"
                      r"\bvanishing\b|\bminimal residual\b|"
                      r"\bno (?:tumou?r|carcinoma|adenocarcinoma) (?:is |was )?(?:identified|seen|found|present) "
                      r"(?:in|on) (?:the )?(?:entire|whole|remaining|additional|further|deeper|all)\b", re.I)
BIOPSY = re.compile(r"\b(?:cores?|biops\w*)\b", re.I)
PROSTATECTOMY = re.compile(r"\bprostatectom\w*\b|\bseminal vesicles?\b", re.I)
OTHER_MALIGNANCY = re.compile(r"\b(lymphoma|leuka?emia|urothelial|transitional cell|bladder (?:cancer|carcinoma|tumou?r)|"
                              r"colon|rectal|colorectal|melanoma|renal cell|sarcoma|second(?:ary)? primary|"
                              r"metasta\w+ (?:from|of) (?!prostat))\b", re.I)
PART_LETTER = re.compile(r"(?<![A-Za-z0-9])([A-Z])[.):]\s+(?=[A-Z][A-Za-z])")
PART_NUMBER = re.compile(r"(?<![\d.])(\d{1,2})[.)]\s+(?=[A-Z][A-Za-z])")


@dataclass
class Flag:
    name: str
    weight: int
    detail: str
    excerpt: str = ""


def excerpt(text: str, start: int, end: int, width: int = 80) -> str:
    s, e = max(0, start - width), min(len(text), end + width)
    return ("..." if s > 0 else "") + " ".join(text[s:e].split()) + ("..." if e < len(text) else "")


def _first(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return excerpt(text, m.start(), m.end()) if m else ""


# --------------------------------------------------------------------------- text measures
def hedging_density(text: str) -> tuple[int, float]:
    """Hedging terms per 1,000 chars, ignoring gross-description phrases
    ("twelve possible lymph nodes") that are not diagnostic hedging."""
    n = max(0, len(HEDGE.findall(text)) - len(HEDGE_EXCLUDE.findall(text)))
    return n, 1000.0 * n / max(len(text), 1)


def part_count(text: str) -> int:
    letters = {m.group(1) for m in PART_LETTER.finditer(text)}
    numbers = {int(m.group(1)) for m in PART_NUMBER.finditer(text)}
    n_letters = sum(1 for i, ch in enumerate("ABCDEFGHIJKLMNOP") if ch in letters and all(c in letters for c in "ABCDEFGHIJKLMNOP"[:i]))
    n_numbers = sum(1 for i in range(1, 30) if i in numbers and all(j in numbers for j in range(1, i)))
    return max(n_letters, n_numbers)


def ocr_score(text: str) -> int:
    """Stray symbols, spaced-out letters and an excess of word fragments
    ("Sur gi cal Pat hol ogy"), which is what OCR damage looks like here."""
    words = WORD.findall(text)
    fragments = sum(1 for w in words if len(w) <= 3 and w.lower() not in SHORT_OK)
    excess = max(0, fragments - int(0.08 * len(words)))
    return len(OCR_SYMBOLS.findall(text)) + 3 * len(OCR_SPACED.findall(text)) + excess


def negated(text: str, m: re.Match) -> bool:
    """Is the match inside a negated clause ("Treatment effect: Not identified")?"""
    before = text[max(0, m.start() - 30):m.start()]
    after = text[m.end():m.end() + 40]
    clause_after = re.split(r"[.;]", after)[0]
    return bool(NEGATED.search(before.split(".")[-1]) or NEGATED.search(clause_after))


def gleason_pairs(text: str) -> list[tuple[int, int, bool]]:
    """Distinct (primary, secondary, is_history) Gleason pairs in the text:
    ``a+b`` forms and synoptic primary/secondary lines. ``is_history`` marks a
    pair that sits next to biopsy / prior / clinical wording."""
    seen: list[tuple[int, int, bool]] = []
    for m in GLEASON_PAIR.finditer(text):
        before = re.split(r"\. (?=[A-Z])", text[max(0, m.start() - 70):m.start()])[-1]   # same sentence only
        hist = bool(HISTORY.search(before))
        pair = (int(m.group(1)), int(m.group(2)), hist)
        if pair not in seen:
            seen.append(pair)
    for m in GLEASON_SYNOPTIC.finditer(text):
        pair = (int(m.group(1)), int(m.group(2)), False)
        if pair not in seen:
            seen.append(pair)
    return seen


def stage_values(text: str, pattern: re.Pattern, prefix: str) -> list[str]:
    seen: list[str] = []
    for m in pattern.finditer(text):
        raw = next((g for g in m.groups() if g), "").upper().replace("O", "0")   # 'NO' is N0 in 100+ reports
        v = parse_stage(prefix + raw, prefix)                   # '3A' -> 'T3A' -> 'pT3a'; 'X' -> 'pNX'
        if v and v not in seen:
            seen.append(v)
    return seen


# --------------------------------------------------------------------------- gold side
def load_gold_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        print(f"warning: gold file not found: {path}; gold heuristics skipped", file=sys.stderr)
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["case_id"]: r for r in csv.DictReader(fh)}


def gdc_conflicts(cache: Path) -> dict[str, str]:
    """case_id -> description, for cases whose GDC diagnosis entries disagree on pT or pN."""
    if not cache.exists():
        print(f"warning: GDC cache not found: {cache}; diagnosis-conflict heuristic skipped", file=sys.stderr)
        return {}
    out: dict[str, str] = {}
    for hit in json.loads(cache.read_text(encoding="utf-8")):
        diags = hit.get("diagnoses") or []
        if len(diags) < 2:
            continue
        stages = [(parse_stage(d.get("ajcc_pathologic_t"), "T"), parse_stage(d.get("ajcc_pathologic_n"), "N"))
                  for d in diags]
        ts = {t for t, _ in stages if t}
        ns = {n for _, n in stages if n}
        if len(ts) > 1 or len(ns) > 1:
            axis = "pN" if len(ns) > 1 else "pT"
            out[hit["submitter_id"]] = f"{axis}: " + " vs ".join(f"{t or '?'} {n or '?'}" for t, n in stages)
    return out


# --------------------------------------------------------------------------- screening
def screen(reports: dict[str, str], gold: dict[str, dict[str, str]], conflicts: dict[str, str],
           *, tails: int = 10) -> dict[str, list[Flag]]:
    """All flags for every report (empty list = nothing odd)."""
    lengths = {c: len(t) for c, t in reports.items()}
    by_len = sorted(lengths, key=lengths.get)
    shortest, longest = set(by_len[:tails]), set(by_len[-tails:]) if tails else set()
    hedges = {c: hedging_density(t) for c, t in reports.items()}
    parts = {c: part_count(t) for c, t in reports.items()}
    ocr = {c: ocr_score(t) for c, t in reports.items()}

    def top5(values: dict[str, float]) -> float:
        xs = sorted(values.values())
        return xs[int(0.95 * (len(xs) - 1))] if xs else 0.0

    hedge_cut, part_cut, ocr_cut = top5({c: d for c, (_, d) in hedges.items()}), top5(parts), top5(ocr)

    # a site where most reports carry addendum language (KK's 'supplemental report to follow') gets no
    # amended_value flag: there the language is the template, so it says nothing about the values
    site_n = Counter(site_of(c) for c in reports)
    site_add = Counter(site_of(c) for c, t in reports.items() if ADDENDUM.search(t))
    template_addendum = {st for st, n in site_n.items() if n >= 5 and site_add[st] / n >= 0.5}

    flags: dict[str, list[Flag]] = {}
    for case_id, text in reports.items():
        f: list[Flag] = []
        g = gold.get(case_id, {})

        add = list(ADDENDUM.finditer(text))
        if add:
            terms = sorted({m.group(0).lower() for m in add})
            f.append(Flag("addendum", 2, f"addendum/amendment language ({len(add)} hit(s): {', '.join(terms)})",
                          excerpt(text, add[0].start(), add[0].end())))
            rev = next((REVISES.search(text, m.end(), m.end() + 300) for m in add
                        if REVISES.search(text, m.end(), m.end() + 300)), None)
            if rev:
                f.append(Flag("addendum_revises", 2, f"the addendum revises or corrects something ('{rev.group(0)}')",
                              excerpt(text, rev.start(), rev.end())))
        if DEFERRED.search(text):
            f.append(Flag("deferred", 3, "diagnosis or result deferred to a later report", _first(DEFERRED, text)))
        n_h, dens = hedges[case_id]
        if n_h >= 3 and dens >= hedge_cut:
            f.append(Flag("hedging", 2, f"hedging density {dens:.1f} per 1,000 chars ({n_h} terms; top 5%)",
                          _first(HEDGE, text)))
        if parts[case_id] >= 6 and parts[case_id] >= part_cut:
            f.append(Flag("parts", 2, f"{parts[case_id]} enumerated specimen parts (top 5%)", _first(PART_LETTER, text)
                          or _first(PART_NUMBER, text)))
        if case_id in shortest:
            f.append(Flag("short", 2, f"among the {tails} shortest reports ({lengths[case_id]:,} chars)",
                          excerpt(text, 0, min(len(text), 200), 0)))
        if case_id in longest:
            f.append(Flag("long", 2, f"among the {tails} longest reports ({lengths[case_id]:,} chars)"))
        if TRANSLATION.search(text):
            f.append(Flag("translation", 2, "translation / language marker", _first(TRANSLATION, text)))
        if ocr[case_id] >= 3 and ocr[case_id] >= ocr_cut:
            frag = next((m for m in re.finditer(r"\b[A-Za-z]{1,3}\b(?: \b[A-Za-z]{1,3}\b){3,}", text)
                         if not all(w.lower() in SHORT_OK for w in m.group(0).split())), None)
            f.append(Flag("ocr", 2, f"OCR damage markers (score {ocr[case_id]}; top 5%)",
                          _first(OCR_SPACED, text) or (excerpt(text, frag.start(), frag.end()) if frag else "")
                          or _first(OCR_SYMBOLS, text)))

        pairs = gleason_pairs(text)
        main_pairs = [(a, b) for a, b, hist in pairs if not hist] or [(a, b) for a, b, _ in pairs]
        gp, gs = parse_pattern(g.get("gleason_primary")), parse_pattern(g.get("gleason_secondary"))
        if len({(a, b) for a, b, _ in pairs}) > 1:
            f.append(Flag("gleason_multi", 2, "several Gleason scores in the text: "
                          + ", ".join(f"{a}+{b}" + (" (history)" if h else "") for a, b, h in pairs),
                          _first(GLEASON_PAIR, text)))
        if main_pairs and gp is not None and gs is not None and (gp, gs) not in main_pairs:
            f.append(Flag("gleason_vs_gold", 3, f"text Gleason {', '.join(f'{a}+{b}' for a, b in main_pairs)} "
                          f"but GDC says {gp}+{gs}", _first(GLEASON_PAIR, text) or _first(GLEASON_SYNOPTIC, text)))
        if TERTIARY.search(text):
            f.append(Flag("tertiary", 1, "tertiary Gleason pattern with a grade", _first(TERTIARY, text)))

        for key, pattern, prefix, label in (("pT", PT_TEXT, "T", "pT"), ("pN", PN_TEXT, "N", "pN")):
            vals = stage_values(text, pattern, prefix)
            gold_v = (g.get(key) or "").strip()
            if len(vals) > 1:
                f.append(Flag(f"{label}_multi", 1, f"several {label} values in the text: {', '.join(vals)}",
                              _first(pattern, text)))
            if vals and gold_v and gold_v not in vals:
                f.append(Flag(f"{label}_vs_gold", 3, f"text says {', '.join(vals)} but GDC says {gold_v}",
                              _first(pattern, text)))
            if g and not gold_v:
                f.append(Flag(f"{label}_missing", 2 if key == "pT" else 1, f"no {label} in GDC"
                              + (f" (text has {', '.join(vals)})" if vals else " (none in the text either)")))

        tm = next((m for m in TREATMENT.finditer(text) if not negated(text, m)), None)
        if tm:
            f.append(Flag("treatment", 2, f"treatment-related finding, not negated ('{tm.group(0)}')",
                          excerpt(text, tm.start(), tm.end())))
        if NO_TUMOR.search(text):
            f.append(Flag("no_tumor", 2, "'no residual tumor' / vanishing-cancer finding", _first(NO_TUMOR, text)))
        if len(BIOPSY.findall(text)) >= 3 and not PROSTATECTOMY.search(text):
            f.append(Flag("biopsy_only", 2, "biopsy/core wording and no prostatectomy or seminal vesicles",
                          _first(BIOPSY, text)))
        if OTHER_MALIGNANCY.search(text):
            f.append(Flag("other_malignancy", 1, "another malignancy mentioned", _first(OTHER_MALIGNANCY, text)))
        if case_id in conflicts:
            f.append(Flag("gdc_conflict", 3 if conflicts[case_id].startswith("pN") else 2,
                          f"GDC diagnosis entries disagree on {conflicts[case_id]}"))
        names = {fl.name for fl in f}
        several_final_gleason = len({(a, b) for a, b, hist in pairs if not hist}) > 1
        if (names & {"addendum", "addendum_revises"} and site_of(case_id) not in template_addendum
                and (names & {"pT_multi", "pN_multi"} or several_final_gleason)):
            f.append(Flag("amended_value", 3, "addendum/correction language and more than one pT, pN or final "
                          "Gleason value in the text: the amendment may change the diagnosis"))
        flags[case_id] = f
    return flags


def site_of(case_id: str) -> str:
    """TCGA tissue source site code: ``TCGA-KK-A8II`` -> ``KK``."""
    parts = case_id.split("-")
    return parts[1] if len(parts) >= 3 else "?"


def discount_site_templates(flags: dict[str, list[Flag]], *, min_site: int = 5, share: float = 0.5) -> None:
    """A flag carried by at least ``share`` of a site's reports is that site's
    template, not an anomaly: its weight drops to 1 and the detail says so."""
    site_sizes = Counter(site_of(c) for c in flags)
    site_flag = Counter((site_of(c), fl.name) for c, fs in flags.items() for fl in fs)
    for case_id, fs in flags.items():
        site = site_of(case_id)
        for fl in fs:
            frac = site_flag[(site, fl.name)] / site_sizes[site]
            if site_sizes[site] >= min_site and frac >= share and fl.weight > 1:
                fl.weight = 1
                fl.detail += f" [site template: {frac:.0%} of {site} reports]"


def score(flags: list[Flag]) -> int:
    return sum(fl.weight for fl in flags)


def gold_summary(g: dict[str, str]) -> str:
    if not g:
        return "no gold row"
    gl = f"{g.get('gleason_primary') or '?'}+{g.get('gleason_secondary') or '?'}={g.get('gleason_total') or '?'}"
    return f"Gleason {gl}, GG {g.get('grade_group') or '?'}, pT {g.get('pT') or '-'}, pN {g.get('pN') or '-'}"


def rank(flags: dict[str, list[Flag]], reports: dict[str, str], top: int, per_site: int) -> list[str]:
    """Case ids by descending score (ties: hedging density, then id), at most ``per_site`` from one site."""
    ordered = sorted((c for c in flags if flags[c]),
                     key=lambda c: (-score(flags[c]), -hedging_density(reports[c])[1], c))
    taken: Counter = Counter()
    chosen: list[str] = []
    for c in ordered:
        if taken[site_of(c)] < per_site:
            chosen.append(c)
            taken[site_of(c)] += 1
        if len(chosen) >= top:
            break
    return chosen


def render(flags: dict[str, list[Flag]], gold: dict[str, dict[str, str]], reports: dict[str, str],
           top: int, reports_dir: Path, per_site: int = 4) -> str:
    ranked = rank(flags, reports, top, per_site)
    n_flagged = sum(1 for fs in flags.values() if fs)
    counts = Counter(fl.name for fs in flags.values() for fl in fs)
    lines = ["# Weird-case candidates", "",
             f"{len(reports)} reports screened from `{reports_dir}`; {n_flagged} carry at least one flag. "
             f"Top {len(ranked)} by total weight below, at most {per_site} per tissue source site. Heuristics "
             f"only, no model calls; a flag is a reason to read the report, not a verdict.", "",
             "Pick with `python -m scripts.working_set add ID ...` (or `pick`), then describe each case in "
             "`data/weird_cases.md`.", "",
             "## How selective each flag is", "", "| flag | reports | weight |", "|---|---:|---:|"]
    weights: dict[str, int] = {}
    for fs in flags.values():
        for fl in fs:
            weights[fl.name] = max(weights.get(fl.name, 0), fl.weight)
    for name, n in counts.most_common():
        lines.append(f"| {name} | {n} | {weights[name]} |")
    lines += ["", "(weight 1 on a case means the flag was discounted as that site's template)", "",
              "## Candidates", ""]
    for position, case_id in enumerate(ranked, 1):
        fs = flags[case_id]
        lines.append(f"### {position}. {case_id}  (score {score(fs)}, {len(reports[case_id]):,} chars)  "
                     f"gold: {gold_summary(gold.get(case_id, {}))}")
        for fl in sorted(fs, key=lambda x: -x.weight):
            line = f"- **{fl.detail}**"
            if fl.excerpt:
                line += f'  \n  "{fl.excerpt}"'
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="packs/prostate_pathology")
    ap.add_argument("--reports-dir", default="data/tcga_prad/reports")
    ap.add_argument("--gold", default=None, help="gold CSV (default: the pack's eval.gold path)")
    ap.add_argument("--gdc-cache", default="data/tcga_prad/gdc_cases.json")
    ap.add_argument("--out", default="data/weird_case_candidates.md")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--tails", type=int, default=10, help="how many shortest and longest reports to flag")
    ap.add_argument("--per-site", type=int, default=4, help="max candidates from one tissue source site")
    args = ap.parse_args(argv)

    reports_dir = Path(args.reports_dir)
    reports = {p.stem: p.read_text(encoding="utf-8") for p in sorted(reports_dir.glob("*.txt"))}
    if not reports:
        print(f"no reports in {reports_dir}; run scripts.fetch_tcga_reports first", file=sys.stderr)
        return 1
    pack = load_pack(args.pack)
    gold_path = Path(args.gold) if args.gold else pack.resolve(pack.eval.gold)
    gold = load_gold_rows(gold_path)
    conflicts = gdc_conflicts(Path(args.gdc_cache))

    flags = screen(reports, gold, conflicts, tails=args.tails)
    discount_site_templates(flags)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(flags, gold, reports, args.top, reports_dir, per_site=args.per_site), encoding="utf-8")
    flagged = sum(1 for fs in flags.values() if fs)
    print(f"screened {len(reports)} reports: {flagged} flagged, top {args.top} -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
