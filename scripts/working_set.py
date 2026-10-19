"""Choose the working set: the case ids the smoke test and the eval runner use.

    python -m scripts.working_set random --n 20        # initial smoke-test draw (fixed seed)
    python -m scripts.working_set pick ID [ID ...]     # exactly these cases, in this order
    python -m scripts.working_set add ID [ID ...]      # append to the current set
    python -m scripts.working_set remove ID [ID ...]
    python -m scripts.working_set show                 # the set, with gold labels and report length

The set is ``data/tcga_prad/working_set.txt``, one case id per line (``#``
starts a comment). Git tracks it, unlike the rest of ``data/``: it holds
public TCGA barcodes, no PHI, and a hand-picked list cannot be regenerated.

Every id must have a staged report (``data/tcga_prad/reports/<id>.txt``); an
unknown id is an error and nothing is written. ``random`` draws only from
cases whose gold labels are complete, so the smoke test can score every
field. ``pick`` and ``add`` accept any staged case and flag the ones whose
gold is missing or incomplete, because the weird cases are often exactly
those. ``random`` refuses to replace an existing set unless ``--force``;
``pick`` always replaces it, which is what picking means.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

from engine.packloader import Pack, PackError, load_pack
from evalharness.runner import load_gold, read_id_file

DEFAULT_PACK = "packs/prostate_pathology"
DEFAULT_REPORTS = "data/tcga_prad/reports"
DEFAULT_OUT = "data/tcga_prad/working_set.txt"

Gold = dict[str, dict[str, Any]]


def normalize_id(raw: str) -> str:
    """``TCGA-XX-1234``, ``TCGA-XX-1234.txt`` or ``data/.../TCGA-XX-1234.txt`` -> ``TCGA-XX-1234``."""
    s = raw.strip()
    if "/" in s or s.endswith(".txt"):
        s = Path(s).name
        if s.endswith(".txt"):
            s = s[:-4]
    return s


def dedupe(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def staged_ids(reports_dir: Path) -> set[str]:
    return {p.stem for p in reports_dir.glob("*.txt")} if reports_dir.exists() else set()


def load_gold_or_none(pack: Pack, path: str | Path | None) -> Gold | None:
    """The gold table, or None (with a warning) when it is not there yet."""
    try:
        return load_gold(pack, path)
    except PackError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        return None


def gold_note(case_id: str, gold: Gold | None, scored: list[str]) -> str:
    if gold is None:
        return "no gold file"
    if case_id not in gold:
        return "no gold labels"
    missing = [f for f in scored if gold[case_id].get(f) is None]
    return "gold incomplete: " + ", ".join(missing) if missing else ""


def complete_ids(staged: set[str], gold: Gold | None, scored: list[str]) -> list[str]:
    """Staged cases whose scored gold labels are all present (all staged cases when there is no gold)."""
    if gold is None:
        return sorted(staged)
    return sorted(c for c in staged if c in gold and all(gold[c].get(f) is not None for f in scored))


def draw_random(eligible: list[str], n: int, seed: int) -> list[str]:
    """``n`` of ``eligible`` with a fixed seed, sorted; all of them when there are at most ``n``."""
    eligible = sorted(eligible)
    if len(eligible) <= n:
        return eligible
    return sorted(random.Random(seed).sample(eligible, n))


def write_set(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")


def render(ids: list[str], pack: Pack, gold: Gold | None, reports_dir: Path) -> str:
    """A plain table: case id, report length, the pack's scored gold fields, a note."""
    scored = list(pack.eval.scored) if pack.eval else []
    header = ["case_id", "chars", *scored, "note"]
    rows: list[list[str]] = []
    for case_id in ids:
        report = reports_dir / f"{case_id}.txt"
        chars = str(len(report.read_text(encoding="utf-8").strip())) if report.exists() else "missing"
        labels = gold.get(case_id, {}) if gold else {}
        cells = ["" if labels.get(f) is None else str(labels[f]) for f in scored]
        rows.append([case_id, chars, *cells, gold_note(case_id, gold, scored)])
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*header).rstrip(), fmt.format(*["-" * w for w in widths]).rstrip()]
    lines += [fmt.format(*r).rstrip() for r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=DEFAULT_PACK)
    ap.add_argument("--reports-dir", default=DEFAULT_REPORTS)
    ap.add_argument("--gold", default=None, help="gold CSV (default: the pack's eval.gold path)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="the working-set file")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_random = sub.add_parser("random", help="draw N complete-label cases with a fixed seed")
    p_random.add_argument("--n", type=int, default=20)
    p_random.add_argument("--seed", type=int, default=0)
    p_random.add_argument("--force", action="store_true", help="replace an existing working set")
    for name, help_ in [("pick", "exactly these cases, in this order (replaces the set)"),
                        ("add", "append these cases"), ("remove", "drop these cases")]:
        sub.add_parser(name, help=help_).add_argument("ids", nargs="+")
    sub.add_parser("show", help="print the current set with its gold labels")
    args = ap.parse_args(argv)

    pack = load_pack(args.pack)
    reports_dir = Path(args.reports_dir)
    out = Path(args.out)
    staged = staged_ids(reports_dir)
    if not staged:
        print(f"no staged reports in {reports_dir}; run scripts.fetch_tcga_reports first", file=sys.stderr)
        return 1
    gold = load_gold_or_none(pack, args.gold)
    scored = list(pack.eval.scored) if pack.eval else []
    current = read_id_file(out) if out.exists() else []

    if args.cmd == "show":
        if not current:
            print(f"no working set yet ({out}); run `python -m scripts.working_set random --n 20` "
                  f"or `python -m scripts.working_set pick ID ...`", file=sys.stderr)
            return 1
        chosen = current
    elif args.cmd == "random":
        if current and not args.force:
            print(f"working set already has {len(current)} case(s) in {out}; pass --force to replace it, "
                  f"or `pick`/`add` to change it", file=sys.stderr)
            return 1
        eligible = complete_ids(staged, gold, scored)
        chosen = draw_random(eligible, args.n, args.seed)
        write_set(out, chosen)
        print(f"working set: {len(chosen)} case(s) drawn from {len(eligible)} with complete labels "
              f"(seed {args.seed}) -> {out}", file=sys.stderr)
    else:
        ids = dedupe([normalize_id(i) for i in args.ids])
        if args.cmd == "remove":
            absent = [i for i in ids if i not in current]
            if absent:
                print(f"not in the working set: {', '.join(absent)}", file=sys.stderr)
            chosen = [c for c in current if c not in ids]
        else:
            unknown = [i for i in ids if i not in staged]
            if unknown:
                print(f"no staged report for: {', '.join(unknown)} (looked in {reports_dir}); nothing written",
                      file=sys.stderr)
                return 1
            chosen = ids if args.cmd == "pick" else dedupe(current + ids)
        write_set(out, chosen)
        print(f"working set: {len(chosen)} case(s) -> {out}", file=sys.stderr)

    if chosen:
        print(render(chosen, pack, gold, reports_dir))
    flagged = [c for c in chosen if gold_note(c, gold, scored)]
    if flagged and gold is not None:
        print(f"note: {len(flagged)} case(s) have missing or incomplete gold; the eval scores "
              f"only the fields it has", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
