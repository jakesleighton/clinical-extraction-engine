"""Stage the TCGA-PRAD subset of the TCGA-Reports corpus.

    python -m scripts.fetch_tcga_reports

1. Download ``TCGA_Reports.csv.zip`` (~11 MB) from the paper's GitHub repo
   (Kefeli & Tatonetti, *Patterns* 2024) into ``data/tcga_reports/``. Skipped
   when the file is already there; ``--zip`` points at a local copy.
2. Read the CSV. Columns: ``patient_filename`` (``<TCGA barcode>.<report uuid>``)
   and ``text`` (one report per patient, 9,523 rows; OCR lines were joined
   with periods, so each report is a single long line).
3. Get the TCGA-PRAD case list from the GDC API (cached at
   ``data/tcga_prad/prad_case_ids.txt``).
4. Write ``data/tcga_prad/reports/<case_id>.txt`` and
   ``data/tcga_prad/manifest.csv`` (case_id, source_key, n_chars).

Everything under ``data/`` is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from scripts._gdc import fetch_project_cases

SOURCE_URL = "https://github.com/jkefeli/tcga-path-reports/raw/main/TCGA_Reports.csv.zip"
PROJECT = "TCGA-PRAD"
_BARCODE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def download(url: str, dest: Path, *, timeout: float = 300.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "clinical-extraction-engine/0.0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  downloading {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
    return dest


def read_corpus(zip_path: Path) -> list[tuple[str, str]]:
    """[(patient_filename, text), ...] from the zipped CSV."""
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
        if not members:
            raise SystemExit(f"{zip_path}: no CSV inside the zip ({z.namelist()})")
        with z.open(members[0]) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            header = next(reader)
            try:
                key_col, text_col = header.index("patient_filename"), header.index("text")
            except ValueError:
                key_col, text_col = 0, 1
                print(f"  note: unexpected header {header}; using the first two columns", file=sys.stderr)
            return [(row[key_col], row[text_col]) for row in reader if len(row) > max(key_col, text_col)]


def case_id_of(key: str) -> str | None:
    m = _BARCODE.match(key.strip())
    return m.group(1) if m else None


def load_project_ids(project: str, cache: Path, *, refresh: bool = False) -> set[str]:
    if cache.exists() and not refresh:
        ids = {line.strip() for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()}
        print(f"  {project}: {len(ids)} case ids from cache {cache}", file=sys.stderr)
        return ids
    hits = fetch_project_cases(project, ["submitter_id"])
    ids = {h["submitter_id"] for h in hits if h.get("submitter_id")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
    return ids


def stage(records: list[tuple[str, str]], project_ids: set[str], reports_dir: Path) -> list[dict[str, object]]:
    """Write one .txt per case; when a case has several reports keep the
    longest. Returns the manifest rows."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    chosen: dict[str, tuple[str, str]] = {}
    duplicates = 0
    for key, text in records:
        case_id = case_id_of(key)
        if case_id is None or case_id not in project_ids:
            continue
        if case_id in chosen:
            duplicates += 1
            if len(text) <= len(chosen[case_id][1]):
                continue
        chosen[case_id] = (key, text)
    if duplicates:
        print(f"  note: {duplicates} extra report(s) for cases already staged; kept the longest", file=sys.stderr)
    manifest = []
    for case_id in sorted(chosen):
        key, text = chosen[case_id]
        (reports_dir / f"{case_id}.txt").write_text(text.strip() + "\n", encoding="utf-8")
        manifest.append({"case_id": case_id, "source_key": key, "n_chars": len(text.strip())})
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="data/tcga_reports/TCGA_Reports.csv.zip", help="where the corpus zip lives")
    ap.add_argument("--url", default=SOURCE_URL)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out-dir", default="data/tcga_prad")
    ap.add_argument("--refresh-ids", action="store_true", help="re-query GDC for the project's case list")
    args = ap.parse_args(argv)

    zip_path = Path(args.zip)
    if zip_path.exists():
        print(f"corpus zip present: {zip_path}", file=sys.stderr)
    else:
        print(f"downloading {args.url} -> {zip_path}", file=sys.stderr)
        download(args.url, zip_path)

    records = read_corpus(zip_path)
    print(f"corpus: {len(records)} reports", file=sys.stderr)

    out_dir = Path(args.out_dir)
    project_ids = load_project_ids(args.project, out_dir / "prad_case_ids.txt", refresh=args.refresh_ids)
    manifest = stage(records, project_ids, out_dir / "reports")
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "source_key", "n_chars"])
        w.writeheader()
        w.writerows(manifest)

    staged = {m["case_id"] for m in manifest}
    print(f"{args.project}: {len(project_ids)} GDC cases, {len(staged)} with a report -> {out_dir / 'reports'}",
          file=sys.stderr)
    print(f"  cases without a report in the corpus: {len(project_ids - staged)}", file=sys.stderr)
    print(f"  manifest: {out_dir / 'manifest.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
