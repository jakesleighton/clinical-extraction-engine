import csv
import io
import zipfile

from scripts.fetch_tcga_reports import case_id_of, read_corpus, stage


def test_case_id_of():
    assert case_id_of("TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0") == "TCGA-BP-5195"
    assert case_id_of("TCGA-EB-A82B.23E186C6-739C-4EF1-8788-79AA89C6E87A") == "TCGA-EB-A82B"
    assert case_id_of("not-a-barcode") is None


def _zip(tmp_path, rows, header=("patient_filename", "text")):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    path = tmp_path / "TCGA_Reports.csv.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("__MACOSX/._TCGA_Reports.csv", "junk")
        z.writestr("TCGA_Reports.csv", buf.getvalue())
    return path


def test_read_corpus_and_stage(tmp_path):
    rows = [("TCGA-AA-0001.uuid1", "Prostate, radical prostatectomy. Gleason 3+4=7."),
            ("TCGA-AA-0001.uuid2", "short"),
            ("TCGA-BB-0002.uuid3", "Kidney, partial nephrectomy."),
            ("TCGA-CC-0003.uuid4", "Prostate biopsy, Gleason 4+4=8.")]
    records = read_corpus(_zip(tmp_path, rows))
    assert records == rows

    manifest = stage(records, {"TCGA-AA-0001", "TCGA-CC-0003", "TCGA-ZZ-9999"}, tmp_path / "reports")
    assert [m["case_id"] for m in manifest] == ["TCGA-AA-0001", "TCGA-CC-0003"]
    assert manifest[0]["source_key"] == "TCGA-AA-0001.uuid1"               # longest report kept
    assert (tmp_path / "reports" / "TCGA-AA-0001.txt").read_text() == rows[0][1] + "\n"
    assert not (tmp_path / "reports" / "TCGA-BB-0002.txt").exists()


def test_read_corpus_falls_back_to_first_two_columns(tmp_path):
    path = _zip(tmp_path, [("TCGA-AA-0001.x", "hello")], header=("id", "report"))
    assert read_corpus(path) == [("TCGA-AA-0001.x", "hello")]
