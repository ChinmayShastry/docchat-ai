"""Tests for per-format text extraction and upload validation."""

import csv

import pytest
from langchain_core.documents import Document

from src import extractor
from src.extractor import (
    UnsupportedFormatError,
    extract_text,
    validate_upload,
)


def _doc(text):
    return Document(page_content=text, metadata={"source": "ignored.pdf", "page": 1})

# ── TXT ───────────────────────────────────────────────────────────────────

def test_extract_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Quarterly revenue reached 42 million dollars.", encoding="utf-8")

    docs = extract_text(str(path), "notes.txt")

    assert len(docs) == 1
    assert "42 million" in docs[0].page_content
    assert docs[0].metadata["source"] == "notes.txt"
    assert docs[0].metadata["page"] == 1


def test_extract_txt_empty_returns_nothing(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n  \n", encoding="utf-8")

    assert extract_text(str(path), "blank.txt") == []


def test_extract_txt_tolerates_bad_encoding(tmp_path):
    """Undecodable bytes must not raise — errors are ignored by design."""
    path = tmp_path / "weird.txt"
    path.write_bytes(b"valid text \xff\xfe more text")

    docs = extract_text(str(path), "weird.txt")
    assert "valid text" in docs[0].page_content


# ── CSV ───────────────────────────────────────────────────────────────────

def test_extract_csv_includes_shape_and_columns(tmp_path):
    path = tmp_path / "data.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "revenue"])
        writer.writerows([["EMEA", 100], ["APAC", 250], ["AMER", 175]])

    docs = extract_text(str(path), "data.csv")
    content = docs[0].page_content

    assert "Rows: 3" in content
    assert "region" in content and "revenue" in content


def test_extract_csv_empty_returns_nothing(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("col_a,col_b\n", encoding="utf-8")

    assert extract_text(str(path), "empty.csv") == []


def test_extract_csv_malformed_does_not_raise(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("a,b\n1,2,3,4,5\n6,7\n", encoding="utf-8")

    # Bad lines are skipped rather than aborting the whole upload.
    docs = extract_text(str(path), "broken.csv")
    assert isinstance(docs, list)


# ── Excel ─────────────────────────────────────────────────────────────────

def test_extract_xlsx_one_document_per_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")

    path = tmp_path / "book.xlsx"
    workbook = openpyxl.Workbook()

    first = workbook.active
    first.title = "Revenue"
    first.append(["region", "amount"])
    first.append(["EMEA", 100])

    second = workbook.create_sheet("Headcount")
    second.append(["team", "people"])
    second.append(["eng", 40])

    workbook.save(path)

    docs = extract_text(str(path), "book.xlsx")

    assert len(docs) == 2
    assert "[Sheet: Revenue]" in docs[0].page_content
    assert "[Sheet: Headcount]" in docs[1].page_content
    assert docs[1].metadata["page"] == 2


# ── DOCX ──────────────────────────────────────────────────────────────────

def test_extract_docx_includes_table_text(tmp_path):
    docx = pytest.importorskip("docx")

    path = tmp_path / "memo.docx"
    document = docx.Document()
    document.add_paragraph("Executive summary paragraph.")

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Revenue"
    table.cell(1, 1).text = "42M"
    document.save(path)

    docs = extract_text(str(path), "memo.docx")
    content = docs[0].page_content

    assert "Executive summary" in content
    assert "42M" in content, "table contents must be extracted, not just paragraphs"


# ── PDF fallback chain ────────────────────────────────────────────────────
#
# Real PDFs are awkward to build in a test, but the behaviour that actually
# matters is the chain: each parser is tried in order, a failure moves to the
# next, and an empty result is treated as a failure rather than success.

def test_pdf_uses_first_parser_that_returns_text(monkeypatch):
    calls = []

    def working(filepath, filename):
        calls.append("pypdf")
        return [_doc("extracted text")]

    def should_not_run(filepath, filename):
        calls.append("later")
        raise AssertionError("later parsers must not run after one succeeds")

    monkeypatch.setattr(extractor, "_pdf_pypdf", working)
    monkeypatch.setattr(extractor, "_pdf_pymupdf", should_not_run)
    monkeypatch.setattr(extractor, "_pdf_plumber", should_not_run)
    monkeypatch.setattr(extractor, "_pdf_ocr", should_not_run)

    docs = extractor._extract_pdf("ignored.pdf", "ignored.pdf")

    assert calls == ["pypdf"]
    assert docs[0].page_content == "extracted text"


def test_pdf_falls_through_on_parser_exception(monkeypatch):
    """A crashing parser must not abort extraction — try the next one."""
    def exploding(filepath, filename):
        raise RuntimeError("corrupt xref table")

    def recovering(filepath, filename):
        return [_doc("recovered by second parser")]

    monkeypatch.setattr(extractor, "_pdf_pypdf", exploding)
    monkeypatch.setattr(extractor, "_pdf_pymupdf", recovering)

    docs = extractor._extract_pdf("ignored.pdf", "ignored.pdf")

    assert docs[0].page_content == "recovered by second parser"


def test_pdf_treats_empty_result_as_failure(monkeypatch):
    """A parser returning no pages has not succeeded — keep going.

    This is the scanned-PDF case: the parser runs cleanly but finds no text
    layer, so the chain must continue rather than report success with nothing.
    """
    monkeypatch.setattr(extractor, "_pdf_pypdf", lambda f, n: [])
    monkeypatch.setattr(extractor, "_pdf_pymupdf", lambda f, n: [])
    monkeypatch.setattr(extractor, "_pdf_plumber", lambda f, n: [_doc("from plumber")])

    docs = extractor._extract_pdf("ignored.pdf", "ignored.pdf")

    assert docs[0].page_content == "from plumber"


def test_pdf_returns_empty_when_every_parser_fails(monkeypatch):
    for name in ("_pdf_pypdf", "_pdf_pymupdf", "_pdf_plumber", "_pdf_ocr"):
        monkeypatch.setattr(extractor, name, lambda f, n: [])

    assert extractor._extract_pdf("ignored.pdf", "ignored.pdf") == []


# ── Format handling ───────────────────────────────────────────────────────

def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")

    with pytest.raises(UnsupportedFormatError, match="Unsupported file type"):
        extract_text(str(path), "archive.zip")


def test_extension_matching_is_case_insensitive(tmp_path):
    path = tmp_path / "NOTES.TXT"
    path.write_text("content here", encoding="utf-8")

    assert len(extract_text(str(path), "NOTES.TXT")) == 1


def test_filename_with_multiple_dots(tmp_path):
    path = tmp_path / "report.final.v2.txt"
    path.write_text("content here", encoding="utf-8")

    assert len(extract_text(str(path), "report.final.v2.txt")) == 1


# ── Upload validation ─────────────────────────────────────────────────────

def test_validate_upload_accepts_supported_file(tmp_path):
    path = tmp_path / "ok.txt"
    path.write_text("small", encoding="utf-8")

    validate_upload(str(path), "ok.txt", max_mb=25)  # must not raise


def test_validate_upload_rejects_oversized(tmp_path):
    path = tmp_path / "big.txt"
    path.write_bytes(b"x" * (2 * 1024 * 1024))

    with pytest.raises(ValueError, match="over the 1 MB limit"):
        validate_upload(str(path), "big.txt", max_mb=1)


def test_validate_upload_rejects_unsupported_type(tmp_path):
    path = tmp_path / "app.exe"
    path.write_bytes(b"MZ")

    with pytest.raises(UnsupportedFormatError):
        validate_upload(str(path), "app.exe", max_mb=25)
