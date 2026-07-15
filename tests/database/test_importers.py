"""Tests for cpp_ai.database importers."""

from __future__ import annotations

from pathlib import Path

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.database import (
    IMPORTER_REGISTRY,
    CPPsite3Importer,
    CSVImporter,
    ColumnMapping,
    ExcelImporter,
    PoseidonImporter,
)


def _write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_csv_basic_import_preserves_metadata_and_provenance(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path / "d.csv",
        "id,Sequence,note",
        ["1,KWKLFKKI,alpha", "2,LLIILRRK,beta"],
    )
    importer = CSVImporter(
        ColumnMapping(sequence="Sequence", identifier="id"), dataset_name="demo"
    )
    report = importer.import_file(csv)

    assert report.n_imported == 2
    assert report.n_skipped == 0
    first = report.peptides[0]
    assert first.sequence == "KWKLFKKI"
    assert first.provenance.dataset == "demo"
    assert first.provenance.original_id == "1"
    assert first.provenance.file_sha256 is not None
    # whole raw row preserved verbatim
    assert first.metadata["note"] == "alpha"
    assert first.metadata["id"] == "1"


def test_csv_skips_empty_sequences(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", "id,Sequence", ["1,KWKL", "2,", "3,   "])
    report = CSVImporter(ColumnMapping("Sequence", "id")).import_file(csv)
    assert report.n_imported == 1
    assert report.n_skipped == 2
    assert all("empty sequence" in s.reason for s in report.skipped)


def test_csv_skips_invalid_sequences_without_crashing(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", "id,Sequence", ["1,KWKL", "2,KW9L", "3,K-L"])
    report = CSVImporter(ColumnMapping("Sequence", "id")).import_file(csv)
    assert report.n_imported == 1
    assert report.n_skipped == 2
    assert all("invalid sequence" in s.reason for s in report.skipped)


def test_non_canonical_imported_but_flagged(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", "id,Sequence", ["1,KWKL", "2,KWXL"])
    report = CSVImporter(ColumnMapping("Sequence", "id")).import_file(csv)
    assert report.n_imported == 2
    assert report.n_non_canonical == 1


def test_column_resolution_is_case_insensitive(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", "ID,SEQUENCE", ["1,KWKL"])
    report = CSVImporter(ColumnMapping("sequence", "id")).import_file(csv)
    assert report.n_imported == 1
    assert report.peptides[0].provenance.original_id == "1"


def test_missing_sequence_column_raises(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", "id,peptide", ["1,KWKL"])
    with pytest.raises(ValidationError):
        CSVImporter(ColumnMapping("Sequence", "id")).import_file(csv)


def test_tab_delimited_is_sniffed(tmp_path: Path) -> None:
    tsv = tmp_path / "d.tsv"
    tsv.write_text("id\tSequence\n1\tKWKLFKKI\n", encoding="utf-8")
    report = CSVImporter(ColumnMapping("Sequence", "id")).import_file(tsv)
    assert report.n_imported == 1
    assert report.peptides[0].sequence == "KWKLFKKI"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CSVImporter().import_file(tmp_path / "nope.csv")


def test_cppsite3_preset(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "cpp.csv", "CPPsite ID,Sequence", ["CPP1,KWKLFKKI"])
    report = CPPsite3Importer().import_file(csv)
    assert report.dataset == "CPPsite3"
    assert report.peptides[0].provenance.dataset == "CPPsite3"
    assert report.peptides[0].provenance.original_id == "CPP1"


def test_poseidon_preset(tmp_path: Path) -> None:
    # Matches POSEIDON's real cargo_encoded.csv schema.
    csv = _write_csv(
        tmp_path / "pos.csv",
        "peptide_name,peptide_sequence,target",
        ["Tat,RKKRRQRRR,650.0"],
    )
    report = PoseidonImporter().import_file(csv)
    assert report.dataset == "POSEIDON"
    assert report.peptides[0].provenance.original_id == "Tat"
    assert report.peptides[0].sequence == "RKKRRQRRR"
    assert report.peptides[0].metadata["target"] == "650.0"


def test_excel_import(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["id", "Sequence", "note"])
    ws.append([1, "KWKLFKKI", "x"])
    ws.append([2, "LLIILRRK", "y"])
    xlsx = tmp_path / "d.xlsx"
    wb.save(xlsx)

    report = ExcelImporter(
        ColumnMapping("Sequence", "id"), dataset_name="xl"
    ).import_file(xlsx)
    assert report.n_imported == 2
    assert report.peptides[0].sequence == "KWKLFKKI"
    assert report.peptides[1].metadata["note"] == "y"


def test_registry_contains_all_sources() -> None:
    for name in ("csv", "excel", "CPPsite3", "POSEIDON"):
        assert name in IMPORTER_REGISTRY


def test_report_summary_is_readable(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path / "d.csv", "id,Sequence", ["1,KWKL", "2,KWXL", "3,"])
    report = CSVImporter(ColumnMapping("Sequence", "id"), dataset_name="demo").import_file(csv)
    text = report.summary()
    assert "demo" in text and "imported 2" in text and "skipped 1" in text
