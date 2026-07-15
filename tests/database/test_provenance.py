"""Tests for cpp_ai.database.provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cpp_ai.database.provenance import file_sha256


def test_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    data = b"KLALKLALKALKAALKLA\n"
    p.write_bytes(data)
    assert file_sha256(p) == hashlib.sha256(data).hexdigest()


def test_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("sequence\nKWKLFKKI\n")
    assert file_sha256(p) == file_sha256(p)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_sha256(tmp_path / "nope.txt")
