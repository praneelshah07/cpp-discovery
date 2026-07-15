"""Tests for cpp_ai.core.types."""

from __future__ import annotations

import pytest

from cpp_ai.core import types


def test_canonical_set_has_exactly_twenty_residues() -> None:
    assert len(types.CANONICAL_AMINO_ACIDS) == 20


def test_ordered_matches_set_and_is_sorted() -> None:
    assert set(types.CANONICAL_AMINO_ACIDS_ORDERED) == types.CANONICAL_AMINO_ACIDS
    assert list(types.CANONICAL_AMINO_ACIDS_ORDERED) == sorted(
        types.CANONICAL_AMINO_ACIDS_ORDERED
    )


def test_canonical_and_ambiguous_are_disjoint() -> None:
    assert types.CANONICAL_AMINO_ACIDS.isdisjoint(types.AMBIGUOUS_CODES)


def test_one_to_three_covers_all_canonical() -> None:
    assert set(types.ONE_TO_THREE) == types.CANONICAL_AMINO_ACIDS


@pytest.mark.parametrize("seq", ["ACDEFGHIKLMNPQRSTVWY", "LLIILL", "K"])
def test_is_canonical_true(seq: str) -> None:
    assert types.is_canonical_sequence(seq)


@pytest.mark.parametrize("seq", ["", "kla", "ACX", "AC-DE", "AC1"])
def test_is_canonical_false(seq: str) -> None:
    assert not types.is_canonical_sequence(seq)


def test_non_canonical_residues_preserves_first_appearance_order() -> None:
    assert types.non_canonical_residues("AXBXA") == ("X", "B")


def test_non_canonical_residues_empty_for_clean_sequence() -> None:
    assert types.non_canonical_residues("ACDEK") == ()
