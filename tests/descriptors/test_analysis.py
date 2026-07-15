"""Tests for descriptor matrix + discriminative ranking utilities."""

from __future__ import annotations

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.descriptors import (
    compute_descriptors,
    descriptor_matrix,
    discriminative_ranking,
)


def _sets() -> list:
    seqs = {
        "a": "LLIILRRRIRKQAHAHSK",
        "b": "LLIILRRRIRKQAHAHSA",  # single K->A
        "c": "KKKKRRRRWWWWLLLLAA",
    }
    return [compute_descriptors(s, peptide_id=n) for n, s in seqs.items()]


def test_matrix_shape_and_labels() -> None:
    sets = _sets()
    dm = descriptor_matrix(sets)
    assert dm.matrix.shape[0] == 3
    assert dm.matrix.shape[1] == len(dm.feature_names)
    assert dm.peptide_ids == ("a", "b", "c")


def test_matrix_requires_input() -> None:
    with pytest.raises(ValidationError):
        descriptor_matrix([])


def test_ranking_orders_most_discriminative_first() -> None:
    ranking = discriminative_ranking(_sets(), by="value_range")
    ranges = [fs.value_range for fs in ranking]
    assert ranges == sorted(ranges, reverse=True)


def test_ranking_identical_peptides_have_zero_spread() -> None:
    same = [compute_descriptors("LLIILRRK", peptide_id=f"p{i}") for i in range(3)]
    ranking = discriminative_ranking(same, by="std")
    assert all(fs.std == pytest.approx(0.0) for fs in ranking)


def test_ranking_requires_two_peptides() -> None:
    with pytest.raises(ValidationError):
        discriminative_ranking([compute_descriptors("LLIILRRK")])


def test_ranking_invalid_key() -> None:
    with pytest.raises(ValidationError):
        discriminative_ranking(_sets(), by="nonsense")


def test_ranking_detects_the_mutated_position_features() -> None:
    # Between the K->A pair only, molecular weight must be a top discriminator.
    a = compute_descriptors("LLIILRRRIRKQAHAHSK", peptide_id="a")
    b = compute_descriptors("LLIILRRRIRKQAHAHSA", peptide_id="b")
    top = [fs.feature for fs in discriminative_ranking([a, b], by="value_range")[:5]]
    assert any("molecular_weight" in f for f in top)
