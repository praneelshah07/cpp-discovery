"""Tests for the VariantGenerator."""

from __future__ import annotations

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.schema import Peptide
from cpp_ai.generation import MutationConstraints, VariantGenerator, net_charge

_PVEC = "LLIILRRRIRKQAHAHSK"


def _ref(seq: str = _PVEC) -> Peptide:
    return Peptide.from_sequence(seq, dataset="ref")


def test_single_mutants_each_have_one_mutation() -> None:
    variants = VariantGenerator("conservative").generate(_ref(), n_mutations=1)
    assert variants
    assert all(v.metadata["n_mutations"] == 1 for v in variants)


def test_variants_differ_from_parent_and_are_unique() -> None:
    variants = VariantGenerator("conservative").generate(_ref(), n_mutations=1)
    seqs = [v.sequence for v in variants]
    assert _PVEC not in seqs
    assert len(seqs) == len(set(seqs))


def test_variant_carries_parent_metadata() -> None:
    ref = _ref()
    v = VariantGenerator("conservative").generate(ref, n_mutations=1)[0]
    assert v.metadata["parent_id"] == ref.peptide_id
    assert v.metadata["parent_sequence"] == _PVEC
    assert v.metadata["strategy"] == "conservative"
    assert len(v.metadata["mutations"]) == 1


def test_locked_motif_is_never_mutated() -> None:
    c = MutationConstraints(locked_motifs=("LLII",))
    variants = VariantGenerator("all_canonical", c).generate(_ref(), n_mutations=1)
    assert variants
    assert all(v.sequence[:4] == "LLII" for v in variants)


def test_charge_constraint_respected() -> None:
    c = MutationConstraints(max_charge=6)
    variants = VariantGenerator("all_canonical", c).generate(_ref(), n_mutations=1)
    assert all(net_charge(v.sequence) <= 6 for v in variants)


def test_charge_preserving_holds_net_charge() -> None:
    variants = VariantGenerator("charge_preserving").generate(_ref(), n_mutations=1)
    assert all(net_charge(v.sequence) == net_charge(_PVEC) for v in variants)


def test_non_canonical_reference_raises() -> None:
    with pytest.raises(ValidationError):
        VariantGenerator("conservative").generate(_ref("LLXILRRK"), n_mutations=1)


def test_n_mutations_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        VariantGenerator().generate(_ref(), n_mutations=0)


def test_too_few_mutable_positions_returns_empty() -> None:
    # lock everything but one position, then request two mutations
    seq = "KKKK"
    c = MutationConstraints(locked_positions=frozenset({0, 1, 2}))
    assert VariantGenerator("all_canonical", c).generate(_ref(seq), n_mutations=2) == []


def test_double_mutants_have_two_mutations() -> None:
    variants = VariantGenerator("conservative", seed=1).generate(
        _ref(), n_mutations=2, max_variants=40
    )
    assert 0 < len(variants) <= 40
    assert all(v.metadata["n_mutations"] == 2 for v in variants)


def test_sampling_is_deterministic() -> None:
    a = VariantGenerator("conservative", seed=7).generate(_ref(), n_mutations=2, max_variants=25)
    b = VariantGenerator("conservative", seed=7).generate(_ref(), n_mutations=2, max_variants=25)
    assert [v.sequence for v in a] == [v.sequence for v in b]


def test_generate_series_covers_all_counts() -> None:
    variants = VariantGenerator("conservative", seed=0).generate_series(
        _ref(), mutation_counts=(1, 2), max_variants_per_count=15
    )
    counts = {v.metadata["n_mutations"] for v in variants}
    assert counts == {1, 2}


def test_generated_variants_are_canonical_by_default() -> None:
    variants = VariantGenerator("all_canonical").generate(_ref(), n_mutations=1)
    assert all(v.is_canonical for v in variants)
