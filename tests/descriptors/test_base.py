"""Tests for the descriptor framework (base.py)."""

from __future__ import annotations

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.registry import Registry
from cpp_ai.descriptors import compute_descriptors
from cpp_ai.descriptors.base import (
    DESCRIPTOR_REGISTRY,
    DescriptorFn,
    DescriptorSet,
    compute_for_peptide,
)
from cpp_ai.core.schema import Peptide


def test_compute_returns_descriptor_set() -> None:
    ds = compute_descriptors("LLIILRRK", blocks=["geometry"])
    assert isinstance(ds, DescriptorSet)
    assert ds["length"] == 8.0
    assert ds.blocks == ("geometry",)


def test_non_canonical_rejected() -> None:
    with pytest.raises(ValidationError):
        compute_descriptors("LLXIRRK")


def test_non_canonical_allowed_when_disabled_does_not_raise_on_guard() -> None:
    # geometry is safe for any letters; the guard is what we're bypassing.
    ds = compute_descriptors("LLXIRRK", blocks=["geometry"], require_canonical=False)
    assert ds["length"] == 7.0


def test_empty_sequence_rejected() -> None:
    with pytest.raises(ValidationError):
        compute_descriptors("   ")


def test_block_selection_subset() -> None:
    ds = compute_descriptors("LLIILRRK", blocks=["geometry", "composition"])
    assert "length" in ds
    assert "frac_L" in ds
    assert "charge_pH7.4_Lehninger" not in ds


def test_vector_orders_by_requested_names() -> None:
    ds = compute_descriptors("LLIILRRK", blocks=["composition"])
    vec = ds.vector(["frac_L", "frac_K"])
    assert vec.shape == (2,)
    assert vec[0] == pytest.approx(3 / 8)  # 3 leucines in LLIILRRK
    assert vec[1] == pytest.approx(1 / 8)


def test_vector_missing_feature_raises() -> None:
    ds = compute_descriptors("LLIILRRK", blocks=["geometry"])
    with pytest.raises(KeyError):
        ds.vector(["nonexistent"])


def test_compute_for_peptide_carries_id() -> None:
    pep = Peptide.from_sequence("LLIILRRK", dataset="t")
    ds = compute_for_peptide(pep, blocks=["geometry"])
    assert ds.peptide_id == pep.peptide_id


def test_non_finite_value_is_rejected() -> None:
    reg: Registry[DescriptorFn] = DESCRIPTOR_REGISTRY
    reg.register("_test_inf", lambda s: {"_inf_feat": float("inf")}, overwrite=True)
    try:
        with pytest.raises(ValidationError):
            compute_descriptors("LLIILRRK", blocks=["_test_inf"])
    finally:
        reg.unregister("_test_inf")


def test_duplicate_feature_across_blocks_rejected() -> None:
    reg = DESCRIPTOR_REGISTRY
    reg.register("_dup_a", lambda s: {"_dup": 1.0}, overwrite=True)
    reg.register("_dup_b", lambda s: {"_dup": 2.0}, overwrite=True)
    try:
        with pytest.raises(ValidationError):
            compute_descriptors("LLIILRRK", blocks=["_dup_a", "_dup_b"])
    finally:
        reg.unregister("_dup_a")
        reg.unregister("_dup_b")


def test_full_battery_has_no_duplicate_feature_names() -> None:
    # If any two real blocks collide on a feature name, this raises.
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG")
    assert len(ds.values) > 100  # rich battery
