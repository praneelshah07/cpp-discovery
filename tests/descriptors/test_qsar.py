"""Tests for QSAR descriptor blocks."""

from __future__ import annotations

from cpp_ai.descriptors import DESCRIPTOR_REGISTRY, compute_descriptors


def test_core_qsar_blocks_registered() -> None:
    for block in ("zscales", "kidera", "vhse", "fasgai", "blosum"):
        assert block in DESCRIPTOR_REGISTRY


def test_zscales_produce_five_features() -> None:
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG", blocks=["zscales"])
    z_features = [k for k in ds.values if k.startswith("Z_")]
    assert len(z_features) == 5


def test_kidera_produces_ten_features() -> None:
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG", blocks=["kidera"])
    kf_features = [k for k in ds.values if k.startswith("KF_")]
    assert len(kf_features) == 10


def test_qsar_differs_for_conservative_substitution() -> None:
    # I->L is a conservative substitution; QSAR axes should still register a shift.
    a = compute_descriptors("IIIIKKKK", blocks=["zscales", "kidera", "vhse"])
    b = compute_descriptors("LLLLKKKK", blocks=["zscales", "kidera", "vhse"])
    assert a.values != b.values
