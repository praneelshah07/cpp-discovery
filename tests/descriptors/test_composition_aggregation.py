"""Tests for composition and aggregation descriptor blocks."""

from __future__ import annotations

import pytest

from cpp_ai.core.types import CANONICAL_AMINO_ACIDS_ORDERED
from cpp_ai.descriptors import compute_descriptors


def test_residue_fractions_sum_to_one() -> None:
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG", blocks=["composition"])
    total = sum(ds[f"frac_{aa}"] for aa in CANONICAL_AMINO_ACIDS_ORDERED)
    assert total == pytest.approx(1.0)


def test_single_residue_fraction() -> None:
    ds = compute_descriptors("KKKKAAAA", blocks=["composition"])
    assert ds["frac_K"] == pytest.approx(0.5)
    assert ds["frac_A"] == pytest.approx(0.5)


def test_cationic_group_fraction() -> None:
    ds = compute_descriptors("KRKRAAAA", blocks=["composition"])
    assert ds["frac_group_cationic"] == pytest.approx(0.5)


def test_aromatic_group_fraction() -> None:
    ds = compute_descriptors("WYFWAAAA", blocks=["composition"])
    assert ds["frac_group_aromatic"] == pytest.approx(0.5)


def test_aggregation_peak_ge_mean() -> None:
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG", blocks=["aggregation"])
    assert (
        ds["aggregation_peak_window_hydrophobicity"]
        >= ds["aggregation_mean_hydrophobicity"]
    )


def test_aggregation_hydrophobic_higher_than_charged() -> None:
    hydro = compute_descriptors("IIIILLLLVVVV", blocks=["aggregation"])
    charged = compute_descriptors("KKKKRRRREEEE", blocks=["aggregation"])
    assert (
        hydro["aggregation_peak_window_hydrophobicity"]
        > charged["aggregation_peak_window_hydrophobicity"]
    )
