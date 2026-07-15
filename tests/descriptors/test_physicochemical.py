"""Tests for physicochemical descriptor blocks."""

from __future__ import annotations

import peptides

from cpp_ai.descriptors import compute_descriptors


def test_length() -> None:
    assert compute_descriptors("KKKK", blocks=["geometry"])["length"] == 4.0


def test_cationic_peptide_is_positively_charged() -> None:
    ds = compute_descriptors("KKKKRRRR", blocks=["charge"])
    # every pK scale should report a strong positive charge
    charges = [v for k, v in ds.values.items() if k.startswith("charge_")]
    assert charges
    assert all(c > 5 for c in charges)


def test_acidic_peptide_is_negatively_charged() -> None:
    ds = compute_descriptors("DDDDEEEE", blocks=["charge"])
    charges = [v for k, v in ds.values.items() if k.startswith("charge_")]
    assert all(c < -5 for c in charges)


def test_charge_covers_every_pk_scale() -> None:
    ds = compute_descriptors("KRKR", blocks=["charge"])
    assert len(ds.values) == len(peptides.tables.PK)


def test_isoelectric_point_high_for_cationic() -> None:
    ds = compute_descriptors("KKKKRRRR", blocks=["isoelectric_point"])
    assert all(v > 10 for v in ds.values.values())


def test_gravy_positive_for_hydrophobic_sequence() -> None:
    ds = compute_descriptors("IIIILLLL", blocks=["biopython_props"])
    assert ds["gravy_kyte_doolittle"] > 0


def test_gravy_negative_for_charged_sequence() -> None:
    ds = compute_descriptors("KKKKRRRR", blocks=["biopython_props"])
    assert ds["gravy_kyte_doolittle"] < 0


def test_secondary_structure_fractions_present() -> None:
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG", blocks=["biopython_props"])
    for k in ("helix_fraction", "turn_fraction", "sheet_fraction"):
        assert k in ds


def test_hydrophobic_moment_present_and_nonnegative() -> None:
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG", blocks=["hydrophobic_moment"])
    assert ds["hydrophobic_moment_alpha"] >= 0
    assert ds["hydrophobic_moment_beta"] >= 0


def test_short_peptide_does_not_break_moment() -> None:
    ds = compute_descriptors("KLW", blocks=["hydrophobic_moment"])
    assert "hydrophobic_moment_alpha" in ds


def test_peptides_props_present() -> None:
    ds = compute_descriptors("KLWKLLKKLLKSAKKLG", blocks=["peptides_props"])
    for k in ("molecular_weight", "aliphatic_index", "boman_index"):
        assert k in ds
    assert ds["molecular_weight"] > 0
