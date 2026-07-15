"""Tests for arrangement (motif) descriptors."""

from __future__ import annotations

from cpp_ai.descriptors import DESCRIPTOR_REGISTRY, compute_descriptors


def test_arrangement_registered() -> None:
    assert "arrangement" in DESCRIPTOR_REGISTRY


def test_longest_runs() -> None:
    ds = compute_descriptors("RRRRAAAAAA", blocks=["arrangement"])
    assert ds["longest_basic_run"] == 4.0
    assert ds["longest_hydrophobic_run"] == 6.0  # AAAAAA


def test_cationic_aromatic_contacts() -> None:
    # RW, WR, KW adjacencies
    ds = compute_descriptors("RWKWRA", blocks=["arrangement"])
    assert ds["cationic_aromatic_contacts"] >= 3.0
    none = compute_descriptors("AAAAAA", blocks=["arrangement"])
    assert none["cationic_aromatic_contacts"] == 0.0


def test_arrangement_distinguishes_same_composition() -> None:
    # identical composition, different arrangement (clustered vs dispersed)
    clustered = compute_descriptors("LLLLKKKK", blocks=["arrangement"])
    dispersed = compute_descriptors("LKLKLKLK", blocks=["arrangement"])
    assert clustered["longest_hydrophobic_run"] > dispersed["longest_hydrophobic_run"]


def test_charge_segregation() -> None:
    segregated = compute_descriptors("RRRRRRDDDDDD", blocks=["arrangement"])
    mixed = compute_descriptors("RDRDRDRDRDRD", blocks=["arrangement"])
    assert segregated["charge_segregation"] > mixed["charge_segregation"]
