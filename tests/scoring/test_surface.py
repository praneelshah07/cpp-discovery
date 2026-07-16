"""Tests for the surface-adsorption (electrostatic) score."""

from __future__ import annotations

from cpp_ai.scoring.surface import charge_adsorption, surface_adsorption


def test_bounded() -> None:
    for q in range(-5, 20):
        assert 0.0 <= charge_adsorption(q) <= 1.0


def test_sweet_spot_peaks_at_plus_four_to_six() -> None:
    peak = max(range(-2, 13), key=charge_adsorption)
    assert 4 <= peak <= 6
    assert charge_adsorption(5) > 0.85


def test_neutral_and_negative_get_low_floor() -> None:
    # exploratory floor, not zero — some CPPs enter via non-electrostatic routes
    for q in (-2, -1, 0):
        assert charge_adsorption(q) <= 0.12
    assert charge_adsorption(0) > 0.0  # kept as exploratory, not discarded


def test_over_charged_tapers_for_toxicity() -> None:
    # very high charge over-adsorbs / turns toxic -> tapered below the sweet spot
    assert charge_adsorption(11) < charge_adsorption(5)
    assert charge_adsorption(12) < charge_adsorption(9)


def test_algae_proven_anchors_score_high() -> None:
    for seq in ("LLIILRRRIRKQAHAHSK", "LLIILARRIRKQAHAHSK", "TNVYNWFQNRRARTKRK"):
        assert surface_adsorption(seq) > 0.8  # pVEC, pVEC-R6A, ClWOX all cationic


def test_neutral_peptide_demoted() -> None:
    # a net-neutral/negative peptide adsorbs poorly regardless of good insertion
    assert surface_adsorption("YQQILTSMPSRNVIQISNDLENLRDLLHVL") < 0.2
