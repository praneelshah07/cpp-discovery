"""Tests for graded safety penalties."""

from __future__ import annotations

import pytest

from cpp_ai.scoring import assess_safety, charge_risk


def test_preferred_window_has_no_charge_risk() -> None:
    assert charge_risk(4) == 0.0
    assert charge_risk(6) == 0.0
    assert charge_risk(7) == 0.0


def test_low_charge_gets_small_penalty_not_exclusion() -> None:
    # +3 was previously dropped by the hard gate; now it gets only a small risk.
    r3 = charge_risk(3)
    assert 0.0 < r3 < 0.25


def test_high_charge_graded() -> None:
    # +8 moderate, +10 strong, +12 stronger — monotonic
    assert charge_risk(8) < charge_risk(10) < charge_risk(12)
    assert charge_risk(8) < 0.5
    assert charge_risk(10) > 0.5


def test_charge_risk_bounded() -> None:
    for q in range(-10, 20):
        assert 0.0 <= charge_risk(q) <= 1.0


def test_assess_safety_lytic_bump() -> None:
    clean = assess_safety(6, lytic=False)
    lytic = assess_safety(6, lytic=True)
    assert clean.overall_risk == 0.0
    assert clean.safety_factor == 1.0
    assert lytic.overall_risk > clean.overall_risk
    assert lytic.safety_factor < clean.safety_factor


def test_assess_safety_factor_complements_risk() -> None:
    s = assess_safety(10, lytic=False)
    assert s.safety_factor == pytest.approx(1.0 - s.overall_risk)
