"""Tests for critical-position (scaffold-mode) scoring."""

from __future__ import annotations

import pytest

from cpp_ai.scoring import (
    CLWOX_CRITICAL,
    CriticalPositionProfile,
    critical_position_score,
    substitution_similarity,
)

_CLWOX = "TNVYNWFQNRRARTKRK"


def test_substitution_similarity() -> None:
    assert substitution_similarity("W", "W") == pytest.approx(1.0)
    assert substitution_similarity("W", "F") > 0.0          # conservative, partial
    assert substitution_similarity("W", "D") == 0.0         # non-conservative
    assert substitution_similarity("K", "R") > 0.0          # conservative charged


def test_identical_scores_one() -> None:
    assert critical_position_score(CLWOX_CRITICAL, _CLWOX) == pytest.approx(1.0)


def test_unrelated_sequence_returns_none() -> None:
    # a totally different peptide should not be positionally scored
    assert critical_position_score(CLWOX_CRITICAL, "DDDDEEEEDDDDEEEE") is None


def test_critical_mutation_hurts_more_than_neutral() -> None:
    # W6->A (critical, weight 3) vs a low-weight tail change
    w6a = _CLWOX[:5] + "A" + _CLWOX[6:]
    tail = _CLWOX[:-1] + "R"  # K17->R, unweighted position
    assert critical_position_score(CLWOX_CRITICAL, w6a) < critical_position_score(CLWOX_CRITICAL, tail)


def test_uniform_profile_is_positional_identity() -> None:
    prof = CriticalPositionProfile.uniform(_CLWOX)
    one_off = _CLWOX[:2] + "A" + _CLWOX[3:]  # V3->A
    s = critical_position_score(prof, one_off)
    assert s is not None and 0.8 < s < 1.0


def test_empty_candidate() -> None:
    assert critical_position_score(CLWOX_CRITICAL, "") is None
