"""Tests for the multi-axis EvidenceScorer."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from cpp_ai.core.schema import Peptide
from cpp_ai.generation import net_charge
from cpp_ai.scoring import EvidenceScorer, evidence_level
from cpp_ai.screening import ScreenCandidate

_LIB = [
    ScreenCandidate("TNVYNWFQNRRARTKRK", "ClWOX", "Cationic", "Endocytosis", "Protein derived"),
    ScreenCandidate("TNVYNWFQNRRARSKRK", "PtWOX", "Cationic", "Endocytosis", "Protein derived"),
    ScreenCandidate("RQIKIWFQNRRMKWKK", "Penetratin", "Cationic", "Endocytosis", "Protein derived"),
    ScreenCandidate("KLAKLAKLAKLA", "amp", "Antimicrobial", "pore formation", "Synthetic"),
    ScreenCandidate("DDDDEEEEDDDDEE", "acidic", "?", "?", "?"),
]


class _StubClassifier:
    def predict_proba(self, peptides: Sequence[Peptide]) -> np.ndarray:
        return np.array(
            [min(0.99, max(0.01, 0.5 + 0.05 * net_charge(p.sequence))) for p in peptides]
        )


def test_profile_axes_present() -> None:
    profiles = EvidenceScorer(_LIB).profile("TNVYNWFQNRRARTKRK")
    top = {p.name: p for p in profiles}["ClWOX"]
    assert top.physchem == pytest.approx(1.0, abs=1e-9)
    assert top.motif_local == pytest.approx(1.0)
    assert top.global_identity == pytest.approx(1.0)
    assert top.cpp_probability is None  # no classifier supplied
    assert top.embedding is None        # no embedding service
    assert top.ad_confidence in {"high", "medium", "low"}


def test_classifier_axis_populated_separately() -> None:
    profiles = EvidenceScorer(_LIB, classifier=_StubClassifier()).profile("TNVYNWFQNRRARTKRK")
    assert all(p.cpp_probability is not None for p in profiles)


def test_sorted_by_shortlist_score() -> None:
    profiles = EvidenceScorer(_LIB).profile("TNVYNWFQNRRARTKRK")
    scores = [p.shortlist_score for p in profiles]
    assert scores == sorted(scores, reverse=True)


def test_lytic_penalized() -> None:
    profiles = {p.name: p for p in EvidenceScorer(_LIB).profile("TNVYNWFQNRRARTKRK")}
    assert profiles["amp"].lytic_risk is True


def test_per_block_exposed() -> None:
    top = EvidenceScorer(_LIB).profile("TNVYNWFQNRRARTKRK")[0]
    assert set(top.physchem_blocks) >= {"charge", "amphipathicity", "hydrophobicity"}


def test_to_record_keys() -> None:
    rec = EvidenceScorer(_LIB, classifier=_StubClassifier()).profile("TNVYNWFQNRRARTKRK")[0].to_record()
    for key in ("physchem", "physchem_pctile", "motif_local", "cpp_probability",
                "ad_confidence", "toxicity_flag", "evidence"):
        assert key in rec


def test_evidence_level_labels() -> None:
    exp = ScreenCandidate("KWKLFKKI", "x", "Cationic", "Endocytosis", "Protein derived")
    eng = ScreenCandidate("KWKLFKKI", "v", "engineered", "engineered_variant", "anchor")
    assert "experimental" in evidence_level(exp)
    assert "computational" in evidence_level(eng)
