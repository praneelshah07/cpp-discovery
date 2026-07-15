"""Phase 8: the explainable ranking engine.

Produces a ranked list of candidates where every entry carries its overall
score, similarity score, mutation summary, confidence, nearest literature
peptides, reasons for selection, predicted strengths, and potential weaknesses.

The rule "never return a score without an explanation" is enforced
structurally: :class:`RankedCandidate` cannot be built without its reasons,
mutation summary, and weaknesses.
"""

from __future__ import annotations

from .candidate import NearestPeptide, RankedCandidate
from .engine import RankingEngine
from .explain import (
    VALIDATION_CAVEAT,
    Evidence,
    RankingThresholds,
    build_reasons,
    build_strengths,
    build_weaknesses,
)

__all__ = [
    "RankingEngine",
    "RankedCandidate",
    "NearestPeptide",
    "RankingThresholds",
    "Evidence",
    "VALIDATION_CAVEAT",
    "build_reasons",
    "build_strengths",
    "build_weaknesses",
]
