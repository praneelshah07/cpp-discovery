"""Interpretable, multi-axis scoring (redesign per expert review).

Decomposes the single "similarity" number into separate, honestly-labelled
signals. Currently provides block-wise physicochemical similarity; motif,
CPP-plausibility, safety, and applicability-domain axes are being added.
"""

from __future__ import annotations

from .blocks import BIOLOGICAL_BLOCKS, BiologicalBlock, all_block_features
from .evidence import EvidenceProfile, EvidenceScorer, evidence_level
from .physchem import BlockScore, BlockSimilarityIndex, PhyschemProfile
from .positional import (
    CLWOX_CRITICAL,
    CriticalPositionProfile,
    critical_position_score,
    substitution_similarity,
)
from .safety import SafetyAssessment, assess_safety, charge_risk

__all__ = [
    "BIOLOGICAL_BLOCKS",
    "BiologicalBlock",
    "all_block_features",
    "BlockSimilarityIndex",
    "PhyschemProfile",
    "BlockScore",
    "EvidenceProfile",
    "EvidenceScorer",
    "evidence_level",
    "charge_risk",
    "assess_safety",
    "SafetyAssessment",
    "CriticalPositionProfile",
    "CLWOX_CRITICAL",
    "critical_position_score",
    "substitution_similarity",
]
