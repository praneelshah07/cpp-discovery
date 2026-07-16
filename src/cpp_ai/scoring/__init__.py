"""Interpretable, multi-axis scoring (redesign per expert review).

Decomposes the single "similarity" number into separate, honestly-labelled
signals. Currently provides block-wise physicochemical similarity; motif,
CPP-plausibility, safety, and applicability-domain axes are being added.
"""

from __future__ import annotations

from .blocks import BIOLOGICAL_BLOCKS, BiologicalBlock, all_block_features
from .context import AlgaeFitScorer, FitContribution, FitTerm
from .disruption import is_trained_model_available, membrane_disruption_prior
from .evidence import EvidenceProfile, EvidenceScorer, evidence_level
from .insertion import insertion_fit
from .physchem import BlockScore, BlockSimilarityIndex, PhyschemProfile
from .positional import (
    CLWOX_CRITICAL,
    CriticalPositionProfile,
    critical_position_score,
    substitution_similarity,
)
from .safety import SafetyAssessment, assess_safety, charge_risk, membrane_lysis_risk
from .surface import charge_adsorption, surface_adsorption

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
    "insertion_fit",
    "AlgaeFitScorer",
    "FitTerm",
    "FitContribution",
    "charge_risk",
    "membrane_lysis_risk",
    "membrane_disruption_prior",
    "is_trained_model_available",
    "surface_adsorption",
    "charge_adsorption",
    "assess_safety",
    "SafetyAssessment",
    "CriticalPositionProfile",
    "CLWOX_CRITICAL",
    "critical_position_score",
    "substitution_similarity",
]
