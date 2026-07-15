"""Rule-based, interpretable explanation generation.

Given the assembled evidence for a candidate, these pure functions produce the
human-readable ``reasons`` / ``strengths`` / ``weaknesses`` shown to the
researcher. The rules are deterministic and transparent on purpose — an
experimentalist can trace every sentence back to a concrete number.

All thresholds live in :class:`RankingThresholds` so they are tunable and
documented in one place. The aggregation and expression signals are labelled
heuristics, never assay results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .candidate import NearestPeptide

# Standing caveat appended to every candidate — the platform never claims truth.
VALIDATION_CAVEAT = (
    "Computational prioritization only — not experimentally validated; "
    "confirm by wet-lab uptake assay."
)


@dataclass(frozen=True)
class RankingThresholds:
    """Tunable cut-offs for turning numbers into strengths/weaknesses/reasons."""

    cpp_high: float = 0.70
    cpp_low: float = 0.40
    similarity_high: float = 0.70
    novelty_identity_max: float = 0.80  # nearest identity below this => "novel"
    charge_favorable_min: int = 4
    charge_recommended_max: int = 8
    uncertainty_high: float = 0.50       # predictive entropy
    aggregation_high: float = 2.5        # peak-window KD hydrophobicity (heuristic)
    cysteine_high: int = 2
    length_min: int = 5
    length_max: int = 40


@dataclass
class Evidence:
    """All facts gathered about one candidate, fed to the explanation rules."""

    sequence: str
    net_charge: int
    length: int
    is_canonical: bool
    cysteine_count: int
    mutation_summary: str
    reference_id: Optional[str] = None
    similarity_score: Optional[float] = None
    cpp_probability: Optional[float] = None
    uncertainty: Optional[float] = None
    epistemic_std: Optional[float] = None
    gravy: Optional[float] = None
    aggregation_peak: Optional[float] = None
    hydrophobic_moment: Optional[float] = None
    nearest: tuple[NearestPeptide, ...] = ()


def build_reasons(ev: Evidence, th: RankingThresholds) -> tuple[str, ...]:
    """Why this candidate was surfaced (never empty)."""
    reasons: list[str] = []
    if ev.cpp_probability is not None and ev.cpp_probability >= th.cpp_high:
        reasons.append(
            f"High predicted CPP likelihood (calibrated P={ev.cpp_probability:.2f})."
        )
    if ev.similarity_score is not None and ev.similarity_score >= th.similarity_high:
        ref = ev.reference_id or "reference"
        reasons.append(f"High similarity to {ref} ({ev.similarity_score:.2f}).")
    if ev.nearest:
        top = ev.nearest[0]
        if top.similarity < th.novelty_identity_max:
            name = top.peptide_id or top.sequence
            reasons.append(
                f"Novel relative to literature (nearest {name} at "
                f"{top.similarity:.0%} identity)."
            )
    if "substitution" in ev.mutation_summary or "Optimized" in ev.mutation_summary:
        reasons.append(f"Traceable design: {ev.mutation_summary}")

    if not reasons:
        # Guarantee a non-empty explanation by naming the dominant component.
        if ev.cpp_probability is not None:
            reasons.append(
                f"Ranked primarily on predicted CPP likelihood "
                f"(P={ev.cpp_probability:.2f})."
            )
        elif ev.similarity_score is not None:
            reasons.append(
                f"Ranked on similarity to {ev.reference_id or 'reference'} "
                f"({ev.similarity_score:.2f})."
            )
        else:
            reasons.append("Included as a candidate for evaluation.")
    return tuple(reasons)


def build_strengths(ev: Evidence, th: RankingThresholds) -> tuple[str, ...]:
    strengths: list[str] = []
    if th.charge_favorable_min <= ev.net_charge <= th.charge_recommended_max:
        strengths.append(f"Favorable cationic net charge (+{ev.net_charge}).")
    if ev.is_canonical and th.length_min <= ev.length <= th.length_max:
        strengths.append(
            f"Genetically encodable and expression-compatible (length {ev.length})."
        )
    if ev.hydrophobic_moment is not None and ev.hydrophobic_moment >= 0.35:
        strengths.append(
            f"Amphipathic (hydrophobic moment {ev.hydrophobic_moment:.2f})."
        )
    if ev.aggregation_peak is not None and ev.aggregation_peak < th.aggregation_high:
        strengths.append("Low predicted aggregation propensity (heuristic).")
    if ev.uncertainty is not None and ev.uncertainty < th.uncertainty_high:
        strengths.append("Model prediction is confident (low uncertainty).")
    return tuple(strengths)


def build_weaknesses(ev: Evidence, th: RankingThresholds) -> tuple[str, ...]:
    weaknesses: list[str] = []
    if not ev.is_canonical:
        weaknesses.append(
            "Contains non-canonical residues — not genetically encodable for the "
            "recombinant mCherry fusion."
        )
    if ev.net_charge > th.charge_recommended_max:
        weaknesses.append(
            f"Net charge +{ev.net_charge} exceeds the recommended +"
            f"{th.charge_recommended_max} (possible cytotoxicity)."
        )
    if ev.uncertainty is not None and ev.uncertainty >= th.uncertainty_high:
        weaknesses.append(
            f"High model uncertainty (predictive entropy {ev.uncertainty:.2f}) — "
            "prioritize experimental validation."
        )
    if ev.epistemic_std is not None and ev.epistemic_std >= 0.30:
        weaknesses.append(
            f"Model ensemble disagrees ({ev.epistemic_std:.2f}) — extrapolation risk."
        )
    if ev.aggregation_peak is not None and ev.aggregation_peak >= th.aggregation_high:
        weaknesses.append(
            f"Elevated aggregation proxy ({ev.aggregation_peak:.2f}) — check solubility."
        )
    if ev.cysteine_count > th.cysteine_high:
        weaknesses.append(
            f"{ev.cysteine_count} cysteines — disulfide/expression complexity."
        )
    if ev.length < th.length_min or ev.length > th.length_max:
        weaknesses.append(
            f"Length {ev.length} is outside the typical CPP range "
            f"[{th.length_min}, {th.length_max}]."
        )
    if ev.cpp_probability is not None and ev.cpp_probability < th.cpp_low:
        weaknesses.append(
            f"Low predicted CPP likelihood (P={ev.cpp_probability:.2f})."
        )
    # Always end with the standing scientific caveat.
    weaknesses.append(VALIDATION_CAVEAT)
    return tuple(weaknesses)
