"""Critical-position ("scaffold mode") scoring.

For candidates that align well to the anchor, residue *identity at
experimentally-important positions* is more directly relevant than generic
descriptors. This implements the review's

    S_critical = Σ ωᵢ · sub_sim(anchorᵢ, candᵢ) / Σ ωᵢ

where important positions get large ωᵢ and ``sub_sim`` is a BLOSUM62-based
substitution similarity (exact preservation = full credit, conservative swap =
partial, non-conservative = none).

It is only meaningful for **alignable** candidates (returns ``None`` otherwise),
which is exactly the scaffold-vs-discovery split: close homologs get positional
scoring; unrelated peptides fall back to descriptors/motifs/CPP/safety.

The ClWOX profile encodes its alanine scan: W6 and T14 are the most critical
(largest effect on internalization), V3 moderate, R11 mild.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from Bio.Align import PairwiseAligner, substitution_matrices

_BLOSUM = substitution_matrices.load("BLOSUM62")


def substitution_similarity(a: str, b: str) -> float:
    """BLOSUM62 substitution similarity in [0, 1] (identity = 1)."""
    try:
        score = float(_BLOSUM[a, b])
        self_score = float(_BLOSUM[a, a])
    except (KeyError, IndexError):
        return 0.0
    if self_score <= 0:
        return 0.0
    return max(0.0, score / self_score)


@dataclass(frozen=True)
class CriticalPositionProfile:
    """Anchor sequence with per-position importance weights (0-based)."""

    anchor: str
    weights: dict[int, float] = field(default_factory=dict)
    baseline: float = 0.5

    def weight_at(self, index: int) -> float:
        return self.weights.get(index, self.baseline)

    @classmethod
    def uniform(cls, anchor: str, *, weight: float = 1.0) -> "CriticalPositionProfile":
        """Every position weighted equally (positional identity to the anchor)."""
        return cls(anchor=anchor, weights={}, baseline=weight)


# ClWOX = TNVYNWFQNRRARTKRK ; alanine-scan-derived weights (0-based indices).
CLWOX_CRITICAL = CriticalPositionProfile(
    anchor="TNVYNWFQNRRARTKRK",
    weights={5: 3.0, 13: 3.0, 2: 1.5, 10: 1.0},  # W6, T14, V3, R11
    baseline=0.5,
)


def _global_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.substitution_matrix = _BLOSUM
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    return aligner


def critical_position_score(
    profile: CriticalPositionProfile,
    candidate: str,
    *,
    min_identity: float = 0.35,
) -> float | None:
    """Weighted positional similarity to the anchor, or ``None`` if not alignable.

    Aligns ``candidate`` to ``profile.anchor``; if identity to the anchor is
    below ``min_identity`` the candidate is treated as unrelated (discovery mode)
    and ``None`` is returned.
    """
    anchor = profile.anchor
    if not candidate:
        return None
    alignment = _global_aligner().align(anchor, candidate)[0]

    anchor_to_cand: dict[int, str] = {}
    for (a_s, a_e), (b_s, b_e) in zip(*alignment.aligned):
        for j in range(a_e - a_s):
            anchor_to_cand[a_s + j] = candidate[b_s + j]

    identity = sum(1 for i, r in anchor_to_cand.items() if anchor[i] == r) / len(anchor)
    if identity < min_identity:
        return None

    total_w = 0.0
    score = 0.0
    for i in range(len(anchor)):
        w = profile.weight_at(i)
        total_w += w
        residue = anchor_to_cand.get(i)
        if residue is not None:
            score += w * substitution_similarity(anchor[i], residue)
    return score / total_w if total_w > 0 else None
