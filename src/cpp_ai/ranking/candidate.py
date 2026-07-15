"""The explainable ranked-candidate type.

The platform's central rule — *never return a score without an explanation* — is
enforced here structurally: a :class:`RankedCandidate` cannot be constructed
without a non-empty ``reasons`` and a ``mutation_summary``. There is no code path
that yields a bare number, because the number and its justification are the same
object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..core.exceptions import ValidationError


@dataclass(frozen=True)
class NearestPeptide:
    """A literature/known peptide close to a candidate."""

    sequence: str
    similarity: float
    peptide_id: str | None = None
    dataset: str | None = None


@dataclass(frozen=True)
class RankedCandidate:
    """A candidate with its overall score and a mandatory full explanation.

    Every field an experimentalist needs to triage a candidate is present:
    the score, the similarity, how confident the model is, what mutations
    produced it, its nearest literature peptides, why it was selected, its
    predicted strengths, and its potential weaknesses.
    """

    sequence: str
    overall_score: float
    similarity_score: float
    confidence: float
    mutation_summary: str
    reasons: tuple[str, ...]
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    nearest_literature: tuple[NearestPeptide, ...]
    components: Mapping[str, float]
    peptide_id: str | None = None
    cpp_probability: float | None = None
    uncertainty: float | None = None
    epistemic_std: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The structural "no score without explanation" guarantee.
        if not self.reasons:
            raise ValidationError(
                "RankedCandidate requires at least one reason — a score may "
                "never be returned without an explanation."
            )
        if not self.mutation_summary:
            raise ValidationError("RankedCandidate requires a mutation_summary.")
        if not self.weaknesses:
            raise ValidationError(
                "RankedCandidate must list weaknesses (at minimum the standing "
                "reminder that predictions require experimental validation)."
            )
