"""The ranking engine: assemble evidence into explained, ranked candidates.

Ties the platform together. For each candidate it gathers predictions (Phase 5),
similarity to a reference and to the literature (Phase 4), descriptors (Phase 2),
and mutation provenance (Phases 6-7), then emits a :class:`RankedCandidate` whose
score is inseparable from its explanation.

Scoring sources are optional and composable: provide a ``classifier`` for CPP
likelihood, a ``reference`` for similarity, or both. Weights over the available
components are user-adjustable.
"""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from ..core.exceptions import ValidationError
from ..core.schema import Peptide
from ..descriptors import compute_descriptors
from ..generation.constraints import net_charge
from ..similarity.composite import CompositeSimilarity
from ..similarity.features import PeptideFeatures
from ..similarity.metrics import SequenceIdentity
from .candidate import NearestPeptide, RankedCandidate
from .explain import Evidence, RankingThresholds, build_reasons, build_strengths, build_weaknesses

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: dict[str, float] = {"cpp_likelihood": 0.6, "similarity": 0.4}


class RankingEngine:
    """Produce an explainable ranked list of candidate peptides."""

    def __init__(
        self,
        *,
        classifier: object | None = None,
        reference: Peptide | None = None,
        literature: Sequence[Peptide] = (),
        weights: Mapping[str, float] | None = None,
        thresholds: RankingThresholds | None = None,
        similarity: CompositeSimilarity | None = None,
        n_nearest: int = 3,
    ) -> None:
        if classifier is None and reference is None:
            raise ValidationError(
                "RankingEngine needs a scoring source: a classifier, a reference, "
                "or both."
            )
        self.classifier = classifier
        self.reference = reference
        self.literature = list(literature)
        self.weights = dict(weights or _DEFAULT_WEIGHTS)
        self.thresholds = thresholds or RankingThresholds()
        # Sequence-based composite is cheap enough to score every candidate.
        self._similarity = similarity or CompositeSimilarity(
            {"sequence_identity": 1.0, "smith_waterman": 1.0, "needleman_wunsch": 1.0}
        )
        self._identity = SequenceIdentity()
        self.n_nearest = n_nearest
        self._ref_features = (
            PeptideFeatures(sequence=reference.sequence) if reference else None
        )

    def rank(
        self, candidates: Sequence[Peptide], *, top_k: int | None = None
    ) -> list[RankedCandidate]:
        """Return candidates ranked by overall score, each fully explained."""
        if not candidates:
            return []
        predictions = (
            self.classifier.predict(candidates) if self.classifier is not None else [None] * len(candidates)  # type: ignore[attr-defined]
        )

        ranked = [
            self._build_candidate(pep, pred)
            for pep, pred in zip(candidates, predictions)
        ]
        ranked.sort(key=lambda c: c.overall_score, reverse=True)
        return ranked[:top_k] if top_k is not None else ranked

    # ------------------------------------------------------------------ #
    def _build_candidate(self, pep: Peptide, prediction: object | None) -> RankedCandidate:
        similarity_score = self._similarity_to_reference(pep)
        nearest = self._nearest_literature(pep)
        evidence = self._gather_evidence(pep, prediction, similarity_score, nearest)

        components = self._components(prediction, similarity_score, nearest)
        overall = self._overall_score(components)
        confidence = self._confidence(prediction, similarity_score, nearest)

        return RankedCandidate(
            sequence=pep.sequence,
            peptide_id=pep.peptide_id,
            overall_score=overall,
            similarity_score=similarity_score if similarity_score is not None else 0.0,
            confidence=confidence,
            mutation_summary=evidence.mutation_summary,
            reasons=build_reasons(evidence, self.thresholds),
            strengths=build_strengths(evidence, self.thresholds),
            weaknesses=build_weaknesses(evidence, self.thresholds),
            nearest_literature=nearest,
            components=components,
            cpp_probability=evidence.cpp_probability,
            uncertainty=evidence.uncertainty,
            epistemic_std=evidence.epistemic_std,
            metadata=dict(pep.metadata),
        )

    def _similarity_to_reference(self, pep: Peptide) -> float | None:
        if self._ref_features is None:
            return None
        feats = PeptideFeatures(sequence=pep.sequence)
        return self._similarity.score(feats, self._ref_features).composite

    def _nearest_literature(self, pep: Peptide) -> tuple[NearestPeptide, ...]:
        if not self.literature:
            return ()
        cand = PeptideFeatures(sequence=pep.sequence)
        scored = [
            NearestPeptide(
                sequence=lit.sequence,
                similarity=self._identity(cand, PeptideFeatures(sequence=lit.sequence)),
                peptide_id=lit.peptide_id,
                dataset=lit.provenance.dataset,
            )
            for lit in self.literature
        ]
        scored.sort(key=lambda n: n.similarity, reverse=True)
        return tuple(scored[: self.n_nearest])

    def _gather_evidence(
        self,
        pep: Peptide,
        prediction: object | None,
        similarity_score: float | None,
        nearest: tuple[NearestPeptide, ...],
    ) -> Evidence:
        gravy = aggregation = moment = None
        if pep.is_canonical:
            ds = compute_descriptors(
                pep.sequence, blocks=["biopython_props", "aggregation", "hydrophobic_moment"]
            )
            gravy = ds["gravy_kyte_doolittle"]
            aggregation = ds["aggregation_peak_window_hydrophobicity"]
            moment = ds["hydrophobic_moment_alpha"]

        return Evidence(
            sequence=pep.sequence,
            net_charge=net_charge(pep.sequence),
            length=len(pep.sequence),
            is_canonical=pep.is_canonical,
            cysteine_count=pep.sequence.count("C"),
            mutation_summary=self._mutation_summary(pep),
            reference_id=self.reference.peptide_id if self.reference else None,
            similarity_score=similarity_score,
            cpp_probability=getattr(prediction, "probability", None),
            uncertainty=getattr(prediction, "predictive_entropy", None),
            epistemic_std=getattr(prediction, "epistemic_std", None),
            gravy=gravy,
            aggregation_peak=aggregation,
            hydrophobic_moment=moment,
            nearest=nearest,
        )

    def _mutation_summary(self, pep: Peptide) -> str:
        md = pep.metadata
        muts = md.get("mutations")
        parent = md.get("parent_id")
        if muts and parent:
            return f"{len(muts)} substitution(s) from {parent}: {', '.join(muts)}"
        if md.get("parents"):
            return f"Optimized variant (derived from {len(md['parents'])} parent(s))."
        if self.reference is not None and len(self.reference.sequence) == len(pep.sequence):
            diff = [
                f"{self.reference.sequence[i]}{i + 1}{pep.sequence[i]}"
                for i in range(len(pep.sequence))
                if self.reference.sequence[i] != pep.sequence[i]
            ]
            if not diff:
                return f"Identical to reference {self.reference.peptide_id}."
            return f"{len(diff)} substitution(s) from {self.reference.peptide_id}: {', '.join(diff)}"
        return "Provided sequence (no recorded parent)."

    def _components(
        self,
        prediction: object | None,
        similarity_score: float | None,
        nearest: tuple[NearestPeptide, ...],
    ) -> dict[str, float]:
        components: dict[str, float] = {}
        if prediction is not None:
            components["cpp_likelihood"] = float(getattr(prediction, "probability"))
        if similarity_score is not None:
            components["similarity"] = similarity_score
        return components

    def _overall_score(self, components: Mapping[str, float]) -> float:
        active = {k: self.weights.get(k, 0.0) for k in components}
        total = sum(active.values())
        if total <= 0:  # no configured weight for the available components
            return sum(components.values()) / len(components)
        return sum(components[k] * (active[k] / total) for k in components)

    def _confidence(
        self,
        prediction: object | None,
        similarity_score: float | None,
        nearest: tuple[NearestPeptide, ...],
    ) -> float:
        conf = getattr(prediction, "confidence", None)
        if conf is not None:
            return float(conf)
        if similarity_score is not None:
            return similarity_score
        return nearest[0].similarity if nearest else 0.5
