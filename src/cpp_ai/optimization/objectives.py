"""Optimization objectives.

An :class:`Objective` maps peptides to a scalar and declares whether that scalar
should be maximized or minimized. Objectives are deliberately thin adapters over
earlier phases (prediction, descriptors, similarity, generation) plus a generic
``FunctionObjective`` escape hatch, so the objective *set* is fully user-defined.

Scientific honesty
------------------
Objectives we can compute rigorously are provided (CPP likelihood, hydrophobicity,
aggregation proxy, net charge, novelty, similarity, expression heuristic).
**Toxicity is intentionally not fabricated** — there is no bundled toxicity
model. Plug a real predictor in as a ``FunctionObjective(direction=MINIMIZE)``
when you have one. The aggregation and expression objectives are labelled
heuristics, not validated assays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Sequence

import numpy as np
import numpy.typing as npt

from ..core.schema import Peptide
from ..descriptors import compute_descriptors
from ..generation.constraints import net_charge


class Direction(Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class Objective(ABC):
    """A named, directional objective over peptides."""

    name: str
    direction: Direction

    @abstractmethod
    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        """Return one raw value per peptide (higher = better iff MAXIMIZE)."""

    def as_minimization(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        """Values transformed so that lower is always better (negate maximize)."""
        values = self.evaluate(peptides)
        return -values if self.direction is Direction.MAXIMIZE else values


class FunctionObjective(Objective):
    """Wrap an arbitrary ``peptide -> float`` scorer (e.g. a toxicity model)."""

    def __init__(
        self, name: str, direction: Direction, fn: Callable[[Peptide], float]
    ) -> None:
        self.name = name
        self.direction = direction
        self._fn = fn

    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        return np.array([float(self._fn(p)) for p in peptides], dtype=float)


class CPPLikelihoodObjective(Objective):
    """Calibrated P(CPP) from a trained classifier — maximize."""

    direction = Direction.MAXIMIZE

    def __init__(self, classifier: object, name: str = "cpp_likelihood") -> None:
        self.name = name
        self._classifier = classifier

    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        # duck-typed against prediction.CPPClassifier to avoid a hard import cycle
        return np.asarray(self._classifier.predict_proba(peptides), dtype=np.float64)  # type: ignore[attr-defined]


class DescriptorObjective(Objective):
    """Optimize a single descriptor feature (e.g. GRAVY, aggregation proxy)."""

    def __init__(
        self, name: str, direction: Direction, block: str, feature: str
    ) -> None:
        self.name = name
        self.direction = direction
        self._block = block
        self._feature = feature

    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        out = np.empty(len(peptides), dtype=float)
        for i, p in enumerate(peptides):
            ds = compute_descriptors(p.sequence, blocks=[self._block])
            out[i] = ds[self._feature]
        return out


class NetChargeObjective(Objective):
    """Net charge, optionally toward a target.

    With ``target`` set, maximizes closeness to it (``-|charge - target|``);
    otherwise returns raw net charge with the given ``direction``.
    """

    def __init__(
        self,
        *,
        target: int | None = None,
        direction: Direction = Direction.MAXIMIZE,
        name: str = "net_charge",
    ) -> None:
        self.name = name
        self._target = target
        self.direction = Direction.MAXIMIZE if target is not None else direction

    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        charges = np.array([net_charge(p.sequence) for p in peptides], dtype=float)
        if self._target is None:
            return charges
        return np.asarray(-np.abs(charges - self._target), dtype=np.float64)


class NoveltyObjective(Objective):
    """Dissimilarity to a known set — maximize.

    Novelty = ``1 - max k-mer (k=2) Jaccard similarity`` to any known sequence.
    A fast, dependency-free proxy; 1.0 means no 2-mer overlap with anything known.
    """

    direction = Direction.MAXIMIZE

    def __init__(
        self, known_sequences: Sequence[str], *, k: int = 2, name: str = "novelty"
    ) -> None:
        self.name = name
        self._k = k
        self._known_kmers = [self._kmers(s) for s in known_sequences]

    def _kmers(self, seq: str) -> frozenset[str]:
        s = seq.strip().upper()
        if len(s) < self._k:
            return frozenset({s})
        return frozenset(s[i : i + self._k] for i in range(len(s) - self._k + 1))

    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        out = np.empty(len(peptides), dtype=float)
        for i, p in enumerate(peptides):
            km = self._kmers(p.sequence)
            best = 0.0
            for known in self._known_kmers:
                union = len(km | known)
                jac = len(km & known) / union if union else 0.0
                best = max(best, jac)
            out[i] = 1.0 - best
        return out


class ExpressionCompatibilityObjective(Objective):
    """Heuristic recombinant-expressibility score in [0, 1] — maximize.

    Rewards genetically-encodable (canonical), moderate-length peptides and
    penalizes cysteine load (disulfide complexity) and long homopolymer runs
    (synthesis/expression difficulty). A **heuristic**, not an expression assay.
    """

    direction = Direction.MAXIMIZE

    def __init__(
        self, *, ideal_min: int = 5, ideal_max: int = 40, name: str = "expression_compat"
    ) -> None:
        self.name = name
        self._ideal_min = ideal_min
        self._ideal_max = ideal_max

    def _score(self, seq: str) -> float:
        from ..core.types import is_canonical_sequence

        if not is_canonical_sequence(seq):
            return 0.0
        length = len(seq)
        if length < self._ideal_min or length > self._ideal_max:
            length_factor = 0.5
        else:
            length_factor = 1.0
        cys_penalty = max(0.0, 1.0 - 0.2 * seq.count("C"))
        longest_run = max((len(r) for r in _runs(seq)), default=1)
        run_penalty = 1.0 if longest_run <= 3 else 0.6
        return length_factor * cys_penalty * run_penalty

    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        return np.array([self._score(p.sequence) for p in peptides], dtype=float)


class SimilarityToReferenceObjective(Objective):
    """Composite similarity to a reference peptide (e.g. pVEC, ClWOX) — maximize.

    Defaults to a **sequence-based** composite (fast, no embeddings/descriptors)
    so it is cheap enough to evaluate across an evolutionary run. Pass a custom
    ``composite`` weighted over any registered metrics for richer notions of
    similarity (the engine-backed feature precompute is the caller's concern).
    """

    direction = Direction.MAXIMIZE

    def __init__(
        self,
        reference: Peptide,
        *,
        composite: object | None = None,
        name: str | None = None,
    ) -> None:
        from ..similarity import CompositeSimilarity
        from ..similarity.features import PeptideFeatures

        self.name = name or f"similarity_to_{reference.peptide_id}"
        self._composite = composite or CompositeSimilarity(
            {"sequence_identity": 1.0, "smith_waterman": 1.0, "needleman_wunsch": 1.0}
        )
        self._ref_features = PeptideFeatures(sequence=reference.sequence)

    def evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        from ..similarity.features import PeptideFeatures

        out = np.empty(len(peptides), dtype=float)
        for i, p in enumerate(peptides):
            feats = PeptideFeatures(sequence=p.sequence)
            out[i] = self._composite.score(feats, self._ref_features).composite  # type: ignore[attr-defined]
        return out


def _runs(seq: str) -> list[str]:
    """Split a sequence into maximal identical-character runs."""
    runs: list[str] = []
    current = ""
    for ch in seq:
        if current and ch == current[-1]:
            current += ch
        else:
            if current:
                runs.append(current)
            current = ch
    if current:
        runs.append(current)
    return runs
