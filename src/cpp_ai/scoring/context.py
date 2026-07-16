"""Context-specific fitness, derived from the evidence ledger's SAR trends.

The recommender's default axes (anchor resemblance, CPP plausibility) answer
"is this a CPP like my anchor?". They do **not** answer "is this the kind of CPP
that works *in algae*?" — and the evidence ledger shows those are different
questions (pure-cationic R9 wins in mammalian cells but loses to amphipathic
pVEC in *Chlamydomonas*).

:class:`AlgaeFitScorer` turns that empirical contrast into a transparent,
data-driven score. It is **fit from the ledger**, not hardcoded: it reads the
winner-vs-non-winner descriptor contrasts for the algae context and uses their
*direction* (with a capped magnitude, because the ledger is small) as weights on
library-standardized descriptors. As the ledger grows, the weights update — this
is the continuous-iteration loop.

Honesty guardrails:

* Weights are **capped** (``tanh``) so a tiny-n contrast cannot dominate.
* Only descriptors whose contrast clears ``min_effect`` are used.
* The score is neutral (0.5) when evidence is absent, and every term is
  inspectable via :meth:`AlgaeFitScorer.explain`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from ..descriptors import compute_descriptors

# Charge-related SAR descriptors excluded from the insertion model — charge is
# modeled explicitly by scoring.surface (see from_ledger for the rationale).
_CHARGE_DESCRIPTORS = frozenset({
    "charge_pH7.4_Lehninger",
    "frac_group_cationic",
    "frac_R",
    "frac_K",
})


@dataclass(frozen=True)
class FitTerm:
    """One descriptor's contribution to the context-fitness weighting."""

    descriptor: str
    weight: float  # signed, capped to (-1, 1); >0 means "higher is better here"
    winner_mean: float
    other_mean: float

    @property
    def prefers(self) -> str:
        return "higher" if self.weight > 0 else "lower"


@dataclass(frozen=True)
class FitContribution:
    """Per-descriptor breakdown of a single peptide's fitness score."""

    descriptor: str
    z: float  # library-standardized descriptor value (clipped)
    weight: float
    contribution: float  # weight * z, normalized share of the raw score


class AlgaeFitScorer:
    """Score how well a peptide matches the *empirical* algae-winner profile.

    Construct via :meth:`from_ledger`. Scores are in ``[0, 1]`` with 0.5 neutral.
    """

    def __init__(
        self,
        terms: Sequence[FitTerm],
        means: dict[str, float],
        sds: dict[str, float],
        *,
        context: str = "algae",
    ) -> None:
        self._terms = list(terms)
        self._means = means
        self._sds = sds
        self.context = context

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_ledger(
        cls,
        ledger: object,
        library_sequences: Sequence[str],
        *,
        context: str = "algae",
        min_effect: float = 0.5,
        max_terms: int = 6,
    ) -> "AlgaeFitScorer":
        """Fit weights from the ledger's winner-vs-other contrasts in ``context``.

        ``library_sequences`` provide the standardization reference so a peptide's
        descriptors are z-scored against the population it will be ranked within.
        """
        # Lazy import keeps the scoring layer from importing evidence at module load.
        from ..evidence import analyze_context  # local import: avoids layering cycle
        from ..evidence.store import EvidenceLedger

        assert isinstance(ledger, EvidenceLedger)
        report = analyze_context(ledger, context)

        # Charge is modeled *explicitly* (see scoring.surface): the tiny ledger's
        # losers (R9/TAT) were extreme polycations, so the SAR spuriously learns
        # "less charge is better" and over-rewards neutral peptides. Exclude the
        # charge descriptors here so this scorer captures only membrane *insertion*
        # (amphipathicity/hydrophobicity/shape); surface_adsorption handles charge.
        used = [
            c for c in report.contrasts
            if abs(c.effect) >= min_effect and c.descriptor not in _CHARGE_DESCRIPTORS
        ]
        used.sort(key=lambda c: abs(c.effect), reverse=True)
        used = used[:max_terms]
        descriptors = [c.descriptor for c in used]
        means, sds = _library_stats(library_sequences, descriptors)

        terms = [
            FitTerm(
                descriptor=c.descriptor,
                weight=math.tanh(c.effect / 2.0),  # cap tiny-n effects
                winner_mean=c.winner_mean,
                other_mean=c.other_mean,
            )
            for c in used
        ]
        return cls(terms, means, sds, context=context)

    @property
    def terms(self) -> list[FitTerm]:
        return list(self._terms)

    @property
    def is_informative(self) -> bool:
        return bool(self._terms)

    # -- scoring ------------------------------------------------------------ #
    def _z(self, descriptor: str, value: float) -> float:
        sd = self._sds.get(descriptor, 0.0)
        if sd <= 0:
            return 0.0
        return max(-3.0, min(3.0, (value - self._means[descriptor]) / sd))

    def explain(self, sequence: str) -> list[FitContribution]:
        if not self._terms:
            return []
        ds = compute_descriptors(sequence, blocks=None)
        total_w = sum(abs(t.weight) for t in self._terms) or 1.0
        out: list[FitContribution] = []
        for t in self._terms:
            if t.descriptor not in ds.values:
                continue
            z = self._z(t.descriptor, ds.values[t.descriptor])
            out.append(
                FitContribution(
                    descriptor=t.descriptor,
                    z=z,
                    weight=t.weight,
                    contribution=t.weight * z / total_w,
                )
            )
        return out

    def score(self, sequence: str) -> float:
        """Return algae-fit in ``[0, 1]`` (0.5 neutral) for one sequence."""
        contribs = self.explain(sequence)
        if not contribs:
            return 0.5
        raw = sum(c.contribution for c in contribs)  # ~[-1, 1]
        return max(0.0, min(1.0, 0.5 + 0.5 * raw))


def _library_stats(
    sequences: Sequence[str], descriptors: Sequence[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Mean/std of each descriptor across the (canonical) library sequences."""
    from ..core.types import is_canonical_sequence

    cols: dict[str, list[float]] = {d: [] for d in descriptors}
    for seq in sequences:
        if not is_canonical_sequence(seq):
            continue
        ds = compute_descriptors(seq, blocks=None)
        for d in descriptors:
            if d in ds.values:
                cols[d].append(ds.values[d])
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    for d, vals in cols.items():
        if vals:
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            means[d] = m
            sds[d] = math.sqrt(var)
        else:
            means[d] = 0.0
            sds[d] = 0.0
    return means, sds
