"""Multi-axis evidence profile — the anti-"one authoritative number" scorer.

Per the expert review, a recommendation should not collapse distinct questions
into a single score. :class:`EvidenceProfile` reports them **separately**:

* **Anchor resemblance** — block-wise physicochemical similarity (+percentile),
  local-motif similarity (Smith-Waterman), global identity, and an *optional*
  ESM-2 "protein-context" similarity (kept as supporting evidence, not half the
  score).
* **CPP plausibility** — the trained CPP-vs-non-CPP classifier's probability
  (separate from uptake strength, which we cannot predict).
* **Applicability domain** — how close the peptide is to known CPPs, with a
  confidence flag, so out-of-distribution scores are not trusted blindly.
* **Safety** — net charge, toxicity flag, membrane-lytic risk.
* **Evidence quality** — is this an experimentally-curated CPP or a computed
  sequence, and in what system.

A ``shortlist_score`` is provided only as a transparent convenience ordering;
every component that formed it is exposed alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ..core.schema import Peptide
from ..screening.candidate import ScreenCandidate
from ..similarity.features import PeptideFeatures
from ..similarity.metrics import SequenceIdentity, SmithWaterman
from .context import AlgaeFitScorer
from .physchem import BlockSimilarityIndex
from .positional import CriticalPositionProfile, critical_position_score
from .safety import assess_safety


def evidence_level(candidate: ScreenCandidate) -> str:
    """Coarse provenance/evidence label for a candidate."""
    src = (candidate.source or "").lower()
    if candidate.category == "engineered" or src in ("anchor", ""):
        return "computational only (engineered/anchor)"
    return "experimental CPP (CPPsite3; mostly mammalian, small cargo)"


@dataclass(frozen=True)
class EvidenceProfile:
    """All scoring axes for one candidate, kept separate and labelled."""

    sequence: str
    name: str
    # anchor resemblance
    physchem: float
    physchem_percentile: float
    physchem_blocks: Mapping[str, float]
    motif_local: float
    global_identity: float
    embedding: float | None
    critical_position: float | None  # scaffold mode; None if not alignable
    # context fitness (empirical, from the evidence ledger); None if not enabled
    algae_fit: float | None
    # CPP plausibility
    cpp_probability: float | None
    # applicability domain
    ad_confidence: str
    ad_nn_distance: float
    # safety (graded, not a hard gate)
    net_charge: int
    toxicity_flag: str
    lytic_risk: bool
    charge_risk: float
    safety_factor: float
    # evidence quality
    evidence: str
    # convenience ordering (transparent; components above)
    shortlist_score: float

    def to_record(self) -> dict[str, object]:
        rec: dict[str, object] = {
            "name": self.name,
            "sequence": self.sequence,
            "shortlist_score": round(self.shortlist_score, 3),
            "physchem": round(self.physchem, 3),
            "physchem_pctile": round(self.physchem_percentile * 100, 0),
            "motif_local": round(self.motif_local, 3),
            "global_identity": round(self.global_identity, 3),
        }
        if self.critical_position is not None:
            rec["critical_position"] = round(self.critical_position, 3)
        if self.algae_fit is not None:
            rec["algae_fit"] = round(self.algae_fit, 3)
        if self.embedding is not None:
            rec["embedding_context"] = round(self.embedding, 3)
        if self.cpp_probability is not None:
            rec["cpp_probability"] = round(self.cpp_probability, 3)
        rec.update({
            "ad_confidence": self.ad_confidence,
            "net_charge": self.net_charge,
            "charge_risk": round(self.charge_risk, 2),
            "toxicity_flag": self.toxicity_flag,
            "lytic_risk": self.lytic_risk,
            "evidence": self.evidence,
        })
        return rec


def _ad_confidence(distance: float, ref: npt.NDArray[np.float64]) -> str:
    """Bucket nearest-neighbor distance into a confidence label vs the library."""
    med, hi = float(np.median(ref)), float(np.quantile(ref, 0.9))
    if distance <= med:
        return "high"
    if distance <= hi:
        return "medium"
    return "low"


class EvidenceScorer:
    """Produce :class:`EvidenceProfile`s for a candidate library against an anchor."""

    def __init__(
        self,
        library: Sequence[ScreenCandidate],
        *,
        embedding_service: object | None = None,
        classifier: object | None = None,
        critical_profile: CriticalPositionProfile | None = None,
        algae_fit_scorer: "AlgaeFitScorer | None" = None,
        sigma: float = 1.0,
    ) -> None:
        self.library = list(library)
        self._block_index = BlockSimilarityIndex([c.sequence for c in self.library], sigma=sigma)
        self._sw = SmithWaterman()
        self._id = SequenceIdentity()
        self._embedding_service = embedding_service
        self._classifier = classifier
        self._critical_profile = critical_profile
        self._algae_fit_scorer = algae_fit_scorer

        # Applicability domain: nearest-neighbor distance in standardized
        # curated-descriptor space (distance to the nearest *other* known CPP).
        std = self._block_index._std_matrix
        self._nn_distances = self._nearest_other(std)

        self._emb_unit: npt.NDArray[np.float64] | None = None
        if embedding_service is not None:
            recs = embedding_service.embed_sequences([c.sequence for c in self.library])  # type: ignore[attr-defined]
            mat = embedding_service.embedding_matrix(recs).astype(np.float64)  # type: ignore[attr-defined]
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._emb_unit = mat / norms

    @staticmethod
    def _nearest_other(std: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n = std.shape[0]
        out = np.empty(n)
        for i in range(n):
            d = np.linalg.norm(std - std[i], axis=1)
            d[i] = np.inf
            out[i] = float(d.min()) if n > 1 else 0.0
        return out

    def profile(
        self,
        anchor_sequence: str,
        *,
        critical_profile: CriticalPositionProfile | None = None,
        top_k: int | None = None,
    ) -> list[EvidenceProfile]:
        crit_profile = critical_profile if critical_profile is not None else self._critical_profile
        phys = {p.sequence: p for p in self._block_index.rank(anchor_sequence)}
        anchor_feat = PeptideFeatures(sequence=anchor_sequence)

        cpp_probs: npt.NDArray[np.float64] | None = None
        if self._classifier is not None:
            peps = [Peptide.from_sequence(c.sequence, dataset="lib") for c in self.library]
            cpp_probs = np.asarray(self._classifier.predict_proba(peps), dtype=np.float64)  # type: ignore[attr-defined]

        anchor_emb_unit: npt.NDArray[np.float64] | None = None
        if self._embedding_service is not None:
            ae = self._embedding_service.embedder.embed(anchor_sequence).astype(np.float64)  # type: ignore[attr-defined]
            na = np.linalg.norm(ae)
            anchor_emb_unit = ae / na if na else ae

        profiles: list[EvidenceProfile] = []
        for i, cand in enumerate(self.library):
            p = phys[cand.sequence]
            cfeat = PeptideFeatures(sequence=cand.sequence)
            motif = self._sw(anchor_feat, cfeat)
            gid = self._id(anchor_feat, cfeat)
            emb = None
            if anchor_emb_unit is not None and self._emb_unit is not None:
                emb = float((self._emb_unit[i] @ anchor_emb_unit + 1.0) / 2.0)
            cpp = None if cpp_probs is None else float(cpp_probs[i])
            blocks = {b.name: b.similarity for b in p.blocks}
            safety = assess_safety(cand.net_charge, cand.lytic_risk)
            crit = (
                None if crit_profile is None
                else critical_position_score(crit_profile, cand.sequence)
            )
            algae_fit = (
                None if self._algae_fit_scorer is None
                else self._algae_fit_scorer.score(cand.sequence)
            )

            profiles.append(
                EvidenceProfile(
                    sequence=cand.sequence,
                    name=cand.name,
                    physchem=p.composite,
                    physchem_percentile=p.percentile,
                    physchem_blocks=blocks,
                    motif_local=motif,
                    global_identity=gid,
                    embedding=emb,
                    critical_position=crit,
                    algae_fit=algae_fit,
                    cpp_probability=cpp,
                    ad_confidence=_ad_confidence(self._nn_distances[i], self._nn_distances),
                    ad_nn_distance=float(self._nn_distances[i]),
                    net_charge=cand.net_charge,
                    toxicity_flag=cand.toxicity_flag,
                    lytic_risk=cand.lytic_risk,
                    charge_risk=safety.charge_risk,
                    safety_factor=safety.safety_factor,
                    evidence=evidence_level(cand),
                    shortlist_score=_shortlist(
                        p.composite, motif, cpp, algae_fit, safety.safety_factor
                    ),
                )
            )
        profiles.sort(key=lambda e: e.shortlist_score, reverse=True)
        return profiles[:top_k] if top_k is not None else profiles


def _shortlist(
    physchem: float,
    motif: float,
    cpp: float | None,
    algae_fit: float | None,
    safety_factor: float,
) -> float:
    """Transparent convenience ordering: mean of available resemblance/plausibility
    axes, scaled by the graded safety factor. All components are exposed.

    When context fitness (``algae_fit``) is enabled it joins the mean as an equal
    axis, so ranking shifts toward the empirical algae-winner profile without any
    single axis silently dominating."""
    parts = [physchem, motif]
    if cpp is not None:
        parts.append(cpp)
    if algae_fit is not None:
        parts.append(algae_fit)
    return float(np.mean(parts)) * safety_factor
