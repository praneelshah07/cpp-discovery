"""Membrane-insertion propensity — a literature-weighted prior (not a fit).

This replaces the ledger-fitted `AlgaeFitScorer` in the ranking. The evidence
review (docs/scoring.md) showed the n≈6 SAR was too small to *fit* insertion
weights and had learned two directions that contradict established membrane
biophysics — an **aromaticity penalty** (wrong: Trp/aromatics are canonical
interfacial anchors that *aid* insertion) and an **aliphatic-index reward**
(a thermostability index, redundant with hydrophobicity). So insertion is now a
small, fixed, biophysically-defensible combination; the ledger is kept only to
*validate* the model, not to set its weights.

The three retained signals, each supported by universal membrane physics:

* **Amphipathicity** (Eisenberg helical hydrophobic moment µH) — the segregated
  hydrophobic face that drives partitioning into the bilayer. Saturating, because
  *more* amphipathicity mainly buys *more lysis* (handled by the separate lysis
  term), not more productive insertion.
* **Helix propensity** — folding into a helix presents that face; kept a *soft*
  contributor because the functional helix is often membrane-induced, so a
  sequence/solution prediction is only a weak proxy.
* **Moderate hydrophobicity** (GRAVY) — some bulk hydrophobicity aids insertion,
  but the relationship is non-monotonic (too hydrophilic → no insertion; too
  hydrophobic → aggregation/lysis), so it is a **bell**, not a ramp.

Charge is deliberately absent — it is modeled explicitly by `scoring.surface`.
Aromatics are neutral (no penalty, no strong reward): they aid insertion but are
dual-use (also lytic), so the lysis term is left to handle their downside.
"""

from __future__ import annotations

import math

from ..core.types import is_canonical_sequence
from ..descriptors import compute_descriptors

# Literature-weighted contributions (sum to 1). Amphipathicity is central to
# membrane entry (weighted highest); helix propensity is a soft supporter;
# hydrophobicity is a moderate, non-monotonic term.
_W_AMPH = 0.5
_W_HELIX = 0.2
_W_HYDRO = 0.3

_MUH_SAT = 0.5    # µH at which the amphipathicity term saturates (~pVEC level)
_GRAVY_OPT = 0.0  # GRAVY optimum: amphipathic peptides need not be net-hydrophobic
_GRAVY_SD = 1.5   # width of the hydrophobicity bell (gentle penalty on extremes)


def membrane_interaction_capacity(sequence: str) -> float:
    """Membrane-insertion propensity in [0, 1] (literature-weighted prior).

    Non-canonical sequences (which the descriptor stack cannot score) return 0.0.
    """
    if not is_canonical_sequence(sequence):
        return 0.0
    d = compute_descriptors(sequence, blocks=("biopython_props", "hydrophobic_moment")).values
    muh = d["hydrophobic_moment_alpha"]
    helix = d["helix_fraction"]
    gravy = d["gravy_kyte_doolittle"]

    amph_term = min(1.0, max(0.0, muh) / _MUH_SAT)      # saturating amphipathicity
    helix_term = max(0.0, min(1.0, helix))               # soft helix propensity
    hydro_term = math.exp(-((gravy - _GRAVY_OPT) ** 2) / (2.0 * _GRAVY_SD**2))  # bell

    return _W_AMPH * amph_term + _W_HELIX * helix_term + _W_HYDRO * hydro_term
