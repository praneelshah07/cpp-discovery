"""Graded safety penalties (replacing the hard charge gate).

Per the review, dropping every peptide outside +4…+7 is too aggressive — it
discards exactly the gently-charged (+3) candidates worth exploring. Instead we
use smooth risk functions in [0, 1] (0 = no concern, 1 = high concern) that
*down-weight* rather than exclude, so the toxicity axis becomes a scored penalty
the user can see.

Charge risk is shaped to the review's guidance:
+3 → small penalty · +4…+7 → preferred (no penalty) · +8 → moderate · +10 → strong.
Membrane-lytic mechanism adds an additive bump. Charge alone does not determine
membrane toxicity, so this is a heuristic prior, not a hemolysis prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_PREFERRED_LO = 4
_PREFERRED_HI = 7
_HI_CENTER = 8.9   # sigmoid center above the preferred window
_HI_SLOPE = 1.0
_LO_SLOPE = 0.12   # gentle penalty below the window
_LO_CAP = 0.5
_LYTIC_BUMP = 0.3

# Membrane-lysis heuristic weights. Grounded in the pVEC-vs-TP10 contrast:
# amphipathicity (µH) is ~equal for both, so it cannot separate them — the real
# discriminator is *net* hydrophobicity (GRAVY) and lack of polar buffering.
# pVEC is net hydrophilic (GRAVY < 0) and buffered; lytic AMP-like peptides
# (TP10, MAP) are net hydrophobic (GRAVY ~ +1) with almost no polar content.
_LYS_W_GRAVY = 0.9
_LYS_W_MUH = 0.8
_LYS_MUH_FLOOR = 0.4
_LYS_W_POLAR = 2.0
_LYS_SLOPE = 3.0
_LYS_CENTER = 0.3


def charge_risk(charge: int) -> float:
    """Graded charge-toxicity risk in [0, 1] (0 = safe)."""
    if _PREFERRED_LO <= charge <= _PREFERRED_HI:
        return 0.0
    if charge > _PREFERRED_HI:
        return 1.0 / (1.0 + math.exp(-_HI_SLOPE * (charge - _HI_CENTER)))
    return min(_LO_CAP, _LO_SLOPE * (_PREFERRED_LO - charge))


def membrane_lysis_risk(sequence: str) -> float:
    """Heuristic membrane-lysis (AMP-like) risk in [0, 1] (0 = gentle).

    A peptide that is **net hydrophobic and amphipathic with little polar
    buffering** behaves like a lytic antimicrobial peptide — it perturbs
    membranes rather than translocating gently. This separates the toxic
    TP10/MAP family (high) from pVEC (low), which gross amphipathicity alone
    cannot. It is an interpretable prior, **not** a measured hemolysis value.
    """
    from ..core.types import is_canonical_sequence
    from ..descriptors import compute_descriptors

    if not is_canonical_sequence(sequence):
        return 0.0
    d = compute_descriptors(
        sequence, blocks=("biopython_props", "hydrophobic_moment", "composition")
    ).values
    gravy = d["gravy_kyte_doolittle"]
    muh = d["hydrophobic_moment_alpha"]
    polar = d["frac_group_polar_uncharged"]

    raw = (
        _LYS_W_GRAVY * max(0.0, gravy)
        + _LYS_W_MUH * max(0.0, muh - _LYS_MUH_FLOOR)
        - _LYS_W_POLAR * polar
    )
    return 1.0 / (1.0 + math.exp(-_LYS_SLOPE * (raw - _LYS_CENTER)))


@dataclass(frozen=True)
class SafetyAssessment:
    """Graded safety signals for a candidate."""

    charge_risk: float
    lytic: bool
    overall_risk: float   # combined, clamped to [0, 1]
    safety_factor: float  # 1 - overall_risk, used to down-weight the shortlist


def assess_safety(charge: int, lytic: bool) -> SafetyAssessment:
    """Combine graded charge risk with a membrane-lytic bump."""
    cr = charge_risk(charge)
    overall = min(1.0, cr + (_LYTIC_BUMP if lytic else 0.0))
    return SafetyAssessment(
        charge_risk=cr, lytic=lytic, overall_risk=overall, safety_factor=1.0 - overall
    )
