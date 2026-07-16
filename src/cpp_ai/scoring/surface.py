"""Surface-adsorption score — electrostatic attraction to the algal surface.

Microalgal surfaces are **negatively charged** (carboxyl / phosphate / sulfated
polysaccharide / acidic glycoprotein groups), so a peptide's first step —
adsorbing onto the surface before it can insert — depends on modest *positive*
charge. This is why the lab's successful anchors (pVEC, pVEC-R6A, ClWOX) are all
cationic, and why a net-neutral or net-negative peptide has little to grab on
with (it must fall back on hydrophobic/receptor mechanisms).

The evidence-ledger SAR could not learn this: its losers (R9, TAT) were *extreme*
polycations, so it spuriously concluded "less charge is better" and over-rewarded
neutral peptides. We therefore model charge **explicitly** here rather than
letting the (confounded) SAR handle it.

`surface_interaction_prior` is a bell in net charge, the product of two real processes:

* a **rise** with positive charge (attraction to the negative surface), and
* a **taper** above ~+8 (over-adsorption → membrane disruption / toxicity),

so the emergent sweet spot is ~**+4 to +6** — almost exactly where the algae-
proven CPPs sit. Neutral/negative peptides keep a small floor so they still
surface as *exploratory* hypotheses rather than being discarded (some CPPs do
enter via hydrophobic/receptor routes).
"""

from __future__ import annotations

import math

from ..descriptors import compute_descriptors
from ..generation import net_charge

_RISE_SLOPE = 1.2
_RISE_MID = 2.5   # adsorption climbs through +2..+4
_FALL_SLOPE = 1.0
_FALL_MID = 8.5   # toxicity taper begins above ~+7
_FLOOR = 0.08     # exploratory floor for neutral/negative peptides

# Order-sensitive local-patch weighting. Colloid electrostatics: a *clustered*
# cationic patch adsorbs to an anionic surface more strongly than the same total
# charge dispersed. Modest — total charge is still the dominant driver.
_W_CHARGE = 0.85
_W_PATCH = 0.15
_PATCH_REF = 4.0  # a contiguous basic run of ~4 = a strong cationic patch


def charge_adsorption(charge: int) -> float:
    """Adsorption likelihood in [0, 1] from net charge alone (the composition term)."""
    rise = 1.0 / (1.0 + math.exp(-_RISE_SLOPE * (charge - _RISE_MID)))
    fall = 1.0 / (1.0 + math.exp(_FALL_SLOPE * (charge - _FALL_MID)))
    return max(_FLOOR, rise * fall)


def surface_interaction_prior(sequence: str) -> float:
    """Likelihood the peptide adsorbs onto a negatively-charged algal surface.

    Blends the net-charge bell (dominant, composition) with an **order-sensitive**
    cationic-patch term (a clustered basic run adsorbs harder than dispersed
    charge). Peaks in the ~+4..+6 charge window; small floor for neutral peptides.
    """
    from ..core.types import is_canonical_sequence

    charge_term = charge_adsorption(net_charge(sequence))
    if not is_canonical_sequence(sequence):
        return charge_term  # patch term needs descriptors; fall back to charge only
    run = compute_descriptors(sequence, blocks=("arrangement",)).values["longest_basic_run"]
    patch_term = min(1.0, max(0.0, run) / _PATCH_REF)
    return _W_CHARGE * charge_term + _W_PATCH * patch_term
