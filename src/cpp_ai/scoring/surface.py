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

from ..generation import net_charge

_RISE_SLOPE = 1.2
_RISE_MID = 2.5   # adsorption climbs through +2..+4
_FALL_SLOPE = 1.0
_FALL_MID = 8.5   # toxicity taper begins above ~+7
_FLOOR = 0.08     # exploratory floor for neutral/negative peptides


def charge_adsorption(charge: int) -> float:
    """Adsorption likelihood in [0, 1] from net charge alone (the core term)."""
    rise = 1.0 / (1.0 + math.exp(-_RISE_SLOPE * (charge - _RISE_MID)))
    fall = 1.0 / (1.0 + math.exp(_FALL_SLOPE * (charge - _FALL_MID)))
    return max(_FLOOR, rise * fall)


def surface_interaction_prior(sequence: str) -> float:
    """Likelihood the peptide adsorbs onto a negatively-charged algal surface.

    Driven by net charge (the dominant electrostatic term). Peaks in the ~+4..+6
    window; small floor for neutral/negative peptides (exploratory, not zero).
    """
    return charge_adsorption(net_charge(sequence))
