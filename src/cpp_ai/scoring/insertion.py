"""Membrane-interaction capacity — an **order-sensitive**, literature-weighted prior.

This is the renamed `insertion_fit`. Per the redesign audit (docs/redesign.md),
the previous version was only ~50 % order-sensitive: it mixed the (order-sensitive)
local hydrophobic moment with `helix_fraction` and whole-sequence GRAVY, both of
which are **composition-only** (invariant under scrambling). That let scrambled
CPPs score like the native sequence.

The rebuilt term is **order-dominant** — a compositionally-scrambled sequence
should score meaningfully lower than the native one:

* **Amphipathic patterning** — the max-local Eisenberg hydrophobic moment (µH over
  an 11-residue sliding window; order-sensitive). The segregated hydrophobic face
  that drives bilayer partitioning.
* **Hydrophobic clustering** — the longest contiguous hydrophobic run
  (order-sensitive): a coherent apolar segment to anchor into the acyl core.
* **Moderate bulk hydrophobicity** — a small GRAVY bell (composition), kept only
  as a minor capacity floor, heavily down-weighted so composition cannot dominate.

`helix_fraction` is **removed** — the audit shows it carries no residue-order
information (biopython derives it from residue fractions). Charge is absent
(modeled by `scoring.surface`); aromatics are neutral (dual-use, left to the
hemolysis/cytotoxicity terms).

**This term is necessary but NOT sufficient for productive delivery** — a peptide
that engages the membrane may still lyse it or fail to release cargo. It must not
be read as an internalization or delivery probability.
"""

from __future__ import annotations

import math

from ..core.types import is_canonical_sequence
from ..descriptors import compute_descriptors

# Weights (sum to 1). Order-sensitive terms (amphipathic patterning + clustering)
# hold 0.80 so a composition-matched scramble cannot score like the native peptide.
_W_AMPH = 0.45      # order-sensitive: max-local hydrophobic moment
_W_CLUSTER = 0.35   # order-sensitive: contiguous hydrophobic segment
_W_HYDRO = 0.20     # composition: minor moderate-hydrophobicity capacity floor

_MUH_SAT = 0.6        # µH saturation (gentle, so real µH differences still register)
_CLUSTER_REF = 6.0    # a contiguous hydrophobic run of ~6 = one+ helical turn's face
_GRAVY_OPT = 0.0
_GRAVY_SD = 1.5


def membrane_interaction_capacity(sequence: str) -> float:
    """Order-sensitive membrane-interaction capacity in [0, 1].

    Necessary-but-not-sufficient for delivery. Non-canonical sequences return 0.0.
    """
    if not is_canonical_sequence(sequence):
        return 0.0
    d = compute_descriptors(
        sequence, blocks=("biopython_props", "hydrophobic_moment", "arrangement")
    ).values
    muh = d["hydrophobic_moment_alpha"]              # max-local µH (order-sensitive)
    run = d["longest_hydrophobic_run"]               # contiguous apolar segment
    gravy = d["gravy_kyte_doolittle"]                # bulk hydrophobicity (composition)

    amph_term = min(1.0, max(0.0, muh) / _MUH_SAT)
    cluster_term = min(1.0, max(0.0, run) / _CLUSTER_REF)
    hydro_term = math.exp(-((gravy - _GRAVY_OPT) ** 2) / (2.0 * _GRAVY_SD**2))

    return _W_AMPH * amph_term + _W_CLUSTER * cluster_term + _W_HYDRO * hydro_term
