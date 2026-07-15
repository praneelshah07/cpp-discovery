"""Hydrophobicity across every published scale.

Rationale
---------
There is no single "correct" hydrophobicity scale — dozens exist (Kyte-Doolittle,
Eisenberg, Hopp-Woods, ...), each derived from a different physical experiment
(partitioning, vapour transfer, buried-surface statistics). They *disagree*, and
that disagreement is exactly what lets us tell seemingly similar peptides apart:
two sequences with near-identical Kyte-Doolittle values can diverge sharply on,
say, the Wimley-White interface scale. Computing all of them turns "hydrophobicity"
into a rich, multi-scale fingerprint for downstream feature selection to mine.
"""

from __future__ import annotations

from typing import Mapping

import peptides
from peptides import Peptide as PeptidesPeptide

from .base import register_descriptor

# All hydrophobicity scales shipped with `peptides` (≈45), keyed by scale name.
_SCALE_NAMES: tuple[str, ...] = tuple(sorted(peptides.tables.HYDROPHOBICITY))


@register_descriptor("hydrophobicity_scales")
def hydrophobicity_scales(sequence: str) -> Mapping[str, float]:
    """Average hydrophobicity of the sequence under every available scale."""
    pep = PeptidesPeptide(sequence)
    return {
        f"hydrophobicity_{scale}": pep.hydrophobicity(scale=scale)
        for scale in _SCALE_NAMES
    }
