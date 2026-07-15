"""Aggregation-propensity proxy.

Scientific honesty note
-----------------------
Rigorous aggregation prediction (TANGO, Aggrescan, Waltz) relies on
experimentally-parameterized models that are not bundled here. Rather than
hard-code constants we cannot verify, this module provides a **transparent,
clearly-labelled proxy**: the most hydrophobic contiguous window, computed from
the validated Kyte-Doolittle scale shipped with ``peptides``.

Aggregation-prone regions (APRs) are typically short, contiguous, highly
hydrophobic stretches, so the peak windowed hydrophobicity is a reasonable
first-order indicator. It is a *heuristic*, not a validated aggregation score;
integrating a validated predictor is a planned enhancement. Treat high values
as "worth checking", never as a claim of aggregation.
"""

from __future__ import annotations

import peptides

from .base import register_descriptor

# Prefer the canonical Kyte-Doolittle scale; fall back gracefully if a given
# `peptides` release names it differently.
_KD_KEY = next(
    (k for k in peptides.tables.HYDROPHOBICITY if k.lower().replace("-", "") == "kytedoolittle"),
    sorted(peptides.tables.HYDROPHOBICITY)[0],
)
_KD: dict[str, float] = peptides.tables.HYDROPHOBICITY[_KD_KEY]
_WINDOW = 5  # typical APR length


@register_descriptor("aggregation")
def aggregation(sequence: str) -> dict[str, float]:
    """Mean and peak windowed Kyte-Doolittle hydrophobicity (APR proxy)."""
    per_residue = [_KD.get(aa, 0.0) for aa in sequence]
    mean_kd = sum(per_residue) / len(per_residue)

    window = min(_WINDOW, len(per_residue))
    window_means = [
        sum(per_residue[i : i + window]) / window
        for i in range(0, len(per_residue) - window + 1)
    ]
    peak = max(window_means) if window_means else mean_kd
    return {
        "aggregation_mean_hydrophobicity": mean_kd,
        "aggregation_peak_window_hydrophobicity": peak,
    }
