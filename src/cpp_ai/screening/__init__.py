"""Screening: the reusable logic behind the web UI.

Pure, testable functions for loading the CPP library, filtering by
toxicity/mechanism/length, ranking against an anchor peptide, diversifying, and
exporting — so the Streamlit app is a thin layer over tested code.
"""

from __future__ import annotations

from .candidate import (
    ScreenCandidate,
    charge_toxicity_flag,
    diversify,
    to_fasta,
)
from .index import ScreeningIndex
from .library import HOMEODOMAIN_MOTIF, apply_filters, load_cppsite3_library

__all__ = [
    "ScreenCandidate",
    "charge_toxicity_flag",
    "diversify",
    "to_fasta",
    "load_cppsite3_library",
    "apply_filters",
    "HOMEODOMAIN_MOTIF",
    "ScreeningIndex",
]
