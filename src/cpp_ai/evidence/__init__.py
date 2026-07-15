"""Curated, literature-sourced CPP evidence ledger and structure-activity analysis.

This package is the platform's ground-truth memory of *what actually worked, and
where* — the empirical counterweight to the similarity recommender. See
:mod:`cpp_ai.evidence.schema` for the record type and ``docs/evidence.md`` for
the curation policy.
"""

from __future__ import annotations

from .schema import (
    CargoType,
    Citation,
    Confidence,
    EvidenceEntry,
    Mechanism,
    Organism,
    OutcomeCall,
    ToxicityCall,
)
from .sar import (
    DescriptorContrast,
    TrendReport,
    analyze_all,
    analyze_context,
)
from .store import DEFAULT_LEDGER_PATH, EvidenceLedger

__all__ = [
    "CargoType",
    "Citation",
    "Confidence",
    "EvidenceEntry",
    "Mechanism",
    "Organism",
    "OutcomeCall",
    "ToxicityCall",
    "DEFAULT_LEDGER_PATH",
    "EvidenceLedger",
    "DescriptorContrast",
    "TrendReport",
    "analyze_all",
    "analyze_context",
]
