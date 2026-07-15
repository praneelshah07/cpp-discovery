"""Phase 1: the database engine.

Import experimentally-curated CPP datasets (CPPsite3, POSEIDON) and arbitrary
CSV/Excel files into one standardized, provenance-preserving store. Raw fields
are never overwritten; every peptide records exactly where it came from.

Importer classes are selectable by name via :data:`IMPORTER_REGISTRY`.
"""

from __future__ import annotations

from .base import (
    IMPORTER_REGISTRY,
    ColumnMapping,
    ImportReport,
    Importer,
    SkippedRow,
)
from .provenance import file_sha256
from .sources import CPPsite3Importer, PoseidonImporter
from .store import (
    AddStats,
    InMemoryPeptideStore,
    PeptideSource,
    PeptideStore,
    SqlitePeptideStore,
    StoredPeptide,
)
from .tabular import CSVImporter, ExcelImporter

__all__ = [
    # base
    "Importer",
    "ColumnMapping",
    "ImportReport",
    "SkippedRow",
    "IMPORTER_REGISTRY",
    # provenance
    "file_sha256",
    # importers
    "CSVImporter",
    "ExcelImporter",
    "CPPsite3Importer",
    "PoseidonImporter",
    # store
    "PeptideStore",
    "InMemoryPeptideStore",
    "SqlitePeptideStore",
    "StoredPeptide",
    "PeptideSource",
    "AddStats",
]
