"""Importer abstractions shared by every data source.

An :class:`Importer` turns an external file (CSV, Excel, a database export)
into a stream of standardized :class:`~cpp_ai.core.schema.Peptide` objects,
attaching full provenance and preserving every original field verbatim in
``metadata`` (honoring "never overwrite raw data").

All importers register themselves in :data:`IMPORTER_REGISTRY`, so a pipeline
can select a source by name from a config file.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..core.exceptions import ValidationError
from ..core.registry import Registry
from ..core.schema import Peptide
from .provenance import file_sha256

logger = logging.getLogger(__name__)

# Registry of importer classes, selectable by dataset name from config.
IMPORTER_REGISTRY: Registry[type["Importer"]] = Registry("importer")


@dataclass(frozen=True)
class ColumnMapping:
    """Which columns of a tabular source carry which standardized fields.

    Only ``sequence`` is required. ``identifier`` names the column holding the
    source database's own ID (preserved as ``ProvenanceRecord.original_id``).
    Every other column is retained verbatim in the peptide's ``metadata``.

    Column matching is case-insensitive and whitespace-insensitive so that
    "Sequence", "sequence", and " SEQUENCE " all resolve identically.
    """

    sequence: str
    identifier: str | None = None

    @staticmethod
    def _norm(name: str) -> str:
        return name.strip().casefold()

    def resolve(self, available: Sequence[str]) -> dict[str, str]:
        """Map standardized field -> actual column name present in the source.

        Raises
        ------
        ValidationError
            If the required sequence column is absent.
        """
        lookup = {self._norm(col): col for col in available}
        resolved: dict[str, str] = {}

        seq_col = lookup.get(self._norm(self.sequence))
        if seq_col is None:
            raise ValidationError(
                f"Sequence column {self.sequence!r} not found. "
                f"Available columns: {list(available)}"
            )
        resolved["sequence"] = seq_col

        if self.identifier is not None:
            id_col = lookup.get(self._norm(self.identifier))
            if id_col is not None:
                resolved["identifier"] = id_col
            else:
                logger.warning(
                    "Identifier column %r not found; original_id will be null.",
                    self.identifier,
                )
        return resolved


@dataclass(frozen=True)
class SkippedRow:
    """A source row that could not be imported, with the reason why."""

    row_index: int
    reason: str
    raw: Mapping[str, Any]


@dataclass
class ImportReport:
    """Result of importing one file: the peptides plus reproducible statistics.

    Skipped rows are reported rather than silently dropped, so a curator can
    audit exactly what was excluded and why.
    """

    dataset: str
    source_file: str
    file_sha256: str
    peptides: list[Peptide] = field(default_factory=list)
    skipped: list[SkippedRow] = field(default_factory=list)

    @property
    def n_imported(self) -> int:
        return len(self.peptides)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)

    @property
    def n_non_canonical(self) -> int:
        """Imported peptides using non-canonical residues (flagged, not dropped)."""
        return sum(1 for p in self.peptides if not p.is_canonical)

    def summary(self) -> str:
        return (
            f"{self.dataset}: imported {self.n_imported} "
            f"({self.n_non_canonical} non-canonical), skipped {self.n_skipped} "
            f"from {Path(self.source_file).name}"
        )


class Importer(ABC):
    """Base class for all data-source importers.

    Subclasses implement :meth:`_iter_raw_rows` (how to read the file) and
    provide a default :meth:`column_mapping`. The base class handles provenance,
    validation, skip-tracking, and metadata preservation uniformly.
    """

    #: Logical dataset name attached to provenance (e.g. "CPPsite3").
    dataset_name: str = "unknown"

    def __init__(self, column_mapping: ColumnMapping | None = None) -> None:
        self._column_mapping = column_mapping or self.default_column_mapping()

    @classmethod
    def default_column_mapping(cls) -> ColumnMapping:
        """The source's expected column layout; override per source."""
        return ColumnMapping(sequence="sequence", identifier="id")

    @property
    def column_mapping(self) -> ColumnMapping:
        return self._column_mapping

    @abstractmethod
    def _iter_raw_rows(self, path: Path) -> Iterator[Mapping[str, Any]]:
        """Yield each source row as an ordered mapping of column -> value."""
        raise NotImplementedError

    def import_file(self, path: str | Path) -> ImportReport:
        """Read ``path`` and produce a fully-provenanced :class:`ImportReport`."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Source file does not exist: {p}")

        digest = file_sha256(p)
        report = ImportReport(
            dataset=self.dataset_name, source_file=str(p), file_sha256=digest
        )

        resolved: dict[str, str] | None = None
        for row_index, raw in enumerate(self._iter_raw_rows(p)):
            if resolved is None:
                resolved = self.column_mapping.resolve(list(raw.keys()))

            raw_seq = raw.get(resolved["sequence"])
            if raw_seq is None or str(raw_seq).strip() == "":
                report.skipped.append(
                    SkippedRow(row_index, "empty sequence", dict(raw))
                )
                continue

            original_id = None
            if "identifier" in resolved:
                id_val = raw.get(resolved["identifier"])
                original_id = None if id_val is None else str(id_val)

            try:
                peptide = Peptide.from_sequence(
                    str(raw_seq),
                    dataset=self.dataset_name,
                    original_id=original_id,
                    source_file=str(p),
                    file_sha256=digest,
                    metadata=dict(raw),  # preserve the whole row verbatim
                )
            except Exception as exc:  # invalid sequence: record, do not crash
                report.skipped.append(
                    SkippedRow(row_index, f"invalid sequence: {exc}", dict(raw))
                )
                continue

            report.peptides.append(peptide)

        logger.info("%s", report.summary())
        return report
