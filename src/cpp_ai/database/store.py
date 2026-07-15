"""Standardized peptide storage.

A :class:`PeptideStore` is the platform's internal home for imported peptides.
Two design commitments shape it:

* **Content-addressed identity with multi-source provenance.** The same
  sequence imported from CPPsite3 and POSEIDON collapses to one entry that
  carries *both* provenance records — nothing is overwritten.
* **Idempotent imports.** Re-importing the same file does not duplicate
  sources, because each source is deduplicated on
  ``(dataset, original_id, file_sha256)``.

Two backends are provided behind one interface: an in-memory store for tests
and small analyses, and a SQLite-backed store (standard library, no ORM) for
persistence.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from pydantic import BaseModel, ConfigDict

from ..core.schema import Peptide, ProvenanceRecord
from ..core.types import is_canonical_sequence


class PeptideSource(BaseModel):
    """One provenance record plus the verbatim source row it came from."""

    model_config = ConfigDict(frozen=True)

    provenance: ProvenanceRecord
    metadata: Mapping[str, Any]


class StoredPeptide(BaseModel):
    """A peptide as held in the store: one sequence, one or more sources."""

    model_config = ConfigDict(frozen=True)

    peptide_id: str
    sequence: str
    sources: tuple[PeptideSource, ...]

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def is_canonical(self) -> bool:
        return is_canonical_sequence(self.sequence)

    @property
    def datasets(self) -> tuple[str, ...]:
        """Distinct source datasets, in first-seen order."""
        seen: dict[str, None] = {}
        for src in self.sources:
            seen.setdefault(src.provenance.dataset, None)
        return tuple(seen)


@dataclass
class AddStats:
    """Counts returned by :meth:`PeptideStore.add_many`."""

    n_new_peptides: int = 0
    n_sources_added: int = 0
    n_duplicate_sources: int = 0


def _source_key(peptide: Peptide) -> tuple[str, str | None, str | None]:
    """Dedup key: a given (dataset, original_id, file) contributes once."""
    prov = peptide.provenance
    return (prov.dataset, prov.original_id, prov.file_sha256)


class PeptideStore(ABC):
    """Interface for a standardized peptide store."""

    @abstractmethod
    def add(self, peptide: Peptide) -> bool:
        """Add one peptide. Return ``True`` if its ``peptide_id`` was new."""

    def add_many(self, peptides: Iterable[Peptide]) -> AddStats:
        """Add many peptides, returning aggregate statistics."""
        stats = AddStats()
        for peptide in peptides:
            before_sources = self._source_count(peptide.peptide_id)
            is_new = self.add(peptide)
            after_sources = self._source_count(peptide.peptide_id)
            if is_new:
                stats.n_new_peptides += 1
            if after_sources > before_sources:
                stats.n_sources_added += 1
            else:
                stats.n_duplicate_sources += 1
        return stats

    @abstractmethod
    def _source_count(self, peptide_id: str) -> int:
        """Number of stored sources for a peptide (0 if absent)."""

    @abstractmethod
    def get(self, peptide_id: str) -> StoredPeptide | None:
        """Return the stored peptide, or ``None`` if not present."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of distinct peptides."""

    @abstractmethod
    def __iter__(self) -> Iterator[StoredPeptide]:
        """Iterate stored peptides in insertion order."""

    def __contains__(self, peptide_id: object) -> bool:
        return isinstance(peptide_id, str) and self.get(peptide_id) is not None


class InMemoryPeptideStore(PeptideStore):
    """Non-persistent store backed by dictionaries. Ideal for tests."""

    def __init__(self) -> None:
        self._sequences: dict[str, str] = {}
        self._sources: dict[str, list[PeptideSource]] = {}
        self._keys: dict[str, set[tuple[str, str | None, str | None]]] = {}

    def add(self, peptide: Peptide) -> bool:
        pid = peptide.peptide_id
        is_new = pid not in self._sequences
        if is_new:
            self._sequences[pid] = peptide.sequence
            self._sources[pid] = []
            self._keys[pid] = set()

        key = _source_key(peptide)
        if key not in self._keys[pid]:
            self._sources[pid].append(
                PeptideSource(provenance=peptide.provenance, metadata=dict(peptide.metadata))
            )
            self._keys[pid].add(key)
        return is_new

    def _source_count(self, peptide_id: str) -> int:
        return len(self._sources.get(peptide_id, ()))

    def get(self, peptide_id: str) -> StoredPeptide | None:
        if peptide_id not in self._sequences:
            return None
        return StoredPeptide(
            peptide_id=peptide_id,
            sequence=self._sequences[peptide_id],
            sources=tuple(self._sources[peptide_id]),
        )

    def __len__(self) -> int:
        return len(self._sequences)

    def __iter__(self) -> Iterator[StoredPeptide]:
        for pid in self._sequences:
            stored = self.get(pid)
            assert stored is not None
            yield stored


class SqlitePeptideStore(PeptideStore):
    """Persistent store backed by SQLite (standard library, no ORM).

    Use ``":memory:"`` for an ephemeral database. The schema keeps peptides and
    their provenance in separate tables so one peptide can carry many sources.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS peptides (
                peptide_id   TEXT PRIMARY KEY,
                sequence     TEXT NOT NULL,
                is_canonical INTEGER NOT NULL,
                length       INTEGER NOT NULL,
                insert_order INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                peptide_id   TEXT NOT NULL REFERENCES peptides(peptide_id),
                dataset      TEXT NOT NULL,
                original_id  TEXT,
                file_sha256  TEXT,
                provenance   TEXT NOT NULL,  -- ProvenanceRecord as JSON
                metadata     TEXT NOT NULL,  -- source row as JSON
                UNIQUE(peptide_id, dataset, original_id, file_sha256)
            );
            """
        )
        self._conn.commit()

    def add(self, peptide: Peptide) -> bool:
        pid = peptide.peptide_id
        cur = self._conn.cursor()
        exists = cur.execute(
            "SELECT 1 FROM peptides WHERE peptide_id = ?", (pid,)
        ).fetchone()
        is_new = exists is None
        if is_new:
            order = cur.execute("SELECT COUNT(*) FROM peptides").fetchone()[0]
            cur.execute(
                "INSERT INTO peptides (peptide_id, sequence, is_canonical, length, insert_order)"
                " VALUES (?, ?, ?, ?, ?)",
                (pid, peptide.sequence, int(peptide.is_canonical), peptide.length, order),
            )

        prov = peptide.provenance
        cur.execute(
            "INSERT OR IGNORE INTO sources"
            " (peptide_id, dataset, original_id, file_sha256, provenance, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                pid,
                prov.dataset,
                prov.original_id,
                prov.file_sha256,
                prov.model_dump_json(),
                json.dumps(_jsonable(peptide.metadata)),
            ),
        )
        self._conn.commit()
        return is_new

    def _source_count(self, peptide_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM sources WHERE peptide_id = ?", (peptide_id,)
        ).fetchone()
        return int(row[0])

    def get(self, peptide_id: str) -> StoredPeptide | None:
        row = self._conn.execute(
            "SELECT sequence FROM peptides WHERE peptide_id = ?", (peptide_id,)
        ).fetchone()
        if row is None:
            return None
        source_rows = self._conn.execute(
            "SELECT provenance, metadata FROM sources WHERE peptide_id = ? ORDER BY rowid",
            (peptide_id,),
        ).fetchall()
        sources = tuple(
            PeptideSource(
                provenance=ProvenanceRecord.model_validate_json(prov_json),
                metadata=json.loads(meta_json),
            )
            for prov_json, meta_json in source_rows
        )
        return StoredPeptide(peptide_id=peptide_id, sequence=row[0], sources=sources)

    def __len__(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM peptides").fetchone()[0])

    def __iter__(self) -> Iterator[StoredPeptide]:
        ids = self._conn.execute(
            "SELECT peptide_id FROM peptides ORDER BY insert_order"
        ).fetchall()
        for (pid,) in ids:
            stored = self.get(pid)
            assert stored is not None
            yield stored

    def close(self) -> None:
        self._conn.close()


def _jsonable(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a source row into JSON-serializable form, stringifying oddities."""
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out
