"""Load, save, and query the curated CPP evidence ledger.

The canonical store is a single pretty-printed JSON array on disk
(``data/curated/cpp_evidence_ledger.json``), committed to git so the curated
knowledge is versioned and diffs are readable. This module is the only writer,
so it keeps the file sorted and stable to avoid churn.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from .schema import EvidenceEntry

#: Default on-disk location, relative to the repo root.
DEFAULT_LEDGER_PATH = Path("data/curated/cpp_evidence_ledger.json")


class EvidenceLedger:
    """An in-memory collection of :class:`EvidenceEntry` with disk round-tripping.

    Entries are kept in a stable order (by peptide name, then organism, then a
    hash of the citation) so serialization is deterministic and git diffs show
    only real changes.
    """

    def __init__(self, entries: Iterable[EvidenceEntry] | None = None) -> None:
        self._entries: list[EvidenceEntry] = list(entries or [])

    # -- container protocol ------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[EvidenceEntry]:
        return iter(self._entries)

    @property
    def entries(self) -> list[EvidenceEntry]:
        return list(self._entries)

    def add(self, entry: EvidenceEntry) -> None:
        self._entries.append(entry)

    # -- queries ------------------------------------------------------------ #
    def for_organism(self, organism: str) -> "EvidenceLedger":
        return EvidenceLedger(e for e in self._entries if e.organism == organism)

    def positives(self) -> "EvidenceLedger":
        return EvidenceLedger(e for e in self._entries if e.is_positive)

    def negatives(self) -> "EvidenceLedger":
        """Losers: failed to enter, or toxic."""
        return EvidenceLedger(
            e for e in self._entries if e.outcome == "fail" or e.toxicity == "toxic"
        )

    def by_peptide(self) -> dict[str, list[EvidenceEntry]]:
        """Group observations by content-addressed peptide id."""
        grouped: dict[str, list[EvidenceEntry]] = defaultdict(list)
        for e in self._entries:
            grouped[e.peptide_id].append(e)
        return dict(grouped)

    def unique_peptides(self) -> int:
        return len({e.peptide_id for e in self._entries})

    # -- ordering + serialization ------------------------------------------ #
    @staticmethod
    def _sort_key(e: EvidenceEntry) -> tuple[str, str, str]:
        return (e.peptide_name.casefold(), e.organism, e.citation.title.casefold())

    def sorted_entries(self) -> list[EvidenceEntry]:
        return sorted(self._entries, key=self._sort_key)

    def to_json(self) -> str:
        payload = [
            e.model_dump(mode="json", exclude_defaults=False)
            for e in self.sorted_entries()
        ]
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    def save(self, path: str | Path = DEFAULT_LEDGER_PATH) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path = DEFAULT_LEDGER_PATH) -> "EvidenceLedger":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        return cls(EvidenceEntry.model_validate(row) for row in raw)

    # -- analysis bridge ---------------------------------------------------- #
    def to_dataframe(self) -> "object":
        """Flat pandas DataFrame (one row per observation) for the SAR layer.

        pandas is imported lazily so importing the ledger never drags in the
        heavy webapp/analysis stack.
        """
        import pandas as pd

        rows = []
        for e in self._entries:
            d = e.model_dump(mode="json")
            cite = d.pop("citation")
            d["citation_title"] = cite.get("title")
            d["doi"] = cite.get("doi")
            d["year"] = cite.get("year")
            rows.append(d)
        return pd.DataFrame(rows)
