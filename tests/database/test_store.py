"""Tests for cpp_ai.database.store (both backends via parametrization)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from cpp_ai.core.schema import Peptide
from cpp_ai.database.store import (
    InMemoryPeptideStore,
    PeptideStore,
    SqlitePeptideStore,
)

StoreFactory = Callable[[], PeptideStore]

_FACTORIES: list[StoreFactory] = [
    InMemoryPeptideStore,
    lambda: SqlitePeptideStore(":memory:"),
]


def _pep(seq: str, dataset: str, original_id: str, file_hash: str = "h0") -> Peptide:
    return Peptide.from_sequence(
        seq, dataset=dataset, original_id=original_id, file_sha256=file_hash
    )


@pytest.fixture(params=_FACTORIES, ids=["memory", "sqlite"])
def store(request: pytest.FixtureRequest) -> PeptideStore:
    return request.param()


def test_add_new_returns_true(store: PeptideStore) -> None:
    assert store.add(_pep("KWKLFKKI", "CPPsite3", "1")) is True
    assert len(store) == 1


def test_add_duplicate_peptide_id_returns_false(store: PeptideStore) -> None:
    store.add(_pep("KWKLFKKI", "CPPsite3", "1"))
    # same sequence, different source -> not a new peptide
    assert store.add(_pep("kwklfkki", "POSEIDON", "9")) is False
    assert len(store) == 1


def test_multi_source_provenance_merges(store: PeptideStore) -> None:
    store.add(_pep("KWKLFKKI", "CPPsite3", "1"))
    store.add(_pep("KWKLFKKI", "POSEIDON", "9"))
    pid = next(iter(store)).peptide_id
    stored = store.get(pid)
    assert stored is not None
    assert len(stored.sources) == 2
    assert stored.datasets == ("CPPsite3", "POSEIDON")


def test_reimport_same_source_is_idempotent(store: PeptideStore) -> None:
    p = _pep("KWKLFKKI", "CPPsite3", "1", file_hash="abc")
    store.add(p)
    store.add(_pep("KWKLFKKI", "CPPsite3", "1", file_hash="abc"))
    pid = next(iter(store)).peptide_id
    stored = store.get(pid)
    assert stored is not None
    assert len(stored.sources) == 1  # not duplicated


def test_add_many_stats(store: PeptideStore) -> None:
    peps = [
        _pep("KWKLFKKI", "CPPsite3", "1"),
        _pep("KWKLFKKI", "POSEIDON", "9"),   # merged source
        _pep("LLIILRRK", "CPPsite3", "2"),
        _pep("KWKLFKKI", "CPPsite3", "1"),   # duplicate source
    ]
    stats = store.add_many(peps)
    assert stats.n_new_peptides == 2
    assert stats.n_sources_added == 3
    assert stats.n_duplicate_sources == 1


def test_get_missing_returns_none(store: PeptideStore) -> None:
    assert store.get("pep_doesnotexist0") is None


def test_contains(store: PeptideStore) -> None:
    p = _pep("KWKLFKKI", "CPPsite3", "1")
    store.add(p)
    assert p.peptide_id in store
    assert "pep_nope" not in store


def test_iteration_order_is_insertion_order(store: PeptideStore) -> None:
    store.add(_pep("KWKLFKKI", "CPPsite3", "1"))
    store.add(_pep("LLIILRRK", "CPPsite3", "2"))
    store.add(_pep("RRRRRRRR", "CPPsite3", "3"))
    seqs = [s.sequence for s in store]
    assert seqs == ["KWKLFKKI", "LLIILRRK", "RRRRRRRR"]


def test_stored_peptide_properties(store: PeptideStore) -> None:
    store.add(_pep("KWXLFKKI", "CPPsite3", "1"))  # X = non-canonical
    stored = next(iter(store))
    assert stored.length == 8
    assert stored.is_canonical is False


def test_sqlite_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "peptides.db"
    s1 = SqlitePeptideStore(db)
    s1.add(_pep("KWKLFKKI", "CPPsite3", "1"))
    s1.close()

    s2 = SqlitePeptideStore(db)
    assert len(s2) == 1
    stored = next(iter(s2))
    assert stored.sequence == "KWKLFKKI"
    assert stored.sources[0].provenance.dataset == "CPPsite3"
    s2.close()
