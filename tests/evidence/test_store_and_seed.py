"""Tests for ledger persistence and the curated seed."""

from __future__ import annotations

from pathlib import Path

from cpp_ai.evidence import Citation, EvidenceEntry, EvidenceLedger
from cpp_ai.evidence.seed import build_seed_ledger


def _entry(name: str, seq: str, organism: str, outcome: str, **over: object) -> EvidenceEntry:
    base: dict[str, object] = dict(
        peptide_name=name,
        sequence=seq,
        organism=organism,
        outcome=outcome,
        citation=Citation(title="paper", doi="10.1/x"),
    )
    base.update(over)
    return EvidenceEntry(**base)  # type: ignore[arg-type]


def test_round_trip_preserves_entries(tmp_path: Path) -> None:
    led = EvidenceLedger(
        [
            _entry("R9", "RRRRRRRRR", "mammalian", "success"),
            _entry("pVEC", "LLIILRRRIRKQAHAHSK", "algae", "success"),
        ]
    )
    path = led.save(tmp_path / "led.json")
    back = EvidenceLedger.load(path)
    assert len(back) == 2
    assert {e.peptide_name for e in back} == {"R9", "pVEC"}


def test_serialization_is_deterministic(tmp_path: Path) -> None:
    led = build_seed_ledger()
    a = led.save(tmp_path / "a.json").read_text()
    # reload and re-save: byte-identical (stable sort, no churn)
    b = EvidenceLedger.load(tmp_path / "a.json").save(tmp_path / "b.json").read_text()
    assert a == b


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    assert len(EvidenceLedger.load(tmp_path / "nope.json")) == 0


def test_queries() -> None:
    led = EvidenceLedger(
        [
            _entry("R9", "RRRRRRRRR", "mammalian", "success"),
            _entry("bad", "AAAA", "algae", "fail"),
            _entry("tox", "KKKK", "algae", "success", toxicity="toxic"),
        ]
    )
    assert led.for_organism("algae").unique_peptides() == 2
    assert {e.peptide_name for e in led.positives()} == {"R9"}
    assert {e.peptide_name for e in led.negatives()} == {"bad", "tox"}


def test_to_dataframe_flattens_citation() -> None:
    led = build_seed_ledger()
    df = led.to_dataframe()
    assert "doi" in df.columns and "citation_title" in df.columns
    assert df["doi"].notna().all()  # every seed row is cited


def test_seed_integrity() -> None:
    led = build_seed_ledger()
    assert len(led) >= 15
    # every entry carries a resolvable citation (verified web pulls only)
    assert all(e.citation.resolvable() for e in led)
    # the context reversal is present: R9 is a mammalian success but an algae failure
    # (Kang 2017: R9 penetrates but does not deliver protein into Chlamydomonas)
    r9 = [e for e in led if e.peptide_name == "R9"]
    by_org = {e.organism: e.outcome for e in r9}
    assert by_org.get("mammalian") == "success"
    assert by_org.get("algae") == "fail"
