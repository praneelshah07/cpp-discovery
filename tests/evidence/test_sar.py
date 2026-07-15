"""Tests for the SAR trend-analysis layer."""

from __future__ import annotations

from cpp_ai.evidence import Citation, EvidenceEntry, EvidenceLedger, analyze_context
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


def test_winners_and_others_split_by_outcome() -> None:
    led = build_seed_ledger()
    rep = analyze_context(led, "algae")
    assert "pVEC" in rep.winners
    assert "R9" in rep.others  # the context reversal, quantified
    assert rep.n_winners >= 1 and rep.n_others >= 1


def test_toxic_success_is_not_a_winner() -> None:
    led = EvidenceLedger(
        [
            _entry("clean", "LLIILRRRIRKQAHAHSK", "algae", "success"),
            _entry("toxwin", "KLALKLALKALKAALKLA", "algae", "success", toxicity="toxic"),
        ]
    )
    rep = analyze_context(led, "algae")
    assert "clean" in rep.winners
    assert "toxwin" in rep.others  # entered cells but toxic -> demoted


def test_contrast_direction_and_effect() -> None:
    led = build_seed_ledger()
    rep = analyze_context(led, "algae")
    top = rep.top_contrasts(3)
    assert top, "expected at least one contrast"
    # effect sizes are sorted by magnitude
    mags = [abs(c.effect) for c in top]
    assert mags == sorted(mags, reverse=True)


def test_underpowered_flag() -> None:
    led = EvidenceLedger(
        [
            _entry("a", "LLIILRRRIRKQAHAHSK", "algae", "success"),
            _entry("b", "RRRRRRRRR", "algae", "fail"),
        ]
    )
    rep = analyze_context(led, "algae")
    assert rep.is_underpowered  # 1 vs 1
    assert "winners" in rep.to_text().lower()


def test_no_contrast_when_all_winners() -> None:
    led = EvidenceLedger(
        [
            _entry("a", "LLIILRRRIRKQAHAHSK", "algae", "success"),
            _entry("b", "RRRRRRRRR", "algae", "success"),
        ]
    )
    rep = analyze_context(led, "algae")
    assert rep.n_others == 0
    assert rep.contrasts == []
