"""Validation: the scoring model must rank known algae outcomes correctly.

Per the evidence review, the tiny ledger is used to *validate* the model, not to
fit its weights. These are the ground-truth orderings the literature-weighted
usable-delivery score must reproduce.
"""

from __future__ import annotations

from cpp_ai.evidence.seed import build_seed_ledger
from cpp_ai.pipeline import usable_delivery
from cpp_ai.scoring import AlgaeFitScorer, EvidenceScorer
from cpp_ai.screening.candidate import ScreenCandidate

# Sequences with a known algae outcome (Kang 2017 / Suresh 2013 / thesis).
_WINNERS = {
    "pVEC": "LLIILRRRIRKQAHAHSK",       # top algae protein carrier
    "pVEC-R6A": "LLIILARRIRKQAHAHSK",   # the lab's construct
}
_LOSERS = {
    "R9": "RRRRRRRRR",                        # penetrates, fails protein delivery
    "TP10": "AGYLLGKINLKALAALAKKIL",          # lytic amphipath (transportan family)
    "MAP": "KLALKLALKALKAALKLA",              # lytic amphipath
}


def _scores() -> dict[str, float]:
    seqs = {**_WINNERS, **_LOSERS}
    lib = [ScreenCandidate(sequence=s, name=n) for n, s in seqs.items()]
    fit = AlgaeFitScorer.from_ledger(build_seed_ledger(), [c.sequence for c in lib])
    profiles = EvidenceScorer(lib, algae_fit_scorer=fit).profile(_WINNERS["pVEC"])
    return {p.name: usable_delivery(p) for p in profiles}


def test_winners_outrank_losers() -> None:
    s = _scores()
    worst_winner = min(s[n] for n in _WINNERS)
    best_loser = max(s[n] for n in _LOSERS)
    assert worst_winner > best_loser, (
        f"a known algae loser out-scored a winner: {s}"
    )


def test_pure_cationic_r9_ranks_low() -> None:
    s = _scores()
    assert s["R9"] < s["pVEC"]  # R9 penetrates but fails protein delivery in algae


def test_lytic_amphipaths_rank_low() -> None:
    s = _scores()
    assert s["TP10"] < s["pVEC"]
    assert s["MAP"] < s["pVEC"]
