"""Tests for the ledger-derived context (algae) fitness scorer."""

from __future__ import annotations

from cpp_ai.evidence.seed import build_seed_ledger
from cpp_ai.scoring import AlgaeFitScorer, EvidenceScorer
from cpp_ai.screening.candidate import ScreenCandidate

# A small stand-in library spanning the winner/loser physicochemistry.
_LIB_SEQS = [
    "LLIILRRRIRKQAHAHSK",  # pVEC (amphipathic winner)
    "LLIILARRIRKQAHAHSK",  # pVEC-R6A
    "RRRRRRRRR",  # R9 (pure cationic loser)
    "YGRKKRRQRRR",  # TAT
    "RQIKIWFQNRRMKWKK",  # penetratin
    "GWTLNSAGYLLGKINLKALAALAKKIL",  # transportan
    "KLALKLALKALKAALKLA",  # MAP
]


def _fit() -> AlgaeFitScorer:
    return AlgaeFitScorer.from_ledger(build_seed_ledger(), _LIB_SEQS)


def test_scorer_is_informative_from_seed() -> None:
    fit = _fit()
    assert fit.is_informative
    assert fit.context == "algae"
    # amphipathicity should be a preferred-higher term
    amph = [t for t in fit.terms if t.descriptor == "hydrophobic_moment_alpha"]
    assert amph and amph[0].weight > 0


def test_pure_cationic_charge_is_penalized() -> None:
    fit = _fit()
    cat = [t for t in fit.terms if t.descriptor == "frac_group_cationic"]
    assert cat and cat[0].weight < 0  # more cationic -> worse in algae


def test_known_winners_beat_known_losers() -> None:
    fit = _fit()
    pvec = fit.score("LLIILRRRIRKQAHAHSK")
    pvec_r6a = fit.score("LLIILARRIRKQAHAHSK")
    r9 = fit.score("RRRRRRRRR")
    tat = fit.score("YGRKKRRQRRR")
    assert pvec > 0.7
    assert pvec_r6a >= pvec  # dropping a charge should not hurt algae fit
    assert r9 < 0.2
    assert tat < 0.3


def test_score_bounded_and_neutral_without_terms() -> None:
    fit = _fit()
    for seq in _LIB_SEQS:
        assert 0.0 <= fit.score(seq) <= 1.0
    # an empty ledger yields a neutral, non-informative scorer
    from cpp_ai.evidence import EvidenceLedger

    empty = AlgaeFitScorer.from_ledger(EvidenceLedger(), _LIB_SEQS)
    assert not empty.is_informative
    assert empty.score("RRRRRRRRR") == 0.5


def test_explain_contributions_sum_toward_score() -> None:
    fit = _fit()
    contribs = fit.explain("LLIILRRRIRKQAHAHSK")
    assert contribs
    raw = sum(c.contribution for c in contribs)
    assert fit.score("LLIILRRRIRKQAHAHSK") == max(0.0, min(1.0, 0.5 + 0.5 * raw))


def test_integration_reranks_toward_algae_winners() -> None:
    lib = [ScreenCandidate(sequence=s, name=s, category="?", mechanism="?", source="cppsite3")
           for s in _LIB_SEQS]
    fit = AlgaeFitScorer.from_ledger(build_seed_ledger(), _LIB_SEQS)

    anchor = "TNVYNWFQNRRARTKRK"  # ClWOX — deliberately not pVEC
    base = EvidenceScorer(lib).profile(anchor)
    algae = EvidenceScorer(lib, algae_fit_scorer=fit).profile(anchor)

    # algae mode exposes the new axis; base mode does not
    assert all(p.algae_fit is None for p in base)
    assert all(p.algae_fit is not None for p in algae)

    # R9 should rank no better under algae mode than under the neutral baseline
    def rank(profiles: list, seq: str) -> int:
        return [p.sequence for p in profiles].index(seq)

    assert rank(algae, "RRRRRRRRR") >= rank(base, "RRRRRRRRR")
