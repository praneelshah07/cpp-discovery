"""Tests for the RankingEngine."""

from __future__ import annotations

from typing import Sequence

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.schema import Peptide
from cpp_ai.prediction.base import PredictionResult
from cpp_ai.ranking import RankedCandidate, RankingEngine

_PVEC = "LLIILRRRIRKQAHAHSK"


def _p(seq: str, dataset: str = "gen") -> Peptide:
    return Peptide.from_sequence(seq, dataset=dataset)


class _StubClassifier:
    """Duck-typed classifier: probability rises with net positive charge."""

    def predict(self, peptides: Sequence[Peptide]) -> list[PredictionResult]:
        results = []
        for p in peptides:
            charge = sum(p.sequence.count(a) for a in "KR") - sum(p.sequence.count(a) for a in "DE")
            prob = min(0.99, max(0.01, 0.5 + 0.05 * charge))
            results.append(
                PredictionResult.from_probability(
                    sequence=p.sequence, probability=prob, model_name="stub", epistemic_std=0.1
                )
            )
        return results


def _candidates() -> list[Peptide]:
    return [_p("LLIILRRRIRKQAHAHSR"), _p("LLIILRRRIRKQAHAHSE"), _p("LLIILKRRIRKQAHAHSK")]


def test_requires_a_scoring_source() -> None:
    with pytest.raises(ValidationError):
        RankingEngine()


def test_rank_returns_ranked_candidates() -> None:
    engine = RankingEngine(classifier=_StubClassifier(), reference=_p(_PVEC, "lit"))
    ranked = engine.rank(_candidates())
    assert all(isinstance(c, RankedCandidate) for c in ranked)
    assert len(ranked) == 3


def test_sorted_by_overall_descending() -> None:
    engine = RankingEngine(classifier=_StubClassifier(), reference=_p(_PVEC, "lit"))
    ranked = engine.rank(_candidates())
    scores = [c.overall_score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_top_k() -> None:
    engine = RankingEngine(classifier=_StubClassifier(), reference=_p(_PVEC, "lit"))
    assert len(engine.rank(_candidates(), top_k=2)) == 2


def test_empty_candidates() -> None:
    engine = RankingEngine(reference=_p(_PVEC, "lit"))
    assert engine.rank([]) == []


def test_reference_only_mode_no_classifier() -> None:
    engine = RankingEngine(reference=_p(_PVEC, "lit"))
    ranked = engine.rank(_candidates())
    assert all("similarity" in c.components for c in ranked)
    assert all("cpp_likelihood" not in c.components for c in ranked)
    assert all(c.reasons for c in ranked)


def test_classifier_only_mode_no_reference() -> None:
    engine = RankingEngine(classifier=_StubClassifier())
    ranked = engine.rank(_candidates())
    assert all("cpp_likelihood" in c.components for c in ranked)
    assert all(c.cpp_probability is not None for c in ranked)


def test_mutation_summary_from_metadata() -> None:
    variant = Peptide.from_sequence(
        "LLIILRRRIRKQAHAHSR",
        dataset="generated",
        metadata={"parent_id": "pep_parent", "mutations": ["K18R"]},
    )
    engine = RankingEngine(reference=_p(_PVEC, "lit"))
    c = engine.rank([variant])[0]
    assert "K18R" in c.mutation_summary
    assert "pep_parent" in c.mutation_summary


def test_mutation_summary_diffed_against_reference() -> None:
    engine = RankingEngine(reference=_p(_PVEC, "lit"))
    c = engine.rank([_p("LLIILRRRIRKQAHAHSR")])[0]  # SK -> SR at last position
    assert "substitution" in c.mutation_summary


def test_nearest_literature_sorted() -> None:
    lit = [_p(_PVEC, "lit"), _p("GRKKRRQRRRPPQ", "lit"), _p("RQIKIWFQNRRMKWKK", "lit")]
    engine = RankingEngine(reference=_p(_PVEC, "lit"), literature=lit, n_nearest=2)
    c = engine.rank([_p("LLIILRRRIRKQAHAHSR")])[0]
    assert len(c.nearest_literature) == 2
    sims = [n.similarity for n in c.nearest_literature]
    assert sims == sorted(sims, reverse=True)


def test_non_canonical_candidate_flagged_without_crashing() -> None:
    engine = RankingEngine(reference=_p(_PVEC, "lit"))
    c = engine.rank([_p("LLXILRRRIRKQAHAHSK")])[0]
    assert any("non-canonical" in w for w in c.weaknesses)


def test_every_candidate_has_full_explanation() -> None:
    engine = RankingEngine(classifier=_StubClassifier(), reference=_p(_PVEC, "lit"))
    for c in engine.rank(_candidates()):
        assert c.reasons and c.mutation_summary and c.weaknesses
