"""Tests for the user-weighted composite similarity."""

from __future__ import annotations

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.similarity import CompositeSimilarity, SimilarityMetric
from cpp_ai.similarity.features import PeptideFeatures


class _ConstMetric(SimilarityMetric):
    """A metric returning a fixed value, for deterministic composite tests."""

    def __init__(self, name: str, value: float, requires=frozenset({"sequence"})) -> None:
        self.name = name
        self._value = value
        self.requires = requires

    def __call__(self, query: PeptideFeatures, target: PeptideFeatures) -> float:
        return self._value


def _f(seq: str = "AAAA") -> PeptideFeatures:
    return PeptideFeatures(sequence=seq)


def test_default_weights_normalized_to_one() -> None:
    comp = CompositeSimilarity()
    assert sum(comp.weights.values()) == pytest.approx(1.0)


def test_default_favors_embedding_and_descriptor() -> None:
    w = CompositeSimilarity().weights
    combined_learned = w["embedding_cosine"] + w["descriptor_similarity"]
    combined_sequence = (
        w["sequence_identity"] + w["smith_waterman"] + w["needleman_wunsch"]
    )
    assert combined_learned > combined_sequence


def test_arbitrary_ratio_weights_normalized() -> None:
    comp = CompositeSimilarity({"sequence_identity": 3, "smith_waterman": 1})
    assert comp.weights["sequence_identity"] == pytest.approx(0.75)
    assert comp.weights["smith_waterman"] == pytest.approx(0.25)


def test_zero_weight_metric_excluded() -> None:
    comp = CompositeSimilarity({"sequence_identity": 1, "smith_waterman": 0})
    assert "smith_waterman" not in comp.weights


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValidationError):
        CompositeSimilarity({"sequence_identity": -1})


def test_all_zero_weights_rejected() -> None:
    with pytest.raises(ValidationError):
        CompositeSimilarity({"sequence_identity": 0})


def test_empty_weights_rejected() -> None:
    with pytest.raises(ValidationError):
        CompositeSimilarity({})


def test_required_features_union() -> None:
    comp = CompositeSimilarity(
        {"sequence_identity": 1, "embedding_cosine": 1, "descriptor_similarity": 1}
    )
    assert comp.required_features == frozenset({"sequence", "embedding", "descriptors"})


def test_sequence_only_needs_no_heavy_features() -> None:
    comp = CompositeSimilarity({"sequence_identity": 1})
    assert comp.required_features == frozenset({"sequence"})


def test_breakdown_contributions_sum_to_composite() -> None:
    overrides = {"a": _ConstMetric("a", 0.8), "b": _ConstMetric("b", 0.4)}
    comp = CompositeSimilarity({"a": 1, "b": 3}, overrides=overrides)
    bd = comp.score(_f(), _f())
    # weighted: 0.8*0.25 + 0.4*0.75 = 0.5
    assert bd.composite == pytest.approx(0.5)
    assert sum(c.contribution for c in bd.components) == pytest.approx(bd.composite)


def test_overrides_inject_custom_metric() -> None:
    comp = CompositeSimilarity({"custom": 1}, overrides={"custom": _ConstMetric("custom", 0.9)})
    assert comp.score(_f(), _f()).composite == pytest.approx(0.9)


def test_top_contributors_orders_by_contribution() -> None:
    overrides = {"a": _ConstMetric("a", 0.1), "b": _ConstMetric("b", 0.9)}
    comp = CompositeSimilarity({"a": 1, "b": 1}, overrides=overrides)
    top = comp.score(_f(), _f()).top_contributors(1)
    assert top[0].metric == "b"


def test_out_of_range_metric_rejected() -> None:
    comp = CompositeSimilarity({"bad": 1}, overrides={"bad": _ConstMetric("bad", 1.5)})
    with pytest.raises(ValidationError):
        comp.score(_f(), _f())
