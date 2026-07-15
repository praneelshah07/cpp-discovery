"""Tests for optimization objectives."""

from __future__ import annotations

import pytest

from cpp_ai.core.schema import Peptide
from cpp_ai.optimization import (
    DescriptorObjective,
    Direction,
    ExpressionCompatibilityObjective,
    FunctionObjective,
    NetChargeObjective,
    NoveltyObjective,
    SimilarityToReferenceObjective,
)


def _p(seq: str) -> Peptide:
    return Peptide.from_sequence(seq, dataset="t")


def test_function_objective() -> None:
    obj = FunctionObjective("len", Direction.MINIMIZE, lambda p: len(p.sequence))
    vals = obj.evaluate([_p("KK"), _p("KKKK")])
    assert list(vals) == [2.0, 4.0]


def test_as_minimization_negates_maximize() -> None:
    obj = FunctionObjective("x", Direction.MAXIMIZE, lambda p: len(p.sequence))
    raw = obj.evaluate([_p("KKK")])
    mini = obj.as_minimization([_p("KKK")])
    assert raw[0] == 3.0 and mini[0] == -3.0


def test_as_minimization_keeps_minimize() -> None:
    obj = FunctionObjective("x", Direction.MINIMIZE, lambda p: len(p.sequence))
    assert obj.as_minimization([_p("KKK")])[0] == 3.0


def test_net_charge_raw() -> None:
    obj = NetChargeObjective(direction=Direction.MAXIMIZE)
    assert list(obj.evaluate([_p("KRKR"), _p("DDEE")])) == [4.0, -4.0]


def test_net_charge_target() -> None:
    obj = NetChargeObjective(target=2)
    # closeness to +2: KRKR (charge 4) -> -2 ; KRAA (charge 2) -> 0 (best)
    vals = obj.evaluate([_p("KRKR"), _p("KRAA")])
    assert vals[1] > vals[0]
    assert obj.direction is Direction.MAXIMIZE


def test_novelty_known_is_zero_novel_is_high() -> None:
    obj = NoveltyObjective(["AAAAAA"])
    vals = obj.evaluate([_p("AAAAAA"), _p("CDEFGH")])
    assert vals[0] == pytest.approx(0.0)
    assert vals[1] > 0.9


def test_expression_non_canonical_is_zero() -> None:
    obj = ExpressionCompatibilityObjective()
    assert obj.evaluate([_p("KWXL")])[0] == 0.0  # X non-canonical


def test_expression_penalizes_cysteine_and_runs() -> None:
    obj = ExpressionCompatibilityObjective()
    clean = obj.evaluate([_p("KWKLFKK")])[0]
    cysteine = obj.evaluate([_p("KCCCCK")])[0]
    assert clean > cysteine
    assert 0.0 < cysteine < 1.0


def test_descriptor_objective_reads_feature() -> None:
    obj = DescriptorObjective("frac_k", Direction.MAXIMIZE, "composition", "frac_K")
    assert obj.evaluate([_p("KKAA")])[0] == pytest.approx(0.5)


def test_similarity_to_reference() -> None:
    ref = _p("LLIILRRRIRKQAHAHSK")
    obj = SimilarityToReferenceObjective(ref)
    vals = obj.evaluate([ref, _p("DDDDEEEEDDDD")])
    assert vals[0] == pytest.approx(1.0)  # identical
    assert vals[1] < vals[0]              # dissimilar scores lower
