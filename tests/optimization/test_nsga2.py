"""Tests for the NSGA-II optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.schema import Peptide
from cpp_ai.generation import MutationConstraints, net_charge
from cpp_ai.optimization import (
    ExpressionCompatibilityObjective,
    NetChargeObjective,
    NoveltyObjective,
    NSGA2Config,
    NSGA2Optimizer,
    non_dominated_sort,
)

_PVEC = "LLIILRRRIRKQAHAHSK"


def _ref() -> Peptide:
    return Peptide.from_sequence(_PVEC, dataset="ref")


def _objectives():
    return [
        NetChargeObjective(target=6),
        NoveltyObjective([_PVEC]),
        ExpressionCompatibilityObjective(),
    ]


def _small_optimizer(constraints=None, seed=0) -> NSGA2Optimizer:
    return NSGA2Optimizer(
        _objectives(),
        strategy="conservative",
        constraints=constraints or MutationConstraints(),
        config=NSGA2Config(population_size=12, n_generations=4, seed=seed),
    )


def test_optimize_returns_nonempty_front() -> None:
    result = _small_optimizer().optimize([_ref()])
    assert len(result.front) > 0
    assert result.objective_names == ("net_charge", "novelty", "expression_compat")


def test_front_sequences_are_unique() -> None:
    result = _small_optimizer().optimize([_ref()])
    seqs = [s.peptide.sequence for s in result.front]
    assert len(seqs) == len(set(seqs))


def test_front_is_mutually_non_dominated() -> None:
    opt = _small_optimizer()
    result = opt.optimize([_ref()])
    # rebuild the minimization matrix for the front and confirm one front only
    F = np.column_stack([o.as_minimization([s.peptide for s in result.front])
                         for o in _objectives()])
    fronts = non_dominated_sort(F)
    assert len(fronts[0]) == len(result.front)


def test_constraints_respected_in_front() -> None:
    c = MutationConstraints(locked_motifs=("LLII",), max_charge=8, canonical_only=True)
    result = _small_optimizer(constraints=c).optimize([_ref()])
    for s in result.front:
        assert s.peptide.sequence[:4] == "LLII"
        assert net_charge(s.peptide.sequence) <= 8
        assert s.peptide.is_canonical


def test_determinism() -> None:
    a = _small_optimizer(seed=5).optimize([_ref()])
    b = _small_optimizer(seed=5).optimize([_ref()])
    assert {s.peptide.sequence for s in a.front} == {s.peptide.sequence for s in b.front}


def test_requires_objectives() -> None:
    with pytest.raises(ValidationError):
        NSGA2Optimizer([])


def test_requires_canonical_seed() -> None:
    opt = _small_optimizer()
    with pytest.raises(ValidationError):
        opt.optimize([])


def test_single_objective_optimizes_toward_target() -> None:
    opt = NSGA2Optimizer(
        [NetChargeObjective(target=3)],
        strategy="all_canonical",
        config=NSGA2Config(population_size=16, n_generations=8, seed=1),
    )
    result = opt.optimize([_ref()])
    best = min(abs(net_charge(s.peptide.sequence) - 3) for s in result.front)
    assert best <= 1  # gets at/near the target charge


def test_solution_carries_all_objective_values() -> None:
    result = _small_optimizer().optimize([_ref()])
    s = result.front[0]
    assert set(s.objective_values) == {"net_charge", "novelty", "expression_compat"}
