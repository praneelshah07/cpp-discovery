"""Tests for the Pareto primitives (minimization convention)."""

from __future__ import annotations

import numpy as np

from cpp_ai.optimization import (
    crowding_distance,
    dominates,
    non_dominated_sort,
    pareto_front_indices,
)


def test_dominates() -> None:
    assert dominates(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    assert dominates(np.array([1.0, 2.0]), np.array([1.0, 3.0]))  # equal + better
    assert not dominates(np.array([1.0, 2.0]), np.array([2.0, 1.0]))  # trade-off
    assert not dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))  # equal


def test_non_dominated_sort_fronts() -> None:
    F = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]])
    fronts = non_dominated_sort(F)
    assert set(fronts[0]) == {0, 1, 2}  # mutually non-dominated
    assert fronts[1] == [3]             # dominated by all of front 0


def test_single_objective_sort() -> None:
    F = np.array([[3.0], [1.0], [2.0]])
    fronts = non_dominated_sort(F)
    assert fronts[0] == [1]  # the minimum


def test_pareto_front_indices() -> None:
    F = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    assert set(pareto_front_indices(F)) == {0, 1}


def test_crowding_distance_boundaries_are_infinite() -> None:
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    cd = crowding_distance(F, [0, 1, 2])
    assert np.isinf(cd[0]) and np.isinf(cd[1])  # extremes
    assert np.isfinite(cd[2])                   # interior


def test_empty_matrix() -> None:
    assert non_dominated_sort(np.empty((0, 2))) == []
    assert pareto_front_indices(np.empty((0, 2))) == []
