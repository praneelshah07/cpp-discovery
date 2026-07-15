"""Pareto machinery: non-dominated sorting and crowding distance.

Multi-objective CPP design has no single "best" — a more CPP-like peptide may be
more aggregation-prone; a more novel one less similar to pVEC. Instead of
collapsing objectives into one score (which hides those trade-offs), we return
the **Pareto front**: the set of candidates that cannot be improved on any
objective without sacrificing another. These are the standard NSGA-II operators.

All functions operate on a **minimization** matrix ``F`` of shape
``(n_solutions, n_objectives)``; objective adapters negate "maximize" objectives
so everything is minimized internally.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def dominates(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> bool:
    """True if ``a`` Pareto-dominates ``b`` (minimization).

    ``a`` dominates ``b`` iff it is no worse on every objective and strictly
    better on at least one.
    """
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F: npt.NDArray[np.float64]) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al.). Returns fronts of row indices.

    Front 0 is the Pareto-optimal set; front 1 is optimal once front 0 is
    removed; and so on.
    """
    n = F.shape[0]
    if n == 0:
        return []
    dominated_by: list[list[int]] = [[] for _ in range(n)]  # solutions p dominates
    domination_count = np.zeros(n, dtype=int)               # how many dominate p

    for p in range(n):
        for q in range(p + 1, n):
            if dominates(F[p], F[q]):
                dominated_by[p].append(q)
                domination_count[q] += 1
            elif dominates(F[q], F[p]):
                dominated_by[q].append(p)
                domination_count[p] += 1

    fronts: list[list[int]] = []
    current = [i for i in range(n) if domination_count[i] == 0]
    while current:
        fronts.append(current)
        nxt: list[int] = []
        for p in current:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        current = nxt
    return fronts


def crowding_distance(
    F: npt.NDArray[np.float64], indices: list[int]
) -> npt.NDArray[np.float64]:
    """Crowding distance for a set of solutions (larger = more isolated).

    Boundary solutions on each objective get infinite distance so extremes are
    always preserved. Used to keep diversity when a front must be truncated.
    """
    m = len(indices)
    distances = np.zeros(m, dtype=float)
    if m == 0:
        return distances
    sub = F[indices]
    for obj in range(F.shape[1]):
        order = np.argsort(sub[:, obj], kind="stable")
        lo, hi = sub[order[0], obj], sub[order[-1], obj]
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        span = hi - lo
        if span == 0:
            continue
        for k in range(1, m - 1):
            distances[order[k]] += (sub[order[k + 1], obj] - sub[order[k - 1], obj]) / span
    return distances


def pareto_front_indices(F: npt.NDArray[np.float64]) -> list[int]:
    """Indices of the non-dominated (first-front) solutions."""
    fronts = non_dominated_sort(F)
    return fronts[0] if fronts else []
