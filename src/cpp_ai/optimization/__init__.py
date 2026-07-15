"""Phase 7: multi-objective optimization.

Return **Pareto-optimal** CPP candidates — the non-dominated trade-off set —
rather than collapsing competing goals into one score. A built-in, constraint-
aware NSGA-II evolves peptides using the Phase 6 substitution operators, scoring
each on a user-chosen set of objectives.

Objectives (all optional, fully user-composed):
CPP likelihood, similarity to a reference (pVEC/ClWOX), hydrophobicity /
aggregation / net charge (descriptors), novelty, expression compatibility, and
any custom `FunctionObjective` (e.g. a toxicity model you supply).
"""

from __future__ import annotations

from .nsga2 import NSGA2Config, NSGA2Optimizer, OptimizationResult, ParetoSolution
from .objectives import (
    CPPLikelihoodObjective,
    DescriptorObjective,
    Direction,
    ExpressionCompatibilityObjective,
    FunctionObjective,
    NetChargeObjective,
    NoveltyObjective,
    Objective,
    SimilarityToReferenceObjective,
)
from .pareto import (
    crowding_distance,
    dominates,
    non_dominated_sort,
    pareto_front_indices,
)

__all__ = [
    # pareto
    "dominates",
    "non_dominated_sort",
    "crowding_distance",
    "pareto_front_indices",
    # objectives
    "Direction",
    "Objective",
    "FunctionObjective",
    "CPPLikelihoodObjective",
    "DescriptorObjective",
    "NetChargeObjective",
    "NoveltyObjective",
    "ExpressionCompatibilityObjective",
    "SimilarityToReferenceObjective",
    # optimizer
    "NSGA2Optimizer",
    "NSGA2Config",
    "OptimizationResult",
    "ParetoSolution",
]
