"""A built-in NSGA-II optimizer over the discrete peptide design space.

Reuses Phase 6 machinery (substitution strategies + constraints) as the genetic
operators, and the Pareto primitives in :mod:`pareto`. Returns the Pareto front
of candidates — the non-dominated trade-off set — never a single "winner".

Why a custom GA (not pymoo): the search space is variable-content amino-acid
strings with domain-specific, constraint-aware mutation/crossover, which a
purpose-built loop expresses more directly (and transparently) than a
vector-oriented framework. The objective interface is backend-agnostic, so a
pymoo backend could be added later without touching objective definitions.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from ..core.schema import Peptide
from ..generation.constraints import MutationConstraints
from ..generation.substitutions import SUBSTITUTION_REGISTRY, SubstitutionStrategy
from .objectives import Objective
from .pareto import crowding_distance, non_dominated_sort, pareto_front_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParetoSolution:
    """A non-dominated candidate with its raw per-objective values."""

    peptide: Peptide
    objective_values: dict[str, float]


@dataclass
class OptimizationResult:
    """The Pareto front plus run metadata."""

    front: list[ParetoSolution]
    n_generations: int
    n_evaluated: int
    objective_names: tuple[str, ...]
    directions: dict[str, str]


@dataclass
class NSGA2Config:
    population_size: int = 50
    n_generations: int = 20
    crossover_prob: float = 0.7
    seed: int = 0
    mutate_attempts: int = 12


class NSGA2Optimizer:
    """Constraint-aware NSGA-II over peptide sequences."""

    def __init__(
        self,
        objectives: Sequence[Objective],
        *,
        strategy: str | SubstitutionStrategy = "conservative",
        constraints: MutationConstraints | None = None,
        config: NSGA2Config | None = None,
    ) -> None:
        if not objectives:
            raise ValidationError("At least one objective is required.")
        self.objectives = list(objectives)
        if isinstance(strategy, str):
            self._strategy: SubstitutionStrategy = SUBSTITUTION_REGISTRY.get(strategy)
        else:
            self._strategy = strategy
        self.constraints = constraints or MutationConstraints()
        self.config = config or NSGA2Config()
        self._cache: dict[str, npt.NDArray[np.float64]] = {}
        self._n_evaluated = 0

    # ------------------------------------------------------------------ #
    def optimize(self, seed_peptides: Sequence[Peptide]) -> OptimizationResult:
        """Evolve from ``seed_peptides`` and return the final Pareto front."""
        canonical_seeds = [p for p in seed_peptides if p.is_canonical or not self.constraints.canonical_only]
        if not canonical_seeds:
            raise ValidationError("Need at least one (canonical) seed peptide.")

        rng = random.Random(self.config.seed)
        population = self._initial_population(canonical_seeds, rng)
        F = self._evaluate(population)

        for _ in range(self.config.n_generations):
            rank, crowd = self._rank_and_crowding(F)
            offspring = self._make_offspring(population, rank, crowd, rng)
            if not offspring:
                break
            combined = population + offspring
            F_combined = self._evaluate(combined)
            population, F = self._environmental_selection(combined, F_combined)

        front_idx = pareto_front_indices(F)
        # Deduplicate the front by sequence (the population may hold several
        # distinct objects sharing one sequence).
        seen: set[str] = set()
        front_peptides: list[Peptide] = []
        for i in front_idx:
            seq = population[i].sequence
            if seq not in seen:
                seen.add(seq)
                front_peptides.append(population[i])
        solutions = self._to_solutions(front_peptides)

        return OptimizationResult(
            front=solutions,
            n_generations=self.config.n_generations,
            n_evaluated=self._n_evaluated,
            objective_names=tuple(o.name for o in self.objectives),
            directions={o.name: o.direction.value for o in self.objectives},
        )

    # ------------------------------------------------------------------ #
    def _initial_population(
        self, seeds: Sequence[Peptide], rng: random.Random
    ) -> list[Peptide]:
        population = list(seeds)
        guard = 0
        while len(population) < self.config.population_size and guard < self.config.population_size * 50:
            guard += 1
            parent = rng.choice(seeds)
            mutant = self._mutate(parent.sequence, rng)
            if mutant is not None and mutant != parent.sequence:
                population.append(self._make_peptide(mutant, [parent.sequence]))
        return population[: self.config.population_size]

    def _make_offspring(
        self,
        population: list[Peptide],
        rank: npt.NDArray[np.int_],
        crowd: npt.NDArray[np.float64],
        rng: random.Random,
    ) -> list[Peptide]:
        offspring: list[Peptide] = []
        guard = 0
        target = self.config.population_size
        while len(offspring) < target and guard < target * 50:
            guard += 1
            a = population[self._tournament(rank, crowd, rng)]
            b = population[self._tournament(rank, crowd, rng)]
            child = a.sequence
            if len(a.sequence) == len(b.sequence) and rng.random() < self.config.crossover_prob:
                child = self._crossover(a.sequence, b.sequence, rng)
            mutated = self._mutate(child, rng)
            if mutated is None:
                continue
            offspring.append(self._make_peptide(mutated, [a.sequence, b.sequence]))
        return offspring

    def _crossover(self, a: str, b: str, rng: random.Random) -> str:
        child = "".join(a[i] if rng.random() < 0.5 else b[i] for i in range(len(a)))
        return child if self.constraints.accepts(child) else a

    def _mutate(self, seq: str, rng: random.Random) -> str | None:
        positions = self.constraints.mutable_positions(seq)
        if not positions:
            return None
        for _ in range(self.config.mutate_attempts):
            pos = rng.choice(positions)
            options = self._strategy(seq[pos])
            if not options:
                continue
            candidate = seq[:pos] + rng.choice(options) + seq[pos + 1 :]
            if candidate != seq and self.constraints.accepts(candidate):
                return candidate
        return None

    def _make_peptide(self, sequence: str, parents: list[str]) -> Peptide:
        return Peptide.from_sequence(
            sequence, dataset="optimized", metadata={"parents": parents}
        )

    # ---- selection helpers ---- #
    def _tournament(
        self, rank: npt.NDArray[np.int_], crowd: npt.NDArray[np.float64], rng: random.Random
    ) -> int:
        i, j = rng.randrange(len(rank)), rng.randrange(len(rank))
        if rank[i] != rank[j]:
            return i if rank[i] < rank[j] else j
        return i if crowd[i] >= crowd[j] else j

    def _environmental_selection(
        self, combined: list[Peptide], F: npt.NDArray[np.float64]
    ) -> tuple[list[Peptide], npt.NDArray[np.float64]]:
        k = self.config.population_size
        selected: list[int] = []
        for front in non_dominated_sort(F):
            if len(selected) + len(front) <= k:
                selected.extend(front)
            else:
                remaining = k - len(selected)
                cd = crowding_distance(F, front)
                order = np.argsort(-cd, kind="stable")
                selected.extend(front[o] for o in order[:remaining])
                break
        return [combined[i] for i in selected], F[selected]

    def _rank_and_crowding(
        self, F: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
        n = F.shape[0]
        rank = np.zeros(n, dtype=int)
        crowd = np.zeros(n, dtype=float)
        for r, front in enumerate(non_dominated_sort(F)):
            cd = crowding_distance(F, front)
            for pos, idx in enumerate(front):
                rank[idx] = r
                crowd[idx] = cd[pos]
        return rank, crowd

    # ---- evaluation with caching ---- #
    def _evaluate(self, peptides: Sequence[Peptide]) -> npt.NDArray[np.float64]:
        uniq: dict[str, Peptide] = {}
        for p in peptides:
            if p.sequence not in self._cache:
                uniq.setdefault(p.sequence, p)
        if uniq:
            peps = list(uniq.values())
            columns = [obj.as_minimization(peps) for obj in self.objectives]
            matrix = np.column_stack(columns)
            self._n_evaluated += len(peps)
            for row, p in zip(matrix, peps):
                self._cache[p.sequence] = row
        return np.array([self._cache[p.sequence] for p in peptides], dtype=float)

    def _to_solutions(self, peptides: Sequence[Peptide]) -> list[ParetoSolution]:
        if not peptides:
            return []
        raw = {obj.name: obj.evaluate(peptides) for obj in self.objectives}
        return [
            ParetoSolution(
                peptide=p,
                objective_values={name: float(vals[i]) for name, vals in raw.items()},
            )
            for i, p in enumerate(peptides)
        ]
