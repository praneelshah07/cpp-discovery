"""VariantGenerator: produce constrained mutational variants of a reference CPP.

Generates single/double/triple (or n-fold) substitution variants of a known CPP
using a substitution strategy, keeping only variants that satisfy the supplied
constraints. Each variant is a :class:`Peptide` whose metadata records its
parent and exact mutations, so downstream ranking/optimization can explain where
every candidate came from.

For large design spaces (e.g. triple mutants of a long peptide), pass
``max_variants`` to switch from exhaustive enumeration to reproducible random
sampling, avoiding combinatorial blow-up.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Iterable

from ..core.exceptions import ValidationError
from ..core.schema import Peptide
from ..core.types import is_canonical_sequence
from .constraints import Mutation, MutationConstraints
from .substitutions import SUBSTITUTION_REGISTRY, SubstitutionStrategy

logger = logging.getLogger(__name__)

_SAFETY_CAP = 100_000  # hard limit on exhaustive enumeration, to bound runaway


class VariantGenerator:
    """Generate constrained substitution variants of a reference peptide."""

    def __init__(
        self,
        strategy: str | SubstitutionStrategy = "conservative",
        constraints: MutationConstraints | None = None,
        *,
        seed: int = 0,
        safety_cap: int = _SAFETY_CAP,
    ) -> None:
        if isinstance(strategy, str):
            self.strategy_name = strategy
            self._strategy: SubstitutionStrategy = SUBSTITUTION_REGISTRY.get(strategy)
        else:
            self.strategy_name = getattr(strategy, "__name__", "custom")
            self._strategy = strategy
        self.constraints = constraints or MutationConstraints()
        self.seed = seed
        self.safety_cap = safety_cap

    def generate(
        self,
        reference: Peptide,
        *,
        n_mutations: int = 1,
        max_variants: int | None = None,
    ) -> list[Peptide]:
        """Return variants of ``reference`` with exactly ``n_mutations`` changes."""
        if n_mutations < 1:
            raise ValidationError("n_mutations must be >= 1.")
        seq = reference.sequence
        if self.constraints.canonical_only and not is_canonical_sequence(seq):
            raise ValidationError(
                f"Reference {reference.peptide_id} is non-canonical; enable "
                "modified residues explicitly or supply a canonical reference."
            )

        positions = self.constraints.mutable_positions(seq)
        options = {p: self._strategy(seq[p]) for p in positions}
        usable = tuple(p for p in positions if options[p])
        if len(usable) < n_mutations:
            return []

        if max_variants is None:
            found = self._enumerate(seq, usable, n_mutations, options)
        else:
            found = self._sample(seq, usable, n_mutations, options, max_variants)

        return [self._build(reference, variant, muts) for variant, muts in found.items()]

    def generate_series(
        self,
        reference: Peptide,
        *,
        mutation_counts: Iterable[int] = (1, 2, 3),
        max_variants_per_count: int | None = None,
    ) -> list[Peptide]:
        """Convenience: generate single, double, and triple mutants in one call."""
        out: list[Peptide] = []
        for n in mutation_counts:
            out.extend(
                self.generate(reference, n_mutations=n, max_variants=max_variants_per_count)
            )
        return out

    # ------------------------------------------------------------------ #
    def _enumerate(
        self,
        seq: str,
        positions: tuple[int, ...],
        n: int,
        options: dict[int, tuple[str, ...]],
    ) -> dict[str, list[Mutation]]:
        results: dict[str, list[Mutation]] = {}
        for combo in itertools.combinations(positions, n):
            for replacements in itertools.product(*(options[p] for p in combo)):
                variant, muts = self._apply(seq, combo, replacements)
                if variant == seq or variant in results:
                    continue
                if not self.constraints.accepts(variant):
                    continue
                results[variant] = muts
                if len(results) >= self.safety_cap:
                    logger.warning(
                        "Enumeration hit safety cap %d; pass max_variants to sample.",
                        self.safety_cap,
                    )
                    return results
        return results

    def _sample(
        self,
        seq: str,
        positions: tuple[int, ...],
        n: int,
        options: dict[int, tuple[str, ...]],
        max_variants: int,
    ) -> dict[str, list[Mutation]]:
        rng = random.Random(self.seed)
        results: dict[str, list[Mutation]] = {}
        max_attempts = max_variants * 50 + 1000
        attempts = 0
        while len(results) < max_variants and attempts < max_attempts:
            attempts += 1
            combo = tuple(sorted(rng.sample(positions, n)))
            replacements = tuple(rng.choice(options[p]) for p in combo)
            variant, muts = self._apply(seq, combo, replacements)
            if variant == seq or variant in results:
                continue
            if not self.constraints.accepts(variant):
                continue
            results[variant] = muts
        return results

    @staticmethod
    def _apply(
        seq: str, positions: tuple[int, ...], replacements: tuple[str, ...]
    ) -> tuple[str, list[Mutation]]:
        chars = list(seq)
        muts: list[Mutation] = []
        for pos, repl in zip(positions, replacements):
            muts.append(Mutation(pos, seq[pos], repl))
            chars[pos] = repl
        return "".join(chars), muts

    def _build(self, reference: Peptide, variant: str, muts: list[Mutation]) -> Peptide:
        return Peptide.from_sequence(
            variant,
            dataset="generated",
            metadata={
                "parent_id": reference.peptide_id,
                "parent_sequence": reference.sequence,
                "strategy": self.strategy_name,
                "n_mutations": len(muts),
                "mutations": [m.notation() for m in muts],
            },
        )
