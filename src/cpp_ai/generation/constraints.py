"""Mutation constraints and the mutation record type.

Constraints let a researcher fence the design space to biologically sensible,
experimentally-tractable variants — e.g. "never touch this functional motif",
"keep net charge at most +8", "canonical residues only".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..core.types import is_canonical_sequence


def net_charge(sequence: str) -> int:
    """Simple integer net charge: (#K + #R) - (#D + #E).

    A fast, interpretable proxy used for design gating. Histidine and terminal
    charges are intentionally excluded; for a pH-aware continuous charge use the
    descriptor module instead.
    """
    pos = sum(sequence.count(a) for a in "KR")
    neg = sum(sequence.count(a) for a in "DE")
    return pos - neg


@dataclass(frozen=True)
class Mutation:
    """A single substitution: 0-based ``position``, ``wild_type`` -> ``mutant``."""

    position: int
    wild_type: str
    mutant: str

    def notation(self) -> str:
        """Standard 1-based mutation string, e.g. ``K5A``."""
        return f"{self.wild_type}{self.position + 1}{self.mutant}"


@dataclass(frozen=True)
class MutationConstraints:
    """Rules restricting which variants are allowed."""

    locked_positions: frozenset[int] = frozenset()
    locked_motifs: tuple[str, ...] = ()
    canonical_only: bool = True
    max_charge: int | None = None
    min_charge: int | None = None
    max_length: int | None = None
    min_length: int | None = None
    #: Extra predicates; a variant is accepted only if all return True.
    custom_filters: tuple[Callable[[str], bool], ...] = field(default_factory=tuple)

    def locked_indices(self, sequence: str) -> frozenset[int]:
        """All positions that must not be mutated (explicit + motif-covered)."""
        locked: set[int] = set(self.locked_positions)
        for motif in self.locked_motifs:
            if not motif:
                continue
            start = sequence.find(motif)
            while start != -1:
                locked.update(range(start, start + len(motif)))
                start = sequence.find(motif, start + 1)
        return frozenset(i for i in locked if 0 <= i < len(sequence))

    def mutable_positions(self, sequence: str) -> tuple[int, ...]:
        """Positions eligible for mutation, in order."""
        locked = self.locked_indices(sequence)
        return tuple(i for i in range(len(sequence)) if i not in locked)

    def accepts(self, sequence: str) -> bool:
        """Whether a candidate variant satisfies every constraint."""
        if self.canonical_only and not is_canonical_sequence(sequence):
            return False
        if self.min_length is not None and len(sequence) < self.min_length:
            return False
        if self.max_length is not None and len(sequence) > self.max_length:
            return False
        if self.max_charge is not None or self.min_charge is not None:
            charge = net_charge(sequence)
            if self.max_charge is not None and charge > self.max_charge:
                return False
            if self.min_charge is not None and charge < self.min_charge:
                return False
        return all(predicate(sequence) for predicate in self.custom_filters)
