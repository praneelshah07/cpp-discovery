"""Sequence-arrangement (motif) descriptors.

The composition/charge descriptors are a "bag of properties": two peptides with
identical amino-acid composition but very different membrane behavior (one
clusters hydrophobic residues, the other disperses them) look the same. These
features restore *arrangement* information:

* longest basic (K/R) run and longest hydrophobic run — clustering vs dispersion
* cationic-aromatic adjacencies (RW/KW/RF/KY…) — a motif repeatedly implicated
  in membrane-active CPPs
* charge segregation between the N- and C-terminal halves

Registered as the ``arrangement`` descriptor block.
"""

from __future__ import annotations

from typing import Mapping

from ..generation.constraints import net_charge
from .base import register_descriptor

_CATIONIC = frozenset("KR")
_AROMATIC = frozenset("WFY")
_HYDROPHOBIC = frozenset("AVLIMFW")


def _longest_run(sequence: str, residues: frozenset[str]) -> int:
    best = current = 0
    for aa in sequence:
        current = current + 1 if aa in residues else 0
        best = max(best, current)
    return best


@register_descriptor("arrangement")
def arrangement(sequence: str) -> Mapping[str, float]:
    """Arrangement-sensitive features that composition alone cannot capture."""
    n = len(sequence)
    contacts = sum(
        1
        for i in range(n - 1)
        if ({sequence[i], sequence[i + 1]} & _CATIONIC)
        and ({sequence[i], sequence[i + 1]} & _AROMATIC)
    )
    half = n // 2
    seg = abs(net_charge(sequence[:half]) - net_charge(sequence[half:])) / n if n else 0.0
    return {
        "longest_basic_run": float(_longest_run(sequence, _CATIONIC)),
        "longest_hydrophobic_run": float(_longest_run(sequence, _HYDROPHOBIC)),
        "cationic_aromatic_contacts": float(contacts),
        "charge_segregation": float(seg),
    }
