"""The screening candidate type and export/diversity helpers.

A :class:`ScreenCandidate` is a peptide plus the annotations the screening UI
surfaces: source database metadata, an (optional) similarity to the query
anchor, and toxicity-relevant flags. Toxicity heuristics encode what we learned
from the lab's own data — high cationic charge and membrane-lytic (antimicrobial
/ direct-translocation) peptides are algae-toxicity risks, whereas endocytic
uptake is gentler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..generation.constraints import net_charge

_ENDOCYTIC = re.compile(r"endocyt|macropinocyt|pinocyt|caveola|clathrin|energy.?dependent", re.I)
_LYTIC = re.compile(r"antimicrob|direct translocation|non.?endocytic|non.?receptor|pore", re.I)


def charge_toxicity_flag(charge: int) -> str:
    """Coarse algae-toxicity flag from net charge (see lab thesis: 9R/pVEC toxic)."""
    if charge >= 8:
        return "HIGH-RISK"
    if charge >= 6:
        return "moderate"
    return "lower"


@dataclass(frozen=True)
class ScreenCandidate:
    """A peptide with screening annotations."""

    sequence: str
    name: str = "?"
    category: str = "?"
    mechanism: str = "?"
    source: str = "?"
    similarity: float | None = None

    @property
    def net_charge(self) -> int:
        return net_charge(self.sequence)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def toxicity_flag(self) -> str:
        return charge_toxicity_flag(self.net_charge)

    @property
    def is_endocytic(self) -> bool:
        return bool(_ENDOCYTIC.search(self.mechanism))

    @property
    def lytic_risk(self) -> bool:
        """True if the mechanism/category suggests membrane lysis (algae-toxic)."""
        return bool(_LYTIC.search(self.mechanism) or _LYTIC.search(self.category))

    def with_similarity(self, value: float) -> "ScreenCandidate":
        return ScreenCandidate(
            self.sequence, self.name, self.category, self.mechanism, self.source, value
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sequence": self.sequence,
            "similarity": None if self.similarity is None else round(self.similarity, 3),
            "net_charge": self.net_charge,
            "length": self.length,
            "toxicity_flag": self.toxicity_flag,
            "lytic_risk": self.lytic_risk,
            "mechanism": self.mechanism,
            "category": self.category,
            "source": self.source,
        }


def to_fasta(candidates: list[ScreenCandidate]) -> str:
    """Serialize candidates to FASTA with annotation-rich headers."""
    lines: list[str] = []
    for c in candidates:
        header = (
            f">{c.name.replace(' ', '_')} | charge={c.net_charge:+d} "
            f"tox={c.toxicity_flag} | {c.mechanism[:40]}"
        )
        lines.append(header)
        lines.append(c.sequence)
    return "\n".join(lines) + ("\n" if lines else "")


def _kmers(seq: str, k: int = 3) -> set[str]:
    return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))} or {seq}


def diversify(
    candidates: list[ScreenCandidate], *, max_jaccard: float = 0.6, limit: int | None = None
) -> list[ScreenCandidate]:
    """Greedily drop near-duplicates (3-mer Jaccard > ``max_jaccard``).

    Input order is preserved as priority (already-ranked lists stay ranked).
    """
    kept: list[ScreenCandidate] = []
    kept_kmers: list[set[str]] = []
    for c in candidates:
        km = _kmers(c.sequence)
        if all(
            len(km & other) / len(km | other) <= max_jaccard for other in kept_kmers
        ):
            kept.append(c)
            kept_kmers.append(km)
        if limit is not None and len(kept) >= limit:
            break
    return kept
