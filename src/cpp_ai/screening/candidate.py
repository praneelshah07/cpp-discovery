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
from dataclasses import dataclass, replace

from ..generation.constraints import net_charge
from .modification import ModificationInfo, classify_modifications

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
    # tested-form chemical modifications (raw CPPsite3 fields; "" == free/none)
    n_term_mod: str = ""
    c_term_mod: str = ""
    chem_mod: str = ""

    @property
    def net_charge(self) -> int:
        return net_charge(self.sequence)

    @property
    def modification(self) -> ModificationInfo:
        """Classified chemical modifications of the tested form."""
        return classify_modifications(self.n_term_mod, self.c_term_mod, self.chem_mod)

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
        return replace(self, similarity=value)

    def to_record(self) -> dict[str, object]:
        mod = self.modification
        return {
            "name": self.name,
            "sequence": self.sequence,
            "similarity": None if self.similarity is None else round(self.similarity, 3),
            "net_charge": self.net_charge,
            "length": self.length,
            "toxicity_flag": self.toxicity_flag,
            "lytic_risk": self.lytic_risk,
            "modification": mod.summary,
            "genetically_encodable": mod.genetically_encodable,
            "mechanism": self.mechanism,
            "category": self.category,
            "source": self.source,
        }


