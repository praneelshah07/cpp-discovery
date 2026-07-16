"""Classify the chemical modifications recorded for a tested CPP.

CPPsite3 records the form a peptide was *actually tested* in — its N-terminal,
C-terminal, and other chemical modifications (``ntermod`` / ``ctermod`` /
``chmod``). The library previously discarded these, so the app treated a naked
one-letter sequence as equivalent to the chemically-modified material that was
measured. That is misleading: a stearoylated or amidated peptide can behave very
differently, and — critically for this lab — such modifications are **not
genetically encodable** in a recombinant mCherry fusion.

This module turns the free-text modification fields into a coarse, honest
classification so the recommender can flag "the tested form ≠ the naked
sequence" and optionally prefer cloneable candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Values that mean "no modification at this position".
_NONE_VALUES = {"", "free", "na", "n/a", "none", "nan", "-"}

# Modification categories, matched against the free-text descriptions.
_LIPID = re.compile(r"stear|palmit|myrist|lipid|cholester|fatty|acyl", re.I)
_FLUORO = re.compile(
    r"fluoresc|fitc|tamra|carboxyfluor|6-?fam|\bcf\b|rhodamin|naphthofluor|\bdye\b|cy[357]",
    re.I,
)
_OTHER_CONJ = re.compile(r"biotin|nanoparticle|\bpmo\b|\bpeg\b|maleimide|azide|alkyne|click", re.I)
_HANDLE = re.compile(r"cysteine|cysteamide|thiol", re.I)
_TERMINAL = re.compile(r"amidation|acetylation|acetyl|amide", re.I)

# Ordered by significance: a peptide's overall class is its most significant part.
_CLASS_PRIORITY = ("noncanonical", "conjugate", "terminal", "none")


@dataclass(frozen=True)
class ModificationInfo:
    """Coarse classification of a tested peptide's chemical modifications."""

    n_term: str
    c_term: str
    chem: str
    modification_class: str  # none | terminal | conjugate | noncanonical
    summary: str  # compact human string, or "none"

    @property
    def is_modified(self) -> bool:
        return self.modification_class != "none"

    @property
    def genetically_encodable(self) -> bool:
        """True iff the tested form is the bare sequence (free termini, no chem mod).

        Terminal caps (amidation/acetylation), conjugates (lipid/fluorophore/
        biotin/nanoparticle), and non-canonical residues cannot be reproduced by
        cloning the amino-acid sequence into an mCherry fusion, so they are all
        flagged non-encodable.
        """
        return self.modification_class == "none"


def _is_none(value: str) -> bool:
    return str(value).strip().lower() in _NONE_VALUES


def _classify_terminal(value: str) -> str:
    """Return the category for one N-/C-terminal modification value."""
    if _LIPID.search(value) or _FLUORO.search(value) or _OTHER_CONJ.search(value) \
            or _HANDLE.search(value):
        return "conjugate"
    if _TERMINAL.search(value):
        return "terminal"
    return "conjugate"  # an unrecognized real terminal mod is still non-native


def classify_modifications(n_term: str, c_term: str, chem: str) -> ModificationInfo:
    """Classify the three CPPsite3 modification fields into one :class:`ModificationInfo`."""
    n_term = "" if _is_none(n_term) else n_term.strip()
    c_term = "" if _is_none(c_term) else c_term.strip()
    chem = "" if _is_none(chem) else chem.strip()

    classes: set[str] = set()
    parts: list[str] = []

    if n_term:
        classes.add(_classify_terminal(n_term))
        parts.append(f"N-term: {n_term}")
    if c_term:
        classes.add(_classify_terminal(c_term))
        parts.append(f"C-term: {c_term}")
    if chem:
        # A chem entry is a conjugate if it names one, else a non-canonical residue.
        if _OTHER_CONJ.search(chem) or _LIPID.search(chem) or _FLUORO.search(chem):
            classes.add("conjugate")
        else:
            classes.add("noncanonical")
        parts.append(f"chem: {chem}")

    klass = next((c for c in _CLASS_PRIORITY if c in classes), "none")
    return ModificationInfo(
        n_term=n_term,
        c_term=c_term,
        chem=chem,
        modification_class=klass,
        summary="; ".join(parts) if parts else "none",
    )
