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
# "functional" conjugates plausibly drive uptake, so the naked cloned sequence
# may not reproduce the tested behavior. "label" conjugates are assay tracers
# (imaging/detection) the sequence does not need to function.
_FUNCTIONAL = re.compile(
    r"stear|palmit|myrist|lipid|cholester|fatty|acyl|nanoparticle|\bpmo\b|\bpeg\b|maleimide",
    re.I,
)
_FLUORO = re.compile(
    r"fluoresc|fitc|tamra|carboxyfluor|6-?fam|\bcf\b|rhodamin|naphthofluor|\bdye\b|cy[357]",
    re.I,
)
_LABEL = re.compile(r"biotin|cysteine|cysteamide|thiol|azide|alkyne|click", re.I)
_TERMINAL = re.compile(r"amidation|acetylation|acetyl|amide", re.I)

# Ordered by significance: a peptide's overall class is its most significant part.
_CLASS_PRIORITY = ("noncanonical", "conjugate", "terminal", "none")

# How much we trust that the *naked cloned sequence* reproduces the tested form,
# per modification kind. Feeds the usable-delivery score (fusion confidence).
_CONF = {
    "noncanonical": 0.15,  # literally a different peptide
    "functional": 0.40,    # uptake may depend on the (non-encodable) conjugate
    "label": 0.80,         # assay tracer the sequence doesn't need
    "terminal": 0.90,      # minor terminal cap
    "none": 1.00,          # bare sequence as tested — cloneable
}


@dataclass(frozen=True)
class ModificationInfo:
    """Coarse classification of a tested peptide's chemical modifications."""

    n_term: str
    c_term: str
    chem: str
    modification_class: str  # none | terminal | conjugate | noncanonical
    fusion_confidence: float  # trust that the naked cloned sequence matches (0..1)
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


def _terminal_kind(value: str) -> str:
    """Confidence key for one N-/C-terminal modification value."""
    if _FUNCTIONAL.search(value):
        return "functional"
    if _FLUORO.search(value) or _LABEL.search(value):
        return "label"
    if _TERMINAL.search(value):
        return "terminal"
    return "label"  # unrecognized terminal mod: moderate, not disqualifying


def classify_modifications(n_term: str, c_term: str, chem: str) -> ModificationInfo:
    """Classify the three CPPsite3 modification fields into one :class:`ModificationInfo`."""
    n_term = "" if _is_none(n_term) else n_term.strip()
    c_term = "" if _is_none(c_term) else c_term.strip()
    chem = "" if _is_none(chem) else chem.strip()

    classes: set[str] = set()
    conf_keys: set[str] = set()
    parts: list[str] = []

    for label, val in (("N-term", n_term), ("C-term", c_term)):
        if not val:
            continue
        kind = _terminal_kind(val)
        conf_keys.add(kind)
        classes.add("terminal" if kind == "terminal" else "conjugate")
        parts.append(f"{label}: {val}")
    if chem:
        if _FUNCTIONAL.search(chem):
            conf_keys.add("functional")
            classes.add("conjugate")
        elif _FLUORO.search(chem):
            conf_keys.add("label")
            classes.add("conjugate")
        else:  # a chem entry we don't recognize is a non-canonical residue
            conf_keys.add("noncanonical")
            classes.add("noncanonical")
        parts.append(f"chem: {chem}")

    klass = next((c for c in _CLASS_PRIORITY if c in classes), "none")
    confidence = min((_CONF[k] for k in conf_keys), default=_CONF["none"])
    return ModificationInfo(
        n_term=n_term,
        c_term=c_term,
        chem=chem,
        modification_class=klass,
        fusion_confidence=confidence,
        summary="; ".join(parts) if parts else "none",
    )
