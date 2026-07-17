"""Cytotoxicity prior — a *curated, literature-grounded* evidence axis.

Separate from `hemolysis_prior` (RBC hemolysis only). The benchmark showed why
both are needed: KLA/(KLAKLAK)ₙ is mitochondrially toxic / pro-apoptotic yet
**not** hemolytic, so the hemolysis model scores it ~0 and it ranks near the top.
Non-hemolytic membrane/organelle toxicity must be represented by curated
evidence, not inferred from descriptors (inference is exactly what let KLA
through).

Design constraints (per the redesign directive):

* **No peptide names inside the scoring path.** The scoring function matches a
  sequence against curated annotations (exact sequence or a data-defined regex
  pattern); the peptide identities live in the annotation table (data), never in
  an `if seq == KLA` branch.
* **Provisional class→factor map is centralized and clearly a prior needing
  validation** — not a per-peptide constant.
* **Non-lytic translocators are distinguished** — Buforin must NOT inherit the
  melittin/KLA penalty.

Everything here is mostly mammalian/bacterial evidence; transfer to algal
membranes/organelles is a documented assumption (docs/redesign.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

CytotoxClass = Literal[
    "cytotoxic_organelle",       # known cytotoxic / organelle (e.g. mitochondrial) disruptive
    "membrane_lytic",            # known strongly membrane-lytic / pore-forming
    "nonlytic_translocator",     # antimicrobial but reported non-lytic translocator
    "conventional_cpp",          # conventional low-lytic CPP
    "insufficient_evidence",     # no curated record (default)
]
ToxicityLocus = Literal["plasma_membrane", "mitochondrial", "general", "unknown"]

# Provisional class → multiplicative factor. **PRIORS REQUIRING VALIDATION** — not
# measured algal values; centralized so they are tuned in one place, never inline.
CYTOTOX_FACTOR: dict[str, float] = {
    "cytotoxic_organelle": 0.2,
    "membrane_lytic": 0.3,
    "nonlytic_translocator": 1.0,
    "conventional_cpp": 1.0,
    "insufficient_evidence": 1.0,
}


@dataclass(frozen=True)
class CytotoxAnnotation:
    scaffold: str
    cytotoxicity_class: CytotoxClass
    toxicity_locus: ToxicityLocus
    evidence_confidence: Literal["high", "medium", "low"]
    evidence_source: str                 # DOI / URL
    experimental_context: str
    sequence: str | None = None          # exact match
    sequence_pattern: str | None = None  # regex family match (DATA, not code)
    note: str = ""


# Curated seed (literature-grounded). Extend as evidence accrues.
CYTOTOX_ANNOTATIONS: tuple[CytotoxAnnotation, ...] = (
    CytotoxAnnotation(
        "KLA / (KLAKLAK)n", "cytotoxic_organelle", "mitochondrial", "high",
        "10.1038/70995", "Ellerby 1999, targeted pro-apoptotic (KLAKLAK)2",
        sequence_pattern=r"(KLAKLAK){1,}", note="mitochondrial membrane disruption, non-hemolytic"),
    CytotoxAnnotation(
        "Melittin", "membrane_lytic", "plasma_membrane", "high",
        "10.1073/pnas.1307010110", "bee venom; hemolysis gold-standard, pore-forming",
        sequence="GIGAVLKVLTTGLPALISWIKRKRQQ"),
    CytotoxAnnotation(
        "Mastoparan", "membrane_lytic", "plasma_membrane", "high",
        "10.1021/bi401406p", "wasp venom, mast-cell degranulating / membrane-active",
        sequence="INLKALAALAKKIL"),
    CytotoxAnnotation(
        "Magainin 2", "membrane_lytic", "plasma_membrane", "high",
        "10.1073/pnas.84.15.5449", "frog AMP, pore-forming",
        sequence="GIGKFLHSAKKFGKAFVGEIMNS"),
    CytotoxAnnotation(
        "Brevinin-2R", "membrane_lytic", "plasma_membrane", "medium",
        "10.1016/j.bbamem.2007.09.006", "frog AMP, membrane-lytic (Rana box)",
        sequence="KLKNFAKGVAQSLLNKASCKLSGQC"),
    CytotoxAnnotation(
        "MAP (model amphipathic)", "membrane_lytic", "plasma_membrane", "medium",
        "10.1096/fj.98-0708com", "designed amphipathic, membrane-perturbing",
        sequence="KLALKLALKALKAALKLA"),
    CytotoxAnnotation(
        "Buforin II", "nonlytic_translocator", "general", "high",
        "10.1073/pnas.150518097", "Park 2000: translocates WITHOUT permeabilization",
        sequence="TRSSRAGLQFPVGRVHRLLRK", note="non-lytic — must not inherit lytic penalty"),
    # conventional CPPs (explicit, so they are not 'insufficient_evidence')
    CytotoxAnnotation("pVEC", "conventional_cpp", "unknown", "medium",
                      "10.1016/j.algal.2017.04.022", "algae protein delivery, low toxicity",
                      sequence="LLIILRRRIRKQAHAHSK"),
    CytotoxAnnotation("Penetratin", "conventional_cpp", "unknown", "medium",
                      "10.1155/2011/414729", "homeodomain CPP", sequence="RQIKIWFQNRRMKWKK"),
    CytotoxAnnotation("TAT", "conventional_cpp", "unknown", "medium",
                      "10.1155/2011/414729", "HIV-1 Tat basic domain", sequence="YGRKKRRQRRR"),
)


def annotate(sequence: str) -> CytotoxAnnotation:
    """Return the curated annotation for a sequence, or an insufficient-evidence stub.

    Matches exact sequence first, then data-defined regex family patterns. No
    peptide identity is hard-coded in this function — it consults the table only.
    """
    seq = sequence.strip().upper()
    for a in CYTOTOX_ANNOTATIONS:
        if a.sequence and a.sequence.upper() == seq:
            return a
    for a in CYTOTOX_ANNOTATIONS:
        if a.sequence_pattern and re.search(a.sequence_pattern, seq):
            return a
    return CytotoxAnnotation(
        "unknown", "insufficient_evidence", "unknown", "low", "", "no curated record")


def cytotoxicity_class(sequence: str) -> CytotoxClass:
    return annotate(sequence).cytotoxicity_class


def cytotoxicity_factor(sequence: str) -> float:
    """Provisional multiplicative factor in (0, 1] from the curated class."""
    return CYTOTOX_FACTOR[annotate(sequence).cytotoxicity_class]
