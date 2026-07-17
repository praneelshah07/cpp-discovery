"""Cargo-delivery evidence — a curated, **advisory** axis (not in the core score).

The sequence model scores whether a peptide *could* interact with and cross a
membrane. It says nothing about whether that peptide has ever delivered the kind
of cargo we care about — a folded ~27 kDa mCherry fusion. That is an
**experimental record**, not a physicochemical property, so it is curated here
and — per the redesign directive — **displayed separately, not folded into the
ranking** in this pass. `cargo_what_if_factor` exists only to *show* how rankings
would change if it were included.

Ordered best→worst; the ordinal encodes "how relevant is the demonstrated cargo
to our goal" (a genetically-fused protein is the exact target; a small dye tells
us little about protein delivery).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CargoClass = Literal[
    "intact_genetic_fusion",   # intact genetically-fused protein delivered (our exact goal)
    "intact_covalent_protein",  # intact covalently-linked protein delivered
    "noncovalent_protein",      # non-covalent protein cargo delivered
    "nucleic_acid_only",        # only nucleic-acid cargo shown
    "small_molecule_only",      # only small molecule / fluorophore shown
    "free_peptide_only",        # only free-peptide uptake shown
    "modification_dependent",   # unclear / modification-dependent evidence
    "no_evidence",              # default
]

# Provisional what-if weights (advisory only — NOT applied to the runtime ranking).
# Priors requiring validation; centralized so they are never inlined per-peptide.
CARGO_WHATIF_WEIGHT: dict[str, float] = {
    "intact_genetic_fusion": 1.00,
    "intact_covalent_protein": 0.90,
    "noncovalent_protein": 0.80,
    "nucleic_acid_only": 0.50,
    "small_molecule_only": 0.35,
    "free_peptide_only": 0.25,
    "modification_dependent": 0.40,
    "no_evidence": 0.50,  # neutral-ish: absence of evidence is not evidence of absence
}


@dataclass(frozen=True)
class CargoAnnotation:
    scaffold: str
    cargo_class: CargoClass
    evidence_confidence: Literal["high", "medium", "low"]
    evidence_source: str            # DOI / URL
    experimental_context: str
    sequence: str | None = None
    note: str = ""


# Curated seed (literature-grounded). Extend as the record grows / lab data lands.
CARGO_ANNOTATIONS: tuple[CargoAnnotation, ...] = (
    CargoAnnotation("pVEC", "noncovalent_protein", "high", "10.1016/j.algal.2017.04.022",
                    "6–150 kDa proteins into Chlamydomonas, non-covalent",
                    sequence="LLIILRRRIRKQAHAHSK"),
    CargoAnnotation("pVEC-R6A", "noncovalent_protein", "medium", "10.1038/s41598-020-78968-x",
                    "RNP/Cas9 delivery into Chlamydomonas", sequence="LLIILARRIRKQAHAHSK"),
    CargoAnnotation("TAT", "intact_covalent_protein", "high", "10.1155/2011/414729",
                    "protein fusions widely delivered (mammalian)", sequence="YGRKKRRQRRR"),
    CargoAnnotation("Penetratin", "intact_covalent_protein", "medium", "10.1155/2011/414729",
                    "protein-cargo delivery (mammalian/plant)", sequence="RQIKIWFQNRRMKWKK"),
    CargoAnnotation("R9", "nucleic_acid_only", "high", "10.1016/j.algal.2017.04.022",
                    "delivers oligonucleotides but FAILED protein into algae",
                    sequence="RRRRRRRRR", note="protein-delivery negative in algae"),
    CargoAnnotation("Buforin II", "nucleic_acid_only", "medium", "10.1073/pnas.150518097",
                    "binds/carries nucleic acids", sequence="TRSSRAGLQFPVGRVHRLLRK"),
)


def cargo_annotation(sequence: str) -> CargoAnnotation:
    seq = sequence.strip().upper()
    for a in CARGO_ANNOTATIONS:
        if a.sequence and a.sequence.upper() == seq:
            return a
    return CargoAnnotation("unknown", "no_evidence", "low", "", "no curated record")


def cargo_class(sequence: str) -> CargoClass:
    return cargo_annotation(sequence).cargo_class


def cargo_what_if_factor(sequence: str) -> float:
    """Advisory factor showing how the ranking WOULD change if cargo evidence were
    folded in. **Not** applied to the runtime score (per directive: display only)."""
    return CARGO_WHATIF_WEIGHT[cargo_annotation(sequence).cargo_class]
