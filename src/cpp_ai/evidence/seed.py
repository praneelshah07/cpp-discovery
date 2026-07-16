"""Curated seed of the CPP evidence ledger — verified literature entries only.

Every entry below was transcribed from a real, resolvable publication (DOI
recorded on the :class:`Citation`). Numbers are only filled in where the paper
reported them; qualitative rankings are recorded as such in ``uptake_metric``
rather than invented as fake percentages.

The scientific point this seed makes concrete: **the best CPP depends on the
context.** R9 is a superb mammalian transporter yet underperforms pVEC for
protein delivery into *Chlamydomonas*; pVEC is only middling for mammalian
protein cargo yet is the strongest CPP in microalgae. Context-stratified
evidence is the whole reason this ledger exists.

Regenerate the on-disk ledger with::

    python -m cpp_ai.evidence.seed
"""

from __future__ import annotations

from .schema import Citation, EvidenceEntry
from .store import DEFAULT_LEDGER_PATH, EvidenceLedger

# --------------------------------------------------------------------------- #
# canonical sequences (widely published constants)
# --------------------------------------------------------------------------- #
SEQ = {
    "R9": "RRRRRRRRR",
    "R8": "RRRRRRRR",
    "TAT": "YGRKKRRQRRR",  # TAT(47-57)
    "Penetratin": "RQIKIWFQNRRMKWKK",  # Antennapedia(43-58)
    "pVEC": "LLIILRRRIRKQAHAHSK",
    "pVEC-R6A": "LLIILARRIRKQAHAHSK",
    "Transportan": "GWTLNSAGYLLGKINLKALAALAKKIL",
    "TP10": "AGYLLGKINLKALAALAKKIL",
    "MAP": "KLALKLALKALKAALKLA",
    "ClWOX": "TNVYNWFQNRRARTKRK",
}

# --------------------------------------------------------------------------- #
# citations (each verified via web pull; DOI recorded)
# --------------------------------------------------------------------------- #
FUTAKI_2001 = Citation(
    title=(
        "Arginine-rich peptides. An abundant source of membrane-permeable "
        "peptides having potential as carriers for intracellular protein delivery"
    ),
    year=2001,
    authors_short="Futaki et al.",
    doi="10.1074/jbc.M007540200",
)
SAALIK_2004 = Citation(
    title="Protein cargo delivery properties of cell-penetrating peptides. A comparative study",
    year=2004,
    authors_short="Säälik et al.",
    doi="10.1021/bc049938y",
)
SURESH_2013 = Citation(
    title="Translocation of cell penetrating peptides on Chlamydomonas reinhardtii",
    year=2013,
    authors_short="Suresh & Kim",
    doi="10.1002/bit.24935",
)
KANG_2017 = Citation(
    title=(
        "A highly efficient cell penetrating peptide pVEC-mediated protein "
        "delivery system into microalgae"
    ),
    year=2017,
    authors_short="Kang, Suresh & Kim",
    doi="10.1016/j.algal.2017.04.022",
)


def build_seed_ledger() -> EvidenceLedger:
    """Return the curated seed ledger (constructed in memory)."""
    e: list[EvidenceEntry] = [
        # ---------------- mammalian uptake (Futaki 2001) ------------------- #
        EvidenceEntry(
            peptide_name="R9",
            sequence=SEQ["R9"],
            organism="mammalian",
            cargo_type="none",
            outcome="success",
            uptake_value=20.0,
            uptake_metric="fold uptake rate vs TAT peptide at 37°C",
            toxicity="mild",
            toxicity_note="Arginine-rich; membrane-perturbing at high peptide:lipid ratios.",
            mechanism="endocytic",
            citation=FUTAKI_2001,
            confidence="high",
            notes="Efficient mammalian uptake, but ~8 arginines is near-optimal — longer is not better.",
        ),
        EvidenceEntry(
            peptide_name="R8",
            sequence=SEQ["R8"],
            organism="mammalian",
            cargo_type="none",
            outcome="success",
            uptake_metric="octaarginine near the optimal arginine chain length for uptake",
            mechanism="endocytic",
            citation=FUTAKI_2001,
            confidence="high",
        ),
        # -------------- mammalian protein cargo (Säälik 2004) -------------- #
        EvidenceEntry(
            peptide_name="TAT",
            sequence=SEQ["TAT"],
            organism="mammalian",
            cell_type="HeLa",
            cargo_type="protein",
            cargo_name="avidin",
            cargo_kda=66.0,
            outcome="success",
            uptake_metric="higher protein transduction than penetratin or pVEC",
            mechanism="endocytic",
            citation=SAALIK_2004,
            confidence="high",
        ),
        EvidenceEntry(
            peptide_name="Transportan",
            sequence=SEQ["Transportan"],
            organism="mammalian",
            cell_type="HeLa",
            cargo_type="protein",
            cargo_name="avidin",
            cargo_kda=66.0,
            outcome="success",
            uptake_metric="higher protein transduction than penetratin or pVEC",
            mechanism="mixed",
            citation=SAALIK_2004,
            confidence="high",
            notes="Chimeric galanin–mastoparan peptide; mastoparan portion is membrane-active.",
        ),
        EvidenceEntry(
            peptide_name="Penetratin",
            sequence=SEQ["Penetratin"],
            organism="mammalian",
            cell_type="HeLa",
            cargo_type="protein",
            cargo_name="avidin",
            cargo_kda=66.0,
            outcome="partial",
            uptake_metric="lower protein transduction than TAT or transportan",
            mechanism="mixed",
            citation=SAALIK_2004,
            confidence="high",
        ),
        EvidenceEntry(
            peptide_name="pVEC",
            sequence=SEQ["pVEC"],
            organism="mammalian",
            cell_type="HeLa",
            cargo_type="protein",
            cargo_name="avidin",
            cargo_kda=66.0,
            outcome="partial",
            uptake_metric="lower protein transduction than TAT or transportan",
            mechanism="unknown",
            citation=SAALIK_2004,
            confidence="high",
            notes="Only middling in mammalian cells — contrast with its top rank in microalgae.",
        ),
        # ---------- algae, small fluorophore cargo (Suresh 2013) ----------- #
        EvidenceEntry(
            peptide_name="pVEC",
            sequence=SEQ["pVEC"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="small_molecule",
            cargo_name="fluorochrome",
            incubation_min=15.0,
            temperature_c=25.0,
            outcome="success",
            uptake_metric="highest translocation vs penetratin, TAT, transportan",
            toxicity="none",
            toxicity_note="Paper reports absence of cytotoxicity for pVEC translocation.",
            mechanism="unknown",
            citation=SURESH_2013,
            confidence="high",
        ),
        EvidenceEntry(
            peptide_name="Penetratin",
            sequence=SEQ["Penetratin"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="small_molecule",
            cargo_name="fluorochrome",
            outcome="partial",
            uptake_metric="lower translocation than pVEC",
            citation=SURESH_2013,
            confidence="high",
        ),
        EvidenceEntry(
            peptide_name="TAT",
            sequence=SEQ["TAT"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="small_molecule",
            cargo_name="fluorochrome",
            outcome="partial",
            uptake_metric="lower translocation than pVEC",
            citation=SURESH_2013,
            confidence="high",
        ),
        EvidenceEntry(
            peptide_name="Transportan",
            sequence=SEQ["Transportan"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="small_molecule",
            cargo_name="fluorochrome",
            outcome="partial",
            uptake_metric="lower translocation than pVEC",
            citation=SURESH_2013,
            confidence="high",
        ),
        # --------- algae, protein cargo — the lab's regime (Kang 2017) ----- #
        EvidenceEntry(
            peptide_name="pVEC",
            sequence=SEQ["pVEC"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii CC-124",
            cargo_type="protein",
            cargo_name="FITC-BSA (and 6–150 kDa proteins)",
            uptake_metric="the ONLY CPP tested that delivered protein; the others failed",
            outcome="success",
            toxicity="mild",
            toxicity_note=(
                "Concentration-dependent: ~45% viability at 40 µM; useful window "
                "~20–60 µM. Excess CPP destabilizes the membrane."
            ),
            mechanism="mixed",
            citation=KANG_2017,
            confidence="high",
            notes=(
                "Tested form is C-terminally AMIDATED. Delivered proteins 6–150 kDa "
                "non-covalently; also worked in Chlorella and Nannochloropsis."
            ),
        ),
        EvidenceEntry(
            peptide_name="pVEC-R6A",
            sequence=SEQ["pVEC-R6A"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            outcome="success",
            uptake_metric="R6→A substitution reported to enhance pVEC uptake",
            mechanism="unknown",
            citation=KANG_2017,
            confidence="medium",
            notes="The lab's own construct; charge +5. Directly relevant to the mCherry-fusion goal.",
        ),
        EvidenceEntry(
            peptide_name="R9",
            sequence=SEQ["R9"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii CC-124",
            cargo_type="protein",
            outcome="fail",
            uptake_metric="did NOT deliver FITC-BSA protein (penetrates but no protein cargo)",
            mechanism="unknown",
            citation=KANG_2017,
            confidence="high",
            notes="The context reversal: strong mammalian transporter, but fails to deliver protein into algae.",
        ),
        EvidenceEntry(
            peptide_name="Transportan",
            sequence=SEQ["Transportan"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii CC-124",
            cargo_type="protein",
            outcome="fail",
            uptake_metric="did NOT deliver protein into algae effectively (only pVEC did)",
            citation=KANG_2017,
            confidence="high",
        ),
        EvidenceEntry(
            peptide_name="TAT",
            sequence=SEQ["TAT"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii CC-124",
            cargo_type="protein",
            outcome="fail",
            uptake_metric="did NOT deliver protein into algae effectively (only pVEC did)",
            citation=KANG_2017,
            confidence="high",
        ),
        EvidenceEntry(
            peptide_name="Penetratin",
            sequence=SEQ["Penetratin"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii CC-124",
            cargo_type="protein",
            outcome="fail",
            uptake_metric="did NOT deliver protein into algae effectively (only pVEC did)",
            citation=KANG_2017,
            confidence="high",
        ),
    ]
    return EvidenceLedger(e)


def main() -> None:
    ledger = build_seed_ledger()
    path = ledger.save(DEFAULT_LEDGER_PATH)
    print(f"Wrote {len(ledger)} entries ({ledger.unique_peptides()} unique peptides) to {path}")


if __name__ == "__main__":
    main()
