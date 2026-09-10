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
    # Lab mCherry-NLS screen peptides (sequences provided by the lab).
    "MAR1": "PPRPPWPPRPPPAPPPSRPP",
    "FUS1": "IALVWSFRMLRHKP",
    "SAG1": "GCAAALGYWGLREQSWAQLG",
    "gAUT": "AQEEFQGVGMVKLKSAFR",
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
KANG_2020 = Citation(
    title=(
        "Development of a pVEC peptide-based ribonucleoprotein (RNP) delivery "
        "system for genome editing using CRISPR/Cas9 in Chlamydomonas reinhardtii"
    ),
    year=2020,
    authors_short="Kang et al.",
    doi="10.1038/s41598-020-78968-x",
)
SIM_2025 = Citation(
    title=(
        "Cell-Penetrating Peptide-Based Triple Nanocomplex Enables Efficient "
        "Nuclear Gene Delivery in Chlamydomonas reinhardtii"
    ),
    year=2025,
    authors_short="Sim et al.",
    doi="10.1002/bit.29019",
)
# The lab's own mCherry-NLS delivery screen (flow cytometry, C. reinhardtii).
# 20 uM free peptide co-incubated with 20 uM mCherry-NLS; readout = % mCherry+
# cells (R2 gate on the YL2-A channel). Not a peer-reviewed publication — an
# internal experimental record, so no DOI. Directly in the lab's target regime.
HILLMAN_2026 = Citation(
    title="Lab mCherry-NLS peptide delivery screen (Round 1), Chlamydomonas reinhardtii",
    year=2026,
    authors_short="Hillman (unpublished lab data)",
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
        # ---- algae, large-protein cargo, pVEC family (Kang 2020, Sim 2025) ---- #
        EvidenceEntry(
            peptide_name="pVEC",
            sequence=SEQ["pVEC"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            cargo_name="Cas9 ribonucleoprotein (RNP)",
            cargo_kda=160.0,
            outcome="success",
            uptake_metric="delivered Cas9 RNP for CRISPR genome editing (functional editing detected)",
            mechanism="unknown",
            citation=KANG_2020,
            confidence="high",
            notes="pVEC carries a very large (~160 kDa) protein complex into algae — supports large-cargo capability.",
        ),
        EvidenceEntry(
            peptide_name="pVEC-R6A",
            sequence=SEQ["pVEC-R6A"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="nucleic_acid",
            cargo_name="plasmid DNA (triple nanocomplex)",
            outcome="success",
            uptake_metric="efficient nuclear gene (DNA) delivery via a CPP-based triple nanocomplex",
            mechanism="unknown",
            citation=SIM_2025,
            confidence="high",
            notes="pVEC-R6A used for nuclear DNA delivery — the lab's construct in a peer-reviewed setting.",
        ),
        # ---- algae, mCherry-NLS protein cargo — the lab's own screen (2026) --- #
        # Flow cytometry, % mCherry+ cells (YL2-A / R2 gate); 20 uM peptide +
        # 20 uM mCherry-NLS, non-covalent. Control (no useful CPP) = 0.247%.
        # Per the lab: weak peptides "don't work well" but are NOT hard failures
        # (all sit above the 0.247% control), so they are recorded as "partial".
        EvidenceEntry(
            peptide_name="pVEC-R6A",
            sequence=SEQ["pVEC-R6A"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            cargo_name="mCherry-NLS",
            cargo_kda=28.0,
            concentration_um=20.0,
            outcome="success",
            uptake_value=15.138,
            uptake_metric="% mCherry+ cells by flow cytometry (YL2-A gate); best in screen, 61x control",
            mechanism="unknown",
            citation=HILLMAN_2026,
            confidence="high",
            notes="The lab's lead construct; strongest mCherry-NLS delivery in the screen.",
        ),
        EvidenceEntry(
            peptide_name="FUS1",
            sequence=SEQ["FUS1"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            cargo_name="mCherry-NLS",
            cargo_kda=28.0,
            concentration_um=20.0,
            outcome="success",
            uptake_value=8.519,
            uptake_metric="% mCherry+ cells by flow cytometry (YL2-A gate); 2nd best, 34x control",
            mechanism="unknown",
            citation=HILLMAN_2026,
            confidence="high",
            notes="A strong second winner with a profile distinct from pVEC (short, aromatic+cationic).",
        ),
        EvidenceEntry(
            peptide_name="SAG1",
            sequence=SEQ["SAG1"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            cargo_name="mCherry-NLS",
            cargo_kda=28.0,
            concentration_um=20.0,
            outcome="partial",
            uptake_value=1.502,
            uptake_metric="% mCherry+ cells by flow cytometry (YL2-A gate); weak, ~6x control",
            mechanism="unknown",
            citation=HILLMAN_2026,
            confidence="high",
            notes="Modest delivery, above control but far below pVEC-R6A/FUS1.",
        ),
        EvidenceEntry(
            peptide_name="gAUT",
            sequence=SEQ["gAUT"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            cargo_name="mCherry-NLS",
            cargo_kda=28.0,
            concentration_um=20.0,
            outcome="partial",
            uptake_value=1.024,
            uptake_metric="% mCherry+ cells by flow cytometry (YL2-A gate); weak, ~4x control",
            mechanism="unknown",
            citation=HILLMAN_2026,
            confidence="high",
            notes="Modest delivery, above control but far below pVEC-R6A/FUS1.",
        ),
        EvidenceEntry(
            peptide_name="ClWOX",
            sequence=SEQ["ClWOX"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            cargo_name="mCherry-NLS",
            cargo_kda=28.0,
            concentration_um=20.0,
            outcome="partial",
            uptake_value=0.832,
            uptake_metric="% mCherry+ cells by flow cytometry (YL2-A gate); weak, ~3.4x control",
            mechanism="unknown",
            citation=HILLMAN_2026,
            confidence="high",
            notes="Notable: the plant-homeoprotein CPP underperforms for mCherry delivery in ALGAE, tempering the ClWOX hypothesis for this organism.",
        ),
        EvidenceEntry(
            peptide_name="MAR1",
            sequence=SEQ["MAR1"],
            organism="algae",
            cell_type="Chlamydomonas reinhardtii",
            cargo_type="protein",
            cargo_name="mCherry-NLS",
            cargo_kda=28.0,
            concentration_um=20.0,
            outcome="partial",
            uptake_value=0.832,
            uptake_metric="% mCherry+ cells by flow cytometry (YL2-A gate); weak, ~3.4x control",
            mechanism="unknown",
            citation=HILLMAN_2026,
            confidence="high",
            notes="Proline-rich scaffold; only modestly above control.",
        ),
    ]
    return EvidenceLedger(e)


def main() -> None:
    ledger = build_seed_ledger()
    path = ledger.save(DEFAULT_LEDGER_PATH)
    print(f"Wrote {len(ledger)} entries ({ledger.unique_peptides()} unique peptides) to {path}")


if __name__ == "__main__":
    main()
