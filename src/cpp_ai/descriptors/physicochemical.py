"""Core physicochemical descriptor blocks.

Biological rationale (why these matter for CPPs)
------------------------------------------------
* **Net charge / isoelectric point** — most cationic CPPs (e.g. penetratin,
  TAT, pVEC) rely on positive charge to engage anionic membrane components.
  Charge is computed at physiological pH 7.4 across several published pK scales
  because the scales disagree by up to ~1 unit and that spread is itself
  informative when discriminating similar peptides.
* **Hydrophobicity (GRAVY) & hydrophobic moment** — amphipathicity drives
  membrane insertion. The hydrophobic *moment* quantifies how the hydrophobic
  face segregates when the backbone is drawn as a helix (angle 100°) or sheet
  (angle 180°).
* **Aliphatic index / instability index** — proxies for stability and, for
  instability, for expression/half-life considerations relevant to the
  downstream recombinant CPP-mCherry fusion.
* **Boman index** — potential for protein-protein interaction / binding.
* **Predicted secondary-structure fractions** — a coarse helix/turn/sheet
  propensity; many efficient CPPs are helical.
"""

from __future__ import annotations

import logging
from typing import Mapping

import peptides
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from peptides import Peptide as PeptidesPeptide

from .base import register_descriptor

logger = logging.getLogger(__name__)

# Physiological pH used for charge; 7.4 reflects extracellular/blood conditions.
_PHYSIOLOGICAL_PH = 7.4

# Every pK scale the installed `peptides` release actually provides. Deriving
# this from the library's own table (rather than hardcoding) means charge and pI
# are computed under all available scales without risking invalid-scale errors.
_PK_SCALES: tuple[str, ...] = tuple(sorted(peptides.tables.PK))


@register_descriptor("geometry")
def geometry(sequence: str) -> Mapping[str, float]:
    """Trivial but foundational: sequence length."""
    return {"length": float(len(sequence))}


@register_descriptor("charge")
def charge(sequence: str) -> Mapping[str, float]:
    """Net charge at pH 7.4 under several pK scales (one feature per scale)."""
    pep = PeptidesPeptide(sequence)
    out: dict[str, float] = {}
    for scale in _PK_SCALES:
        try:
            out[f"charge_pH7.4_{scale}"] = pep.charge(pH=_PHYSIOLOGICAL_PH, pKscale=scale)
        except (ValueError, KeyError):  # scale not in this peptides release
            logger.debug("charge pK scale %r unavailable; skipping", scale)
    return out


@register_descriptor("isoelectric_point")
def isoelectric_point(sequence: str) -> Mapping[str, float]:
    """Isoelectric point under several pK scales (one feature per scale)."""
    pep = PeptidesPeptide(sequence)
    out: dict[str, float] = {}
    for scale in _PK_SCALES:
        try:
            out[f"pI_{scale}"] = pep.isoelectric_point(pKscale=scale)
        except (ValueError, KeyError):
            logger.debug("pI pK scale %r unavailable; skipping", scale)
    return out


@register_descriptor("hydrophobic_moment")
def hydrophobic_moment(sequence: str) -> Mapping[str, float]:
    """Eisenberg hydrophobic moment for helical (100°) and sheet (180°) geometry.

    The window is capped at the peptide length so short peptides are handled.
    """
    pep = PeptidesPeptide(sequence)
    window = min(11, len(sequence))
    return {
        "hydrophobic_moment_alpha": pep.hydrophobic_moment(window=window, angle=100),
        "hydrophobic_moment_beta": pep.hydrophobic_moment(window=window, angle=180),
    }


@register_descriptor("peptides_props")
def peptides_props(sequence: str) -> Mapping[str, float]:
    """Stability / binding indices from the ``peptides`` library."""
    pep = PeptidesPeptide(sequence)
    return {
        "molecular_weight": pep.molecular_weight(),
        "aliphatic_index": pep.aliphatic_index(),
        "boman_index": pep.boman(),
        "instability_index_peptides": pep.instability_index(),
    }


@register_descriptor("biopython_props")
def biopython_props(sequence: str) -> Mapping[str, float]:
    """GRAVY, aromaticity, instability, pI, and secondary-structure fractions.

    These come from biopython's ProtParam, an independent implementation from
    the ``peptides`` values above; keeping both provides complementary signal.
    """
    pa = ProteinAnalysis(sequence)
    helix, turn, sheet = pa.secondary_structure_fraction()
    return {
        "gravy_kyte_doolittle": pa.gravy(),
        "aromaticity": pa.aromaticity(),
        "instability_index_biopython": pa.instability_index(),
        "molecular_weight_biopython": pa.molecular_weight(),
        "isoelectric_point_biopython": pa.isoelectric_point(),
        "helix_fraction": helix,
        "turn_fraction": turn,
        "sheet_fraction": sheet,
    }
