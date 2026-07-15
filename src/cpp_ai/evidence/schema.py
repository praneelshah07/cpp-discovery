"""Schema for the curated CPP evidence ledger.

The ledger is the platform's *ground-truth memory*: one row per **observation of
one peptide in one experimental context**, sourced from the literature with a
citation. Unlike the bulk databases (POSEIDON = mammalian small-dye uptake only;
CPPsite3 = positives only), the ledger deliberately records **outcomes and
failures** — R9 underperforming, pVEC being *Chlamydomonas*-toxic — so the
scoring logic can be iterated against what actually worked, and where.

Design mirrors :mod:`cpp_ai.core.schema`:

* Entries are ``frozen`` — a curated fact is immutable; corrections make a new
  entry (and the ledger is versioned in git).
* Provenance is mandatory. For literature entries the "source" is a
  :class:`Citation` (DOI/URL), preserved verbatim.
* The peptide is content-addressed via :func:`cpp_ai.core.schema.compute_peptide_id`
  so the same sequence from two papers resolves to the same peptide with two
  observations — exactly the aggregation semantics the SAR layer needs.

Nothing here is a claim of efficacy; every entry is a recorded experimental
result with its citation, and every field is honestly nullable when a paper did
not report it.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.schema import compute_peptide_id

# --------------------------------------------------------------------------- #
# controlled vocabularies (kept small + honest; extend as the ledger grows)
# --------------------------------------------------------------------------- #

#: Biological system the observation was made in. ``algae`` is the target
#: domain and is called out explicitly because it is the whole point of the lab.
Organism = Literal["mammalian", "plant", "algae", "bacterial", "fungal", "other"]

#: What was carried into the cell. Cargo size dominates mechanism, so it is a
#: first-class axis rather than a note (a small dye behaves nothing like a
#: ~27 kDa mCherry fusion).
CargoType = Literal[
    "none",  # peptide alone / label directly on the peptide
    "small_molecule",  # FITC, TAMRA, doxorubicin, ...
    "peptide",
    "protein",  # GFP/mCherry/Cre/... the lab's regime
    "nucleic_acid",  # siRNA, plasmid, ASO
    "nanoparticle",
    "other",
]

#: Normalized qualitative call, assigned by the curator from the paper's own
#: conclusion. This is the label the SAR layer learns winners-vs-losers from.
#: ``toxic`` is separate from ``fail`` on purpose: a peptide can enter cells well
#: yet be unusable because it kills them (the pVEC/9R-in-algae case).
OutcomeCall = Literal["success", "partial", "fail", "toxic", "inconclusive"]

#: Observed cytotoxicity, independent of uptake.
ToxicityCall = Literal["none", "mild", "toxic", "not_reported"]

#: Reported/most-supported internalization mechanism.
Mechanism = Literal["endocytic", "direct", "mixed", "unknown"]

#: How much the curator trusts this row (clarity of the source, directness of the
#: measurement, whether the sequence is unambiguous).
Confidence = Literal["high", "medium", "low"]


class Citation(BaseModel):
    """A literature source for an evidence entry.

    At least one of ``doi`` or ``url`` must be present so every entry is
    auditable back to a real, resolvable source (verified web pulls only — no
    remembered numbers).
    """

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., min_length=1)
    year: Optional[int] = Field(default=None, ge=1950, le=2100)
    authors_short: Optional[str] = Field(
        default=None, description="e.g. 'Futaki et al.' — for compact display."
    )
    doi: Optional[str] = Field(default=None, description="Bare DOI, e.g. '10.1074/jbc.M...'.")
    url: Optional[str] = Field(default=None, description="Resolvable URL (PubMed, publisher, DB).")

    @field_validator("doi")
    @classmethod
    def _strip_doi_prefix(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = value.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if v.lower().startswith(prefix):
                v = v[len(prefix):]
        return v or None

    def resolvable(self) -> str | None:
        """Best single URL to cite, if any."""
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url


class EvidenceEntry(BaseModel):
    """One recorded experimental outcome for one peptide in one context.

    Every quantitative or condition field is optional because real papers omit
    them; the curator records exactly what was reported and nothing more. The
    required core — sequence, system, a normalized ``outcome`` call, and a
    citation — is what makes the row usable as ground truth.
    """

    model_config = ConfigDict(frozen=True)

    # --- identity ---
    peptide_name: str = Field(..., min_length=1, description="Common name, e.g. 'R9', 'TAT', 'pVEC'.")
    sequence: str = Field(..., min_length=1, description="One-letter sequence as tested.")
    peptide_id: str = Field(default="", description="Content hash of the sequence; auto-filled.")

    # --- experimental context ---
    organism: Organism
    cell_type: Optional[str] = Field(
        default=None, description="Specific line/strain, e.g. 'HeLa', 'C. reinhardtii CC-125'."
    )
    cargo_type: CargoType = "none"
    cargo_name: Optional[str] = Field(default=None, description="e.g. 'FITC', 'mCherry', 'siRNA'.")
    cargo_kda: Optional[float] = Field(default=None, ge=0, description="Approx cargo mass in kDa.")
    concentration_um: Optional[float] = Field(default=None, ge=0)
    incubation_min: Optional[float] = Field(default=None, ge=0)
    temperature_c: Optional[float] = Field(default=None)

    # --- outcome ---
    outcome: OutcomeCall
    uptake_value: Optional[float] = Field(
        default=None, description="Quantitative result, if reported (unit in uptake_metric)."
    )
    uptake_metric: Optional[str] = Field(
        default=None, description="What uptake_value means, e.g. '% internalized', 'fold vs TAT'."
    )
    toxicity: ToxicityCall = "not_reported"
    toxicity_note: Optional[str] = None
    mechanism: Mechanism = "unknown"

    # --- provenance + curation ---
    citation: Citation
    confidence: Confidence = "medium"
    notes: Optional[str] = None

    @field_validator("sequence")
    @classmethod
    def _canonicalize(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("Peptide sequence must not be empty.")
        if not normalized.isalpha():
            raise ValueError(f"Peptide sequence contains non-letter characters: {value!r}")
        return normalized

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        """Fill the content-addressed id from the sequence if not supplied."""
        expected = compute_peptide_id(self.sequence)
        if not self.peptide_id:
            object.__setattr__(self, "peptide_id", expected)
        elif self.peptide_id != expected:
            raise ValueError(
                f"peptide_id {self.peptide_id!r} does not match the sequence hash "
                f"(expected {expected!r})."
            )

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def is_positive(self) -> bool:
        """True for a usable win: entered cells and was not toxic."""
        return self.outcome in ("success", "partial") and self.toxicity != "toxic"
