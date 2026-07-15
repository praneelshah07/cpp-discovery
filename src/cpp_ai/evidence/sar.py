"""Structure–activity trend analysis over the evidence ledger.

Given the curated outcomes, this asks the empirical question the recommender
cannot: *within a biological context, which physicochemical properties separate
the CPPs that worked from the ones that didn't?* It joins each ledger peptide to
the tested descriptor battery and contrasts winners against the rest.

Honesty guardrails baked in:

* Contrasts are **stratified by organism** — a mammalian winner and an algae
  winner are different questions (POSEIDON already showed conditions carry as
  much signal as sequence).
* Peptides are **de-duplicated within a context** so a heavily-studied peptide
  cannot dominate a mean.
* Sample sizes are reported on every trend. With a seed-sized ledger these are
  **descriptive contrasts, not statistical inference**, and the report says so.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..descriptors import compute_descriptors
from .store import EvidenceLedger

# A compact, interpretable descriptor panel. Deliberately small: these are the
# axes a bench scientist can reason about, not the full 210-feature battery.
TREND_DESCRIPTORS: tuple[str, ...] = (
    "length",
    "charge_pH7.4_Lehninger",
    "frac_group_cationic",
    "frac_R",
    "frac_K",
    "frac_group_hydrophobic",
    "aromaticity",
    "hydrophobic_moment_alpha",
    "gravy_kyte_doolittle",
    "boman_index",
    "aliphatic_index",
)

# Human-readable direction hints for the headline reading.
_NICE_NAME = {
    "charge_pH7.4_Lehninger": "net charge (pH 7.4)",
    "frac_group_cationic": "cationic fraction",
    "frac_group_hydrophobic": "hydrophobic fraction",
    "hydrophobic_moment_alpha": "amphipathicity (helical µH)",
    "gravy_kyte_doolittle": "GRAVY hydrophobicity",
    "boman_index": "Boman (binding) index",
    "aliphatic_index": "aliphatic index",
}

_SUCCESS = "success"


def _nice(name: str) -> str:
    return _NICE_NAME.get(name, name)


@dataclass(frozen=True)
class DescriptorContrast:
    """How one descriptor differs between winners and non-winners in a context."""

    descriptor: str
    winner_mean: float
    other_mean: float
    pooled_sd: float

    @property
    def delta(self) -> float:
        return self.winner_mean - self.other_mean

    @property
    def effect(self) -> float:
        """Standardized mean difference (Cohen's-d-like); 0 if no spread."""
        return self.delta / self.pooled_sd if self.pooled_sd > 0 else 0.0

    @property
    def direction(self) -> str:
        if self.delta > 0:
            return "higher"
        if self.delta < 0:
            return "lower"
        return "same"

    def sentence(self) -> str:
        return (
            f"{_nice(self.descriptor)}: winners {self.direction} "
            f"({self.winner_mean:.2f} vs {self.other_mean:.2f}, d={self.effect:+.2f})"
        )


@dataclass(frozen=True)
class TrendReport:
    """SAR contrast for one biological context."""

    organism: str
    winners: list[str]  # peptide names counted as winners (success, non-toxic)
    others: list[str]  # peptide names counted as non-winners
    contrasts: list[DescriptorContrast] = field(default_factory=list)

    @property
    def n_winners(self) -> int:
        return len(self.winners)

    @property
    def n_others(self) -> int:
        return len(self.others)

    @property
    def is_underpowered(self) -> bool:
        return self.n_winners < 2 or self.n_others < 2

    def top_contrasts(self, k: int = 5) -> list[DescriptorContrast]:
        return sorted(self.contrasts, key=lambda c: abs(c.effect), reverse=True)[:k]

    def to_text(self) -> str:
        lines = [
            f"=== {self.organism.upper()} — winners vs others ===",
            f"winners ({self.n_winners}): {', '.join(self.winners) or '—'}",
            f"others  ({self.n_others}): {', '.join(self.others) or '—'}",
        ]
        if self.n_winners == 0 or self.n_others == 0:
            lines.append("(no contrast possible — need both winners and non-winners)")
            return "\n".join(lines)
        if self.is_underpowered:
            lines.append("⚠ underpowered: descriptive only, not statistical inference.")
        lines.append("top separating properties:")
        for c in self.top_contrasts():
            lines.append(f"  • {c.sentence()}")
        return "\n".join(lines)


def _winner_status(ledger: EvidenceLedger, organism: str) -> tuple[dict[str, str], dict[str, bool]]:
    """Return {peptide_name -> sequence} and {peptide_name -> is_winner} in a context.

    A peptide counts as a winner if *any* of its observations in this context is
    a non-toxic success; otherwise it is a non-winner. Toxic-anywhere demotes.
    """
    seqs: dict[str, str] = {}
    best_success: dict[str, bool] = {}
    toxic: dict[str, bool] = {}
    for e in ledger.for_organism(organism):
        seqs[e.peptide_name] = e.sequence
        if e.outcome == _SUCCESS and e.toxicity != "toxic":
            best_success[e.peptide_name] = True
        best_success.setdefault(e.peptide_name, False)
        if e.toxicity == "toxic":
            toxic[e.peptide_name] = True
    is_winner = {
        name: (best_success.get(name, False) and not toxic.get(name, False)) for name in seqs
    }
    return seqs, is_winner


def analyze_context(ledger: EvidenceLedger, organism: str) -> TrendReport:
    """Contrast winner vs non-winner descriptors within one organism context."""
    seqs, is_winner = _winner_status(ledger, organism)
    winners = sorted(n for n, w in is_winner.items() if w)
    others = sorted(n for n, w in is_winner.items() if not w)

    # descriptor vectors per unique peptide (skip non-canonical, which can't be scored)
    vecs: dict[str, dict[str, float]] = {}
    for name, seq in seqs.items():
        try:
            ds = compute_descriptors(seq, blocks=None)
        except Exception:
            continue
        vecs[name] = {d: ds.values[d] for d in TREND_DESCRIPTORS if d in ds.values}

    contrasts: list[DescriptorContrast] = []
    if winners and others:
        for d in TREND_DESCRIPTORS:
            w_vals = [vecs[n][d] for n in winners if n in vecs and d in vecs[n]]
            o_vals = [vecs[n][d] for n in others if n in vecs and d in vecs[n]]
            if not w_vals or not o_vals:
                continue
            all_vals = w_vals + o_vals
            pooled_sd = statistics.pstdev(all_vals) if len(all_vals) > 1 else 0.0
            contrasts.append(
                DescriptorContrast(
                    descriptor=d,
                    winner_mean=statistics.fmean(w_vals),
                    other_mean=statistics.fmean(o_vals),
                    pooled_sd=pooled_sd,
                )
            )

    return TrendReport(organism=organism, winners=winners, others=others, contrasts=contrasts)


def analyze_all(ledger: EvidenceLedger) -> list[TrendReport]:
    """One trend report per organism present in the ledger, ordered by name."""
    organisms = sorted({e.organism for e in ledger})
    return [analyze_context(ledger, org) for org in organisms]
