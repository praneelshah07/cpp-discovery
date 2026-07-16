"""Shared recommendation backbone — one logic path for every front-end.

`recommend_for_algae` is *the* entry point: it loads the CPP library, the curated
evidence ledger, and the algae-fit model, runs the evidence-backed scoring, and
returns a ranked, filtered result. The Streamlit app and the CLI both call it, so
the science lives in exactly one place.

It answers two questions your grad student actually asked:

* **"beyond just pVEC variants"** — ``max_identity`` drops near-duplicates of the
  anchor so genuinely different peptides surface.
* **"which are more beneficial to algae, not just similar-looking"** —
  ``rank_by="algae_fit"`` orders by the empirical algae-winner profile (from the
  ledger's SAR), not by resemblance to the anchor.

Honesty is unchanged: with little algae data these are prioritized hypotheses for
wet-lab testing, not efficacy predictions. Run as a batch job:

    python -m cpp_ai.pipeline --anchor pVEC --rank-by algae_fit --top 25 \
        --out algae_candidates.csv --report algae_report.md
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, Sequence

from .evidence import EvidenceLedger
from .scoring import AlgaeFitScorer, EvidenceProfile, EvidenceScorer
from .scoring.positional import CriticalPositionProfile
from .screening import load_cppsite3_library
from .screening.candidate import ScreenCandidate
from .similarity.features import PeptideFeatures
from .similarity.metrics import SequenceIdentity

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data" / "raw" / "cppsite3_api.json"
_LEDGER_PATH = _ROOT / "data" / "curated" / "cpp_evidence_ledger.json"
_CLASSIFIER = _ROOT / "data" / "processed" / "cpp_classifier.pkl"

RankBy = Literal["blend", "algae_fit"]

# Canonical anchors. pVEC is the default: it has real *Chlamydomonas* data,
# whereas ClWOX is not yet demonstrated in algae (plant vs algal cell-wall
# chemistry differ), so ClWOX stays available but is not the default.
ANCHOR_PRESETS: dict[str, str] = {
    "pVEC": "LLIILRRRIRKQAHAHSK",
    "pVEC-R6A": "LLIILARRIRKQAHAHSK",
    "pVEC-wt": "LLIILRRRIRKQAHAHSK",
    "ClWOX": "TNVYNWFQNRRARTKRK",
}


def resolve_anchor(anchor: str) -> str:
    """Accept a preset name or a raw one-letter sequence; return the sequence."""
    if anchor in ANCHOR_PRESETS:
        return ANCHOR_PRESETS[anchor]
    seq = anchor.strip().upper()
    if seq and seq.isalpha():
        return seq
    raise ValueError(
        f"Anchor {anchor!r} is neither a known preset ({', '.join(ANCHOR_PRESETS)}) "
        "nor a valid one-letter sequence."
    )


@dataclass(frozen=True)
class AlgaeRecommendation:
    """Ranked, filtered recommendation output shared by all front-ends."""

    anchor: str
    algae_mode: bool
    rank_by: RankBy
    profiles: list[EvidenceProfile]
    n_library: int
    n_before_filter: int
    fit_scorer: AlgaeFitScorer | None

    def to_dataframe(self) -> Any:
        """Flat pandas DataFrame, one row per recommended peptide (pandas lazy)."""
        import pandas as pd

        rows: list[dict[str, object]] = []
        for p in self.profiles:
            row: dict[str, object] = {
                "peptide": p.name,
                "sequence": p.sequence,
                "length": len(p.sequence),
                "net_charge": p.net_charge,
            }
            if p.algae_fit is not None:
                row["algae_fit"] = round(p.algae_fit, 3)
            row.update(
                {
                    "physchem_similarity": round(p.physchem, 3),
                    "shared_motif": round(p.motif_local, 3),
                    "sequence_identity": round(p.global_identity, 3),
                    "overall_match": round(p.shortlist_score, 3),
                }
            )
            if p.cpp_probability is not None:
                row["cpp_likelihood"] = round(p.cpp_probability, 3)
            row["toxicity_risk"] = p.toxicity_flag
            row["lysis_risk"] = round(p.lysis_risk, 2)
            row["confidence"] = p.ad_confidence
            row["evidence"] = p.evidence
            rows.append(row)
        return pd.DataFrame(rows)

    def to_markdown(self, top_k: int = 25) -> str:
        """A short, honest report suitable for a lab notebook or a PR."""
        shown = self.profiles[:top_k]
        lines = [
            f"# Algae-delivery CPP candidates (anchor: {self.anchor})",
            "",
            f"- ranked by: **{self.rank_by}**"
            + ("  (empirical algae-winner profile)" if self.rank_by == "algae_fit" else ""),
            f"- algae-fit model: {'on' if self.algae_mode else 'off'}",
            f"- library screened: {self.n_library} CPPs → {self.n_before_filter} after "
            f"filters → showing top {len(shown)}",
            "",
            "> Computational hypotheses for wet-lab testing, **not** algae-uptake "
            "predictions. The algae-fit signal is learned from a small curated "
            "evidence ledger and sharpens with new data.",
            "",
        ]
        if self.algae_mode and self.fit_scorer is not None and self.fit_scorer.is_informative:
            prefs = ", ".join(
                f"{t.descriptor} {t.prefers}" for t in self.fit_scorer.terms
            )
            lines += [f"**Algae-fit prefers:** {prefs}", ""]
        header = "| # | peptide | seq | len | chg | algae_fit | lysis | phys | motif | id | tox |"
        sep = "|---|---|---|---|---|---|---|---|---|---|---|"
        lines += [header, sep]
        for i, p in enumerate(shown, 1):
            af = f"{p.algae_fit:.2f}" if p.algae_fit is not None else "—"
            lines.append(
                f"| {i} | {p.name} | `{p.sequence}` | {len(p.sequence)} | "
                f"{p.net_charge:+d} | {af} | {p.lysis_risk:.2f} | {p.physchem:.2f} | "
                f"{p.motif_local:.2f} | {p.global_identity:.2f} | {p.toxicity_flag} |"
            )
        return "\n".join(lines) + "\n"


def _load_classifier() -> Any:
    if not _CLASSIFIER.exists():
        return None
    with open(_CLASSIFIER, "rb") as fh:
        return pickle.load(fh)


def _algae_priority(p: EvidenceProfile) -> float:
    """Algae benefit discounted by membrane-lysis risk.

    A peptide only counts as algae-good if it is *also* gentle: an amphipathic,
    net-hydrophobic, poorly-buffered peptide (TP10/MAP) can score high algae-fit
    yet be membrane-lytic. Discounting by ``lysis_risk`` demotes exactly those
    false positives, separating them from pVEC (low lysis risk)."""
    return (p.algae_fit or 0.0) * (1.0 - p.lysis_risk)


def _collapse_families(
    profiles: Sequence[EvidenceProfile], threshold: float
) -> list[EvidenceProfile]:
    """Greedily keep the top-ranked representative of each near-duplicate family.

    Uses a fast stdlib string-similarity ratio (not biological alignment): family
    collapse only needs "is this basically the same scaffold?", so difflib is
    both sufficient and cheap. Assumes ``profiles`` is already ranked best-first.
    """
    reps: list[EvidenceProfile] = []
    for p in profiles:
        if all(
            SequenceMatcher(None, p.sequence, r.sequence).ratio() < threshold for r in reps
        ):
            reps.append(p)
    return reps


def filter_and_rank(
    profiles: Sequence[EvidenceProfile],
    anchor_seq: str,
    *,
    low_toxicity: bool = True,
    max_identity: float | None = None,
    max_lysis_risk: float | None = None,
    rank_by: RankBy = "blend",
    collapse_families: float | None = None,
) -> list[EvidenceProfile]:
    """The shared filter + rank *policy*, used by both the app and the CLI.

    Kept separate from scoring so the (expensive, cached) EvidenceScorer step can
    be reused by front-ends while this cheap policy runs every interaction. Drops
    the anchor, optionally toxic / membrane-lytic candidates and near-variants of
    the anchor, orders by ``rank_by`` (algae-fit is discounted by lysis risk),
    then optionally collapses near-duplicate scaffolds to one representative.
    """
    identity = SequenceIdentity()
    anchor_feat = PeptideFeatures(sequence=anchor_seq)

    kept: list[EvidenceProfile] = []
    for p in profiles:
        if p.sequence == anchor_seq:
            continue
        if low_toxicity and (p.lytic_risk or p.toxicity_flag == "HIGH-RISK"):
            continue
        if max_lysis_risk is not None and p.lysis_risk >= max_lysis_risk:
            continue
        if max_identity is not None:
            gid = identity(anchor_feat, PeptideFeatures(sequence=p.sequence))
            if gid >= max_identity:
                continue
        kept.append(p)

    if rank_by == "algae_fit":
        kept.sort(key=lambda p: (_algae_priority(p), p.shortlist_score), reverse=True)
    # "blend" preserves EvidenceScorer's existing shortlist ordering.

    if collapse_families is not None:
        kept = _collapse_families(kept, collapse_families)
    return kept


def recommend_for_algae(
    anchor: str = "pVEC",
    *,
    library: Sequence[ScreenCandidate] | None = None,
    ledger: EvidenceLedger | None = None,
    classifier: Any | None = None,
    top_k: int | None = 25,
    algae_mode: bool = True,
    low_toxicity: bool = True,
    rank_by: RankBy = "blend",
    max_identity: float | None = None,
    max_lysis_risk: float | None = None,
    collapse_families: float | None = None,
    critical_profile: CriticalPositionProfile | None = None,
    embedding_service: object | None = None,
) -> AlgaeRecommendation:
    """Rank library CPPs for algae delivery against an anchor. The one logic path.

    Parameters
    ----------
    anchor: preset name (``pVEC``, ``pVEC-R6A``, ``ClWOX``) or a raw sequence.
    top_k: keep this many after filtering; ``None`` returns all (front-ends slice).
    algae_mode: enable the ledger-derived algae-fit axis.
    rank_by: ``"blend"`` (transparent multi-axis order) or ``"algae_fit"`` (order
        by the empirical algae-winner profile — "beneficial, not just similar").
    max_identity: drop candidates whose global identity to the anchor is >= this,
        to look **beyond near-variants** of the anchor.
    """
    anchor_seq = resolve_anchor(anchor)
    lib = list(library) if library is not None else load_cppsite3_library(_DATA)
    led = ledger if ledger is not None else EvidenceLedger.load(_LEDGER_PATH)
    clf = classifier if classifier is not None else _load_classifier()

    fit: AlgaeFitScorer | None = None
    if algae_mode and len(led):
        fit = AlgaeFitScorer.from_ledger(led, [c.sequence for c in lib])
        if not fit.is_informative:
            fit = None

    if rank_by == "algae_fit" and fit is None:
        rank_by = "blend"  # honest fallback: no informative algae signal available

    scorer = EvidenceScorer(
        lib, embedding_service=embedding_service, classifier=clf, algae_fit_scorer=fit
    )
    profiles = scorer.profile(anchor_seq, critical_profile=critical_profile)
    n_before = len(profiles)

    kept = filter_and_rank(
        profiles,
        anchor_seq,
        low_toxicity=low_toxicity,
        max_identity=max_identity,
        max_lysis_risk=max_lysis_risk,
        rank_by=rank_by,
        collapse_families=collapse_families,
    )

    if top_k is not None:
        kept = kept[:top_k]

    return AlgaeRecommendation(
        anchor=anchor_seq,
        algae_mode=algae_mode,
        rank_by=rank_by,
        profiles=kept,
        n_library=len(lib),
        n_before_filter=n_before,
        fit_scorer=fit,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cpp_ai.pipeline",
        description="Rank CPPs for algae delivery (evidence-backed). Writes CSV + report.",
    )
    p.add_argument("--anchor", default="pVEC",
                   help=f"preset ({', '.join(ANCHOR_PRESETS)}) or a raw sequence [pVEC]")
    p.add_argument("--top", type=int, default=25, help="how many candidates to keep [25]")
    p.add_argument("--rank-by", choices=["blend", "algae_fit"], default="blend",
                   help="'algae_fit' = most algae-beneficial profile, not just similar [blend]")
    p.add_argument("--max-identity", type=float, default=None, metavar="FRAC",
                   help="drop candidates with >= this identity to the anchor (e.g. 0.6) "
                        "to look beyond near-variants")
    p.add_argument("--max-lysis", type=float, default=None, metavar="RISK",
                   help="drop candidates with membrane-lysis risk >= this (e.g. 0.5) "
                        "to exclude AMP-like/lytic scaffolds such as the TP10 family")
    p.add_argument("--collapse-families", type=float, default=None, metavar="RATIO",
                   help="collapse near-duplicate scaffolds (e.g. 0.7) to one "
                        "representative each, for a diverse panel")
    p.add_argument("--no-algae", action="store_true", help="disable the algae-fit axis")
    p.add_argument("--allow-toxic", action="store_true",
                   help="keep high-charge / membrane-lytic candidates")
    p.add_argument("--out", default="algae_candidates.csv", help="CSV output path")
    p.add_argument("--report", default=None, help="optional markdown report path")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rec = recommend_for_algae(
        anchor=args.anchor,
        top_k=args.top,
        algae_mode=not args.no_algae,
        low_toxicity=not args.allow_toxic,
        rank_by=args.rank_by,
        max_identity=args.max_identity,
        max_lysis_risk=args.max_lysis,
        collapse_families=args.collapse_families,
    )
    df = rec.to_dataframe()
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(
        f"Ranked {rec.n_library} CPPs vs {args.anchor} "
        f"(rank_by={rec.rank_by}, algae={'on' if rec.algae_mode else 'off'}); "
        f"wrote {len(df)} candidates -> {out}"
    )
    if args.report:
        rp = Path(args.report)
        rp.write_text(rec.to_markdown(top_k=args.top), encoding="utf-8")
        print(f"Wrote report -> {rp}")
    if rec.profiles:
        top = rec.profiles[0]
        af = f", algae_fit={top.algae_fit:.2f}" if top.algae_fit is not None else ""
        print(f"Top hit: {top.name} ({top.sequence}){af}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
