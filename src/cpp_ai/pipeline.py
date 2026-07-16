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


# --------------------------------------------------------------------------- #
# family tag, "why recommended", and hypothesis categories
# --------------------------------------------------------------------------- #
# Coarse mechanistic family from sequence — a heuristic label so a panel of ten
# hits reveals as ~three mechanisms, not ten. Not a phylogeny.
def peptide_family(sequence: str) -> str:
    s = sequence.upper()
    n = len(s) or 1
    frac_rk = sum(c in "RK" for c in s) / n
    if "WFQN" in s:
        return "Homeodomain"
    if "INLKALAALAKKIL" in s or "GYLLG" in s:
        return "Transportan"
    if s.startswith("LLIIL") or "RKQAHAH" in s:
        return "pVEC-like"
    if frac_rk > 0.7:
        return "Polyarginine/cationic"
    if "YGRKKRR" in s or "RRRQRRR" in s:
        return "TAT-like"
    return "Amphipathic/other"


# nice display names for the algae-fit descriptors referenced in reasons
_NICE_DESC = {
    "hydrophobic_moment_alpha": "amphipathicity",
    "aliphatic_index": "aliphatic content",
    "aromaticity": "low aromaticity",
    "gravy_kyte_doolittle": "hydrophobicity balance",
    "frac_group_hydrophobic": "hydrophobic fraction",
    "frac_group_cationic": "moderate charge",
}


@dataclass(frozen=True)
class Reason:
    """One human-readable reason a candidate was (or wasn't) recommended."""

    text: str
    positive: bool


def explain_profile(
    profile: EvidenceProfile, *, fit_scorer: AlgaeFitScorer | None = None
) -> list[Reason]:
    """Turn a candidate's scoring axes into plain positive/negative reasons.

    This is the "why was this recommended?" panel — it makes the ranking a
    transparent, critiqueable argument rather than a black-box number.
    """
    r: list[Reason] = []
    if profile.physchem_percentile >= 0.7:
        r.append(Reason(
            f"High physicochemical similarity to anchor "
            f"({profile.physchem_percentile * 100:.0f}th pct)", True))
    if profile.algae_fit is not None:
        if profile.algae_fit >= 0.6:
            extra = ""
            if fit_scorer is not None:
                top = sorted(
                    fit_scorer.explain(profile.sequence),
                    key=lambda c: c.contribution, reverse=True,
                )
                if top and top[0].contribution > 0:
                    extra = f" (esp. {_NICE_DESC.get(top[0].descriptor, top[0].descriptor)})"
            r.append(Reason(f"Matches the algae-winner profile{extra}", True))
        elif profile.algae_fit < 0.4:
            r.append(Reason("Weak match to the algae-winner profile", False))
    if profile.lysis_risk < 0.3:
        r.append(Reason("Low predicted membrane-lysis (gentle)", True))
    elif profile.lysis_risk > 0.6:
        r.append(Reason("Membrane-lytic (AMP-like) risk", False))
    if profile.cpp_probability is not None and profile.cpp_probability >= 0.8:
        r.append(Reason(f"High CPP-classifier probability "
                        f"({profile.cpp_probability * 100:.0f}%)", True))
    if profile.motif_local >= 0.5:
        r.append(Reason("Shares a sequence motif with the anchor", True))
    elif profile.motif_local < 0.2:
        r.append(Reason("Low motif similarity to the anchor", False))
    if profile.ad_confidence == "high":
        r.append(Reason("In the well-characterized region (high confidence)", True))
    elif profile.ad_confidence == "low":
        r.append(Reason("Unusual sequence — treat scores cautiously (low confidence)", False))
    if "experimental" in profile.evidence:
        r.append(Reason("Experimentally-studied CPP", True))
    else:
        r.append(Reason("Computational candidate (no direct evidence)", False))
    if profile.genetically_encodable:
        r.append(Reason("Cloneable — tested as the bare sequence (mCherry-fusion ready)", True))
    else:
        r.append(Reason(
            f"Tested form is modified ({profile.modification}); the naked-sequence "
            f"scores may not transfer to a plain fusion", False))
    return r


@dataclass(frozen=True)
class Category:
    """A labelled bucket of candidates representing one testable hypothesis."""

    key: str
    title: str
    rationale: str
    profiles: list[EvidenceProfile]


def categorize(
    profiles: Sequence[EvidenceProfile],
    anchor_seq: str,
    *,
    per_bucket: int = 3,
    keys: Sequence[str] | None = None,
) -> list[Category]:
    """Partition ranked candidates into distinct hypothesis buckets.

    The point (per the review): give the wet lab *options* — a few genuinely
    different bets — instead of twenty near-identical peptides. A peptide may
    appear in more than one bucket; that overlap is itself informative.
    """
    pool = list(profiles)
    cats: list[Category] = []

    def add(key: str, title: str, rationale: str, ordered: list[EvidenceProfile]) -> None:
        cats.append(Category(key, title, rationale, ordered[:per_bucket]))

    add("closest", "Closest to the anchor", "Highest physicochemical similarity",
        sorted(pool, key=lambda p: p.physchem, reverse=True))
    add("novel", "Most novel scaffold", "Least sequence-similar to the anchor",
        sorted(pool, key=lambda p: SequenceMatcher(None, p.sequence, anchor_seq).ratio()))
    add("gentle", "Gentlest (lowest lysis risk)", "Least membrane-perturbing",
        sorted(pool, key=lambda p: p.lysis_risk))
    encodable = [p for p in pool if p.genetically_encodable]
    if encodable:
        add("cloneable", "Cloneable (encodable as tested)",
            "Tested as the bare sequence — mCherry-fusion ready",
            sorted(encodable, key=lambda p: p.shortlist_score, reverse=True))
    if any(p.algae_fit is not None for p in pool):
        add("algae", "Strongest algae profile",
            "Best algae-winner descriptors, discounted by lysis risk",
            sorted(pool, key=lambda p: (p.algae_fit or 0.0) * (1.0 - p.lysis_risk), reverse=True))
    if any(p.cpp_probability is not None for p in pool):
        add("cpp", "Highest CPP confidence", "Top CPP-classifier probability",
            sorted(pool, key=lambda p: p.cpp_probability or 0.0, reverse=True))
    else:
        add("confident", "Highest applicability confidence",
            "In the well-characterized region of known CPPs",
            sorted(pool, key=lambda p: (p.ad_confidence == "high", p.physchem), reverse=True))

    if keys is not None:
        cats = [c for c in cats if c.key in keys]
    return cats


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
                "family": peptide_family(p.sequence),
                "length": len(p.sequence),
                "net_charge": p.net_charge,
            }
            if p.algae_fit is not None:
                row["usable_delivery"] = round(usable_delivery(p), 3)
                row["algae_suitability"] = round(p.algae_fit, 3)
                row["fusion_confidence"] = round(p.fusion_confidence, 2)
            row.update(
                {
                    "physchem_similarity": round(p.physchem, 3),
                    "shared_motif": round(p.motif_local, 3),
                    "sequence_identity": round(p.global_identity, 3),
                    "composite_score": round(p.shortlist_score, 3),
                }
            )
            if p.cpp_probability is not None:
                row["cpp_likelihood"] = round(p.cpp_probability, 3)
            row["toxicity_risk"] = p.toxicity_flag
            row["lysis_risk"] = round(p.lysis_risk, 2)
            row["confidence"] = p.ad_confidence
            row["evidence"] = p.evidence
            row["tested_form"] = p.modification
            row["genetically_encodable"] = p.genetically_encodable
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

    def to_markdown_categorized(
        self, *, per_bucket: int = 3, keys: Sequence[str] | None = None
    ) -> str:
        """Report as distinct hypothesis buckets, each with reasons — options, not a list."""
        cats = categorize(self.profiles, self.anchor, per_bucket=per_bucket, keys=keys)
        lines = [
            f"# Algae-delivery hypotheses (anchor: {self.anchor})",
            "",
            "> Distinct bets for the wet lab, not one ranked list. Computational "
            "hypotheses for testing, **not** algae-uptake predictions.",
            "",
        ]
        for c in cats:
            lines += [f"## {c.title}", f"_{c.rationale}_", ""]
            for p in c.profiles:
                af = f", algae-suit {p.algae_fit:.2f}" if p.algae_fit is not None else ""
                clone = "cloneable" if p.genetically_encodable else "modified-as-tested"
                lines.append(
                    f"- **{p.name}** `{p.sequence}` "
                    f"({peptide_family(p.sequence)}, {p.net_charge:+d}, "
                    f"lysis {p.lysis_risk:.2f}, {clone}{af})"
                )
                reasons = explain_profile(p, fit_scorer=self.fit_scorer)
                for rsn in reasons:
                    lines.append(f"    - {'✓' if rsn.positive else '✕'} {rsn.text}")
            lines.append("")
        return "\n".join(lines) + "\n"


def _load_classifier() -> Any:
    if not _CLASSIFIER.exists():
        return None
    with open(_CLASSIFIER, "rb") as fh:
        return pickle.load(fh)


def usable_delivery(p: EvidenceProfile) -> float:
    """Usable algae-delivery score: ``algae_fit × (1 − lysis)² × fusion_confidence``.

    The **squared** lysis term (per external review) penalizes membrane-lytic
    peptides aggressively — melittin/transportan should not sit near a balanced
    peptide just because the linear lysis discount was gentle. ``fusion_confidence``
    folds modification-awareness in: a peptide whose function may depend on a
    non-encodable conjugate is discounted, so the naked cloneable candidates win.
    Returns ``algae_fit`` unchanged when no algae signal is available (fit None)."""
    if p.algae_fit is None:
        return 0.0
    return p.algae_fit * (1.0 - p.lysis_risk) ** 2 * p.fusion_confidence


def _algae_priority(p: EvidenceProfile) -> float:
    return usable_delivery(p)


@dataclass(frozen=True)
class FamilyGroup:
    """A near-duplicate scaffold: a top representative plus all ranked members."""

    representative: EvidenceProfile
    members: list[EvidenceProfile]  # ranked, includes the representative

    @property
    def size(self) -> int:
        return len(self.members)


def group_families(
    profiles: Sequence[EvidenceProfile], threshold: float = 0.7
) -> list[FamilyGroup]:
    """Group near-duplicate scaffolds, keeping the top-ranked one as representative.

    Uses a fast stdlib string-similarity ratio (not biological alignment): grouping
    only needs "is this basically the same scaffold?". Assumes ``profiles`` is
    already ranked best-first, so each group's first member is its representative.
    Lets the UI show a diverse list yet expand any scaffold to its ranked variants.
    """
    groups: list[tuple[EvidenceProfile, list[EvidenceProfile]]] = []
    for p in profiles:
        for rep, members in groups:
            if SequenceMatcher(None, p.sequence, rep.sequence).ratio() >= threshold:
                members.append(p)
                break
        else:
            groups.append((p, [p]))
    return [FamilyGroup(representative=r, members=m) for r, m in groups]


def _collapse_families(
    profiles: Sequence[EvidenceProfile], threshold: float
) -> list[EvidenceProfile]:
    """Keep the top-ranked representative of each near-duplicate family."""
    return [g.representative for g in group_families(profiles, threshold)]


def filter_and_rank(
    profiles: Sequence[EvidenceProfile],
    anchor_seq: str,
    *,
    low_toxicity: bool = True,
    max_identity: float | None = None,
    max_lysis_risk: float | None = None,
    require_encodable: bool = False,
    rank_by: RankBy = "blend",
    collapse_families: float | None = None,
) -> list[EvidenceProfile]:
    """The shared filter + rank *policy*, used by both the app and the CLI.

    Kept separate from scoring so the (expensive, cached) EvidenceScorer step can
    be reused by front-ends while this cheap policy runs every interaction. Drops
    the anchor, optionally toxic / membrane-lytic / non-encodable candidates and
    near-variants of the anchor, orders by ``rank_by`` (algae-fit is discounted by
    lysis risk), then optionally collapses near-duplicate scaffolds.
    """
    identity = SequenceIdentity()
    anchor_feat = PeptideFeatures(sequence=anchor_seq)

    kept: list[EvidenceProfile] = []
    for p in profiles:
        if p.sequence == anchor_seq:
            continue
        if low_toxicity and (p.lytic_risk or p.toxicity_flag == "HIGH-RISK"):
            continue
        if require_encodable and not p.genetically_encodable:
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
    require_encodable: bool = False,
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
        require_encodable=require_encodable,
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
    p.add_argument("--encodable-only", action="store_true",
                   help="keep only candidates tested as the bare sequence (cloneable, "
                        "mCherry-fusion ready — no amidation/lipidation/conjugation)")
    p.add_argument("--no-algae", action="store_true", help="disable the algae-fit axis")
    p.add_argument("--allow-toxic", action="store_true",
                   help="keep high-charge / membrane-lytic candidates")
    p.add_argument("--out", default="algae_candidates.csv", help="CSV output path")
    p.add_argument("--report", default=None, help="optional markdown report path")
    p.add_argument("--categorize", action="store_true",
                   help="write the report as distinct hypothesis buckets (options, not a list)")
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
        require_encodable=args.encodable_only,
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
        md = rec.to_markdown_categorized() if args.categorize else rec.to_markdown(top_k=args.top)
        rp.write_text(md, encoding="utf-8")
        print(f"Wrote {'categorized ' if args.categorize else ''}report -> {rp}")
    if rec.profiles:
        top = rec.profiles[0]
        af = f", algae_fit={top.algae_fit:.2f}" if top.algae_fit is not None else ""
        print(f"Top hit: {top.name} ({top.sequence}){af}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
