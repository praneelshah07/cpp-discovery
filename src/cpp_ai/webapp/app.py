"""CPP Discovery — a recommendation platform (Streamlit).

Given an anchor peptide (e.g. ClWOX), it recommends the most similar cloneable
CPPs from the CPPsite3 database, with each piece of evidence shown as its own
plain-language column. Thin layer over the tested cpp_ai.scoring / screening
modules. Run:  streamlit run src/cpp_ai/webapp/app.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

# --- import bootstrap (must precede any cpp_ai import) --------------------- #
# Make the source tree importable directly, independent of how the package was
# installed. Streamlit Cloud can leave a stale non-editable `cpp_ai` in
# site-packages that predates newly-added subpackages (e.g. cpp_ai.evidence);
# prepending src/ ensures the live source — which always has every module —
# wins on sys.path.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from cpp_ai.evidence import EvidenceLedger  # noqa: E402
from cpp_ai.generation import net_charge  # noqa: E402
from cpp_ai.scoring import (  # noqa: E402
    AlgaeFitScorer,
    EvidenceScorer,
    is_trained_model_available,
)
from cpp_ai.pipeline import (  # noqa: E402
    CARGO_CONTEXT,
    CELL_WALL_CONTEXT,
    charge_density,
    explain_profile,
    filter_and_rank,
    fusion_charge_estimate,
    group_families,
    peptide_family,
    usable_delivery,
)
from cpp_ai.screening import load_cppsite3_library  # noqa: E402
from cpp_ai.screening.candidate import ScreenCandidate  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data" / "raw" / "cppsite3_api.json"
_LEDGER = _ROOT / "data" / "curated" / "cpp_evidence_ledger.json"

_PRESETS = {
    "pVEC-R6A (your construct — algae-proven)": "LLIILARRIRKQAHAHSK",
    "pVEC (wild-type — algae-proven)": "LLIILRRRIRKQAHAHSK",
    "ClWOX (plant homeoprotein; not yet shown in algae)": "TNVYNWFQNRRARTKRK",
    "Custom…": "",
}

st.set_page_config(page_title="CPP Discovery — Recommendations", layout="wide")


# --------------------------------------------------------------------------- #
# data + heavy resources (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _library() -> list[ScreenCandidate]:
    return load_cppsite3_library(_DATA) if _DATA.exists() else []


@st.cache_resource(show_spinner=False)
def _classifier() -> Any:
    path = _ROOT / "data" / "processed" / "cpp_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_resource(show_spinner=False)
def _ledger() -> EvidenceLedger:
    return EvidenceLedger.load(_LEDGER) if _LEDGER.exists() else EvidenceLedger()


@st.cache_resource(show_spinner=False)
def _algae_fit() -> AlgaeFitScorer | None:
    led = _ledger()
    if not len(led):
        return None
    return AlgaeFitScorer.from_ledger(led, [c.sequence for c in _library()])


@st.cache_resource(show_spinner=True)
def _scorer(use_algae_fit: bool) -> EvidenceScorer:
    fit = _algae_fit() if use_algae_fit else None
    return EvidenceScorer(_library(), classifier=_classifier(), algae_fit_scorer=fit)


@st.cache_data(show_spinner="Scoring the library for algae delivery…")
def _profiles(anchor: str, use_algae_fit: bool) -> list[Any]:
    """Score every library peptide against the anchor — cached by (anchor, mode).

    Each candidate's scores are intrinsic: they don't change when you move a
    display slider or open a candidate to inspect it. Caching the whole scan here
    means only the *first* look at a given anchor pays the cost; every later
    interaction is instant instead of re-scoring ~2,200 peptides each time.
    """
    return _scorer(use_algae_fit).profile(anchor)


@st.cache_data(show_spinner="Building the shortlist…")
def _ranked_groups(anchor: str, use_algae_fit: bool) -> list[Any]:
    """Score → filter → group near-duplicate scaffolds, cached by (anchor, mode).

    Grouping is anchor-fixed and independent of how many rows you choose to show,
    so it is cached here and merely sliced by the display slider — the slider and
    the candidate inspector no longer trigger a re-score or a re-group.
    """
    profiles = _profiles(anchor, use_algae_fit)
    ranked = filter_and_rank(profiles, anchor, low_toxicity=False, rank_by="algae_fit")
    return group_families(ranked, 0.7)


# --------------------------------------------------------------------------- #
# display helpers
# --------------------------------------------------------------------------- #
def _toxicity_label(profile: Any) -> str:
    if profile.lytic_risk:
        return "⚠ membrane-lytic"
    return {"HIGH-RISK": "High", "moderate": "Moderate", "lower": "Low"}[profile.toxicity_flag]


def _evidence_label(profile: Any) -> str:
    return "Experimental CPP" if "experimental" in profile.evidence else "Computational"


def _pct(x: float | None) -> int | None:
    return None if x is None else round(x * 100)


def _table(profiles: list[Any]) -> pd.DataFrame:
    """Self-explanatory CSV: plain column names with the good/bad direction baked in."""
    rows = []
    for p in profiles:
        row: dict[str, object] = {
            "Candidate": p.name,
            "Sequence": p.sequence,
            "Type": peptide_family(p.sequence),
            "Length": len(p.sequence),
            "Net charge": p.net_charge,
        }
        if p.algae_fit is not None:
            row["Delivery score (0-100, higher=better)"] = _pct(usable_delivery(p))
            row["Surface binding (0-100)"] = _pct(p.surface_interaction_prior)
            row["Membrane entry (0-100)"] = _pct(p.algae_fit)
        row["Damage risk (0-100, lower=better)"] = _pct(p.lysis_risk)
        row["Toxicity flag"] = _toxicity_label(p)
        row["Similarity to your peptide (%)"] = _pct(p.physchem)
        row["Shared motif (%)"] = _pct(p.motif_local)
        row["Sequence identity (%)"] = _pct(p.global_identity)
        if p.cpp_probability is not None:
            row["CPP likelihood (%)"] = _pct(p.cpp_probability)
        if p.critical_position is not None:
            row["Key-residue match (%)"] = _pct(p.critical_position)
        row["Prediction confidence"] = p.ad_confidence
        row["Evidence"] = _evidence_label(p)
        row["mCherry-ready (cloneable)"] = "yes" if p.genetically_encodable else "no"
        row["Tested form"] = p.modification
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
st.title("🧬 CPP Discovery — algae-delivery candidates")
st.markdown(
    "Pick a peptide that works (an **anchor**) and this tool ranks ~2,200 known, "
    "cloneable cell-penetrating peptides for **delivery into microalgae** — "
    "balancing similarity, an evidence-based algae profile, a trained membrane-"
    "hemolysis (toxicity) prior, "
    "and whether the peptide is genetically encodable for an mCherry fusion."
)
st.info(
    "Computational **hypotheses for wet-lab testing**, not algae-uptake predictions. "
    "Click any candidate to see its properties and *why* it was ranked where it is.",
    icon="🔬",
)
st.caption(CELL_WALL_CONTEXT)
st.caption(CARGO_CONTEXT)

if not _DATA.exists():
    st.error(f"CPPsite3 data not found at {_DATA}. See docs/data_sources.md.")
    st.stop()

lib = _library()

# ---- sidebar (intentionally minimal — the logic is the star) ----
st.sidebar.header("Your peptide")
choice = st.sidebar.selectbox("Anchor peptide (a CPP that works)", list(_PRESETS))
anchor = st.sidebar.text_input("Sequence (one-letter)", value=_PRESETS[choice]).strip().upper()
n_show = st.sidebar.slider("How many candidates", 5, 40, 15)
st.sidebar.metric("Cloneable CPPs screened", len(lib))
st.sidebar.caption(
    "Ranked by **Delivery score** = surface binding × membrane entry × safety × "
    "mCherry-readiness. Membrane-damaging (hemolytic-looking) peptides are kept "
    "but flagged ⚠. Near-identical scaffolds are grouped into one row — expand any "
    "candidate to see its variants."
)

if not anchor:
    st.info("Enter an anchor peptide sequence in the sidebar to begin.")
    st.stop()

st.markdown(
    f"**Anchor:** `{anchor}` · net charge **{net_charge(anchor):+d}** · length **{len(anchor)}**"
)

# ---- score (always optimized for algae delivery) ----
# Scoring + grouping are cached by (anchor, mode); the slider only slices the
# result, so moving it or inspecting a candidate is instant.
_algae_on = _algae_fit() is not None
groups = _ranked_groups(anchor, _algae_on)[:n_show]
reps = [g.representative for g in groups]

_LYSIS_WARN = 0.5


def _reasons_md(profile: Any) -> str:
    reasons = explain_profile(profile)
    return "\n".join(f"- {'✓' if r.positive else '✕'} {r.text}" for r in reasons)


def _lean_table(family_groups: list[Any]) -> pd.DataFrame:
    rows = []
    for g in family_groups:
        p = g.representative
        risk = round((p.lysis_risk or 0.0) * 100)
        rows.append({
            "Candidate": p.name,
            "Type": peptide_family(p.sequence),
            "Delivery score": _pct(usable_delivery(p)) if p.algae_fit is not None else None,
            "Surface binding": _pct(p.surface_interaction_prior),
            "Membrane entry": _pct(p.algae_fit),
            "Damage risk": ("⚠ " if p.lysis_risk >= _LYSIS_WARN else "") + str(risk),
            "Net charge": p.net_charge,
            "mCherry-ready": "✓" if p.genetically_encodable else "—",
            "Confidence": p.ad_confidence,
            "Similar variants": g.size,
        })
    return pd.DataFrame(rows)


# Plain-language column meanings, shown on hover — so someone who doesn't live in
# CPP jargon can read the table without a glossary. All 0–100 scores are "higher
# is better" *except* Damage risk (lower is better), which is called out.
def _column_config() -> dict[str, Any]:
    return {
        "Type": st.column_config.TextColumn(
            "Type", help="Rough mechanism family guessed from the sequence "
            "(a convenience label, not a strict classification)."),
        "Delivery score": st.column_config.ProgressColumn(
            "Delivery score", min_value=0, max_value=100, format="%d",
            help="The headline ranking, 0–100 (higher = more promising). Combines "
            "surface binding × membrane entry × safety × mCherry-readiness, so a "
            "peptide must do well on ALL of them to score high. It's a way to "
            "prioritize what to test — NOT a probability that delivery will work."),
        "Surface binding": st.column_config.NumberColumn(
            "Surface binding", format="%d",
            help="Step 1: how strongly the peptide is pulled onto the algal cell "
            "surface (which is negatively charged). 0–100, higher = better. "
            "Strongest around a mild positive charge of +4 to +6."),
        "Membrane entry": st.column_config.NumberColumn(
            "Membrane entry", format="%d",
            help="Step 2: how well the peptide's shape lets it slip into the "
            "membrane. 0–100, higher = better. Needed for delivery, but not a "
            "guarantee of it on its own."),
        "Damage risk": st.column_config.TextColumn(
            "Damage risk ⬇", help="Chance the peptide ruptures/damages membranes — "
            "a trained toxicity estimate shown as 0–100 where LOWER is better. "
            "⚠ marks 50+ : it looks like a membrane-damaging peptide. Such peptides "
            "are kept (they may still work) but flagged so you can weigh the risk."),
        "Net charge": st.column_config.NumberColumn(
            "Net charge", format="%+d",
            help="Overall electrical charge from its amino acids (+ from R/K, "
            "− from D/E). A mild positive charge (+4 to +6) binds the algal "
            "surface best."),
        "mCherry-ready": st.column_config.TextColumn(
            "mCherry-ready", help="✓ = can be used directly as a gene fused to "
            "mCherry (no chemical synthesis, tags, or modifications). Only about a "
            "third of the library qualifies — even pVEC was tested in a modified form."),
        "Confidence": st.column_config.TextColumn(
            "Confidence", help="How similar this sequence is to well-studied CPPs. "
            "'low' means an unusual peptide, so read its scores with extra caution."),
        "Similar variants": st.column_config.NumberColumn(
            "Similar variants", format="%d",
            help="How many near-identical sequences are folded into this one row. "
            "Open the candidate under 'Inspect' to see them all, ranked."),
    }


st.subheader(f"Top {len(reps)} algae-delivery candidates")
st.caption(
    "A **diverse** shortlist — near-identical scaffolds are grouped into one row "
    "(see the Variants count). ⚠ marks membrane-lytic peptides that may be toxic "
    "and need testing."
)
if is_trained_model_available():
    st.caption("Toxicity axis: **trained hemolysis prior** (HemoPI2) ✓")
else:
    st.caption(
        "⚠ Toxicity axis: **heuristic fallback** — the trained model failed to load "
        "(likely a scikit-learn version mismatch). Hemolysis-prior scores are the crude "
        "GRAVY heuristic, not the trained prior."
    )
st.dataframe(_lean_table(groups), use_container_width=True, hide_index=True,
             column_config=_column_config())
st.caption(
    "💡 Hover over any column header for a plain-language explanation. Every 0–100 "
    "score is *higher = better* — except **Damage risk ⬇**, where lower is better."
)

# ---- inspect one candidate: properties + why it fits algae + variants ----
st.markdown("### 🔬 Inspect a candidate")
sel = st.selectbox("Pick a candidate to see its properties and why it ranks here",
                   ["—"] + [g.representative.name for g in groups])
if sel != "—":
    g = next(g for g in groups if g.representative.name == sel)
    p = g.representative
    if p.lysis_risk >= _LYSIS_WARN:
        st.warning(
            f"High **membrane-damage risk ({_pct(p.lysis_risk)}%)** — this peptide "
            "looks like membrane-damaging (hemolytic/AMP) peptides. It may still enter "
            "cells, but could also harm membranes / be toxic. Not disqualifying — a "
            "strong-entry peptide with high damage risk is a real (if risky) candidate; "
            "verify toxicity in algae.",
            icon="⚠️",
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Delivery score", _pct(usable_delivery(p)) if p.algae_fit is not None else "—",
              help="Headline 0–100 ranking (higher = better). Prioritization heuristic, not a probability.")
    c1.metric("Surface binding", _pct(p.surface_interaction_prior),
              help="How strongly it sticks to the negatively-charged algal surface (step 1).")
    c2.metric("Membrane entry", _pct(p.algae_fit),
              help="How well its shape lets it enter the membrane (step 2).")
    c2.metric("Damage risk", _pct(p.lysis_risk),
              help="Membrane-rupture (toxicity) estimate, 0–100 — lower is better.")
    c3.metric("Net charge", f"{p.net_charge:+d}",
              help="Overall charge; +4 to +6 binds the algal surface best.")
    c3.metric("Cloning confidence", f"{p.fusion_confidence:.2f}",
              help="How well the plain cloned sequence matches the form actually tested (1.0 = identical).")
    st.markdown(
        f"`{p.sequence}` · **{peptide_family(p.sequence)}** · length {len(p.sequence)} · "
        f"{'cloneable (mCherry-fusion ready)' if p.genetically_encodable else 'tested form: ' + p.modification}"
    )
    st.caption(
        f"Fusion advisory (not used to rank): CPP charge **{p.net_charge:+d}** · "
        f"charge density **{charge_density(p):.2f}** /residue · est. CPP–mCherry "
        f"complex charge **{fusion_charge_estimate(p):+d}** (mCherry ≈ −6). "
        "Surface binding is driven by the CPP's local cationic patch, not the "
        "complex net charge — so a near-neutral complex is *not* a failed construct."
    )
    st.markdown("**Why it ranks here:**")
    st.markdown(_reasons_md(p))
    if g.size > 1:
        with st.expander(f"🧬 See all {g.size} variants of this scaffold (ranked)"):
            st.dataframe(_lean_table([group_families([m], 1.0)[0] for m in g.members]),
                         use_container_width=True, hide_index=True,
                         column_config=_column_config())

filtered = reps  # used by the download section below

with st.expander("ℹ️ What the columns mean (plain language)"):
    st.markdown(
        "Think of delivery as an obstacle course: a peptide has to **stick to the "
        "cell**, then **get through the membrane**, without **killing the cell**, and "
        "still be **easy to build**. The **Delivery score** multiplies those steps "
        "together, so a peptide has to do well on *all* of them to rank high.\n\n"
        "- **Delivery score** (0–100, higher better) — the headline ranking = surface "
        "binding × membrane entry × safety × mCherry-readiness. A way to prioritize "
        "what to test, **not** a probability that delivery will work.\n"
        "- **Surface binding** (0–100, higher better) — pull toward the *negatively "
        "charged* algal surface, the essential first step. Strongest at net charge "
        "**+4 to +6**; neutral/negative peptides score low and show up only as "
        "exploratory. (This is why the algae-proven peptides are all mildly positive.)\n"
        "- **Membrane entry** (0–100, higher better) — how well the peptide's shape "
        "lets it slip into the membrane, learned from CPPs that worked in microalgae. "
        "Charge is handled by Surface binding, not here.\n"
        "- **Damage risk ⬇** (0–100, *lower* better) — a **trained toxicity estimate** "
        "(model trained on real hemolysis data, ROC-AUC ~0.85): does it look like a "
        "membrane-damaging peptide? ⚠ marks 50+. It's trained on **human red-blood-cell** "
        "damage, so it's a borrowed cross-kingdom estimate for algae — not a measured "
        "algal toxicity — and it only sees membrane-*rupturing* toxicity (some toxins "
        "that kill by other routes score near zero). Delivery and damage are shown "
        "**separately** so you can weigh the trade-off yourself.\n"
        "- **mCherry-ready** (✓ / —) — whether the peptide can be used as the bare "
        "gene fused to mCherry (✓) or was only tested in a modified form that may not "
        "transfer (a dye tag, amidation, lipidation, unusual residues). Only ~30% of "
        "the library is fully cloneable — even pVEC was tested tagged + amidated.\n"
        "- **Type** — a rough mechanism family from the sequence (a convenience label). "
        "**Similar variants** — how many near-identical sequences are grouped under this "
        "row (expand via *Inspect*).\n"
        "- **Confidence** — how close the peptide is to ones the tool has seen well; "
        "'low' means an unusual sequence — treat its scores cautiously."
    )

# ---- downloads ----
if filtered:
    df = _table(filtered)
    fasta = "".join(
        f">{p.name.replace(' ', '_')} | usable={usable_delivery(p)*100:.0f} "
        f"charge={p.net_charge:+d} lysis={p.lysis_risk:.2f} "
        f"{'cloneable' if p.genetically_encodable else 'modified'}\n{p.sequence}\n"
        for p in filtered
    )
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download CSV", df.to_csv(index=False),
                       file_name="cpp_recommendations.csv", mime="text/csv")
    c2.download_button("⬇ Download FASTA", fasta,
                       file_name="cpp_recommendations.fasta", mime="text/plain")
