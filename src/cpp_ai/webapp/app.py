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
from cpp_ai.scoring import AlgaeFitScorer, EvidenceScorer  # noqa: E402
from cpp_ai.pipeline import (  # noqa: E402
    CELL_WALL_CONTEXT,
    explain_profile,
    filter_and_rank,
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
    rows = []
    for p in profiles:
        row: dict[str, object] = {
            "Peptide": p.name,
            "Sequence": p.sequence,
            "Family": peptide_family(p.sequence),
            "Composite score": _pct(p.shortlist_score),
            "Similarity to anchor": _pct(p.physchem),
            "Shared motif": _pct(p.motif_local),
        }
        if p.algae_fit is not None:
            row["Algae suitability"] = _pct(p.algae_fit)
        if p.cpp_probability is not None:
            row["CPP likelihood"] = _pct(p.cpp_probability)
        if p.critical_position is not None:
            row["Key-residue match"] = _pct(p.critical_position)
        row["Net charge"] = p.net_charge
        row["Lysis risk"] = round(p.lysis_risk, 2)
        row["Toxicity risk"] = _toxicity_label(p)
        row["Confidence"] = p.ad_confidence
        row["Evidence"] = _evidence_label(p)
        row["Cloneable"] = "✓" if p.genetically_encodable else "—"
        row["Tested form"] = p.modification
        row["Sequence identity"] = _pct(p.global_identity)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
st.title("🧬 CPP Discovery — algae-delivery candidates")
st.markdown(
    "Pick a peptide that works (an **anchor**) and this tool ranks ~2,200 known, "
    "cloneable cell-penetrating peptides for **delivery into microalgae** — "
    "balancing similarity, an evidence-based algae profile, membrane-lysis safety, "
    "and whether the peptide is genetically encodable for an mCherry fusion."
)
st.info(
    "Computational **hypotheses for wet-lab testing**, not algae-uptake predictions. "
    "Click any candidate to see its properties and *why* it was ranked where it is.",
    icon="🔬",
)
st.caption(CELL_WALL_CONTEXT)

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
    "Ranked by **usable delivery** = surface binding × insertion fit × "
    "(1 − lysis)² × cloneability. Membrane-lytic peptides are kept but flagged ⚠. "
    "Near-identical scaffolds are grouped — expand any candidate to see its variants."
)

if not anchor:
    st.info("Enter an anchor peptide sequence in the sidebar to begin.")
    st.stop()

st.markdown(
    f"**Anchor:** `{anchor}` · net charge **{net_charge(anchor):+d}** · length **{len(anchor)}**"
)

# ---- score (always optimized for algae delivery) ----
_algae_on = _algae_fit() is not None
with st.spinner("Scoring the library for algae delivery…"):
    profiles = _scorer(_algae_on).profile(anchor)

# usable-delivery ranking; keep lytic peptides (warn, don't exclude).
ranked = filter_and_rank(profiles, anchor, low_toxicity=False, rank_by="algae_fit")
groups = group_families(ranked, 0.7)[:n_show]
reps = [g.representative for g in groups]

_LYSIS_WARN = 0.5


def _reasons_md(profile: Any) -> str:
    reasons = explain_profile(profile)
    return "\n".join(f"- {'✓' if r.positive else '✕'} {r.text}" for r in reasons)


def _lean_table(family_groups: list[Any]) -> pd.DataFrame:
    rows = []
    for g in family_groups:
        p = g.representative
        rows.append({
            "Peptide": p.name,
            "Family": peptide_family(p.sequence),
            "Usable delivery": _pct(usable_delivery(p)) if p.algae_fit is not None else None,
            "Surface binding": _pct(p.surface_adsorption),
            "Insertion fit": _pct(p.algae_fit),
            "Lysis": ("⚠ " if p.lysis_risk >= _LYSIS_WARN else "") + f"{p.lysis_risk:.2f}",
            "Charge": p.net_charge,
            "Cloneable": "✓" if p.genetically_encodable else "—",
            "Confidence": p.ad_confidence,
            "Variants": g.size,
        })
    return pd.DataFrame(rows)


st.subheader(f"Top {len(reps)} algae-delivery candidates")
st.caption(
    "A **diverse** shortlist — near-identical scaffolds are grouped into one row "
    "(see the Variants count). ⚠ marks membrane-lytic peptides that may be toxic "
    "and need testing."
)
st.dataframe(_lean_table(groups), use_container_width=True, hide_index=True)

# ---- inspect one candidate: properties + why it fits algae + variants ----
st.markdown("### 🔬 Inspect a candidate")
sel = st.selectbox("Pick a candidate to see its properties and why it ranks here",
                   ["—"] + [g.representative.name for g in groups])
if sel != "—":
    g = next(g for g in groups if g.representative.name == sel)
    p = g.representative
    if p.lysis_risk >= _LYSIS_WARN:
        st.warning(
            "This peptide looks **membrane-lytic** (AMP-like). It may enter cells but "
            "could also damage membranes / be toxic — treat as a hypothesis needing "
            "wet-lab toxicity testing.",
            icon="⚠️",
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Usable delivery", _pct(usable_delivery(p)) if p.algae_fit is not None else "—")
    c1.metric("Surface binding", _pct(p.surface_adsorption))
    c2.metric("Insertion fit", _pct(p.algae_fit))
    c2.metric("Lysis risk", f"{p.lysis_risk:.2f}")
    c3.metric("Net charge", f"{p.net_charge:+d}")
    c3.metric("Fusion confidence", f"{p.fusion_confidence:.2f}")
    st.markdown(
        f"`{p.sequence}` · **{peptide_family(p.sequence)}** · length {len(p.sequence)} · "
        f"{'cloneable (mCherry-fusion ready)' if p.genetically_encodable else 'tested form: ' + p.modification}"
    )
    st.markdown("**Why it ranks here:**")
    st.markdown(_reasons_md(p))
    if g.size > 1:
        with st.expander(f"🧬 See all {g.size} variants of this scaffold (ranked)"):
            st.dataframe(_lean_table([group_families([m], 1.0)[0] for m in g.members]),
                         use_container_width=True, hide_index=True)

filtered = reps  # used by the download section below

with st.expander("ℹ️ How the ranking works"):
    st.markdown(
        "Candidates are ordered by **usable delivery = surface binding × insertion "
        "fit × (1 − lysis risk)² × fusion confidence** — three separable biological "
        "steps plus cloneability, so a peptide has to clear *all* of them to rank high.\n\n"
        "- **Usable delivery** — the headline 0–100 ranking score above. A design "
        "heuristic, **not** a delivery probability.\n"
        "- **Surface binding** — electrostatic attraction to the *negatively charged* "
        "algal surface (the first step: no adsorption → no uptake). Peaks at net "
        "charge **+4 to +6**; neutral/negative peptides score low and appear only as "
        "exploratory. This is why the algae-proven anchors are all cationic.\n"
        "- **Insertion fit** — match to the membrane-*insertion* profile of CPPs that "
        "worked in microalgae (amphipathic/hydrophobic/shape), learned from the "
        "evidence ledger. Charge is handled by Surface binding, not here.\n"
        "- **Lysis** — heuristic membrane-lysis (AMP-like) risk. ⚠ (≥0.50) = perturbs "
        "membranes (TP10/MAP-like) rather than entering gently; kept but flagged for "
        "toxicity testing. A prior, not a measured hemolysis value.\n"
        "- **Cloneable / Fusion confidence** — whether the peptide was tested as the "
        "bare sequence (✓, mCherry-fusion ready) or in a form that may not transfer "
        "(fluorescein tag, amidation, lipidation, non-canonical residues). Only ~30% "
        "of the library is fully cloneable — even pVEC was tested tagged + amidated.\n"
        "- **Family** — a coarse mechanistic tag (heuristic). **Variants** — how many "
        "near-identical scaffolds are grouped under this row (expand via *Inspect*).\n"
        "- **Confidence** — how close the peptide is to ones the tool has seen; 'low' "
        "means an unusual sequence — treat its scores cautiously."
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
