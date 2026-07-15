"""CPP Discovery — a recommendation platform (Streamlit).

Given an anchor peptide (e.g. ClWOX), it recommends the most similar cloneable
CPPs from the CPPsite3 database, with each piece of evidence shown as its own
plain-language column. Thin layer over the tested cpp_ai.scoring / screening
modules. Run:  streamlit run src/cpp_ai/webapp/app.py
"""

from __future__ import annotations

import importlib.util
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
    CLWOX_CRITICAL,
    AlgaeFitScorer,
    CriticalPositionProfile,
    EvidenceScorer,
)
from cpp_ai.screening import load_cppsite3_library  # noqa: E402
from cpp_ai.screening.candidate import ScreenCandidate  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data" / "raw" / "cppsite3_api.json"
_LEDGER = _ROOT / "data" / "curated" / "cpp_evidence_ledger.json"
_EMB_CACHE = _ROOT / "data" / "processed" / "emb_cache"

_PRESETS = {
    "ClWOX (plant homeoprotein, best — 30%)": "TNVYNWFQNRRARTKRK",
    "pVEC-R6A (your construct)": "LLIILARRIRKQAHAHSK",
    "pVEC (wild-type)": "LLIILRRRIRKQAHAHSK",
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


def _embeddings_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("esm") is not None
    )


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
def _scorer(use_embeddings: bool, use_algae_fit: bool) -> EvidenceScorer:
    service = None
    if use_embeddings:
        from cpp_ai.embeddings import ESM2Embedder, EmbeddingCache, EmbeddingService

        service = EmbeddingService(ESM2Embedder("esm2_t6_8M"), EmbeddingCache(str(_EMB_CACHE)))
    fit = _algae_fit() if use_algae_fit else None
    return EvidenceScorer(
        _library(), embedding_service=service, classifier=_classifier(), algae_fit_scorer=fit
    )


# --------------------------------------------------------------------------- #
# display helpers
# --------------------------------------------------------------------------- #
def _toxicity_label(profile: Any) -> str:
    if profile.lytic_risk:
        return "⚠ membrane-lytic"
    return {"HIGH-RISK": "High", "moderate": "Moderate", "lower": "Low"}[profile.toxicity_flag]


def _evidence_label(profile: Any) -> str:
    return "Experimental CPP" if "experimental" in profile.evidence else "Computational"


def _pct(x: float | None) -> object:
    return None if x is None else round(x * 100)


def _table(profiles: list[Any]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        row: dict[str, object] = {
            "Peptide": p.name,
            "Sequence": p.sequence,
            "Overall match": _pct(p.shortlist_score),
            "Similarity to anchor": _pct(p.physchem),
            "Shared motif": _pct(p.motif_local),
            "Sequence identity": _pct(p.global_identity),
        }
        if p.algae_fit is not None:
            row["Algae-delivery fit"] = _pct(p.algae_fit)
        if p.cpp_probability is not None:
            row["CPP likelihood"] = _pct(p.cpp_probability)
        if p.critical_position is not None:
            row["Key-residue match"] = _pct(p.critical_position)
        row["Net charge"] = p.net_charge
        row["Toxicity risk"] = _toxicity_label(p)
        row["Confidence"] = p.ad_confidence
        row["Evidence"] = _evidence_label(p)
        rows.append(row)
    return pd.DataFrame(rows)


def _style(df: pd.DataFrame) -> Any:
    if "Toxicity risk" not in df.columns:
        return df

    def color(v: object) -> str:
        return {
            "Low": "background-color:#d4edda",
            "Moderate": "background-color:#fff3cd",
            "High": "background-color:#f8d7da",
            "⚠ membrane-lytic": "background-color:#f8d7da",
        }.get(str(v), "")

    styler = df.style
    elementwise = getattr(styler, "map", None) or styler.applymap  # pandas <>=2.1
    return elementwise(color, subset=["Toxicity risk"])


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
st.title("🧬 CPP Discovery — peptide recommendations")
st.markdown(
    "Pick a peptide that works (an **anchor**) and this tool recommends the most "
    "similar cell-penetrating peptides from a database of ~2,200 known CPPs — the "
    "kind you could clone and test. Each recommendation shows *why* it was picked, "
    "column by column."
)
st.info(
    "These are **computational suggestions for wet-lab testing** — not predictions "
    "that a peptide will deliver cargo into algae. Use them to prioritize what to try.",
    icon="🔬",
)

if not _DATA.exists():
    st.error(f"CPPsite3 data not found at {_DATA}. See docs/data_sources.md.")
    st.stop()

lib = _library()

# ---- sidebar (kept intentionally minimal) ----
st.sidebar.header("Your peptide")
choice = st.sidebar.selectbox("Anchor peptide", list(_PRESETS))
anchor = st.sidebar.text_input("Sequence (one-letter)", value=_PRESETS[choice]).strip().upper()
n_show = st.sidebar.slider("How many recommendations", 5, 40, 15)
low_tox = st.sidebar.checkbox("Prioritize lower-toxicity candidates", value=True)
algae_mode = st.sidebar.checkbox(
    "Optimize for algae delivery (evidence-based)",
    value=False,
    help=(
        "Re-rank using the physicochemical profile that actually worked in "
        "microalgae in the literature — amphipathicity/hydrophobicity up, "
        "pure cationic charge and aromaticity down. Learned from the curated "
        "evidence ledger, not hardcoded."
    ),
    disabled=_algae_fit() is None,
)

with st.sidebar.expander("Advanced"):
    scaffold = st.checkbox("Scaffold mode (score key residues W6/T14 vs ClWOX)", value=False)
    exclude_homeodomain = st.checkbox("Exclude the well-known homeodomain family", value=False)
    use_emb = (
        st.checkbox("Use ESM-2 embeddings (slower)", value=False)
        if _embeddings_available()
        else False
    )
    if not _embeddings_available():
        st.caption("ESM-2 embeddings unavailable here (torch not installed).")

st.sidebar.metric("CPPs in the library", len(lib))

if not anchor:
    st.info("Enter an anchor peptide sequence in the sidebar to begin.")
    st.stop()

st.markdown(
    f"**Anchor:** `{anchor}` · net charge **{net_charge(anchor):+d}** · length **{len(anchor)}**"
)

# ---- score + filter ----
crit_profile = None
if scaffold:
    crit_profile = (
        CLWOX_CRITICAL if anchor == "TNVYNWFQNRRARTKRK"
        else CriticalPositionProfile.uniform(anchor)
    )

with st.spinner("Finding and scoring similar CPPs…"):
    profiles = _scorer(use_emb, algae_mode).profile(anchor, critical_profile=crit_profile)

if algae_mode:
    st.success(
        "**Algae-delivery mode on.** Ranking now favors the physicochemical "
        "profile that worked in microalgae (amphipathic + hydrophobic, lower pure "
        "charge) — the empirical opposite of the generic 'add arginine' prior. "
        "Derived from a small curated evidence ledger, so treat it as a "
        "hypothesis-sharpener, not proof.",
        icon="🌱",
    )

filtered = []
for p in profiles:
    if p.sequence == anchor:
        continue
    if low_tox and (p.lytic_risk or p.toxicity_flag == "HIGH-RISK"):
        continue
    if exclude_homeodomain and ("WFQN" in p.sequence):
        continue
    filtered.append(p)
filtered = filtered[:n_show]

st.subheader(f"Top {len(filtered)} recommended peptides")
st.dataframe(_style(_table(filtered)), use_container_width=True, hide_index=True)

with st.expander("ℹ️ What do these columns mean?"):
    st.markdown(
        "- **Overall match** — a convenience 0–100 blend of the columns below (how it's "
        "ranked). Always read the individual columns, not just this.\n"
        "- **Similarity to anchor** — how close its physical/chemical make-up (charge, "
        "hydrophobicity, amphipathicity, shape, composition, arrangement) is to your "
        "anchor. 100 = essentially the same profile.\n"
        "- **Shared motif** — does it contain a similar short stretch of sequence to the "
        "anchor (a shared 'motif')?\n"
        "- **Sequence identity** — overall % of amino acids that match when the two are "
        "lined up end-to-end.\n"
        "- **Algae-delivery fit** *(algae mode)* — how well the peptide matches the "
        "physicochemical profile of CPPs that actually worked in microalgae "
        "(amphipathic/hydrophobic, lower pure charge). Learned from the curated "
        "evidence ledger; small dataset, so it's a hypothesis-sharpener.\n"
        "- **CPP likelihood** — a trained model's estimate that it's a genuine "
        "cell-penetrating peptide (not how well it enters algae).\n"
        "- **Key-residue match** *(scaffold mode)* — how well it preserves the residues "
        "shown to matter in ClWOX (W6, T14).\n"
        "- **Net charge** — positive minus negative residues. Very high charge tends to be "
        "toxic to algae.\n"
        "- **Toxicity risk** — a heuristic flag (Low / Moderate / High, or membrane-lytic) "
        "from charge and uptake mechanism. Not a measured value.\n"
        "- **Confidence** — how similar this peptide is to ones the tool has seen; 'low' "
        "means an unusual sequence — treat its scores cautiously.\n"
        "- **Evidence** — an experimentally-studied CPP, or a computational candidate."
    )

# ---- downloads ----
if filtered:
    df = _table(filtered)
    fasta = "".join(
        f">{p.name.replace(' ', '_')} | match={p.shortlist_score*100:.0f} "
        f"charge={p.net_charge:+d} toxicity={_toxicity_label(p)}\n{p.sequence}\n"
        for p in filtered
    )
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download CSV", df.to_csv(index=False),
                       file_name="cpp_recommendations.csv", mime="text/csv")
    c2.download_button("⬇ Download FASTA", fasta,
                       file_name="cpp_recommendations.fasta", mime="text/plain")
