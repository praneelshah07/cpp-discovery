"""CPP Discovery — algae candidate screening UI (Streamlit).

Thin layer over cpp_ai.screening. Run:  streamlit run src/cpp_ai/webapp/app.py
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from cpp_ai.generation import MutationConstraints, VariantGenerator, net_charge
from cpp_ai.scoring import EvidenceScorer
from cpp_ai.screening import (
    ScreenCandidate,
    ScreeningIndex,
    apply_filters,
    diversify,
    load_cppsite3_library,
    to_fasta,
)

_ROOT = Path(__file__).resolve().parents[3]
_DATA = _ROOT / "data" / "raw" / "cppsite3_api.json"
_EMB_CACHE = _ROOT / "data" / "processed" / "emb_cache"

_PRESETS = {
    "ClWOX (plant homeoprotein, best — 30%)": "TNVYNWFQNRRARTKRK",
    "pVEC-R6A (your construct)": "LLIILARRIRKQAHAHSK",
    "pVEC (wild-type)": "LLIILRRRIRKQAHAHSK",
    "Custom…": "",
}

_KNOWN_ANCHORS = {  # excluded from "non-obvious" screens
    "TNVYNWFQNRRARTKRK", "LLIILARRIRKQAHAHSK", "LLIILRRRIRKQAHAHSK",
}

st.set_page_config(page_title="CPP Discovery — Algae Screening", layout="wide")


@st.cache_data(show_spinner=False)
def _library() -> list[ScreenCandidate]:
    if not _DATA.exists():
        return []
    return load_cppsite3_library(_DATA)


@st.cache_resource(show_spinner=True)
def _index(use_embeddings: bool) -> ScreeningIndex:
    lib = _library()
    service = None
    if use_embeddings:
        from cpp_ai.embeddings import ESM2Embedder, EmbeddingCache, EmbeddingService

        service = EmbeddingService(ESM2Embedder("esm2_t6_8M"), EmbeddingCache(str(_EMB_CACHE)))
    return ScreeningIndex.build(lib, embedding_service=service)


@st.cache_resource(show_spinner=False)
def _classifier() -> Any:
    path = _ROOT / "data" / "processed" / "cpp_classifier.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


@st.cache_resource(show_spinner=True)
def _evidence_scorer(use_embeddings: bool) -> EvidenceScorer:
    service = None
    if use_embeddings:
        from cpp_ai.embeddings import ESM2Embedder, EmbeddingCache, EmbeddingService

        service = EmbeddingService(ESM2Embedder("esm2_t6_8M"), EmbeddingCache(str(_EMB_CACHE)))
    return EvidenceScorer(_library(), embedding_service=service, classifier=_classifier())


def _anchor_input() -> str:
    choice = st.sidebar.selectbox("Anchor peptide", list(_PRESETS))
    default = _PRESETS[choice]
    seq = st.sidebar.text_input("Sequence (one-letter)", value=default).strip().upper()
    return seq


def _results_frame(candidates: list[ScreenCandidate]) -> pd.DataFrame:
    return pd.DataFrame([c.to_record() for c in candidates])


def _downloads(candidates: list[ScreenCandidate], stem: str) -> None:
    col1, col2 = st.columns(2)
    col1.download_button(
        "⬇ Download CSV", _results_frame(candidates).to_csv(index=False),
        file_name=f"{stem}.csv", mime="text/csv",
    )
    col2.download_button(
        "⬇ Download FASTA", to_fasta(candidates),
        file_name=f"{stem}.fasta", mime="text/plain",
    )


def _style(df: pd.DataFrame) -> Any:
    if "toxicity_flag" not in df.columns:
        return df

    def color(v: object) -> str:
        return {"HIGH-RISK": "background-color:#f8d7da", "moderate": "background-color:#fff3cd",
                "lower": "background-color:#d4edda"}.get(str(v), "")

    styler = df.style
    # pandas >=2.1 renamed Styler.applymap -> Styler.map (applymap later removed).
    elementwise = getattr(styler, "map", None) or styler.applymap
    return elementwise(color, subset=["toxicity_flag"])


# --------------------------------------------------------------------------- #
st.title("🧬 CPP Discovery — algae candidate screening")
st.caption(
    "Mine the CPPsite3 database (grounded in ClWOX / pVEC-R6A) for cloneable, "
    "toxicity-aware wet-lab candidates. Outputs are computational hypotheses — "
    "not validated. High charge and membrane-lytic peptides are flagged as algae "
    "toxicity risks."
)

if not _DATA.exists():
    st.error(f"CPPsite3 data not found at {_DATA}. Download it first (see docs/data_sources.md).")
    st.stop()

def _embeddings_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("esm") is not None
    )


lib = _library()
st.sidebar.metric("Library peptides (CPPsite3)", len(lib))
if _embeddings_available():
    use_emb = st.sidebar.checkbox("Use ESM-2 embeddings (slower first run)", value=False)
else:
    use_emb = False
    st.sidebar.caption("ESM-2 embeddings unavailable here (torch not installed).")
anchor = _anchor_input()
mode = st.sidebar.radio(
    "Mode",
    ["Evidence profile (multi-axis)", "Anchor similarity search",
     "Diverse gentle-entry screen", "Engineer variants"],
)

# shared filters
st.sidebar.subheader("Filters")
lo, hi = st.sidebar.slider("Net charge range (toxicity)", -5, 12, (4, 7))
lmin, lmax = st.sidebar.slider("Length range", 5, 40, (10, 25))
endo = st.sidebar.checkbox("Endocytic uptake only (gentler)", value=False)
excl_homeo = st.sidebar.checkbox("Exclude homeodomain family (WFQN)", value=False)
excl_lytic = st.sidebar.checkbox("Exclude membrane-lytic (toxic)", value=False)
top_k = st.sidebar.slider("Max results", 5, 50, 15)

if not anchor:
    st.info("Enter an anchor peptide sequence in the sidebar to begin.")
    st.stop()
st.markdown(f"**Anchor:** `{anchor}`  · net charge **{net_charge(anchor):+d}** · length **{len(anchor)}**")

# --------------------------------------------------------------------------- #
if mode == "Evidence profile (multi-axis)":
    st.subheader("Multi-axis evidence profile")
    st.caption(
        "Every signal is shown **separately** — physicochemical resemblance (per "
        "block), local motif, global identity, CPP-classifier probability, "
        "applicability-domain confidence, and graded safety. `shortlist_score` is "
        "only a convenience ordering; read the axes. Not an algae-uptake prediction."
    )
    max_risk = st.slider(
        "Max charge risk (graded penalty, not a hard gate)", 0.0, 1.0, 1.0, 0.05
    )
    scaffold = st.checkbox(
        "Scaffold mode (positional W6/T14 weighting for ClWOX-alignable candidates)",
        value=False,
    )
    from cpp_ai.scoring import CLWOX_CRITICAL, CriticalPositionProfile
    crit_profile = None
    if scaffold:
        crit_profile = (
            CLWOX_CRITICAL if anchor == "TNVYNWFQNRRARTKRK"
            else CriticalPositionProfile.uniform(anchor)
        )
    with st.spinner("Scoring all axes across the library…"):
        scorer = _evidence_scorer(use_emb)
        profiles = scorer.profile(anchor, critical_profile=crit_profile)
    profiles = [
        p for p in profiles
        if p.sequence != anchor and lmin <= len(p.sequence) <= lmax and p.charge_risk <= max_risk
    ][:top_k]
    df = pd.DataFrame([p.to_record() for p in profiles])
    st.dataframe(_style(df), use_container_width=True)
    if profiles:
        col1, col2 = st.columns(2)
        col1.download_button("⬇ Download CSV", df.to_csv(index=False),
                             file_name="evidence_profiles.csv", mime="text/csv")
        fasta = "".join(
            f">{p.name.replace(' ', '_')} | shortlist={p.shortlist_score:.3f} "
            f"physchem={p.physchem:.2f} charge={p.net_charge:+d} risk={p.charge_risk:.2f}\n{p.sequence}\n"
            for p in profiles
        )
        col2.download_button("⬇ Download FASTA", fasta,
                             file_name="evidence_profiles.fasta", mime="text/plain")

elif mode == "Anchor similarity search":
    st.subheader("CPPs most similar to the anchor")
    embed_w = st.slider("Embedding weight (functional vs physicochemical)", 0.0, 1.0, 0.5) if use_emb else 0.0
    with st.spinner("Building index / ranking…"):
        idx = _index(use_emb)
        ranked = idx.rank(anchor, embed_weight=embed_w)
    filtered = apply_filters(
        ranked, min_len=lmin, max_len=lmax, min_charge=lo, max_charge=hi,
        endocytic_only=endo, exclude_homeodomain=excl_homeo, exclude_lytic=excl_lytic,
        exclude_sequences=_KNOWN_ANCHORS,
    )
    results = diversify(filtered, limit=top_k)
    st.dataframe(_style(_results_frame(results)), use_container_width=True)
    _downloads(results, "anchor_similarity_candidates")

elif mode == "Diverse gentle-entry screen":
    st.subheader("Diverse, gentle-entry database candidates")
    st.caption("Filter-first, then diversify — surfaces non-obvious peptides from varied families.")
    filtered = apply_filters(
        lib, min_len=lmin, max_len=lmax, min_charge=lo, max_charge=hi,
        endocytic_only=endo or True, aromatic_only=True, exclude_homeodomain=True,
        exclude_lytic=excl_lytic, exclude_sequences=_KNOWN_ANCHORS,
    )
    st.write(f"{len(filtered)} peptides pass the gentle-entry filters.")
    results = diversify(sorted(filtered, key=lambda c: c.net_charge), limit=top_k)
    st.dataframe(_style(_results_frame(results)), use_container_width=True)
    _downloads(results, "diverse_gentle_candidates")

else:  # Engineer variants
    st.subheader("Engineer conservative variants of the anchor")
    strategy = st.selectbox("Substitution strategy", ["conservative", "charge_preserving", "hydrophobic", "all_canonical"])
    n_mut = st.slider("Mutations per variant", 1, 3, 1)
    lock_motif = st.text_input("Lock motif (never mutate), optional", value="WFQN" if anchor == _PRESETS[list(_PRESETS)[0]] else "")
    max_variants = st.slider("Max variants", 10, 500, 100)
    try:
        constraints = MutationConstraints(
            locked_motifs=tuple(m for m in [lock_motif.strip().upper()] if m),
            canonical_only=True, min_charge=lo, max_charge=hi,
        )
        from cpp_ai.core.schema import Peptide
        ref = Peptide.from_sequence(anchor, dataset="anchor")
        variants = VariantGenerator(strategy, constraints, seed=0).generate(
            ref, n_mutations=n_mut, max_variants=max_variants
        )
        cands = [
            ScreenCandidate(v.sequence, name=", ".join(v.metadata["mutations"]),
                            category="engineered", mechanism="engineered_variant", source="anchor")
            for v in variants
        ]
        results = diversify(cands, limit=top_k)
        st.write(f"{len(variants)} variants generated; showing {len(results)} diverse.")
        st.dataframe(_style(_results_frame(results)), use_container_width=True)
        _downloads(results, "engineered_variants")
    except Exception as exc:  # noqa: BLE001 - surface constraint errors to the user
        st.error(f"Generation failed: {exc}")
