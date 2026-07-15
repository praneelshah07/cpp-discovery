# Transfer packet — CPP Discovery platform

Paste this at the start of a new conversation to continue work. It captures the
project, its current state, the environment quirks, the science, and where we're
headed. The active focus is **refining the Streamlit recommendation app** and
**deciding project direction**.

---

## 1. What this is

`cpp-ai` — a research-grade, modular platform that recommends **Cell Penetrating
Peptides (CPPs)** to test in the lab. Ultimate goal: deliver a **recombinant
mCherry-fusion protein into algae**. Every output is a *computational hypothesis
for wet-lab validation*, never a claim of efficacy.

The user (Anil/Praneel, praneelshah101@gmail.com) is a researcher; explanations
should stay scientifically honest and, for the app, novice-readable.

## 2. Where everything lives

- **Local repo:** `~/cpp-discovery` (its own git repo; the parent `~` is *also* a
  git repo for unrelated TradingView work — keep all work inside `~/cpp-discovery`).
- **GitHub:** `github.com/praneelshah07/cpp-discovery` (**private**), branch `main`.
  SSH key is set up (`~/.ssh/id_ed25519`, no passphrase); `git push` works
  non-interactively.
- **Streamlit Community Cloud:** deployed from the repo. Main file path:
  `src/cpp_ai/webapp/app.py`. Cloud auto-redeploys on push. (The user has the app URL.)
- **venv:** `~/cpp-discovery/.venv` (activate: `source .venv/bin/activate`).

## 3. Environment quirks (Intel macOS — important)

Platform: **Intel x86_64 macOS**, Python 3.10.9. Consequences already handled:
- **`numpy<2`** is pinned (1.26.4) because PyTorch's last Intel-mac build is 2.2.2,
  which crashes under numpy 2. The `embeddings` extra auto-pins this on Darwin x86.
- **`pyarrow<15`** pinned in the `webapp` extra (no newer x86-mac wheel).
- **XGBoost** installed but **won't load** (its `libxgboost.dylib` picks up a stale
  `libomp.dylib` from `~/anaconda3`). It's guarded/optional — the model zoo skips it.
- **umap-learn** won't build (numba/llvmlite). UMAP is a guarded option; PCA/t-SNE
  (sklearn) are the always-available fallbacks.
- Local **pandas is 2.3.3** (Styler.applymap removed → use `.map`; already handled).

Run checks: `python -m pytest -m "not slow"` · `python -m mypy src` · `python -m ruff check src tests`.
Current state: **383 tests pass**, mypy --strict clean (68 files), ruff clean.
Launch app locally: `streamlit run src/cpp_ai/webapp/app.py`.

## 4. The data (in `data/raw/`, mostly gitignored)

- `poseidon_cargo_encoded.csv` — POSEIDON: 1,316 quantitative uptake measurements,
  571 unique peptides. Mammalian cells, mostly small-fluorophore cargo (only 3 rows
  mCherry). `PoseidonImporter` preset matches its schema.
- `cppsite3_api.json` (**committed** for the cloud app) — CPPsite3 full dump, 6,788
  records, 2,814 unique canonical CPPs. `upef` (uptake) field is free-text/unusable
  as a label. This is the app's library.
- `cppsite3_natural.csv` — CPPsite3 sequences only (id/pepseq/dssp/SMILES).
- `uniprot_human_pool.fasta` — 12,279 human proteins, used as classifier negatives.
- 4 `.tgz` — CPPsite3 3D structures (future structure features).
- `data/processed/`: `cpp_classifier.pkl` (RF, gitignored), `emb_cache/` (ESM-2
  vectors), and candidate CSV/FASTA outputs.

## 5. The science (critical context)

- **Anchors:** `ClWOX = TNVYNWFQNRRARTKRK` (plant homeoprotein 3rd-α-helix CPP, the
  best in the Landry-lab ClWOX paper — 30% internalization in plant cells,
  endocytic, gentle) and `pVEC-R6A = LLIILARRIRKQAHAHSK` (the user's construct;
  confirmed via DNA translation; it's pVEC with R6→A, charge +5). Original pVEC =
  `LLIILRRRIRKQAHAHSK` (+6).
- **Toxicity is a hard constraint:** the user's PhD-thesis data shows **9R and pVEC
  are cytotoxic to *Chlamydomonas*** at 5–10 µM. So high-charge / membrane-lytic
  peptides are algae-toxicity risks; the app flags them and prefers moderate charge
  + endocytic mechanism (like ClWOX).
- **Domain mismatch (be honest about this):** POSEIDON = mammalian + small dye;
  CPPsite3 = mixed; **zero algae data**. The tool therefore **cannot predict algae
  uptake** — it is a *similarity-grounded recommender* + safety flags, not a
  predictor. Say this plainly.
- **What the recommender surfaces:** ClWOX belongs to the homeodomain family (shared
  `WFQN…RR` motif), so top recommendations are Penetratin, Engrailed-2, HoxA-13,
  CUPID, PDX-1-PTD, etc. A cross-database standout is `MDCRWRWKCCKK` (crotamine
  RW-motif, high in both POSEIDON and CPPsite3).
- **CPP-vs-non-CPP classifier:** RandomForest on CPPsite3 (2,547 pos) vs
  length-matched UniProt negatives (2,547). ROC-AUC **0.953 random-CV**, **0.940
  family-split** (leakage-checked, holds up). Saved to `data/processed/cpp_classifier.pkl`.

## 6. Architecture (all under `src/cpp_ai/`)

core · database · descriptors (+`motifs.py` arrangement block) · embeddings
(ESM-2/ProtT5, lazy, cached) · similarity · prediction (+`validation.py`
family-split) · generation · optimization (built-in NSGA-II) · ranking · **scoring**
(the active area) · visualization · screening · webapp. Each module has tests in
`tests/` and a doc in `docs/`.

## 7. How the app scores (the current focus)

The app (`webapp/app.py`) is a **single recommendation view** built on
`scoring.EvidenceScorer.profile(anchor)`. Each recommendation shows separate,
plain-language columns (all computed with classical/interpretable methods — **not
DeepChem, not deep learning by default**):

- **Similarity to anchor** — block-wise physicochemical. Descriptors come from
  `biopython` (ProtParam) + `peptides.py` (amino-acid property lookup tables), grouped
  into 7 biological blocks (charge, hydrophobicity, amphipathicity, structure,
  composition, arrangement, aggregation). Per block: z-standardize across the
  library, Gaussian on RMS distance (`exp(-d²/2σ²)` — magnitude-aware), weighted
  average, reported as a library-calibrated 0–100. Code: `scoring/physchem.py`,
  `scoring/blocks.py`.
- **Shared motif** — Smith-Waterman local alignment (biopython + BLOSUM62),
  self-normalized. **Sequence identity** — global alignment identity %.
  Code: `similarity/metrics.py`.
- **CPP likelihood** — the RF classifier's calibrated probability (separate axis;
  blank on cloud where the pkl isn't committed).
- **Key-residue match** (scaffold mode) — BLOSUM62-weighted positional similarity
  at ClWOX's alanine-scan-critical residues (W6, T14). Code: `scoring/positional.py`.
- **Applicability-domain confidence** — nearest-neighbor distance to known CPPs →
  high/medium/low. **Toxicity risk** — graded charge risk + lytic flag
  (`scoring/safety.py`). **Evidence** — experimental vs computational.
- `shortlist_score` (shown as "Overall match") is a transparent convenience blend;
  the individual axes are the point. Code: `scoring/evidence.py`.
- **ESM-2 embeddings** — the only deep-learning option; off by default, disabled on
  cloud (no torch).

## 8. Expert review — status

A detailed review said the app looked "more authoritative than it is" and to
separate collapsed questions into interpretable axes. **Done:** block-wise
similarity (fix redundancy/imbalance), distance-aware (not just cosine),
library-calibrated percentile, multi-axis `EvidenceProfile`, graded safety
penalties (no hard charge gate), critical-position scaffold mode, motif/arrangement
features, family-split re-validation (AUC 0.94), and a novice-friendly single-view
UI. **Still pending:** Pareto-style *categorized* shortlist; consensus CPP models;
a real hemolysis/membrane-lysis predictor (needs data); and the **algae
active-learning loop** (needs the user's wet-lab results). Full review notes are in
the conversation history and `docs/scoring.md`.

## 9. Open questions / where to take it (discuss with user)

1. **Algae data loop** — highest value. Once the user has ~20–40 consistently
   measured algae results, build a small algae-specific ranking/active-learning
   model. Until then, the app stays a recommender.
2. **Score *any* peptide, not just library ones** — let the user paste a candidate
   and get its full evidence profile (currently it ranks the library vs an anchor).
3. **Safety upgrade** — replace the heuristic toxicity flag with a proper
   hemolysis/membrane-lysis model (HemoPI-style datasets).
4. **Cloud polish** — optionally commit `cpp_classifier.pkl` so the "CPP likelihood"
   column populates online; confirm resource limits are OK.
5. **Categorized shortlist** (Pareto) — "closest ClWOX analogs / lower-risk
   exploratory / mechanistically diverse" instead of one ranked list.
6. **Cargo/expression compatibility** — the downstream is a recombinant mCherry
   fusion; prioritize genetically-encodable, expression-friendly candidates.

## 10. Working style / conventions

- Modular, tested, typed. Add tests + a `docs/*.md` for new modules; keep
  `mypy --strict` and `ruff` clean. Commit messages end with the Co-Authored-By
  trailer. Commit/push only when asked.
- The app must stay honest (hypotheses, not algae predictions) and novice-readable.
- Don't reintroduce the removed app modes (diverse screen, variant engineering) or
  the max-charge-risk slider — the user wants a clean recommendation platform.
