# Recommendation backbone (`cpp_ai.pipeline`)

One logic path for every front-end. `recommend_for_algae` loads the CPP library,
the curated evidence ledger, and the ledger-derived algae-fit model, runs the
evidence-backed scoring, and returns a ranked, filtered `AlgaeRecommendation`.
The Streamlit app and the CLI both call into it (via `filter_and_rank`), so the
science lives in exactly one place and neither front-end can drift from it.

## Why a backbone

The deliverable the lab actually wants is *"a model that outputs the best
peptides for algae, given computationally-backed logic"* — outputs, not
interactivity. So the pipeline is the product; the app is just an optional
interactive skin over it, and the CLI is the batch/report form. UI effort stays
near zero; all logic changes land here and both front-ends inherit them.

## The two questions it answers

- **"beyond just pVEC variants."** `max_identity` drops candidates whose global
  identity to the anchor is at/above the cutoff (e.g. 0.6), so genuinely
  different scaffolds surface instead of near-duplicates.
- **"which are more beneficial to algae, not just similar-looking."**
  `rank_by="algae_fit"` orders by the empirical algae-winner profile (from the
  ledger SAR) rather than resemblance to the anchor, **discounted by membrane-
  lysis risk** so amphipathic-but-lytic scaffolds (the TP10/MAP family) don't win
  on amphipathicity alone. Falls back to `"blend"` when the ledger has no
  informative algae signal (honest, not silent).
- **"make the top hits proper, not eight copies of one lytic scaffold."**
  `max_lysis_risk` drops AMP-like candidates; `collapse_families` keeps one
  representative per near-duplicate scaffold (fast difflib similarity) for a
  diverse panel.

## Anchor policy

`pVEC` is the default anchor: it has real *Chlamydomonas* data. `ClWOX` is
available but **not** the default — it is not yet demonstrated in algae (plant
and algal cell-wall chemistry differ), pending the lab's own ClWOX-in-algae
results.

## CLI

```bash
# gentle, diverse, genuinely different scaffolds ranked by the algae profile
python -m cpp_ai.pipeline --anchor pVEC --rank-by algae_fit --max-identity 0.6 \
    --max-lysis 0.6 --collapse-families 0.7 \
    --top 25 --out algae_candidates.csv --report algae_report.md
```

Flags: `--anchor` (preset or raw sequence), `--top`, `--rank-by {blend,algae_fit}`,
`--max-identity FRAC`, `--max-lysis RISK`, `--collapse-families RATIO`,
`--encodable-only`, `--categorize`, `--no-algae`, `--allow-toxic`, `--out`,
`--report`.

Outputs a ranked CSV and (optionally) a short, caveat-carrying markdown report.

## Options, not a list — categorized hypotheses

Per review, a flat top-20 is mostly near-duplicates. `categorize(profiles,
anchor)` partitions the ranked pool into distinct **testable bets**, a few
peptides each: *Closest to the anchor · Most novel scaffold · Gentlest (lowest
lysis) · Strongest algae profile · Highest CPP confidence*. A peptide may appear
in more than one bucket — that overlap is informative. `--categorize` (CLI) and
the app's "Group into hypotheses" toggle render them; the app lets you pick which
buckets to show.

## Why was this recommended?

`explain_profile(profile, fit_scorer=...)` turns the scoring axes into plain
✓/✕ reasons ("Matches the algae-winner profile (esp. amphipathicity)", "Low
predicted membrane-lysis", "Computational candidate — no direct evidence"),
making each recommendation a transparent, critiqueable argument instead of a
number. Shown inline in the categorized view and via the app's
"Why was a peptide recommended?" picker.

`peptide_family(seq)` is a coarse mechanistic tag (Homeodomain / Transportan /
pVEC-like / Polyarginine / …) so a panel reveals as a few mechanisms, not ten.

## Usable-delivery score (the default algae ranking)

`usable_delivery(p) = algae_fit × (1 − lysis_risk)² × fusion_confidence`.

- The **squared** lysis term (per external review) penalizes membrane-lytic
  peptides aggressively — transportan drops from ~29 to ~8 — so a lytic peptide
  never sits next to a balanced one on amphipathicity alone.
- `fusion_confidence` folds modification-awareness into the *logic* (no toggle):
  cloneable 1.0 · terminal cap 0.9 · tracer label 0.8 · functional conjugate
  (lipid/nanoparticle) 0.4 · non-canonical residue 0.15. A peptide whose function
  may depend on a non-encodable conjugate is discounted automatically.
- Honest limit: melittin stays mid-pack — its hydrophobic-helix + cationic-tail
  architecture defeats a GRAVY-based lysis proxy, and a `longest_hydrophobic_run`
  penalty backfires (pVEC has a longer run). Proper separation needs a real
  hemolysis model (deferred).

`group_families(profiles, threshold)` returns `FamilyGroup(representative,
members)` so the UI shows a diverse list of representatives yet can expand any
scaffold to its ranked variants.

## Modification-awareness (the tested form ≠ the naked sequence)

CPPsite3 records the form each peptide was *actually tested* in
(`ntermod`/`ctermod`/`chmod`), which the loader now carries.
`screening.modification.classify_modifications` buckets them into
`none` / `terminal` (amidation, acetylation) / `conjugate` (lipid, fluorophore,
biotin, nanoparticle) / `noncanonical` (non-standard residues), with a
`genetically_encodable` flag (true only for `none` — the bare sequence as
tested). **Only ~30% of the library is encodable as tested** — even pVEC was
fluorescein-tagged and amidated — so a modified peptide's measured behavior may
not transfer to a plain mCherry fusion.

Surfaced as the `tested_form` / `genetically_encodable` columns and an
`explain_profile` reason; `require_encodable` (CLI `--encodable-only`, app
"Cloneable only") filters to cloneable candidates; and a **"Cloneable
(encodable as tested)"** hypothesis bucket ranks the mCherry-fusion-ready ones.

## Naming (honest labels)

- **Algae suitability** (was "algae-delivery fit") — a design heuristic, not a
  delivery probability.
- **Composite score** (was "overall match") — a transparent convenience blend,
  not a probability. CSV columns: `algae_suitability`, `composite_score`.

## Shape

- `recommend_for_algae(...) -> AlgaeRecommendation` — the full path (load → score
  → filter → rank). `.to_dataframe()` / `.to_markdown()` /
  `.to_markdown_categorized()`.
- `filter_and_rank(profiles, anchor, ...)` — the shared filter+rank **policy**,
  factored out so the app can reuse its own cached `EvidenceScorer` (the
  expensive step) while still sharing the exact ranking policy.
- `categorize`, `explain_profile`, `peptide_family` — presentation logic shared
  by both front-ends.

## Honesty

Every output is a prioritized wet-lab hypothesis, not an algae-uptake
prediction. The algae-fit signal comes from a small curated ledger and sharpens
as real data arrives — the report says so on every run.

## Testing

`tests/test_pipeline.py`: anchor resolution, anchor exclusion, algae-fit axis on/
off, `rank_by="algae_fit"` ordering + fallback, the beyond-variants filter,
truncation, dataframe/markdown rendering, and that `filter_and_rank` reproduces
the pipeline ordering (the app/CLI share one policy).
