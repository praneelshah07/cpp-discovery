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
`--no-algae`, `--allow-toxic`, `--out`, `--report`.

Outputs a ranked CSV and (optionally) a short, caveat-carrying markdown report.

## Shape

- `recommend_for_algae(...) -> AlgaeRecommendation` — the full path (load → score
  → filter → rank). `AlgaeRecommendation.to_dataframe()` / `.to_markdown()`.
- `filter_and_rank(profiles, anchor, ...)` — the shared filter+rank **policy**,
  factored out so the app can reuse its own cached `EvidenceScorer` (the
  expensive step) while still sharing the exact ranking policy.

## Honesty

Every output is a prioritized wet-lab hypothesis, not an algae-uptake
prediction. The algae-fit signal comes from a small curated ledger and sharpens
as real data arrives — the report says so on every run.

## Testing

`tests/test_pipeline.py`: anchor resolution, anchor exclusion, algae-fit axis on/
off, `rank_by="algae_fit"` ordering + fallback, the beyond-variants filter,
truncation, dataframe/markdown rendering, and that `filter_and_rank` reproduces
the pipeline ordering (the app/CLI share one policy).
