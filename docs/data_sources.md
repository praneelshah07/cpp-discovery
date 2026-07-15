# Getting the training data (CPPsite 2.0 & POSEIDON)

This platform learns from experimentally-curated CPP databases. Here is exactly
what to download and what the ML layer (Phase 5) needs. Once you have the files,
drop them in `data/raw/` and the importers ([`cpp_ai.database`](database.md))
will ingest them.

## Current status (what we have)

- ✅ **POSEIDON** `poseidon_cargo_encoded.csv` is in `data/raw/`. `PoseidonImporter`
  ingests it: **863 usable records → 385 unique peptides** (453 rows skipped —
  chemically-modified entries that aren't plain sequences; 38 flagged
  non-canonical). Provides **quantitative uptake** (`target`) → regression-ready.
- ✅ **CPPsite structures** — four `.tgz` archives of PDB 3D structures (~1558
  peptides) are in `data/raw/`. Useful for future structure-based features;
  sequences are extractable from them.
- ⛳ **Still needed for classification:** the CPPsite *annotation table*
  (sequence + metadata as CSV/TSV) was not obtained — only structures — and a
  **negative set** (non-CPP peptides). Until then, classification runs on
  synthetic data; POSEIDON's uptake supports regression directly.

## What the model actually needs

| Task | Label needed | Source |
|------|--------------|--------|
| **Classification** ("is this a CPP?") | positives **and** negatives | CPPsite (positives) + a negative set |
| **Regression** ("how efficient is uptake?") | quantitative uptake values | POSEIDON |

> **Important:** CPPsite is a **positive-only** database (every entry is a known
> CPP). To train a CPP-vs-non-CPP classifier you must also supply **negatives** —
> commonly random peptides sampled from UniProt/SwissProt that are not known
> CPPs, or the negative sets used by published CPP predictors. POSEIDON, by
> contrast, has quantitative uptake values suitable for regression directly.

## 1. CPPsite 2.0 (Raghava lab, IIIT-Delhi)

- **Site:** http://crdd.osdd.net/raghava/cppsite/ (also mirrored via
  webs.iiitd.edu.in). ~1855 entries of experimentally validated CPPs.
- **How:** open the site's **Download** section and export the complete dataset
  as **CSV/TSV** (an SDF is also offered; CSV/TSV is easiest for us). If only an
  in-browser table is available, use "download complete dataset".
- **Columns we use:** peptide `Sequence` (one-letter) and its ID. Everything
  else is preserved as metadata. Useful fields to keep if present: uptake
  efficiency, cargo, N-/C-terminal modifications, cell line, uptake mechanism.
- The site uses an old TLS config, so a browser download is the reliable path.

## 2. POSEIDON (Moreira lab)

- **Site:** https://moreiralab.com/resources/poseidon/
- **Data file (recommended):** the GitHub repo
  https://github.com/MoreiraLAB/poseidon — folder **`curated_data/`**, file
  **`cargo_encoded.csv`** (~333 KB; the main dataset). 2,300+ entries with
  **quantitative uptake values** and physicochemical properties.
- **Columns we use:** the peptide sequence column and the uptake-value column
  (for regression); the rest is kept as metadata.

## Handing the data to the platform

1. Save the files under `data/raw/` (kept out of git; treated as immutable).
2. Tell me the filenames. I'll confirm the exact column names (the importers
   resolve columns case-insensitively, so minor header differences are fine) and
   wire up the presets `CPPsite3Importer` / `PoseidonImporter`.
3. For classification I'll help assemble a negative set (e.g. length-matched
   random UniProt peptides) so we can train beyond the synthetic data.

Until then, Phase 5 runs on `make_synthetic_cpp_dataset()`, and real data plugs
in at the identical `(list[Peptide], labels)` interface.
