# `cpp_ai.webapp` — Phase 10: the screening dashboard

A Streamlit UI that lets you run the CPP screening workflows yourself — anchor
similarity search, diverse gentle-entry mining, and variant engineering — over
the CPPsite3 database, with toxicity flags and CSV/FASTA download.

Per the platform's architecture it is a **thin layer**: all logic lives in the
tested [`cpp_ai.screening`](../src/cpp_ai/screening) module; `app.py` only wires
widgets to those functions.

## Launch

```bash
pip install -e ".[webapp]"          # streamlit, pandas, pyarrow
streamlit run src/cpp_ai/webapp/app.py
```

Requires `data/raw/cppsite3_api.json` (see [data_sources.md](data_sources.md)).
On Intel macOS the `webapp` extra pins `pyarrow<15` (newer versions have no x86
wheel).

## The three modes

1. **Anchor similarity search** — rank the whole CPPsite3 library against an
   anchor peptide (presets: ClWOX, pVEC-R6A, pVEC-WT, or type your own). Ranking
   uses physicochemical-descriptor similarity by default; tick **Use ESM-2
   embeddings** to add functional (pLM) similarity, with a slider to weight the
   two. Filters and diversity are applied to the ranked list.

2. **Diverse gentle-entry screen** — filter-first, then diversify: surfaces
   *non-obvious* candidates from varied protein families that share ClWOX's
   gentle-entry profile (endocytic, moderate charge, aromatic, non-homeodomain).
   This is the "mine the database for things I wouldn't pick by hand" mode.

3. **Engineer variants** — generate constrained conservative/charge-preserving/
   hydrophobic variants of the anchor (lock a motif, bound charge), then view and
   download them.

## Toxicity awareness (baked in)

Grounded in the lab's own finding that 9R/pVEC are cytotoxic to *Chlamydomonas*:

- **Charge flag** — `toxicity_flag` colors each row (green ≤ +5, yellow +6/+7,
  red ≥ +8).
- **`lytic_risk`** — flags antimicrobial / direct-translocation / pore-forming
  peptides that enter by membrane disruption (algae-toxic) even at low charge.
- The net-charge filter defaults to +4…+7, and "exclude membrane-lytic" is one
  click.

## Outputs

Every screen shows a sortable table and offers **Download CSV** and **Download
FASTA** (annotation-rich headers) — ready to PCR/clone as mCherry fusions.

> All outputs are computational hypotheses for wet-lab validation, never claims
> of efficacy.

## Testing

The UI logic is tested in `tests/screening/` (candidate flags, library loading &
filters, and the similarity index — 17 tests). The Streamlit script itself is
thin glue; it was verified to boot and render live.
