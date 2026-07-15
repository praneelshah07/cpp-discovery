# `cpp_ai.descriptors` — Phase 2: physicochemical & QSAR descriptors

This module turns each peptide into a broad numerical fingerprint. The guiding
decision (per project direction) is **breadth**: compute *every* major scale so
downstream analysis can discover which properties best distinguish
seemingly-similar candidates, rather than committing early to one scale.

A single peptide yields **~200 features across 25 descriptor blocks**.

## Why breadth beats a single default

There is no universally "correct" hydrophobicity scale or pK set — each encodes
a different experiment and they disagree. Two peptides with near-identical
Kyte-Doolittle hydrophobicity can diverge on the Wimley-White interface scale;
a conservative I→L substitution barely moves gross charge but shifts several
QSAR axes. Computing the full battery preserves that discriminating signal for
feature selection (Phase 5), similarity (Phase 4), and visualization (Phase 9).

## The battery

| Block(s) | What | Biological relevance |
|----------|------|----------------------|
| `geometry` | length | baseline |
| `charge` | net charge at pH 7.4 under **all 9 pK scales** | cationic uptake is the dominant CPP mechanism |
| `isoelectric_point` | pI under all 9 pK scales | charge state across pH |
| `hydrophobicity_scales` | mean hydrophobicity under **~45 scales** | amphipathicity / membrane insertion |
| `hydrophobic_moment` | Eisenberg moment, helix (100°) & sheet (180°) | how the hydrophobic face segregates |
| `biopython_props` | GRAVY, aromaticity, instability, MW, pI, helix/turn/sheet fractions | structure & stability |
| `peptides_props` | MW, aliphatic index, Boman index, instability | stability, binding potential |
| `composition` | 20 residue fractions + interpretable group fractions (cationic, aromatic, …) | Arg/Lys/Trp content |
| `aggregation` | mean & peak windowed KD hydrophobicity | **labelled heuristic** APR proxy (see below) |
| QSAR: `zscales`, `kidera`, `vhse`, `fasgai`, `tscales`, `stscales`, `blosum`, `cruciani`, `atchley`, `ms_whim`, `protfp`, `sneath`, `physical`, `pcp`, `svger`, `vstpv` | multidimensional physicochemical embeddings | sharpest tool for separating near-identical peptides |

## Architecture

- **Every descriptor is a registered pure function** `str -> {feature: value}`
  in `DESCRIPTOR_REGISTRY`, individually testable and swappable. Select any
  subset by name, or run the whole battery (`blocks=None`).
- `compute_descriptors(seq)` / `compute_for_peptide(pep)` return an immutable
  `DescriptorSet` with `.vector(feature_names)` for building ML matrices.
- **Guards:** non-canonical sequences are rejected (their properties are
  ill-defined and libraries misbehave); non-finite values and duplicate feature
  names raise rather than silently corrupt matrices.

## Finding what discriminates (`analysis`)

Directly serving the "which property best separates these peptides?" question:

```python
from cpp_ai.descriptors import compute_descriptors, discriminative_ranking

peptides = {"pVEC": "LLIILRRRIRKQAHAHSK", "variant": "LLIILRRRIRKQAHAHSA"}
sets = [compute_descriptors(s, peptide_id=n) for n, s in peptides.items()]

for fs in discriminative_ranking(sets, by="value_range")[:5]:
    print(fs.feature, fs.value_range)
# -> molecular_weight, aliphatic_index, charge_pH7.4_*, ...  (the K→A signal)
```

`descriptor_matrix(sets)` builds an aligned `(n_peptides × n_features)` matrix
over shared features; `discriminative_ranking(sets, by=...)` ranks features by
spread (`coefficient_of_variation`, `std`, or `value_range`). This is
exploratory — principled, label-aware feature selection arrives in Phase 5.

## Aggregation: an honest proxy

Rigorous aggregation predictors (TANGO, Aggrescan, Waltz) use
experimentally-parameterized models not bundled here. Rather than hard-code
constants we cannot verify, the `aggregation` block reports the most hydrophobic
contiguous window (from the validated Kyte-Doolittle scale) as a **transparent,
clearly-labelled proxy** for aggregation-prone regions. Treat high values as
"worth checking", never as a claim of aggregation. Integrating a validated
predictor is a planned enhancement.

## Testing
`tests/descriptors/` (46 tests) covers the framework (canonical guard, finite &
uniqueness guards, vector ordering), physicochemical sanity (cationic→positive
charge, hydrophobic→positive GRAVY), composition (fractions sum to 1), QSAR
dimensionality, the aggregation proxy, and the discriminative-ranking analysis.
