# `cpp_ai.generation` — Phase 6: constrained candidate generation

Generate mutational variants of known CPPs (pVEC, ClWOX, …) to explore the
sequence neighborhood of experimentally successful peptides — the raw material
the ranking (Phase 8) and optimization (Phase 7) phases select from.

## Substitution strategies (`SUBSTITUTION_REGISTRY`)

Each strategy is a registered pure function `residue -> allowed replacements`
(never itself), encoding a design intent:

| Strategy | Replacements | Intent |
|----------|--------------|--------|
| `all_canonical` | any of the other 19 residues | exhaustive scan |
| `conservative` | positive-BLOSUM62 residues | safest edits; likely preserve fold/function |
| `charge_preserving` | same charge class only | hold net charge (the dominant CPP driver) constant |
| `hydrophobic` | hydrophobic residues (AVLIMFW) | probe increased amphipathicity |

`charge_preserving`'s classes are defined to **exactly match** the `net_charge`
metric (only K/R = +1, D/E = −1, everything else neutral incl. His), so it
genuinely preserves net charge — verified in tests.

## Constraints (`MutationConstraints`)

Fence the design space to sensible, expressible variants:

- `locked_positions` / `locked_motifs` — never mutate a functional motif (e.g.
  lock `"LLII"`); all motif occurrences are protected.
- `canonical_only` (default **True**) — keep candidates genetically encodable
  for the recombinant CPP-mCherry fusion.
- `max_charge` / `min_charge` — e.g. "max charge +8".
- `max_length` / `min_length`.
- `custom_filters` — arbitrary `sequence -> bool` predicates.

`net_charge(seq) = (#K + #R) − (#D + #E)` is a fast integer proxy for gating
(His and termini excluded; use the descriptor module for pH-aware charge).

## Generator (`VariantGenerator`)

```python
from cpp_ai.core.schema import Peptide
from cpp_ai.generation import VariantGenerator, MutationConstraints

pVEC = Peptide.from_sequence("LLIILRRRIRKQAHAHSK", dataset="ref")
gen = VariantGenerator(
    "conservative",
    MutationConstraints(locked_motifs=("LLII",), max_charge=8),
)

singles = gen.generate(pVEC, n_mutations=1)                       # exhaustive
doubles = gen.generate(pVEC, n_mutations=2, max_variants=500)     # sampled
series  = gen.generate_series(pVEC, mutation_counts=(1, 2, 3))    # convenience
```

- Only **substitutions** (length is preserved), so `fixed length` is inherent.
- `max_variants=None` enumerates exhaustively (with a 100k safety cap); passing
  `max_variants` switches to **reproducible random sampling** (seeded) to avoid
  combinatorial blow-up on double/triple mutants of long peptides.
- Results are deduplicated, exclude the parent, and each is a `Peptide` whose
  metadata records `parent_id`, `parent_sequence`, `strategy`, `n_mutations`,
  and the exact `mutations` (e.g. `["K18A"]`) — full traceability for the
  ranking rationale.
- A non-canonical reference is rejected (unless `canonical_only=False`).

## Testing
`tests/generation/` (30 tests) covers each strategy (self-exclusion, biochemical
correctness, charge-class integrity), constraints (net charge, motif/position
locking, canonical/charge/length/custom acceptance), and the generator (mutation
counts, motif protection, charge limits, canonical guard, bounded/deterministic
sampling, parent metadata, dedup).
