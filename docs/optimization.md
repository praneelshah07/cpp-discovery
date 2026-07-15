# `cpp_ai.optimization` — Phase 7: multi-objective optimization

CPP design has no single "best" peptide: making a candidate more CPP-like may
make it more aggregation-prone; making it more novel makes it less similar to
pVEC. Collapsing these into one weighted score hides the trade-offs. Instead
this phase returns the **Pareto front** — the set of candidates that cannot be
improved on any objective without sacrificing another.

## Approach: a built-in NSGA-II

A constraint-aware NSGA-II evolves peptides using the **Phase 6 substitution
operators** as mutation, plus uniform crossover between equal-length sequences.
We use a purpose-built GA rather than `pymoo` because the search space is
variable-content amino-acid strings with domain-specific, constraint-aware
operators — a custom loop expresses that more directly and transparently, with
no extra dependency. The objective interface is backend-agnostic, so a pymoo
backend could be added later without changing any objective definitions.

## Objectives (`Objective`, all optional & user-composed)

| Objective | Direction | Source |
|-----------|-----------|--------|
| `CPPLikelihoodObjective(classifier)` | maximize | Phase 5 calibrated P(CPP) |
| `SimilarityToReferenceObjective(ref)` | maximize | Phase 4 composite similarity (sequence-based default) |
| `DescriptorObjective(name, dir, block, feature)` | either | Phase 2 descriptors (e.g. GRAVY, aggregation proxy) |
| `NetChargeObjective(target=…)` | maximize/closeness | net charge |
| `NoveltyObjective(known)` | maximize | 1 − k-mer Jaccard to a known set |
| `ExpressionCompatibilityObjective()` | maximize | recombinant-expressibility heuristic |
| `FunctionObjective(name, dir, fn)` | either | **any** custom scorer |

Each declares a `Direction`; the engine negates "maximize" objectives so
everything is minimized internally.

### Scientific honesty
- **Toxicity is deliberately not fabricated** — no bundled toxicity model. When
  you have one, plug it in as `FunctionObjective("toxicity", MINIMIZE, model_fn)`.
- The `aggregation` and `expression_compat` objectives are **labelled
  heuristics**, not validated assays.

## Pareto primitives (`pareto.py`)
`dominates`, `non_dominated_sort` (Deb fast sort → fronts), `crowding_distance`
(diversity preservation, boundary-infinite), `pareto_front_indices`. All operate
on a minimization matrix and are independently tested.

## Usage

```python
from cpp_ai.optimization import (NSGA2Optimizer, NSGA2Config, Direction,
    SimilarityToReferenceObjective, NoveltyObjective, DescriptorObjective,
    ExpressionCompatibilityObjective, CPPLikelihoodObjective)
from cpp_ai.generation import MutationConstraints

objectives = [
    CPPLikelihoodObjective(trained_classifier),
    SimilarityToReferenceObjective(pVEC, name="sim_pVEC"),
    NoveltyObjective([p.sequence for p in known_cpps]),
    DescriptorObjective("aggregation", Direction.MINIMIZE,
                        "aggregation", "aggregation_peak_window_hydrophobicity"),
    ExpressionCompatibilityObjective(),
]
opt = NSGA2Optimizer(
    objectives,
    strategy="conservative",
    constraints=MutationConstraints(locked_motifs=("LLII",), max_charge=8),
    config=NSGA2Config(population_size=60, n_generations=30, seed=0),
)
result = opt.optimize([pVEC])          # seed from a known CPP
for sol in result.front:               # Pareto-optimal trade-off set
    print(sol.peptide.sequence, sol.objective_values)
```

## Behaviour & guarantees
- Returns the **deduplicated** Pareto front, each `ParetoSolution` carrying its
  raw per-objective values (with the direction of each objective reported).
- **Constraints are enforced throughout** — locked motifs/positions, charge and
  length bounds, canonical-only — so every candidate stays valid.
- **Reproducible** (seeded) and **cached** (each unique sequence is scored once
  across the whole run).
- Offspring record their `parents` in metadata for traceability.

## Testing
`tests/optimization/` (25 tests): the Pareto primitives (domination, fronts,
crowding), every objective (directions, `as_minimization`, novelty, expression
heuristic, similarity, descriptor read-through), and the optimizer (non-empty &
unique front, mutual non-domination, constraint enforcement, determinism,
single-objective convergence to a target, input guards).
