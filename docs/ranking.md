# `cpp_ai.ranking` — Phase 8: the explainable ranking engine

The layer that ties the platform together. It takes candidate peptides (from
generation or optimization) and produces a ranked list where **every entry is
inseparable from its explanation**.

## The structural guarantee

The project rule — *never return a score without an explanation* — is enforced
by the type system, not by convention. `RankedCandidate` cannot be constructed
without a non-empty `reasons`, a `mutation_summary`, and `weaknesses` (its
`__post_init__` raises otherwise). There is no code path that emits a bare
number.

Every `RankedCandidate` carries exactly what the spec requires:

- `overall_score` and `similarity_score`
- `confidence`
- `mutation_summary` — e.g. *"1 substitution from pep_x: K11R"*
- `nearest_literature` — closest known/literature peptides with % identity
- `reasons` — why it was selected
- `strengths` — predicted positives
- `weaknesses` — potential negatives (always ending with the standing caveat
  that this is a computational prediction requiring wet-lab validation)
- supporting evidence: `cpp_probability`, `uncertainty`, `epistemic_std`,
  `components` (the sub-scores that formed the overall score)

## How explanations are built (`explain.py`)

Deterministic, transparent rules turn concrete numbers into sentences, with all
cut-offs in a tunable `RankingThresholds`:

- **reasons** — high CPP likelihood, high similarity to the reference, novelty
  vs. literature, and the traceable design path. Guaranteed non-empty.
- **strengths** — favorable cationic charge, expression compatibility,
  amphipathicity, low aggregation (heuristic), low model uncertainty.
- **weaknesses** — non-canonical residues, charge above the recommended +8,
  high predictive entropy, ensemble disagreement (extrapolation risk), elevated
  aggregation proxy, cysteine load, off-range length, low CPP likelihood — plus
  the always-present validation caveat.

An experimentalist can trace every sentence back to a number.

## The engine (`RankingEngine`)

Scoring sources are optional and composable — provide a `classifier` (Phase 5)
for CPP likelihood, a `reference` (e.g. pVEC) for similarity, or both:

```python
from cpp_ai.ranking import RankingEngine

engine = RankingEngine(
    classifier=trained_cpp_classifier,     # Phase 5 (optional)
    reference=pVEC,                        # Phase 4 similarity (optional)
    literature=[pVEC, TAT, penetratin],    # for nearest-literature
    weights={"cpp_likelihood": 0.6, "similarity": 0.4},   # user-adjustable
)
ranked = engine.rank(candidate_peptides, top_k=20)

for c in ranked:
    print(c.sequence, round(c.overall_score, 3), c.mutation_summary)
    print("  reasons:", c.reasons)
    print("  strengths:", c.strengths)
    print("  weaknesses:", c.weaknesses)
```

- **overall score** = weighted blend of the available components (normalized
  over whatever is present); needs at least one scoring source.
- **similarity & nearest literature** use a fast sequence-based composite, so
  ranking doesn't require embeddings (though a custom composite can be passed).
- **mutation summary** is read from generation/optimization metadata, or diffed
  against the reference when metadata is absent.
- **non-canonical candidates** are ranked without crashing (descriptors are
  skipped) and flagged as a weakness.

## Testing
`tests/ranking/` (28 tests): the structural guarantee (empty reasons / mutation
summary / weaknesses all rejected), the explanation rules (each reason,
strength, and weakness path; caveat always present), and the engine (sorting,
top-k, classifier-only / reference-only / combined modes, mutation-summary
sourcing, nearest-literature ordering, non-canonical handling, and the
every-candidate-has-a-full-explanation invariant).
