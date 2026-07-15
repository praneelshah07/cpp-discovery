# `cpp_ai.similarity` — Phase 4: the similarity engine

Rank peptides by how similar they are to a reference (pVEC, pVEC-R6A, ClWOX, …)
using a **user-weighted blend of complementary metrics**, with a full
explanation attached to every score. This is where the platform first becomes
queryable: *"rank all known CPPs by similarity to pVEC."*

## Metrics (each registered & swappable in `SIMILARITY_REGISTRY`)

All metrics return a similarity in **[0, 1]** (1 = identical) so they compose.

| Metric | Family | What it measures | Needs |
|--------|--------|------------------|-------|
| `sequence_identity` | sequence | identical-position fraction of the global alignment | sequence |
| `smith_waterman` | sequence | local alignment (BLOSUM62), self-normalized | sequence |
| `needleman_wunsch` | sequence | global alignment (BLOSUM62), self-normalized | sequence |
| `embedding_cosine` | embedding | cosine of pLM vectors, mapped to [0, 1] | embedding |
| `descriptor_similarity` | descriptor | `1/(1+distance)` in standardized descriptor space | descriptors |

Alignment scores are normalized by self-alignment score, so identical sequences
score exactly 1 and length differences are handled gracefully.

## Composite: maximum adjustability

`CompositeSimilarity` blends metrics with **fully user-adjustable weights**:

- **Any subset** of metrics; **arbitrary non-negative weights** (normalized
  internally — `{"embedding_cosine": 3, "sequence_identity": 1}` is a valid 3:1
  blend); zero weights drop a metric.
- **Custom-parameterized metric instances** via `overrides` (e.g. Smith-Waterman
  with different gap penalties, or an entirely new registered metric).
- **Explainable by construction:** every result is a `SimilarityBreakdown` with
  each metric's raw score, weight, and weighted contribution — a composite score
  is never returned without saying how it was reached.

### Default weights
Per project direction, the default favors **embedding + descriptor** similarity
(0.30 each → 0.60 combined) over the three sequence-alignment metrics (0.40
combined), because learned and physicochemical closeness track *functional* CPP
similarity better than raw sequence identity. Fully overridable.

## Engine

`SimilarityEngine.rank(reference, candidates, top_k=...)` returns an explainable,
ranked `list[SimilarityHit]`. It precomputes **only** the features the active
metrics need:

- **Embeddings** via the cached `EmbeddingService` (so vectors are reused across
  queries). Required only if an embedding metric has non-zero weight — otherwise
  no embedder is needed at all.
- **Descriptors** computed for the reference + candidates, then **standardized
  per feature across the population** so different-scale descriptors contribute
  comparably to the distance. Non-canonical candidates are rejected with a clear
  error (their descriptors are ill-defined).

## Usage

```python
from cpp_ai.similarity import SimilarityEngine, CompositeSimilarity
from cpp_ai.embeddings import ESM2Embedder, EmbeddingCache, EmbeddingService

svc = EmbeddingService(ESM2Embedder("esm2_t33_650M"), EmbeddingCache("data/processed/emb"))
engine = SimilarityEngine(CompositeSimilarity(), embedding_service=svc)

for hit in engine.rank(pVEC, known_cpps, top_k=20):
    print(hit.target_sequence, round(hit.composite, 3),
          hit.breakdown.top_contributors(2))
```

Sequence-only (no embedder, no descriptors needed):

```python
engine = SimilarityEngine(CompositeSimilarity({"smith_waterman": 1, "sequence_identity": 1}))
```

Offline full composite with the mock embedder (no torch): pass
`EmbeddingService(MockEmbedder(), cache)`.

## Testing
`tests/similarity/` (36 tests) covers each metric (identity=1, unit-interval
bounds, alignment symmetry, embedding cosine at 1/0.5/0, descriptor distance
monotonicity, missing-feature errors), the composite (normalization, ratio
weights, exclusions, breakdown sums, overrides, out-of-range guard), and the
engine (ranking order, top_k, feature precomputation, non-canonical guard,
sequence-only path, id propagation).
