# `cpp_ai.embeddings` — Phase 3: protein language-model embeddings

Protein language models (pLMs) map a sequence to a fixed-length vector that
encodes structural and evolutionary context far beyond hand-crafted descriptors.
These vectors feed the similarity engine (Phase 4) and can be optional inputs to
the ML models (Phase 5).

## What's here

| Component | Purpose |
|-----------|---------|
| `Embedder` | model-agnostic interface; subclasses implement batched `_embed_sequences` |
| `EmbeddingRecord` | a vector tagged with its model + peptide |
| `EmbeddingCache` | content-addressed on-disk cache (compute once, reuse forever) |
| `EmbeddingService` | ties an embedder to a cache: fetch-or-compute, batch misses |
| `MockEmbedder` | deterministic, dependency-free; runs the whole pipeline offline |
| `ESM2Embedder` | ESM-2 (fair-esm), default `esm2_t33_650M`, CPU-capable |
| `ProtT5Embedder` | ProtT5 encoder (transformers) |

All embedders register in `EMBEDDER_REGISTRY` (`mock`, `prott5`, `esm2_t6_8M`,
`esm2_t12_35M`, `esm2_t30_150M`, `esm2_t33_650M`).

## Design decisions

- **Cache everything.** Embeddings are deterministic per (model, sequence) but
  expensive (CPU inference is seconds per peptide). The cache keys on a hash of
  model name + sequence so vectors from different models never collide, and
  writes are atomic (temp file + rename) so an interrupted run can't corrupt a
  vector. `EmbeddingService` computes only cache misses, once, in a batch.
- **Lazy heavy imports.** torch / fair-esm / transformers are imported *inside*
  methods, so importing this package (e.g. to list models) is free. They live
  behind the `embeddings` install extra.
- **Mock as a first-class tool.** `MockEmbedder` is not just for tests — it lets
  similarity, ML, and ranking be developed and run with no torch and no
  multi-gigabyte downloads. It carries **no biological meaning**; never use it
  for real analysis.
- **Pooling.** Per-residue representations are reduced to one vector by `mean`
  (default) or `max` pooling. For ESM-2 the BOS/EOS tokens are excluded.

## Usage

```python
from cpp_ai.embeddings import ESM2Embedder, EmbeddingCache, EmbeddingService

service = EmbeddingService(
    ESM2Embedder("esm2_t33_650M"),          # default; downloads weights on first use
    EmbeddingCache("data/processed/emb_cache"),
)
records = service.embed_sequences(["LLIILRRRIRKQAHAHSK", "KWKLFKKI"])
matrix = service.embedding_matrix(records)   # (2, 1280)
```

Offline / fast development:

```python
from cpp_ai.embeddings import MockEmbedder, EmbeddingCache, EmbeddingService
service = EmbeddingService(MockEmbedder(dim=320), EmbeddingCache("/tmp/cache"))
```

## Hardware notes

- **No GPU required.** ESM-2 650M runs on CPU (slower, seconds/peptide); the
  cache means you pay that cost only once per peptide.
- **Intel macOS:** PyTorch's last x86-macOS build is 2.2.x, which was compiled
  against NumPy 1.x and crashes under NumPy 2. The `embeddings` extra therefore
  pins `numpy<2` on that platform automatically. (Apple Silicon and Linux are
  unaffected.)
- Model sizes: 8M (320-D, tests), 35M (480-D), 150M (640-D), 650M (1280-D,
  default). ProtT5 is 1024-D and ~2-3 GB.

## Testing

`tests/embeddings/` covers the cache (roundtrip, namespacing, atomic overwrite),
the interface + pooling, `MockEmbedder` determinism, and the `EmbeddingService`
cache-first / batch-misses / order-preservation logic — all offline. Real ESM-2
is exercised by `test_esm2_real.py` (8M checkpoint), marked `slow` and skipped
unless fair-esm + torch are installed; it is **verified passing on CPU**.
```
pytest -m "not slow"   # fast, no ML deps
pytest -m slow         # real ESM-2 (needs the embeddings extra)
```
