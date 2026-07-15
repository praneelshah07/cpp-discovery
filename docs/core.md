# `cpp_ai.core` — foundational primitives

`core` is the dependency-light foundation every other module builds on. It has
no scientific logic of its own; instead it defines the *contracts* (data
schema), the *plumbing* (plugin registry), and the *vocabulary* (biological
constants and exceptions) that keep the rest of the platform modular and
reproducible.

## Why this module exists

A research platform that will be extended by multiple labs needs a stable
shared spine. If every module invents its own peptide representation, the
system rots. `core` prevents that by making one immutable, provenance-carrying
`Peptide` the single currency exchanged between phases.

## Components

### `types` — biological constants
The single source of truth for the **20 canonical (genetically-encodable)
amino acids**. This matters biologically: the intended downstream experiment is
a recombinant **CPP–mCherry fusion protein**, which can only encode standard
residues. Generation and optimization therefore default to these 20; anything
else must be an explicit, flagged opt-in.

- `CANONICAL_AMINO_ACIDS` / `CANONICAL_AMINO_ACIDS_ORDERED`
- `AMBIGUOUS_CODES` — B, J, O, U, X, Z: real database placeholders that are
  **flagged, never silently dropped**.
- `is_canonical_sequence()`, `non_canonical_residues()`

### `schema` — the data contract
- **`ProvenanceRecord`** — immutable origin record (dataset, original id,
  source file + SHA-256, UTC import timestamp, verbatim extras). Provenance is
  *mandatory*: a peptide cannot exist without one.
- **`Peptide`** — immutable (`frozen`) peptide. Key properties:
  - **Content-addressed identity.** `peptide_id` is derived from the sequence,
    so the *same peptide from two databases resolves to the same ID* (with two
    provenance records). This is exactly the dedup semantics we want.
  - **Canonicalized but not censored.** Sequences are upper-cased/stripped and
    validated as letters-only, but ambiguity codes are allowed and surfaced via
    `is_canonical` / `non_canonical_residues` rather than discarded — honoring
    "never overwrite raw data."
- `compute_peptide_id()` — the hashing function behind identity.

### `registry` — swappable algorithms
`Registry[T]` is a generic, type-safe name→component map. Each family of
interchangeable parts (descriptors, similarity metrics, models, mutation
operators, objectives) gets its own registry. Higher layers select components
**by configuration**, not imports — this is how "every algorithm should be
replaceable" becomes real. Duplicate registrations raise unless
`overwrite=True`, because silently shadowing a scientific component is a
reproducibility hazard.

### `exceptions` — one rooted hierarchy
All errors derive from `CppAiError`, so callers can catch everything with one
`except` while still distinguishing `ValidationError`, `ProvenanceError`, and
registry errors.

## Guarantees

- **Immutability:** raw records cannot be mutated in place (`frozen` models).
- **Provenance:** every `Peptide` carries where it came from.
- **Determinism:** IDs, ordered residues, and registry listings are stable.
- **Light footprint:** only `pydantic` + `numpy`; safe to import anywhere.

## Testing
`tests/core/` covers types, schema (identity, canonicalization, immutability,
ambiguity handling, provenance), and registry behavior — 41 tests, `mypy
--strict` and `ruff` clean.
