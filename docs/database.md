# `cpp_ai.database` — Phase 1: the database engine

Before any modeling, the platform needs a trustworthy, reproducible home for
the experimental data it learns from. This module imports curated CPP datasets
and arbitrary spreadsheets into one **standardized, provenance-preserving
store**.

## Why this comes first

Models are only as credible as their training data. A publication-grade
platform must be able to answer, for any peptide it reasons about: *where did
this come from, and exactly which file/row produced it?* That auditability is
built into the data layer rather than bolted on later.

## Guarantees

- **Raw data is never overwritten.** Every imported peptide keeps its *entire*
  original row verbatim in `metadata`.
- **Provenance is mandatory and file-exact.** Each record stores its dataset,
  the source's own ID, the source path, and the **SHA-256 of the source file**,
  plus a UTC import timestamp.
- **Content-addressed identity with multi-source merge.** The same sequence
  from CPPsite3 and POSEIDON becomes *one* peptide carrying *both* provenance
  records (verified end-to-end).
- **Idempotent imports.** Re-importing the same file adds no duplicate sources
  (dedup on `dataset + original_id + file_sha256`).
- **Nothing is silently dropped.** Empty or malformed sequences are recorded as
  `SkippedRow`s with reasons; non-canonical sequences are imported but flagged.

## Components

### Importers
All importers subclass `Importer`, which handles provenance, validation,
skip-tracking, and metadata preservation uniformly. Subclasses only say *how to
read rows*.

| Class | Source | Notes |
|-------|--------|-------|
| `CSVImporter` | delimited text | stdlib only; delimiter sniffed (`,` `\t` `;` `|`) |
| `ExcelImporter` | `.xlsx` | needs `openpyxl` (`pip install "cpp-ai[database]"`) |
| `CPPsite3Importer` | CPPsite3 export | preset mapping `Sequence` / `CPPsite ID` |
| `PoseidonImporter` | POSEIDON export | preset mapping `Sequence` / `ID` |

`ColumnMapping` maps a source's columns to standardized fields and resolves them
**case- and whitespace-insensitively**, so header-spelling drift between
database releases doesn't break imports. Only the sequence column is required.

All importers are registered in `IMPORTER_REGISTRY`, so a pipeline can select a
source by name from config.

### Store
`PeptideStore` is the storage interface, with two backends:
- `InMemoryPeptideStore` — for tests and small analyses.
- `SqlitePeptideStore` — persistent, standard-library SQLite (no ORM). Peptides
  and their provenance live in separate tables so one peptide can carry many
  sources.

`StoredPeptide` aggregates a sequence with its `PeptideSource`s and exposes
`length`, `is_canonical`, and `datasets`.

## Example

```python
from cpp_ai.database import CPPsite3Importer, PoseidonImporter, SqlitePeptideStore

store = SqlitePeptideStore("data/processed/peptides.db")
store.add_many(CPPsite3Importer().import_file("data/raw/cppsite3.csv").peptides)
store.add_many(PoseidonImporter().import_file("data/raw/poseidon.csv").peptides)

for sp in store:
    print(sp.peptide_id, sp.sequence, sp.datasets)
```

For a custom file:

```python
from cpp_ai.database import CSVImporter, ColumnMapping

importer = CSVImporter(ColumnMapping(sequence="peptide_seq", identifier="acc"),
                       dataset_name="my_lab_2026")
report = importer.import_file("data/raw/my_data.csv")
print(report.summary())          # imported N (M non-canonical), skipped K
```

## Testing
`tests/database/` covers provenance hashing, all importers (skips, invalid
sequences, non-canonical flagging, case-insensitive columns, delimiter
sniffing, Excel, presets), and both store backends (merge, idempotency,
ordering, persistence) — parametrized across memory + SQLite.
