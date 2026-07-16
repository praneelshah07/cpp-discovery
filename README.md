# cpp-ai

A research-grade, open-source platform for the **computational discovery and
optimization of Cell Penetrating Peptides (CPPs)**.

> **Scientific disclaimer.** Every output of this platform is a *computationally
> prioritized hypothesis for wet-lab validation*, never a claim of biological
> function. Predictions come with confidence and rationale, and are intended to
> feed the experimental loop:
>
> design → CPP–mCherry fusion cloning → protein purification → delivery into
> algae → confocal microscopy → iteratively improve the model.

## Status

Early development. Built module-by-module with tests and docs at each step.

| Module | Role | Status |
|--------|------|--------|
| `core` (schema, registry, provenance, constants) | foundation | ✅ built + tested |
| `database` (CPPsite3, POSEIDON, CSV/Excel importers) | data provenance | ✅ built + tested |
| `descriptors` (physicochemical + QSAR, ~200 features) | features | ✅ built + tested |
| `embeddings` (ESM-2, ProtT5, cached) | optional features | ✅ built + tested |
| `similarity` (alignment, embedding, descriptor, composite) | resemblance | ✅ built + tested |
| `prediction` (RF/LGBM/SVM/NN + calibration + uncertainty) | CPP classifier | ✅ built + tested |
| `generation` (constrained variants) | sequence utils | ✅ built + tested |
| `scoring` (block similarity, safety, evidence profile, algae-fit) | ranking core | ✅ built + tested |
| `evidence` (curated literature ledger + SAR trends) | ground truth | ✅ built + tested |
| `screening` (library loading, candidates) | app data | ✅ built + tested |
| `webapp` (Streamlit recommendation app) | product | ✅ built + tested |

> **Scope note.** The `ranking`, `visualization`, and `optimization` (NSGA-II)
> modules were removed once the product settled into a focused recommendation
> platform — `scoring` superseded `ranking`, and the app renders no plots or
> variant-engineering. They remain in git history at tag `pre-prune-0.0.2`.

## Design principles

- **Modular & replaceable.** Every algorithm registers itself and is selected
  by config, not imports.
- **Reproducible.** Immutable raw data, mandatory provenance, content-addressed
  peptide identity.
- **Lightweight core.** Heavy ML/embedding stacks live behind optional extras.
- **Explainable by construction.** Every recommendation is decomposed into
  separate, labelled evidence axes rather than one opaque score.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # core + dev tooling

# opt into heavy stacks only when needed:
pip install -e ".[database]"       # importers
pip install -e ".[embeddings]"     # torch + ESM-2 / ProtT5
pip install -e ".[all]"            # everything
```

## Develop

```bash
pytest -q            # tests
mypy src             # strict type checking
ruff check src tests # lint
```

## Run the screening dashboard

```bash
pip install -e ".[webapp]"
streamlit run src/cpp_ai/webapp/app.py
```

Mine CPPsite3 for toxicity-aware, cloneable candidates (anchor search, diverse
gentle-entry screen, variant engineering) with CSV/FASTA export. See
[docs/webapp.md](docs/webapp.md).

## Documentation

Per-module docs (with biological rationale) live in [`docs/`](docs/). Start with
[`docs/core.md`](docs/core.md).

## License

MIT.
