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

| Phase | Module | Status |
|------|--------|--------|
| — | `core` (schema, registry, provenance, constants) | ✅ built + tested |
| 1 | `database` (CPPsite3, POSEIDON, CSV/Excel importers) | ✅ built + tested |
| 2 | `descriptors` (physicochemical + QSAR, ~200 features) | ✅ built + tested |
| 3 | `embeddings` (ESM-2, ProtT5, cached) | ✅ built + tested |
| 4 | `similarity` (alignment, embedding, descriptor, composite) | ✅ built + tested |
| 5 | `prediction` (RF/XGB/LGBM/SVM/NN + calibration + uncertainty) | ✅ built + tested |
| 6 | `generation` (constrained variants) | ✅ built + tested |
| 7 | `optimization` (multi-objective / Pareto) | ✅ built + tested |
| 8 | `ranking` (explainable) | ✅ built + tested |
| 9 | `visualization` (UMAP/t-SNE, heatmaps, radar) | ✅ built + tested |
| 10 | `webapp` (Streamlit screening dashboard) | ✅ built + tested |

## Design principles

- **Modular & replaceable.** Every algorithm registers itself and is selected
  by config, not imports.
- **Reproducible.** Immutable raw data, mandatory provenance, content-addressed
  peptide identity.
- **Lightweight core.** Heavy ML/embedding stacks live behind optional extras.
- **Explainable by construction.** The ranking layer cannot emit a score
  without a rationale.

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
