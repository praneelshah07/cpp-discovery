# `cpp_ai.visualization` — Phase 9: visualization

Turns the platform's outputs into figures for exploration and publication.
Every function is **pure** (builds and returns a figure, never displays it) and
supports **two backends**:

- `backend="plotly"` (default) — interactive HTML, ideal for the Phase 10 web app
- `backend="matplotlib"` — static, publication-quality figures for papers

Matplotlib figures are built via `matplotlib.figure.Figure` directly (not the
pyplot state machine), so no display/interactive backend is needed — they render
headlessly (e.g. in CI).

## Functions

| Function | What | Covers (spec) |
|----------|------|---------------|
| `embedding_map(X, method=…)` | reduce features to 2-D and scatter, colored by score or grouped by label | UMAP, t-SNE, embedding viz, similarity maps |
| `scatter_2d(coords, …)` | low-level 2-D scatter | — |
| `property_distribution(groups, kind=…)` | overlaid histograms / box plots of a property across groups | property distributions |
| `radar_plot(axes, series)` | spider chart of candidates across property axes | radar plots |
| `single_mutation_scan_matrix` + `mutation_heatmap` | deep-mutational-scan matrix (positions × 20 residues) and its heatmap | mutation heatmaps |
| `calibration_plot(curve)` | reliability curve from a Phase-5 `CalibrationCurve` | calibration curves |
| `pareto_scatter(solutions, x, y)` | Pareto front in two objective dimensions | multi-objective viz |

## Dimensionality reduction (`reduce_dimensions`)

`method="pca"` and `method="tsne"` come from scikit-learn and are always
available (t-SNE auto-scales its perplexity for small sets). `method="umap"` is
optional — if `umap-learn` isn't installed (it can be hard to build on some
platforms, e.g. Intel macOS) it raises a clear error pointing to PCA/t-SNE
rather than failing at import.

## Usage

```python
from cpp_ai.visualization import embedding_map, mutation_heatmap, single_mutation_scan_matrix

# Map ESM-2 embeddings, colored by predicted CPP likelihood, for the web app
fig = embedding_map(embedding_matrix, method="umap", color=cpp_probs, backend="plotly")

# Deep mutational scan of predicted CPP likelihood, as a publication figure
matrix = single_mutation_scan_matrix(pVEC.sequence,
                                     lambda s: classifier.predict_proba([Peptide.from_sequence(s, dataset="scan")])[0])
fig = mutation_heatmap(matrix, colorbar_label="P(CPP)", backend="matplotlib")
```

## Testing
`tests/visualization/` (29 tests): reduction (PCA/t-SNE shapes, determinism,
guards, UMAP-absent error) and every plot in **both backends** (correct figure
type returned, trace/structure checks, and input-validation guards). All
headless — no display required.
