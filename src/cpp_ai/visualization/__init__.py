"""Phase 9: visualization.

Pure figure-building functions with **dual backends** — Plotly (interactive,
for the web app) and Matplotlib (static, for publication) — selected via
``backend=``. Every function returns a figure object without displaying it, so
it is testable headlessly.

Includes: embedding maps (PCA / t-SNE / UMAP), property distributions, radar
profiles, deep-mutational-scan heatmaps, calibration curves, and Pareto-front
scatters. Behind the ``visualization`` extra (plotly, matplotlib, scikit-learn;
umap-learn optional).
"""

from __future__ import annotations

from .plots import (
    calibration_plot,
    embedding_map,
    mutation_heatmap,
    pareto_scatter,
    property_distribution,
    radar_plot,
    scatter_2d,
    single_mutation_scan_matrix,
)
from .reduce import reduce_dimensions

__all__ = [
    "reduce_dimensions",
    "scatter_2d",
    "embedding_map",
    "property_distribution",
    "radar_plot",
    "single_mutation_scan_matrix",
    "mutation_heatmap",
    "calibration_plot",
    "pareto_scatter",
]
