"""Plotting functions with dual backends: Plotly (interactive) and Matplotlib
(publication).

Every function accepts ``backend="plotly"`` (default; interactive HTML, ideal for
the Phase 10 web app) or ``backend="matplotlib"`` (static figures for papers) and
returns the corresponding figure object. The functions are **pure** — they build
and return a figure without displaying it — so they are testable headlessly.

Matplotlib figures are built via ``matplotlib.figure.Figure`` directly (not the
pyplot state machine), so no interactive backend or display is required.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ..core.exceptions import ValidationError
from ..core.types import CANONICAL_AMINO_ACIDS_ORDERED
from .reduce import reduce_dimensions

_BACKENDS = ("plotly", "matplotlib")


def _check_backend(backend: str) -> None:
    if backend not in _BACKENDS:
        raise ValidationError(f"backend must be one of {_BACKENDS}, got {backend!r}.")


def _mpl_figure() -> Any:
    from matplotlib.figure import Figure

    return Figure(figsize=(7, 5), tight_layout=True)


def _go() -> Any:
    import plotly.graph_objects as go

    return go


# --------------------------------------------------------------------------- #
# 2-D scatter / embedding map
# --------------------------------------------------------------------------- #
def scatter_2d(
    coords: npt.NDArray[np.float64],
    *,
    color: Sequence[float] | npt.NDArray[np.float64] | None = None,
    labels: Sequence[str] | None = None,
    text: Sequence[str] | None = None,
    backend: str = "plotly",
    title: str = "",
    axis_labels: tuple[str, str] = ("dim 1", "dim 2"),
) -> Any:
    """Scatter 2-D coordinates, optionally colored by a continuous ``color`` or
    categorical ``labels``, with ``text`` hover annotations."""
    _check_backend(backend)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValidationError("coords must be (n, >=2).")
    x, y = coords[:, 0], coords[:, 1]

    if backend == "plotly":
        go = _go()
        fig = go.Figure()
        if labels is not None:
            for lab in dict.fromkeys(labels):
                mask = [i for i, la in enumerate(labels) if la == lab]
                fig.add_trace(
                    go.Scatter(
                        x=x[mask], y=y[mask], mode="markers", name=str(lab),
                        text=None if text is None else [text[i] for i in mask],
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="markers", text=text,
                    marker=dict(color=color, colorscale="Viridis",
                                showscale=color is not None),
                )
            )
        fig.update_layout(title=title, xaxis_title=axis_labels[0], yaxis_title=axis_labels[1])
        return fig

    fig = _mpl_figure()
    ax = fig.add_subplot(111)
    if labels is not None:
        for lab in dict.fromkeys(labels):
            mask = [i for i, la in enumerate(labels) if la == lab]
            ax.scatter(x[mask], y[mask], label=str(lab), s=30)
        ax.legend()
    elif color is not None:
        sc = ax.scatter(x, y, c=color, cmap="viridis", s=30)
        fig.colorbar(sc, ax=ax)
    else:
        ax.scatter(x, y, s=30)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    return fig


def embedding_map(
    X: npt.NDArray[np.float64],
    *,
    method: str = "pca",
    color: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    text: Sequence[str] | None = None,
    standardize: bool = True,
    backend: str = "plotly",
    title: str | None = None,
    seed: int = 0,
) -> Any:
    """Reduce high-dimensional features to 2-D and scatter them.

    Covers UMAP / t-SNE / PCA embedding and similarity maps. Color by a score
    (e.g. CPP likelihood) or group by ``labels`` (e.g. dataset).
    """
    coords = reduce_dimensions(X, method=method, standardize=standardize, seed=seed)
    return scatter_2d(
        coords, color=color, labels=labels, text=text, backend=backend,
        title=title or f"{method.upper()} projection",
        axis_labels=(f"{method}-1", f"{method}-2"),
    )


# --------------------------------------------------------------------------- #
# Property distributions
# --------------------------------------------------------------------------- #
def property_distribution(
    groups: Mapping[str, Sequence[float]],
    *,
    kind: str = "hist",
    backend: str = "plotly",
    title: str = "Property distribution",
    xlabel: str = "value",
    bins: int = 20,
) -> Any:
    """Overlaid histograms (``kind='hist'``) or box plots (``kind='box'``) of a
    property across one or more named groups of peptides."""
    _check_backend(backend)
    if kind not in ("hist", "box"):
        raise ValidationError("kind must be 'hist' or 'box'.")
    if not groups:
        raise ValidationError("Need at least one group to plot.")

    if backend == "plotly":
        go = _go()
        fig = go.Figure()
        for name, values in groups.items():
            arr = np.asarray(list(values), dtype=float)
            if kind == "hist":
                fig.add_trace(go.Histogram(x=arr, name=name, opacity=0.6, nbinsx=bins))
            else:
                fig.add_trace(go.Box(y=arr, name=name))
        fig.update_layout(title=title, xaxis_title=xlabel, barmode="overlay")
        return fig

    fig = _mpl_figure()
    ax = fig.add_subplot(111)
    if kind == "hist":
        for name, values in groups.items():
            ax.hist(np.asarray(list(values), dtype=float), bins=bins, alpha=0.5, label=name)
        ax.legend()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
    else:
        data = [np.asarray(list(v), dtype=float) for v in groups.values()]
        ax.boxplot(data, labels=list(groups))
        ax.set_ylabel(xlabel)
    ax.set_title(title)
    return fig


# --------------------------------------------------------------------------- #
# Radar / spider plot
# --------------------------------------------------------------------------- #
def radar_plot(
    axes: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    backend: str = "plotly",
    title: str = "Property profile",
) -> Any:
    """Radar chart comparing several candidates across shared property axes.

    ``series`` maps a candidate name to its value on each of ``axes`` (same
    order). Values should be pre-normalized to a common scale for a fair shape.
    """
    _check_backend(backend)
    n = len(axes)
    if n < 3:
        raise ValidationError("A radar plot needs at least 3 axes.")
    for name, vals in series.items():
        if len(vals) != n:
            raise ValidationError(f"Series {name!r} has {len(vals)} values, expected {n}.")

    if backend == "plotly":
        go = _go()
        fig = go.Figure()
        for name, vals in series.items():
            fig.add_trace(
                go.Scatterpolar(
                    r=list(vals) + [vals[0]],
                    theta=list(axes) + [axes[0]],
                    fill="toself", name=name,
                )
            )
        fig.update_layout(title=title, polar=dict(radialaxis=dict(visible=True)))
        return fig

    fig = _mpl_figure()
    ax = fig.add_subplot(111, projection="polar")
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])
    for name, vals in series.items():
        v = np.concatenate([np.asarray(vals, dtype=float), np.asarray(vals[:1], dtype=float)])
        ax.plot(closed, v, label=name)
        ax.fill(closed, v, alpha=0.1)
    ax.set_xticks(angles)
    ax.set_xticklabels(list(axes))
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig


# --------------------------------------------------------------------------- #
# Mutation heatmap
# --------------------------------------------------------------------------- #
def single_mutation_scan_matrix(
    reference: str,
    score_fn: "Any",
    *,
    residues: Sequence[str] = CANONICAL_AMINO_ACIDS_ORDERED,
) -> npt.NDArray[np.float64]:
    """Build a (positions x residues) matrix of a score for every single mutant.

    ``score_fn(sequence) -> float`` is called on each single-substitution variant
    (the wild-type residue cell holds the reference score). Handy for deep
    mutational-scan heatmaps of, e.g., predicted CPP likelihood.
    """
    ref = reference.strip().upper()
    ref_score = float(score_fn(ref))
    matrix = np.empty((len(ref), len(residues)), dtype=float)
    for i, wt in enumerate(ref):
        for j, aa in enumerate(residues):
            if aa == wt:
                matrix[i, j] = ref_score
            else:
                variant = ref[:i] + aa + ref[i + 1 :]
                matrix[i, j] = float(score_fn(variant))
    return matrix


def mutation_heatmap(
    matrix: npt.NDArray[np.float64],
    *,
    position_labels: Sequence[str] | None = None,
    residue_labels: Sequence[str] = CANONICAL_AMINO_ACIDS_ORDERED,
    backend: str = "plotly",
    title: str = "Mutation heatmap",
    colorbar_label: str = "score",
) -> Any:
    """Heatmap of a (positions x residues) matrix."""
    _check_backend(backend)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValidationError("matrix must be 2-D (positions x residues).")
    positions = (
        list(position_labels)
        if position_labels is not None
        else [str(i + 1) for i in range(matrix.shape[0])]
    )

    if backend == "plotly":
        go = _go()
        fig = go.Figure(
            data=go.Heatmap(
                z=matrix, x=list(residue_labels), y=positions,
                colorscale="Viridis", colorbar=dict(title=colorbar_label),
            )
        )
        fig.update_layout(title=title, xaxis_title="residue", yaxis_title="position")
        return fig

    fig = _mpl_figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(residue_labels)))
    ax.set_xticklabels(list(residue_labels))
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels(positions)
    ax.set_xlabel("residue")
    ax.set_ylabel("position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    return fig


# --------------------------------------------------------------------------- #
# Calibration curve
# --------------------------------------------------------------------------- #
def calibration_plot(curve: Any, *, backend: str = "plotly", title: str = "Calibration") -> Any:
    """Reliability curve from a Phase-5 ``CalibrationCurve`` (predicted vs.
    observed frequency, with the perfect-calibration diagonal)."""
    _check_backend(backend)
    pred = np.asarray(curve.prob_predicted, dtype=float)
    obs = np.asarray(curve.prob_observed, dtype=float)
    label = f"model (Brier={curve.brier_score:.3f})"

    if backend == "plotly":
        go = _go()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                 line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=pred, y=obs, mode="lines+markers", name=label))
        fig.update_layout(title=title, xaxis_title="predicted probability",
                          yaxis_title="observed frequency")
        return fig

    fig = _mpl_figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], "--", label="perfect")
    ax.plot(pred, obs, "o-", label=label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend()
    return fig


# --------------------------------------------------------------------------- #
# Pareto scatter
# --------------------------------------------------------------------------- #
def pareto_scatter(
    solutions: Sequence[Any],
    x_objective: str,
    y_objective: str,
    *,
    color_objective: str | None = None,
    text: Sequence[str] | None = None,
    backend: str = "plotly",
    title: str = "Pareto front",
) -> Any:
    """Scatter Pareto solutions (``ParetoSolution``) in two objective dimensions."""
    _check_backend(backend)
    if not solutions:
        raise ValidationError("No solutions to plot.")
    x = np.array([s.objective_values[x_objective] for s in solutions], dtype=float)
    y = np.array([s.objective_values[y_objective] for s in solutions], dtype=float)
    color = (
        np.array([s.objective_values[color_objective] for s in solutions], dtype=float)
        if color_objective
        else None
    )
    hover = text if text is not None else [s.peptide.sequence for s in solutions]
    return scatter_2d(
        np.column_stack([x, y]), color=color, text=hover, backend=backend,
        title=title, axis_labels=(x_objective, y_objective),
    )
