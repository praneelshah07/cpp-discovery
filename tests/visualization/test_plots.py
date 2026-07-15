"""Tests for the plotting functions (both backends, headless)."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.core.schema import Peptide
from cpp_ai.prediction.evaluation import CalibrationCurve
from cpp_ai.optimization import ParetoSolution
from cpp_ai.visualization import (
    calibration_plot,
    embedding_map,
    mutation_heatmap,
    pareto_scatter,
    property_distribution,
    radar_plot,
    scatter_2d,
    single_mutation_scan_matrix,
)

_BACKENDS = ["plotly", "matplotlib"]


def _module(fig) -> str:
    return type(fig).__module__.split(".")[0]


# --- scatter / embedding ---
@pytest.mark.parametrize("backend", _BACKENDS)
def test_embedding_map_returns_backend_figure(backend: str) -> None:
    X = np.random.default_rng(0).normal(size=(15, 8))
    fig = embedding_map(X, method="pca", color=np.arange(15), backend=backend)
    assert _module(fig) == backend


def test_scatter_invalid_backend() -> None:
    with pytest.raises(ValidationError):
        scatter_2d(np.zeros((3, 2)), backend="ascii")


def test_scatter_requires_2d_coords() -> None:
    with pytest.raises(ValidationError):
        scatter_2d(np.zeros((3, 1)))


def test_scatter_labels_make_multiple_traces() -> None:
    coords = np.random.default_rng(0).normal(size=(6, 2))
    fig = scatter_2d(coords, labels=["a", "a", "b", "b", "c", "c"], backend="plotly")
    assert len(fig.data) == 3  # one trace per label group


# --- property distribution ---
@pytest.mark.parametrize("backend", _BACKENDS)
def test_property_distribution(backend: str) -> None:
    groups = {"cpp": [1.0, 2, 3], "non": [2.0, 3, 4]}
    assert _module(property_distribution(groups, backend=backend)) == backend


def test_property_distribution_invalid_kind() -> None:
    with pytest.raises(ValidationError):
        property_distribution({"a": [1.0]}, kind="pie")


def test_property_distribution_empty() -> None:
    with pytest.raises(ValidationError):
        property_distribution({})


# --- radar ---
@pytest.mark.parametrize("backend", _BACKENDS)
def test_radar_plot(backend: str) -> None:
    fig = radar_plot(["a", "b", "c"], {"x": [1.0, 2, 3]}, backend=backend)
    assert _module(fig) == backend


def test_radar_needs_three_axes() -> None:
    with pytest.raises(ValidationError):
        radar_plot(["a", "b"], {"x": [1.0, 2]})


def test_radar_series_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        radar_plot(["a", "b", "c"], {"x": [1.0, 2]})


# --- mutation heatmap ---
def test_single_mutation_scan_matrix() -> None:
    ref = "KWK"
    mat = single_mutation_scan_matrix(ref, lambda s: float(s.count("K")))
    assert mat.shape == (3, 20)
    # wild-type cells hold the reference score (2 lysines)
    from cpp_ai.core.types import CANONICAL_AMINO_ACIDS_ORDERED as AAs
    assert mat[0, AAs.index("K")] == 2.0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_mutation_heatmap(backend: str) -> None:
    mat = np.random.default_rng(0).normal(size=(5, 20))
    assert _module(mutation_heatmap(mat, backend=backend)) == backend


def test_mutation_heatmap_requires_2d() -> None:
    with pytest.raises(ValidationError):
        mutation_heatmap(np.zeros(10))


# --- calibration ---
@pytest.mark.parametrize("backend", _BACKENDS)
def test_calibration_plot(backend: str) -> None:
    curve = CalibrationCurve(
        prob_predicted=np.array([0.1, 0.5, 0.9]),
        prob_observed=np.array([0.15, 0.5, 0.85]),
        brier_score=0.1,
    )
    assert _module(calibration_plot(curve, backend=backend)) == backend


# --- pareto scatter ---
def _sol(seq: str, a: float, b: float) -> ParetoSolution:
    return ParetoSolution(
        peptide=Peptide.from_sequence(seq, dataset="t"),
        objective_values={"objA": a, "objB": b},
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_pareto_scatter(backend: str) -> None:
    sols = [_sol("KWKL", 0.1, 0.9), _sol("LLII", 0.8, 0.2)]
    fig = pareto_scatter(sols, "objA", "objB", backend=backend)
    assert _module(fig) == backend


def test_pareto_scatter_empty() -> None:
    with pytest.raises(ValidationError):
        pareto_scatter([], "objA", "objB")
