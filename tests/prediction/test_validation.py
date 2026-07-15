"""Tests for family-split cross-validation."""

from __future__ import annotations

import numpy as np
import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.prediction import cluster_sequences, family_split_cross_validate
from cpp_ai.prediction.base import MODEL_REGISTRY


def test_cluster_groups_similar_sequences() -> None:
    seqs = ["KWKLFKKIRR", "KWKLFKKIRK", "KWKLFKKIRA",  # one family
            "DDDDEEEEDD", "DDDDEEEEDE"]                 # another
    clusters = cluster_sequences(seqs, threshold=0.4)
    assert clusters[0] == clusters[1] == clusters[2]   # same family
    assert clusters[3] == clusters[4]
    assert clusters[0] != clusters[3]                  # different families


def test_singletons_get_distinct_clusters() -> None:
    seqs = ["KWKLFKKI", "DDDDEEEE", "RRRRRRRR"]
    clusters = cluster_sequences(seqs, threshold=0.6)
    assert len(set(clusters)) == 3


def test_family_split_cv_runs_and_reports() -> None:
    rng = np.random.default_rng(0)
    # 6 families x 6 near-duplicates; label correlates with a feature
    seqs, X, y = [], [], []
    bases = ["KWKLFKKIRR", "LLIILRRKAA", "DDDDEEEEDD", "GGGGSSSSPP", "RRWWKKFFYY", "AAVVLLIIMM"]
    for f, base in enumerate(bases):
        for j in range(6):
            seqs.append(base[:-1] + "ACDEFG"[j])
            X.append([f + rng.normal(0, 0.1), rng.normal()])
            y.append(f % 2)
    report, n_clusters = family_split_cross_validate(
        MODEL_REGISTRY.get("logistic_regression"),
        np.array(X), np.array(y), seqs, n_splits=3, threshold=0.5,
    )
    assert n_clusters >= 3
    assert report.folds
    assert 0.0 <= report.summary()["roc_auc"][0] <= 1.0


def test_too_few_clusters_raises() -> None:
    seqs = ["KWKLFKKIRR"] * 10  # all identical -> 1 cluster
    X = np.random.default_rng(0).normal(size=(10, 2))
    y = np.array([0, 1] * 5)
    with pytest.raises(ValidationError):
        family_split_cross_validate(
            MODEL_REGISTRY.get("logistic_regression"), X, y, seqs, n_splits=3
        )
