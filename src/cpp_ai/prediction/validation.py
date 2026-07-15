"""Leakage-aware, family-split cross-validation.

CPP databases are full of near-duplicates (truncations, alanine scans, scrambled
and repeated variants, close homologs). With random CV, a family member seen in
training can leak into the test fold, inflating the reported AUC. The honest
question is "can the model recognize a *new* CPP family?", which requires
grouping related sequences and keeping whole groups together across folds.

This module clusters sequences (fast k-mer Jaccard, single-linkage) and runs
GroupKFold so no cluster spans train and test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold

from ..core.exceptions import ValidationError
from .base import MODEL_REGISTRY, EstimatorFactory
from .evaluation import CVReport, _METRICS


def _kmers(seq: str, k: int) -> frozenset[str]:
    s = seq.strip().upper()
    if len(s) < k:
        return frozenset({s})
    return frozenset(s[i : i + k] for i in range(len(s) - k + 1))


def cluster_sequences(
    sequences: Sequence[str], *, threshold: float = 0.4, k: int = 3
) -> list[int]:
    """Greedy single-linkage clustering by k-mer Jaccard; returns a cluster id per sequence.

    Two sequences join the same cluster if their k-mer Jaccard exceeds
    ``threshold`` against the cluster's first member (fast, order-dependent, but
    adequate to keep obvious families together).
    """
    reps: list[frozenset[str]] = []
    assignments: list[int] = []
    for seq in sequences:
        km = _kmers(seq, k)
        assigned = -1
        for cid, rep in enumerate(reps):
            union = len(km | rep)
            if union and len(km & rep) / union >= threshold:
                assigned = cid
                break
        if assigned == -1:
            assigned = len(reps)
            reps.append(km)
        assignments.append(assigned)
    return assignments


def family_split_cross_validate(
    factory: EstimatorFactory,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int_],
    sequences: Sequence[str],
    *,
    model_name: str = "model",
    n_splits: int = 5,
    threshold: float = 0.4,
) -> tuple[CVReport, int]:
    """Cross-validate with clustered (family) splits. Returns (report, n_clusters)."""
    if isinstance(factory, str):
        factory = MODEL_REGISTRY.get(factory)
    y = np.asarray(y).astype(int)
    groups = np.asarray(cluster_sequences(sequences, threshold=threshold))
    n_clusters = int(groups.max()) + 1 if len(groups) else 0
    if n_clusters < n_splits:
        raise ValidationError(
            f"Only {n_clusters} sequence clusters — need >= n_splits ({n_splits}) "
            "for a family split. Lower the threshold or n_splits."
        )

    gkf = GroupKFold(n_splits=n_splits)
    report = CVReport(model_name=f"{model_name} (family-split)")
    for train_idx, test_idx in gkf.split(X, y, groups):
        est = factory()
        est.fit(X[train_idx], y[train_idx])
        prob = est.predict_proba(X[test_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        yt = y[test_idx]
        if len(set(yt.tolist())) < 2:
            continue  # a fold with one class can't score AUC
        report.folds.append({name: fn(yt, prob, pred) for name, fn in _METRICS.items()})
    return report, n_clusters
