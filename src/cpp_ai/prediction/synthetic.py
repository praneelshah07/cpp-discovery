"""Synthetic labeled CPP dataset generator.

**This data is fake.** It exists so the ML framework can be built, tested, and
demonstrated end-to-end *before* real CPPsite3/POSEIDON data is loaded, and as a
reproducible fixture. It must never be used to make biological claims.

The signal is deliberately CPP-*plausible* but simplified: positives lean
cationic/amphipathic (Arg/Lys/Trp/Leu, like many real CPPs), negatives lean
acidic/polar. Crucially the two profiles **share a large common baseline** and
labels are flipped with probability ``label_noise``, so the classes overlap and
the task is learnable but *non-trivial* — models should reach a realistic
ROC-AUC (~0.8-0.9), predictions show a genuine probability spread, and
calibration/uncertainty are actually exercised. Swap this for real labeled data
at the same interface: ``(list[Peptide], np.ndarray[int])``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..core.schema import Peptide

# A shared, near-uniform baseline over all 20 residues keeps the classes
# overlapping; small class-specific bumps provide the (weak) learnable signal.
_BASELINE: dict[str, float] = {aa: 1.0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
_POSITIVE_BUMP: dict[str, float] = {"R": 1.6, "K": 1.6, "L": 0.8, "W": 0.7, "F": 0.5}
_NEGATIVE_BUMP: dict[str, float] = {"D": 1.6, "E": 1.6, "S": 0.8, "T": 0.7, "N": 0.5}


def _profile(bump: dict[str, float]) -> dict[str, float]:
    profile = dict(_BASELINE)
    for residue, extra in bump.items():
        profile[residue] += extra
    return profile


_POSITIVE_PROFILE = _profile(_POSITIVE_BUMP)
_NEGATIVE_PROFILE = _profile(_NEGATIVE_BUMP)


def _sample_sequence(
    profile: dict[str, float], rng: np.random.Generator, length: int
) -> str:
    residues = list(profile)
    weights = np.array([profile[r] for r in residues], dtype=float)
    weights /= weights.sum()
    return "".join(rng.choice(residues, size=length, p=weights))


def make_synthetic_cpp_dataset(
    n_per_class: int = 150,
    *,
    min_length: int = 8,
    max_length: int = 24,
    label_noise: float = 0.15,
    seed: int = 0,
) -> tuple[list[Peptide], npt.NDArray[np.int_]]:
    """Return ``(peptides, labels)`` — a balanced synthetic CPP/non-CPP set.

    The *generating* profile (which residue distribution a sequence is drawn
    from) is the true class; each observed ``label`` is then flipped with
    probability ``label_noise`` to create irreducible uncertainty. Label 1 =
    CPP-like, 0 = non-CPP-like.
    """
    if not 0.0 <= label_noise < 0.5:
        raise ValueError("label_noise must be in [0, 0.5).")
    rng = np.random.default_rng(seed)
    peptides: list[Peptide] = []
    labels: list[int] = []

    for true_label, profile in ((1, _POSITIVE_PROFILE), (0, _NEGATIVE_PROFILE)):
        for _ in range(n_per_class):
            length = int(rng.integers(min_length, max_length + 1))
            seq = _sample_sequence(profile, rng, length)
            observed = 1 - true_label if rng.random() < label_noise else true_label
            peptides.append(
                Peptide.from_sequence(
                    seq, dataset="synthetic", metadata={"synthetic_true_label": true_label}
                )
            )
            labels.append(observed)

    order = rng.permutation(len(peptides))
    peptides = [peptides[i] for i in order]
    labels_arr = np.array(labels, dtype=int)[order]
    return peptides, labels_arr
