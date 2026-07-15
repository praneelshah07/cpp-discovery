"""Feature container passed to similarity metrics.

Different metrics need different inputs: alignment metrics need only the
sequence, embedding cosine needs the pLM vector, descriptor similarity needs a
(standardized) descriptor vector. Bundling them in one immutable object lets the
engine precompute each feature once and hand the same object to every metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, eq=False)
class PeptideFeatures:
    """Precomputed features for one peptide.

    ``embedding`` and ``descriptors`` are optional: a metric that needs one will
    raise a clear error if it is absent, rather than silently returning a
    meaningless score.
    """

    sequence: str
    peptide_id: str | None = None
    embedding: npt.NDArray[np.float32] | None = None
    descriptors: npt.NDArray[np.float64] | None = None
