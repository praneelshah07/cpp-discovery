"""cpp_ai: a research-grade platform for computational CPP discovery.

The top-level package intentionally exposes only the lightweight :mod:`core`
primitives. Heavier subpackages (embeddings, prediction, ...) are imported
explicitly by callers and gated behind optional install extras.

Every output of this platform is a *computationally prioritized hypothesis for
wet-lab validation*, never a claim of biological function.
"""

from __future__ import annotations

from . import core

__version__ = "0.0.1"

__all__ = ["core", "__version__"]
