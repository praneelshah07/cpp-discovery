"""Screening: loading the CPP library into tested candidate objects.

Filtering, ranking, diversifying and exporting are handled downstream by
:mod:`cpp_ai.pipeline`; this package just turns the raw database into
:class:`ScreenCandidate`s.
"""

from __future__ import annotations

from .candidate import ScreenCandidate, charge_toxicity_flag
from .library import load_cppsite3_library

__all__ = [
    "ScreenCandidate",
    "charge_toxicity_flag",
    "load_cppsite3_library",
]
