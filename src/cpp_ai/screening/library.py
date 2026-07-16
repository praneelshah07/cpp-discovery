"""Load the CPP screening library (CPPsite3).

Turns the CPPsite3 API JSON into :class:`ScreenCandidate`s. Filtering/ranking is
handled downstream by :mod:`cpp_ai.pipeline`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.types import is_canonical_sequence
from .candidate import ScreenCandidate


def load_cppsite3_library(
    json_path: str | Path,
    *,
    min_len: int = 10,
    max_len: int = 40,
    l_chirality_only: bool = True,
) -> list[ScreenCandidate]:
    """Load unique canonical CPPs from the CPPsite3 API JSON."""
    data = json.loads(Path(json_path).read_text())["data"]
    seen: dict[str, ScreenCandidate] = {}
    for r in data:
        seq = str(r.get("pepseq", "")).strip().upper()
        if not is_canonical_sequence(seq) or not (min_len <= len(seq) <= max_len):
            continue
        if l_chirality_only and r.get("chiral") != "L":
            continue
        if seq in seen:
            continue
        seen[seq] = ScreenCandidate(
            sequence=seq,
            name=str(r.get("pepnam", "?")),
            category=str(r.get("cat", "?")),
            mechanism=str(r.get("proupmech", "?")),
            source=str(r.get("source", "?")),
            n_term_mod=str(r.get("ntermod", "")),
            c_term_mod=str(r.get("ctermod", "")),
            chem_mod=str(r.get("chmod", "")),
        )
    return list(seen.values())


