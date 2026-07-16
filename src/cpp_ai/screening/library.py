"""Load and filter the CPP screening library (CPPsite3).

Turns the CPPsite3 API JSON into :class:`ScreenCandidate`s and provides the
filters the UI exposes: length, charge (toxicity), endocytic-only, aromatic,
exclude the homeodomain family, and name exclusions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from ..core.types import is_canonical_sequence
from .candidate import ScreenCandidate

# The homeodomain 3rd-helix signature (ClWOX's own family — often excluded to
# surface non-obvious, non-textbook candidates).
HOMEODOMAIN_MOTIF = re.compile(r"WFQ.R|WFQN")


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


def apply_filters(
    candidates: Iterable[ScreenCandidate],
    *,
    min_len: int = 1,
    max_len: int = 1000,
    min_charge: int | None = None,
    max_charge: int | None = None,
    endocytic_only: bool = False,
    aromatic_only: bool = False,
    exclude_homeodomain: bool = False,
    exclude_lytic: bool = False,
    exclude_name_regex: str | None = None,
    exclude_sequences: Iterable[str] = (),
) -> list[ScreenCandidate]:
    """Return candidates passing every enabled filter."""
    name_re = re.compile(exclude_name_regex, re.I) if exclude_name_regex else None
    excluded = {s.strip().upper() for s in exclude_sequences}
    out: list[ScreenCandidate] = []
    for c in candidates:
        if not (min_len <= c.length <= max_len):
            continue
        if min_charge is not None and c.net_charge < min_charge:
            continue
        if max_charge is not None and c.net_charge > max_charge:
            continue
        if endocytic_only and not c.is_endocytic:
            continue
        if aromatic_only and not any(a in c.sequence for a in "WYF"):
            continue
        if exclude_homeodomain and HOMEODOMAIN_MOTIF.search(c.sequence):
            continue
        if exclude_lytic and c.lytic_risk:
            continue
        if name_re and name_re.search(c.name):
            continue
        if c.sequence in excluded:
            continue
        out.append(c)
    return out
