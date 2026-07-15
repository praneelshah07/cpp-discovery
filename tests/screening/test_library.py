"""Tests for library loading and filtering."""

from __future__ import annotations

import json
from pathlib import Path

from cpp_ai.screening import ScreenCandidate, apply_filters, load_cppsite3_library


def _write_lib(path: Path) -> Path:
    data = {"data": [
        {"pepseq": "TNVYNWFQNRRARTKRK", "pepnam": "ClWOX", "chiral": "L",
         "cat": "Cationic", "proupmech": "Endocytosis", "source": "Protein derived"},
        {"pepseq": "RRRRRRRRR", "pepnam": "R9", "chiral": "L",
         "cat": "Cationic", "proupmech": "Direct translocation", "source": "Synthetic"},
        {"pepseq": "KLAKLAK", "pepnam": "amp", "chiral": "L",
         "cat": "Antimicrobial", "proupmech": "pore formation", "source": "Synthetic"},
        {"pepseq": "ACDXEFG", "pepnam": "bad", "chiral": "L", "cat": "?", "proupmech": "?", "source": "?"},
        {"pepseq": "GRKKRRQRRRPPQ", "pepnam": "TAT", "chiral": "D", "cat": "?", "proupmech": "?", "source": "?"},
    ]}
    path.write_text(json.dumps(data))
    return path


def test_load_filters_noncanonical_and_chirality(tmp_path: Path) -> None:
    lib = load_cppsite3_library(_write_lib(tmp_path / "l.json"), min_len=5, max_len=40)
    seqs = {c.sequence for c in lib}
    assert "TNVYNWFQNRRARTKRK" in seqs
    assert "ACDXEFG" not in seqs        # non-canonical (X) dropped
    assert "GRKKRRQRRRPPQ" not in seqs  # D-chirality dropped


def _cands() -> list[ScreenCandidate]:
    return [
        ScreenCandidate("TNVYNWFQNRRARTKRK", "ClWOX", "Cationic", "Endocytosis"),
        ScreenCandidate("RRRRRRRRR", "R9", "Cationic", "Direct translocation"),
        ScreenCandidate("KLAKLAKLA", "amp", "Antimicrobial", "pore formation"),
    ]


def test_filter_charge() -> None:
    out = apply_filters(_cands(), max_charge=7)
    assert all(c.net_charge <= 7 for c in out)
    assert "RRRRRRRRR" not in {c.sequence for c in out}


def test_filter_endocytic_only() -> None:
    out = apply_filters(_cands(), endocytic_only=True)
    assert {c.name for c in out} == {"ClWOX"}


def test_filter_exclude_homeodomain() -> None:
    out = apply_filters(_cands(), exclude_homeodomain=True)
    assert "TNVYNWFQNRRARTKRK" not in {c.sequence for c in out}  # WFQN motif


def test_filter_exclude_lytic() -> None:
    out = apply_filters(_cands(), exclude_lytic=True)
    names = {c.name for c in out}
    assert "amp" not in names and "R9" not in names  # both lytic-ish


def test_filter_exclude_sequences_and_name() -> None:
    out = apply_filters(_cands(), exclude_sequences=["TNVYNWFQNRRARTKRK"])
    assert "ClWOX" not in {c.name for c in out}
    out2 = apply_filters(_cands(), exclude_name_regex="R9|amp")
    assert {c.name for c in out2} == {"ClWOX"}
