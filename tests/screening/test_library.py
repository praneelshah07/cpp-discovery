"""Tests for library loading."""

from __future__ import annotations

import json
from pathlib import Path

from cpp_ai.screening import load_cppsite3_library


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
