"""Tests for block-wise physicochemical similarity."""

from __future__ import annotations

import pytest

from cpp_ai.core.exceptions import ValidationError
from cpp_ai.scoring import (
    BIOLOGICAL_BLOCKS,
    BlockSimilarityIndex,
    all_block_features,
)

_LIB = [
    "TNVYNWFQNRRARTKRK",   # ClWOX
    "TNVYNWFQNRRARSKRK",   # PtWOX (1 residue off ClWOX)
    "KNVFYWFQNHKARERQK",   # AtWUS
    "RRRRRRRRRRRR",        # extreme cationic
    "DDDDEEEEDDDDEEEE",    # acidic, very different
    "LLIILARRIRKQAHAHSK",  # pVEC-R6A
]


def test_blocks_metadata() -> None:
    assert len(BIOLOGICAL_BLOCKS) >= 5
    # amphipathicity is up-weighted vs charge (per review)
    w = {b.name: b.default_weight for b in BIOLOGICAL_BLOCKS}
    assert w["amphipathicity"] > w["charge"]


def test_all_features_deduplicated() -> None:
    feats = all_block_features()
    assert len(feats) == len(set(feats))


def _index() -> BlockSimilarityIndex:
    return BlockSimilarityIndex(_LIB)


def test_anchor_matches_itself_perfectly() -> None:
    profiles = _index().rank("TNVYNWFQNRRARTKRK")
    top = profiles[0]
    assert top.sequence == "TNVYNWFQNRRARTKRK"
    assert top.composite == pytest.approx(1.0, abs=1e-9)
    assert top.percentile == pytest.approx(1.0)
    assert all(0.0 <= b.similarity <= 1.0 for b in top.blocks)


def test_ptwox_ranks_above_acidic() -> None:
    profiles = {p.sequence: p for p in _index().rank("TNVYNWFQNRRARTKRK")}
    assert profiles["TNVYNWFQNRRARSKRK"].composite > profiles["DDDDEEEEDDDDEEEE"].composite


def test_extreme_charge_lowers_charge_block() -> None:
    profiles = {p.sequence: p for p in _index().rank("TNVYNWFQNRRARTKRK")}
    # ClWOX (+6): PtWOX (+6, same charge) should match the charge block far
    # better than poly-R (+12) — distance-aware, magnitude matters.
    assert profiles["RRRRRRRRRRRR"].block("charge") < profiles["TNVYNWFQNRRARSKRK"].block("charge")


def test_weights_change_composite() -> None:
    idx = _index()
    default = {p.sequence: p.composite for p in idx.rank("TNVYNWFQNRRARTKRK")}
    charge_only = {
        p.sequence: p.composite
        for p in idx.rank("TNVYNWFQNRRARTKRK", weights={"charge": 1.0})
    }
    assert default != charge_only


def test_per_block_scores_present_and_bounded() -> None:
    p = _index().rank("LLIILARRIRKQAHAHSK")[0]
    names = {b.name for b in p.blocks}
    assert names == {b.name for b in BIOLOGICAL_BLOCKS}
    assert all(0.0 <= b.similarity <= 1.0 for b in p.blocks)


def test_zero_weights_rejected() -> None:
    with pytest.raises(ValidationError):
        _index().rank("TNVYNWFQNRRARTKRK", weights={"charge": 0.0})


def test_empty_library_rejected() -> None:
    with pytest.raises(ValidationError):
        BlockSimilarityIndex([])
