"""Tests for the ScreeningIndex (descriptor-only, no torch needed)."""

from __future__ import annotations

from cpp_ai.screening import ScreenCandidate, ScreeningIndex

_BLOCKS = ["charge", "composition", "zscales"]


def _candidates() -> list[ScreenCandidate]:
    return [
        ScreenCandidate("TNVYNWFQNRRARTKRK", "ClWOX"),
        ScreenCandidate("KNVFYWFQNHKARERQK", "AtWUS"),
        ScreenCandidate("DDDDEEEEDDDDEEEE", "acidic"),
        ScreenCandidate("RRRRRRRRR", "R9"),
    ]


def test_build_and_rank_orders_by_similarity() -> None:
    idx = ScreeningIndex.build(_candidates(), descriptor_blocks=_BLOCKS)
    ranked = idx.rank("TNVYNWFQNRRARTKRK")
    sims = [c.similarity for c in ranked]
    assert sims == sorted(sims, reverse=True)
    # the anchor itself (identical candidate) should top the list
    assert ranked[0].name == "ClWOX"
    assert ranked[0].similarity is not None and ranked[0].similarity > 0.99


def test_dissimilar_ranks_low() -> None:
    idx = ScreeningIndex.build(_candidates(), descriptor_blocks=_BLOCKS)
    ranked = idx.rank("TNVYNWFQNRRARTKRK")
    # the acidic peptide should not be the closest to a cationic homeodomain peptide
    assert ranked[0].name != "acidic"


def test_top_k() -> None:
    idx = ScreeningIndex.build(_candidates(), descriptor_blocks=_BLOCKS)
    assert len(idx.rank("KNVFYWFQNHKARERQK", top_k=2)) == 2


def test_no_embeddings_by_default() -> None:
    idx = ScreeningIndex.build(_candidates(), descriptor_blocks=_BLOCKS)
    assert not idx.has_embeddings
