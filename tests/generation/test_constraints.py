"""Tests for mutation constraints and net charge."""

from __future__ import annotations

from cpp_ai.generation import Mutation, MutationConstraints, net_charge


def test_net_charge() -> None:
    assert net_charge("KRKR") == 4
    assert net_charge("DDEE") == -4
    assert net_charge("KKDDE") == -1
    assert net_charge("AAAA") == 0
    assert net_charge("HHHH") == 0  # His excluded


def test_mutation_notation() -> None:
    assert Mutation(4, "K", "A").notation() == "K5A"  # 1-based


def test_locked_indices_from_positions_and_motif() -> None:
    c = MutationConstraints(locked_positions=frozenset({0}), locked_motifs=("RRR",))
    seq = "LLIRRRAA"
    locked = c.locked_indices(seq)
    assert 0 in locked
    assert {3, 4, 5} <= locked  # RRR occupies indices 3-5


def test_locked_motif_all_occurrences() -> None:
    c = MutationConstraints(locked_motifs=("AA",))
    seq = "AAKKAA"
    assert c.locked_indices(seq) == frozenset({0, 1, 4, 5})


def test_mutable_positions_excludes_locked() -> None:
    c = MutationConstraints(locked_positions=frozenset({0, 1}))
    assert c.mutable_positions("KKKK") == (2, 3)


def test_accepts_canonical_only() -> None:
    c = MutationConstraints(canonical_only=True)
    assert c.accepts("KWKL")
    assert not c.accepts("KWXL")


def test_accepts_charge_bounds() -> None:
    c = MutationConstraints(max_charge=2, min_charge=0)
    assert c.accepts("KRAA")  # charge 2
    assert not c.accepts("KRKA")  # charge 3
    assert not c.accepts("DDAA")  # charge -2


def test_accepts_length_bounds() -> None:
    c = MutationConstraints(min_length=4, max_length=6)
    assert c.accepts("KKKK")
    assert not c.accepts("KKK")
    assert not c.accepts("KKKKKKK")


def test_accepts_custom_filter() -> None:
    no_cys = MutationConstraints(custom_filters=(lambda s: "C" not in s,))
    assert no_cys.accepts("KWKL")
    assert not no_cys.accepts("KWCL")
