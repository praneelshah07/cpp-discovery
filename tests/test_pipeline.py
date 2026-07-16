"""Tests for the shared recommendation backbone (cpp_ai.pipeline)."""

from __future__ import annotations

import pytest

from cpp_ai.evidence.seed import build_seed_ledger
from cpp_ai.pipeline import (
    ANCHOR_PRESETS,
    filter_and_rank,
    recommend_for_algae,
    resolve_anchor,
)
from cpp_ai.scoring import AlgaeFitScorer, EvidenceScorer
from cpp_ai.screening.candidate import ScreenCandidate

# A small in-memory library spanning amphipathic winners and cationic losers,
# plus a couple of near-pVEC variants to exercise the beyond-variants filter.
_LIB = [
    ScreenCandidate(sequence=s, name=n, category="?", mechanism="?", source="cppsite3")
    for n, s in [
        ("pVEC-var1", "LLIILRRRIRKQAHAHSA"),  # ~pVEC (1 change)
        ("pVEC-var2", "LLIILARRIRKQAHAHSK"),  # pVEC-R6A
        ("TP10-like", "AGYLLGKINLKALAALAKKIL"),  # amphipathic, unlike pVEC
        ("R9", "RRRRRRRRR"),  # pure cationic
        ("TAT", "YGRKKRRQRRR"),
        ("MAP", "KLALKLALKALKAALKLA"),  # amphipathic
    ]
]
_PVEC = "LLIILRRRIRKQAHAHSK"


def _rec(**over: object):
    kw: dict[str, object] = dict(
        anchor=_PVEC, library=_LIB, ledger=build_seed_ledger(), classifier=None, top_k=None
    )
    kw.update(over)
    return recommend_for_algae(**kw)  # type: ignore[arg-type]


def test_resolve_anchor_presets_and_sequences() -> None:
    assert resolve_anchor("pVEC") == ANCHOR_PRESETS["pVEC"]
    assert resolve_anchor("lliilrrrirkqahahsk") == _PVEC  # raw seq, normalized
    with pytest.raises(ValueError):
        resolve_anchor("not a seq!")


def test_anchor_excluded_from_results() -> None:
    rec = _rec()
    assert all(p.sequence != _PVEC for p in rec.profiles)


def test_algae_mode_populates_fit_axis() -> None:
    assert all(p.algae_fit is not None for p in _rec(algae_mode=True).profiles)
    assert all(p.algae_fit is None for p in _rec(algae_mode=False).profiles)


def test_rank_by_algae_fit_orders_by_algae_profile() -> None:
    # keep toxic candidates so R9 (charge +9) is present to test its low rank
    rec = _rec(algae_mode=True, rank_by="algae_fit", low_toxicity=False)
    fits = [p.algae_fit for p in rec.profiles]
    assert fits == sorted(fits, reverse=True)
    # pure-cationic R9 should not be near the top under the algae profile
    names = [p.name for p in rec.profiles]
    assert names.index("R9") > names.index("TP10-like")


def test_rank_by_algae_fit_falls_back_without_evidence() -> None:
    from cpp_ai.evidence import EvidenceLedger

    rec = _rec(ledger=EvidenceLedger(), algae_mode=True, rank_by="algae_fit")
    assert rec.rank_by == "blend"  # honest fallback: no informative signal


def test_beyond_variants_filter_drops_near_anchor() -> None:
    with_variants = {p.name for p in _rec(max_identity=None).profiles}
    beyond = {p.name for p in _rec(max_identity=0.6).profiles}
    assert "pVEC-var1" in with_variants  # ~identical to anchor, kept by default
    assert "pVEC-var1" not in beyond  # dropped as a near-variant
    assert "TP10-like" in beyond  # a genuinely different scaffold survives


def test_top_k_truncates() -> None:
    assert len(_rec(top_k=2).profiles) == 2


def test_dataframe_and_markdown_render() -> None:
    rec = _rec(top_k=5, algae_mode=True, rank_by="algae_fit", low_toxicity=False)
    df = rec.to_dataframe()
    assert len(df) == 5 and "algae_fit" in df.columns
    md = rec.to_markdown()
    assert "Algae-delivery CPP candidates" in md
    assert "not** algae-uptake predictions" in md  # the honesty caveat survives


def test_filter_and_rank_matches_pipeline_policy() -> None:
    # The shared helper (used by the app) reproduces the pipeline's ordering.
    fit = AlgaeFitScorer.from_ledger(build_seed_ledger(), [c.sequence for c in _LIB])
    profiles = EvidenceScorer(_LIB, algae_fit_scorer=fit).profile(_PVEC)
    kept = filter_and_rank(profiles, _PVEC, rank_by="algae_fit")
    via_pipeline = _rec(algae_mode=True, rank_by="algae_fit").profiles
    assert [p.sequence for p in kept] == [p.sequence for p in via_pipeline]
