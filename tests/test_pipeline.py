"""Tests for the shared recommendation backbone (cpp_ai.pipeline)."""

from __future__ import annotations

import pytest

from cpp_ai.evidence.seed import build_seed_ledger
from cpp_ai.pipeline import (
    ANCHOR_PRESETS,
    categorize,
    explain_profile,
    filter_and_rank,
    group_families,
    peptide_family,
    recommend_for_algae,
    resolve_anchor,
    usable_delivery,
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
    # a gentle, high-fit amphipath (pVEC-R6A) should outrank pure-cationic R9
    # under the lysis-discounted algae profile
    names = [p.name for p in rec.profiles]
    assert names.index("pVEC-var2") < names.index("R9")


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
    assert len(df) == 5 and "insertion_fit" in df.columns
    md = rec.to_markdown()
    assert "Algae-delivery CPP candidates" in md
    assert "not** algae-uptake predictions" in md  # the honesty caveat survives


def test_max_lysis_filter_drops_lytic_amphipaths() -> None:
    kept = {p.name for p in _rec(max_lysis_risk=0.6, low_toxicity=False).profiles}
    # MAP is a lytic amphipath and should be excluded by the lysis filter
    assert "MAP" not in kept
    # pVEC-R6A (gentle) survives
    assert "pVEC-var2" in kept


def test_lysis_discount_demotes_lytic_under_algae_fit() -> None:
    # MAP has high algae_fit but high lysis; the discount should rank it below a
    # gentler amphipath. Compare its rank to TP10-like (gentler here).
    rec = _rec(algae_mode=True, rank_by="algae_fit", low_toxicity=False)
    names = [p.name for p in rec.profiles]
    assert names.index("TP10-like") < names.index("MAP")


def test_collapse_families_reduces_near_duplicates() -> None:
    full = _rec(low_toxicity=False).profiles
    collapsed = _rec(low_toxicity=False, collapse_families=0.7).profiles
    # the two near-identical pVEC variants collapse to one representative
    assert len(collapsed) < len(full)
    pvec_like = [p for p in collapsed if p.name.startswith("pVEC-var")]
    assert len(pvec_like) <= 1


def test_lysis_risk_present_on_profiles() -> None:
    for p in _rec().profiles:
        assert 0.0 <= p.lysis_risk <= 1.0


def test_peptide_family_tags() -> None:
    assert peptide_family("RQIKIWFQNRRMKWKK") == "Homeodomain"  # WFQN motif
    assert peptide_family("AGYLLGKINLKALAALAKKIL") == "Transportan"
    assert peptide_family("LLIILRRRIRKQAHAHSK") == "pVEC-like"
    assert peptide_family("RRRRRRRRR") == "Polyarginine/cationic"


def test_explain_profile_has_reasons() -> None:
    rec = _rec(algae_mode=True, top_k=None)
    pvec_r6a = next(p for p in rec.profiles if p.name == "pVEC-var2")
    reasons = explain_profile(pvec_r6a, fit_scorer=rec.fit_scorer)
    assert reasons
    assert any(r.positive for r in reasons)  # a gentle amphipath has positives
    # a lytic peptide should carry a negative membrane-lysis reason
    rec2 = _rec(algae_mode=True, low_toxicity=False, top_k=None)
    map_p = next((p for p in rec2.profiles if p.name == "MAP"), None)
    if map_p is not None:
        texts = [r.text for r in explain_profile(map_p) if not r.positive]
        assert any("lyt" in t.lower() for t in texts)


def test_usable_delivery_squares_lysis_and_applies_fusion() -> None:
    rec = _rec(algae_mode=True, low_toxicity=False, top_k=None)
    p = rec.profiles[0]
    expected = (
        p.surface_adsorption * p.algae_fit  # type: ignore[operator]
        * (1 - p.lysis_risk) ** 2 * p.fusion_confidence
    )
    assert abs(usable_delivery(p) - expected) < 1e-9
    # a lytic peptide is penalized harder by the squared term than the linear one
    lytic = next((x for x in rec.profiles if x.lysis_risk > 0.5), None)
    if lytic is not None and lytic.algae_fit is not None:
        squared = usable_delivery(lytic)
        linear = lytic.surface_adsorption * lytic.algae_fit * (1 - lytic.lysis_risk) \
            * lytic.fusion_confidence
        assert squared < linear


def test_usable_delivery_zero_without_algae_fit() -> None:
    rec = _rec(algae_mode=False, top_k=1)
    assert usable_delivery(rec.profiles[0]) == 0.0


def test_group_families_representative_and_members() -> None:
    rec = _rec(algae_mode=True, low_toxicity=False, top_k=None)
    groups = group_families(rec.profiles, 0.7)
    # representatives are unique-ish scaffolds; members re-expand to the full set
    assert sum(g.size for g in groups) == len(rec.profiles)
    assert all(g.representative is g.members[0] for g in groups)
    # near-identical pVEC variants land in one group
    reps = [g.representative.name for g in groups]
    assert len(reps) == len(set(reps))


def test_require_encodable_filters_modified() -> None:
    lib = [
        ScreenCandidate(sequence="LLIILARRIRKQAHAHSK", name="bare"),  # encodable
        ScreenCandidate(sequence="GWTLNSAGYLLGKINLKALAALAKKIL", name="tagged",
                        c_term_mod="Amidation"),  # modified
    ]
    rec = recommend_for_algae(
        anchor=_PVEC, library=lib, ledger=build_seed_ledger(), classifier=None,
        top_k=None, require_encodable=True, low_toxicity=False,
    )
    names = {p.name for p in rec.profiles}
    assert "bare" in names and "tagged" not in names


def test_cloneable_bucket_only_has_encodable() -> None:
    lib = [
        ScreenCandidate(sequence="LLIILARRIRKQAHAHSK", name="bare"),
        ScreenCandidate(sequence="AGYLLGKINLKALAALAKKIL", name="tagged", n_term_mod="Stearylation"),
    ]
    rec = recommend_for_algae(
        anchor=_PVEC, library=lib, ledger=build_seed_ledger(), classifier=None,
        top_k=None, low_toxicity=False,
    )
    cats = categorize(rec.profiles, _PVEC)
    cloneable = [c for c in cats if c.key == "cloneable"]
    assert cloneable and all(p.genetically_encodable for p in cloneable[0].profiles)


def test_encodable_reason_present() -> None:
    lib = [ScreenCandidate(sequence="AGYLLGKINLKALAALAKKIL", name="tagged", n_term_mod="Stearylation")]
    rec = recommend_for_algae(
        anchor=_PVEC, library=lib, ledger=build_seed_ledger(), classifier=None,
        top_k=None, low_toxicity=False,
    )
    reasons = explain_profile(rec.profiles[0])
    assert any("modified" in r.text.lower() and not r.positive for r in reasons)


def test_categorize_returns_distinct_buckets() -> None:
    rec = _rec(algae_mode=True, low_toxicity=False, top_k=None)
    cats = categorize(rec.profiles, _PVEC, per_bucket=3)
    keys = {c.key for c in cats}
    assert {"closest", "novel", "gentle", "algae"} <= keys
    for c in cats:
        assert len(c.profiles) <= 3
    # the "closest" bucket should be more anchor-similar than the "novel" bucket
    closest = next(c for c in cats if c.key == "closest")
    novel = next(c for c in cats if c.key == "novel")
    assert closest.profiles[0].physchem >= novel.profiles[0].physchem


def test_categorize_key_filter() -> None:
    rec = _rec(algae_mode=True, top_k=None)
    cats = categorize(rec.profiles, _PVEC, keys=["gentle"])
    assert [c.key for c in cats] == ["gentle"]


def test_categorized_markdown_renders() -> None:
    rec = _rec(algae_mode=True, low_toxicity=False, top_k=None)
    md = rec.to_markdown_categorized(per_bucket=2)
    assert "Algae-delivery hypotheses" in md
    assert "Gentlest" in md and "✓" in md  # buckets + reasons present


def test_dataframe_uses_renamed_columns() -> None:
    df = _rec(algae_mode=True, top_k=5).to_dataframe()
    assert "composite_score" in df.columns
    # mechanistic decomposition columns
    assert {"usable_delivery", "surface_binding", "insertion_fit"} <= set(df.columns)
    assert "family" in df.columns
    assert "overall_match" not in df.columns  # old name gone


def test_filter_and_rank_matches_pipeline_policy() -> None:
    # The shared helper (used by the app) reproduces the pipeline's ordering.
    fit = AlgaeFitScorer.from_ledger(build_seed_ledger(), [c.sequence for c in _LIB])
    profiles = EvidenceScorer(_LIB, algae_fit_scorer=fit).profile(_PVEC)
    kept = filter_and_rank(profiles, _PVEC, rank_by="algae_fit")
    via_pipeline = _rec(algae_mode=True, rank_by="algae_fit").profiles
    assert [p.sequence for p in kept] == [p.sequence for p in via_pipeline]
