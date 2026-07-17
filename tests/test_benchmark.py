"""Validation gate: the known-peptide benchmark.

These tests do NOT assert a passing model — the model does not yet cleanly rank
good CPPs above disruptive peptides (separation AUC ~0.63). They assert the
behaviours that *do* hold, and pin the *known failures* as explicit regression
trackers so a future change that fixes (or breaks) them is visible.
"""

from __future__ import annotations

from cpp_ai.benchmark import (
    BENCHMARK_PANEL,
    benchmark_metrics,
    run_benchmark,
    scramble_margin,
    separation_auc,
)


def _by_name() -> dict[str, object]:
    return {r.name: r for r in run_benchmark()}


def test_all_scored_and_bounded() -> None:
    results = run_benchmark()
    assert len(results) == len(BENCHMARK_PANEL)
    for r in results:
        for v in (r.surface, r.insertion, r.hemolysis, r.cytotox, r.usable):
            assert 0.0 <= v <= 1.0


def test_clear_hemolytic_peptides_are_buried() -> None:
    # The model DOES get the RBC-hemolytic ones right (the trained prior catches
    # them): melittin, MAP, magainin sit at the bottom.
    r = _by_name()
    for name in ("Melittin", "MAP", "Magainin 2"):
        assert r[name].usable < 0.05, name  # type: ignore[attr-defined]


def test_inert_controls_rank_low() -> None:
    r = _by_name()
    for name in ("polyGS", "anionic"):
        assert r[name].usable < 0.05, name  # type: ignore[attr-defined]


def test_KNOWN_FAILURE_hemolysis_misses_KLA() -> None:
    # KLA is mitochondrially toxic but NOT hemolytic, so the HemoPI2 prior scores
    # it near-zero disruption — a documented blind spot (hemolysis != membrane
    # toxicity). If a future model fixes this, update the test.
    kla = _by_name()["KLA (KLAKLAK)2"]
    assert kla.hemolysis < 0.15  # type: ignore[attr-defined]  # blind spot present


def test_KLA_low_hemolysis_but_elevated_cytotoxicity() -> None:
    # The Stage-3 fix: KLA's hemolysis is ~0 (the blind spot), but the curated
    # cytotoxicity axis flags it (factor < 1) so it is demoted by evidence.
    from cpp_ai.scoring.cytotoxicity import cytotoxicity_class, cytotoxicity_factor
    kla = "KLAKLAKKLAKLAK"
    assert cytotoxicity_class(kla) == "cytotoxic_organelle"
    assert cytotoxicity_factor(kla) < 0.5


def test_melittin_high_hemolysis() -> None:
    from cpp_ai.scoring.disruption import hemolysis_prior, is_trained_model_available
    if is_trained_model_available():
        assert hemolysis_prior("GIGAVLKVLTTGLPALISWIKRKRQQ") > 0.7


def test_buforin_not_penalized_like_lytic_amps() -> None:
    # Non-lytic translocator must keep factor 1.0, unlike melittin/KLA.
    from cpp_ai.scoring.cytotoxicity import cytotoxicity_class, cytotoxicity_factor
    assert cytotoxicity_class("TRSSRAGLQFPVGRVHRLLRK") == "nonlytic_translocator"
    assert cytotoxicity_factor("TRSSRAGLQFPVGRVHRLLRK") == 1.0


def test_cytotoxicity_demotes_lytic_families() -> None:
    from cpp_ai.scoring.cytotoxicity import cytotoxicity_factor
    for lytic in ("GIGAVLKVLTTGLPALISWIKRKRQQ", "INLKALAALAKKIL", "KLALKLALKALKAALKLA"):
        assert cytotoxicity_factor(lytic) < 1.0


def test_exponent_choice_is_not_load_bearing() -> None:
    # KLA/melittin ranks stable across exponents 1..3 — the cytotoxicity axis, not
    # the exponent, does the work (redesign G).
    from cpp_ai.benchmark import exponent_sensitivity
    rep = exponent_sensitivity()
    assert len({rep[e]["KLA_rank"] for e in rep}) == 1  # KLA rank identical across exponents
    assert all(rep[e]["auc"] > 0.63 for e in rep)


def test_pvec_beats_its_scrambles_on_average() -> None:
    # After the order-sensitive rebuild (Stage 2), native pVEC should out-score the
    # AVERAGE of its composition-matched scrambles by a clear margin.
    from cpp_ai.benchmark import scramble_margin
    assert scramble_margin("LLIILRRRIRKQAHAHSK", n=20) > 5.0


def test_membrane_interaction_capacity_is_order_sensitive() -> None:
    # MIC must change under scrambling (it did not, meaningfully, before Stage 2).
    import random
    from cpp_ai.scoring.insertion import membrane_interaction_capacity as mic
    native = "LLIILRRRIRKQAHAHSK"
    scr = []
    for k in range(15):
        letters = list(native)
        random.Random(k).shuffle(letters)
        scr.append(mic("".join(letters)))
    assert mic(native) > sum(scr) / len(scr)  # native arrangement scores higher


def test_RESIDUAL_a_single_adversarial_scramble_can_still_score_high() -> None:
    # Honest limitation: the AVERAGE margin is positive, but not every individual
    # scramble falls below native (per review: do not tune to one scramble).
    scr = _by_name()["scrambled-pVEC"]
    assert scr.usable > 0.2  # type: ignore[attr-defined]  # this particular shuffle still ranks up


def test_separation_is_weak_but_computed() -> None:
    # Documents that good-vs-disruptive separation is currently poor (not asserting
    # a passing threshold — that would be dishonest today).
    auc = separation_auc(run_benchmark())
    assert 0.4 <= auc <= 1.0


def test_metrics_report_shape() -> None:
    m = benchmark_metrics()
    assert set(m) == {"separation_auc", "ranks", "scramble_margin"}
    assert "KLA (KLAKLAK)2" in m["ranks"]  # type: ignore[operator]
    assert set(m["scramble_margin"]) == {"pVEC", "Penetratin", "TAT"}  # type: ignore[arg-type]


def test_scramble_margin_reproducible() -> None:
    # seeded, so the reported margin is stable across runs
    a = scramble_margin("LLIILRRRIRKQAHAHSK", n=10, base_seed=0)
    b = scramble_margin("LLIILRRRIRKQAHAHSK", n=10, base_seed=0)
    assert a == b
