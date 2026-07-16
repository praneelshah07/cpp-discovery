"""Benchmark panel — validate the scoring model against *known* peptide biology.

Per external review: stop adding descriptors, start validating. This is a fixed
panel of well-characterized peptides with an expected class, so any change to the
scoring logic can be checked against biology instead of tweaked blindly.

The panel deliberately mixes three classes:

* ``good_cpp``     — CPPs with demonstrated intracellular (often protein) delivery.
* ``disruptive``   — membrane-lytic / cytotoxic peptides (venoms, AMPs, KLA).
* ``control``      — scrambled / compositionally-inert sequences that should NOT
  rank as delivery candidates.

The question the panel answers: **does the model rank good CPPs above disruptive
peptides and controls for productive delivery?** As of writing it does *not*
cleanly (see docs/benchmark.md) — the panel exists to make that failure visible
and to guard against regressions, not to assert a passing model.

Scores are computed on the bare peptide (``fusion_confidence`` = 1.0), so this
isolates the sequence-level biology (Boxes 1–3), not cargo/encodability.

Run:  ``python -m cpp_ai.benchmark``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .scoring.disruption import membrane_disruption_prior
from .scoring.insertion import insertion_fit
from .scoring.surface import surface_adsorption

Category = Literal["good_cpp", "disruptive", "control"]


@dataclass(frozen=True)
class BenchmarkPeptide:
    name: str
    sequence: str
    category: Category
    note: str = ""


# Sequences are the canonical published forms. Class labels are coarse and a few
# are genuinely borderline (transportan is a CPP *and* mastoparan-derived/toxic;
# R9 enters mammalian cells but fails algal protein delivery) — noted inline.
BENCHMARK_PANEL: tuple[BenchmarkPeptide, ...] = (
    BenchmarkPeptide("pVEC", "LLIILRRRIRKQAHAHSK", "good_cpp", "algae protein delivery (Kang 2017)"),
    BenchmarkPeptide("Penetratin", "RQIKIWFQNRRMKWKK", "good_cpp", "Antennapedia homeodomain"),
    BenchmarkPeptide("TAT", "YGRKKRRQRRR", "good_cpp", "classic fusion tag"),
    BenchmarkPeptide("R9", "RRRRRRRRR", "good_cpp", "borderline: fails algal protein delivery"),
    BenchmarkPeptide("Buforin II", "TRSSRAGLQFPVGRVHRLLRK", "good_cpp", "non-lytic translocator"),
    BenchmarkPeptide("MPG", "GALFLGFLGAAGSTMGAWSQPKKKRKV", "good_cpp", "primary amphipathic"),
    BenchmarkPeptide("Transportan", "GWTLNSAGYLLGKINLKALAALAKKIL", "good_cpp",
                     "borderline: mastoparan-derived, membrane-active"),
    BenchmarkPeptide("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ", "disruptive", "bee venom, hemolytic"),
    BenchmarkPeptide("KLA (KLAKLAK)2", "KLAKLAKKLAKLAK", "disruptive",
                     "pro-apoptotic, mitochondrially toxic, NON-hemolytic"),
    BenchmarkPeptide("TP10", "AGYLLGKINLKALAALAKKIL", "disruptive", "transportan-10, membrane-active"),
    BenchmarkPeptide("Mastoparan", "INLKALAALAKKIL", "disruptive", "wasp venom"),
    BenchmarkPeptide("Magainin 2", "GIGKFLHSAKKFGKAFVGEIMNS", "disruptive", "frog AMP"),
    BenchmarkPeptide("Brevinin-2R", "KLKNFAKGVAQSLLNKASCKLSGQC", "disruptive", "frog AMP"),
    BenchmarkPeptide("MAP", "KLALKLALKALKAALKLA", "disruptive", "model amphipathic, lytic"),
    BenchmarkPeptide("scrambled-pVEC", "IHRLKAQIRLSILRAHKL", "control", "pVEC composition, shuffled"),
    BenchmarkPeptide("polyGS", "GSGSGSGSGSGSGSGS", "control", "inert linker-like"),
    BenchmarkPeptide("anionic", "DDEGSDDEGSDDEGSDE", "control", "net-negative, no membrane activity"),
)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    category: Category
    surface: float
    insertion: float
    disruption: float
    usable: float
    note: str


def _usable(surface: float, insertion: float, disruption: float) -> float:
    """usable_delivery on a *bare* peptide (fusion_confidence = 1.0)."""
    return surface * insertion * (1.0 - disruption) ** 2


def run_benchmark() -> list[BenchmarkResult]:
    """Score the panel and return results sorted by usable-delivery (desc)."""
    out: list[BenchmarkResult] = []
    for p in BENCHMARK_PANEL:
        s = surface_adsorption(p.sequence)
        i = insertion_fit(p.sequence)
        d = membrane_disruption_prior(p.sequence)
        out.append(BenchmarkResult(p.name, p.category, s, i, d, _usable(s, i, d), p.note))
    out.sort(key=lambda r: r.usable, reverse=True)
    return out


def separation_auc(results: list[BenchmarkResult]) -> float:
    """P(a random good_cpp outranks a random disruptive) by usable-delivery.

    0.5 = no separation, 1.0 = perfect. Computed as the Mann–Whitney statistic.
    """
    good = [r.usable for r in results if r.category == "good_cpp"]
    bad = [r.usable for r in results if r.category == "disruptive"]
    if not good or not bad:
        return float("nan")
    wins = sum((g > b) + 0.5 * (g == b) for g in good for b in bad)
    return wins / (len(good) * len(bad))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    results = run_benchmark()
    print(f"{'rank':>4} {'peptide':16s} {'class':11s} "
          f"{'surf':>5s} {'ins':>5s} {'disr':>5s} {'USABLE':>7s}")
    for rank, r in enumerate(results, 1):
        print(f"{rank:>4} {r.name[:16]:16s} {r.category:11s} "
              f"{r.surface:5.2f} {r.insertion:5.2f} {r.disruption:5.2f} {r.usable * 100:7.1f}")
    for c in ("good_cpp", "disruptive", "control"):
        print(f"mean USABLE [{c}]: {_mean([r.usable for r in results if r.category == c]) * 100:.1f}")
    print(f"good-vs-disruptive separation AUC: {separation_auc(results):.2f}  (0.5 = none)")


if __name__ == "__main__":
    main()
