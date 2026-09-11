"""Leave-one-out formula-testing harness for algae mCherry delivery.

Purpose: turn "which scoring formula is best?" into something we MEASURE against
the real labeled data, instead of guessing. Every candidate formula is scored on
how well it rank-orders the known peptides by delivery.

Two honesty rules for tiny n (6 quantitative points):
  1. Parameter-free formulas (single descriptor, or the current fixed pipeline
     formula) are evaluated on all 6 directly — there is nothing to overfit.
  2. Fitted formulas (a regression with weights learned from the data) are
     evaluated by LEAVE-ONE-OUT: fit on 5, predict the held-out 1, repeat. This
     is the only fair way to estimate how a fitted model generalizes here.

Metrics:
  - pair_acc: fraction of peptide PAIRS ordered correctly (most interpretable at
    n=6; ties in the target are excluded).
  - spearman: rank correlation of formula score vs log10(%mCherry).
  - kang_ok: does the formula rank pVEC ABOVE all 4 Kang-2017 fails? (0..1)

Nothing here is a conclusion — 6 points can rule formulas OUT, not prove one.
"""

from __future__ import annotations

import itertools
import numpy as np

from cpp_ai.descriptors import compute_descriptors
from cpp_ai.evidence.store import EvidenceLedger
from cpp_ai.pipeline import _DATA, _LEDGER_PATH, load_cppsite3_library, AlgaeFitScorer, HEMOLYSIS_EXPONENT
from cpp_ai.scoring.context import AlgaeFitScorer as _AFS  # noqa: F401 (type clarity)
from cpp_ai.scoring.cytotoxicity import cytotoxicity_factor
from cpp_ai.scoring.disruption import hemolysis_prior
from cpp_ai.scoring.insertion import membrane_interaction_capacity
from cpp_ai.scoring.surface import surface_interaction_prior

SCREEN = {
    "pVEC-R6A": ("LLIILARRIRKQAHAHSK", 15.138),
    "FUS1":     ("IALVWSFRMLRHKP", 8.519),
    "SAG1":     ("GCAAALGYWGLREQSWAQLG", 1.502),
    "gAUT":     ("AQEEFQGVGMVKLKSAFR", 1.024),
    "ClWOX":    ("TNVYNWFQNRRARTKRK", 0.832),
    "MAR1":     ("PPRPPWPPRPPPAPPPSRPP", 0.832),
}
KANG_FAILS = {
    "R9": "RRRRRRRRR", "TAT": "YGRKKRRQRRR",
    "Transportan": "GWTLNSAGYLLGKINLKALAALAKKIL", "Penetratin": "RQIKIWFQNRRMKWKK",
}
KANG_WIN = ("pVEC", "LLIILRRRIRKQAHAHSK")


def _netq(s: str) -> int:
    return (s.count("K") + s.count("R")) - (s.count("D") + s.count("E"))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if xr.std() == 0 or yr.std() == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def _pair_acc(score: np.ndarray, target: np.ndarray) -> float:
    ok = tot = 0
    for i, j in itertools.combinations(range(len(target)), 2):
        if target[i] == target[j]:
            continue  # tied truth: no ordering to get right
        tot += 1
        if np.sign(score[i] - score[j]) == np.sign(target[i] - target[j]):
            ok += 1
    return ok / tot if tot else 0.0


def build_context() -> dict[str, dict]:
    """Precompute every feature each formula might use, once per peptide."""
    led = EvidenceLedger.load(_LEDGER_PATH)
    lib = load_cppsite3_library(_DATA)
    scorer = AlgaeFitScorer.from_ledger(led, [c.sequence for c in lib])
    ctx = {}
    for name, (seq, pct) in SCREEN.items():
        d = compute_descriptors(seq, blocks=None).values
        ctx[name] = {
            "seq": seq, "pct": pct, "logpct": float(np.log10(pct)),
            "desc": d, "charge": _netq(seq),
            "surface": surface_interaction_prior(seq),
            "membrane": membrane_interaction_capacity(seq),
            "algae_sar": scorer.score(seq),
            "hemolysis": hemolysis_prior(seq),
            "cytotox": cytotoxicity_factor(seq),
        }
    return ctx, scorer


# ---- parameter-free candidate formulas: ctx_entry -> score ------------------- #
def _membrane_blend(c, w=0.5):
    return (1 - w) * c["membrane"] + w * c["algae_sar"]

def _selectivity(c):
    return (1 - c["hemolysis"]) ** HEMOLYSIS_EXPONENT * c["cytotox"]

FORMULAS = {
    "current_usable_delivery":   lambda c: c["surface"] * _membrane_blend(c) * _selectivity(c),
    "current_minus_charge":      lambda c: _membrane_blend(c) * _selectivity(c),
    "membrane_blend_only":       lambda c: _membrane_blend(c),
    "mechanistic_membrane_only": lambda c: c["membrane"],
    "algae_sar_only":            lambda c: c["algae_sar"],
    "aliphatic_index_only":      lambda c: c["desc"]["aliphatic_index"],
    "hydrophobic_moment_only":   lambda c: c["desc"]["hydrophobic_moment_alpha"],
    "longest_hydro_run_only":    lambda c: c["desc"]["longest_hydrophobic_run"],
    "gravy_only":                lambda c: c["desc"]["gravy_kyte_doolittle"],
    "charge_only":               lambda c: c["charge"],
    "surface_only":              lambda c: c["surface"],
}

# features used by the fitted (LOO) formulas
FITTED = {
    "LOO_linreg[aliphatic]":            ["aliphatic_index"],
    "LOO_linreg[aliphatic,muH]":        ["aliphatic_index", "hydrophobic_moment_alpha"],
    "LOO_linreg[aliphatic,muH,run,gravy]": [
        "aliphatic_index", "hydrophobic_moment_alpha",
        "longest_hydrophobic_run", "gravy_kyte_doolittle"],
}


def _feature_matrix(ctx, names, feats):
    X = np.array([[ctx[n]["desc"][f] for f in feats] for n in names])
    # standardize columns for numerical stability
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def eval_all(ctx, scorer):
    names = list(SCREEN)
    logpct = np.array([ctx[n]["logpct"] for n in names])

    rows = []
    for fname, fn in FORMULAS.items():
        s = np.array([fn(ctx[n]) for n in names])
        rows.append((fname, _pair_acc(s, logpct), _spearman(s, logpct), False))

    for fname, feats in FITTED.items():
        X = _feature_matrix(ctx, names, feats)
        preds = np.zeros(len(names))
        for i in range(len(names)):
            tr = [j for j in range(len(names)) if j != i]
            A = np.c_[X[tr], np.ones(len(tr))]
            coef, *_ = np.linalg.lstsq(A, logpct[tr], rcond=None)
            preds[i] = np.r_[X[i], 1.0] @ coef
        rows.append((fname, _pair_acc(preds, logpct), _spearman(preds, logpct), True))

    # Kang cross-check: fraction of the 4 fails that each formula ranks BELOW pVEC.
    kwin = _ctx_for(KANG_WIN[1], scorer)
    kfails = {n: _ctx_for(s, scorer) for n, s in KANG_FAILS.items()}
    kang = {}
    for fname, fn in FORMULAS.items():
        wv = fn(kwin)
        below = sum(1 for c in kfails.values() if fn(c) < wv)
        kang[fname] = below / len(kfails)

    rows.sort(key=lambda r: (-r[1], -r[2]))
    print(f"{'formula':38} {'pair_acc':>9} {'spearman':>9} {'fitted':>7} {'kang_ok':>8}")
    for fname, pa, sp, fit in rows:
        ks = f"{kang.get(fname, float('nan')):.2f}" if fname in kang else "  n/a"
        print(f"{fname:38} {pa:9.2f} {sp:9.2f} {str(fit):>7} {ks:>8}")
    print("\npair_acc/spearman are over the 6 screen peptides (14 ordered pairs; the "
          "ClWOX/MAR1 tie excluded).\nkang_ok = fraction of 4 Kang fails ranked below pVEC (1.00 = all correct).")


def _ctx_for(seq: str, scorer) -> dict:
    d = compute_descriptors(seq, blocks=None).values
    return {
        "seq": seq, "desc": d, "charge": _netq(seq),
        "surface": surface_interaction_prior(seq),
        "membrane": membrane_interaction_capacity(seq),
        "algae_sar": scorer.score(seq),
        "hemolysis": hemolysis_prior(seq),
        "cytotox": cytotoxicity_factor(seq),
    }


def main() -> None:
    ctx, scorer = build_context()
    eval_all(ctx, scorer)


if __name__ == "__main__":
    main()
