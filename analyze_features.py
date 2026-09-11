"""Critical feature analysis: which descriptors separate algae-delivery winners
from losers, using the real labeled data we have.

Two data regimes (kept separate on purpose — different assays):
  A. The lab mCherry-NLS screen (quantitative %mCherry+ cells, one experiment).
  B. Kang 2017 protein delivery (binary: pVEC works, R9/TAT/Transportan/
     Penetratin fail) — a cross-check in a different cargo/assay.

Everything here is HYPOTHESIS-GENERATION: n is tiny (6 quantitative points),
so treat correlations as leads to test, not conclusions. Outputs a ranked CSV.
"""

from __future__ import annotations

import numpy as np

from cpp_ai.descriptors import compute_descriptors

# --- A. quantitative lab screen (% mCherry+ cells, C. reinhardtii) ------------ #
SCREEN = {
    "pVEC-R6A": ("LLIILARRIRKQAHAHSK", 15.138),
    "FUS1":     ("IALVWSFRMLRHKP", 8.519),
    "SAG1":     ("GCAAALGYWGLREQSWAQLG", 1.502),
    "gAUT":     ("AQEEFQGVGMVKLKSAFR", 1.024),
    "ClWOX":    ("TNVYNWFQNRRARTKRK", 0.832),
    "MAR1":     ("PPRPPWPPRPPPAPPPSRPP", 0.832),
}
# --- B. Kang 2017 binary cross-check ----------------------------------------- #
KANG = {
    "pVEC":        ("LLIILRRRIRKQAHAHSK", 1),   # only CPP that delivered protein
    "R9":          ("RRRRRRRRR", 0),
    "TAT":         ("YGRKKRRQRRR", 0),
    "Transportan": ("GWTLNSAGYLLGKINLKALAALAKKIL", 0),
    "Penetratin":  ("RQIKIWFQNRRMKWKK", 0),
}

WINNERS = ["pVEC-R6A", "FUS1"]           # strong in the mCherry screen
WEAK = ["SAG1", "gAUT", "ClWOX", "MAR1"]  # above control but poor


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if xr.std() == 0 or yr.std() == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    names = list(SCREEN)
    feats = {n: compute_descriptors(SCREEN[n][0], blocks=None).values for n in names}
    pct = np.array([SCREEN[n][1] for n in names])
    log_pct = np.log10(pct)  # uptake is right-skewed; log is the honest scale

    all_desc = sorted(feats[names[0]])
    rows = []
    for d in all_desc:
        vals = np.array([feats[n][d] for n in names])
        if vals.std() == 0:
            continue
        win = np.array([feats[n][d] for n in WINNERS])
        weak = np.array([feats[n][d] for n in WEAK])
        pooled = np.concatenate([win, weak])
        sd = pooled.std()
        cohen = (win.mean() - weak.mean()) / sd if sd else 0.0
        rows.append({
            "descriptor": d,
            "spearman_logpct": _spearman(vals, log_pct),
            "pearson_logpct": _pearson(vals, log_pct),
            "winner_mean": win.mean(),
            "weak_mean": weak.mean(),
            "sep_cohen_d": cohen,
        })

    rows.sort(key=lambda r: -abs(r["spearman_logpct"]))
    print(f"Analyzed {len(all_desc)} descriptors over {len(names)} screen peptides "
          f"(winners={WINNERS} vs weak={WEAK}).\n")
    print(f"{'descriptor':32} {'spearman':>9} {'pearson':>8} {'win_mean':>10} {'weak_mean':>10} {'cohen_d':>8}")
    for r in rows[:25]:
        print(f"{r['descriptor']:32} {r['spearman_logpct']:9.2f} {r['pearson_logpct']:8.2f} "
              f"{r['winner_mean']:10.3f} {r['weak_mean']:10.3f} {r['sep_cohen_d']:8.2f}")

    # write full ranked table
    import csv
    with open("feature_analysis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nFull ranked table ({len(rows)} descriptors) -> feature_analysis.csv")

    # Kang binary cross-check: do the top screen descriptors also separate
    # pVEC (win) from the 4 fails?
    print("\n=== Kang 2017 cross-check (pVEC=win vs R9/TAT/Transportan/Penetratin=fail) ===")
    kfeats = {n: compute_descriptors(KANG[n][0], blocks=None).values for n in KANG}
    for r in rows[:8]:
        d = r["descriptor"]
        pvec = kfeats["pVEC"][d]
        fails = np.array([kfeats[n][d] for n in KANG if KANG[n][1] == 0])
        arrow = "higher" if pvec > fails.mean() else "lower"
        print(f"  {d:30} pVEC={pvec:8.3f}  fails_mean={fails.mean():8.3f}  (pVEC {arrow})")


if __name__ == "__main__":
    main()
