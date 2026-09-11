"""Hunt for patterns BEYOND greasiness in the algae-delivery data.

Three questions:
  Q1. After removing the greasiness (aliphatic) signal, what ELSE correlates
      with delivery? (residual analysis -> finds a genuine 2nd lever)
  Q2. Which plain, interpretable sequence traits separate winners from losers?
  Q3. Do the extra literature points (Kang 2017 fails) agree?

n is small, so this is lead-generation, not proof. A pattern that shows up in
BOTH the quantitative screen AND the binary literature set is the credible kind.
"""

from __future__ import annotations

import numpy as np
from cpp_ai.descriptors import compute_descriptors

# quantitative screen (%mCherry+), C. reinhardtii
SCREEN = {
    "pVEC-R6A": ("LLIILARRIRKQAHAHSK", 15.138),
    "FUS1":     ("IALVWSFRMLRHKP", 8.519),
    "SAG1":     ("GCAAALGYWGLREQSWAQLG", 1.502),
    "gAUT":     ("AQEEFQGVGMVKLKSAFR", 1.024),
    "ClWOX":    ("TNVYNWFQNRRARTKRK", 0.832),
    "MAR1":     ("PPRPPWPPRPPPAPPPSRPP", 0.832),
}
# literature protein-delivery in algae (Kang 2017): pVEC works, these fail
KANG_FAILS = {
    "R9": "RRRRRRRRR", "TAT": "YGRKKRRQRRR",
    "Transportan": "GWTLNSAGYLLGKINLKALAALAKKIL", "Penetratin": "RQIKIWFQNRRMKWKK",
}
PVEC = ("pVEC", "LLIILRRRIRKQAHAHSK")

HYDRO = set("AILMFWVYC")   # greasy residues
BASIC = set("KR")
ACIDIC = set("DE")


def spearman(x, y):
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if xr.std() == 0 or yr.std() == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def traits(seq):
    d = compute_descriptors(seq, blocks=None).values
    n = len(seq)
    return {
        "len": n,
        "charge": (seq.count("K") + seq.count("R")) - (seq.count("D") + seq.count("E")),
        "%greasy": round(100 * sum(c in HYDRO for c in seq) / n),
        "%pro": round(100 * seq.count("P") / n),
        "#Trp": seq.count("W"),
        "#acidic": sum(c in ACIDIC for c in seq),
        "aliphatic": round(d["aliphatic_index"]),
        "amphipathy(muH)": round(d["hydrophobic_moment_alpha"], 2),
        "charge_seg": round(d.get("charge_segregation", float("nan")), 2),
        "hydro_run": round(d["longest_hydrophobic_run"]),
    }


def main():
    names = list(SCREEN)
    log = np.array([np.log10(SCREEN[n][1]) for n in names])
    feats = {n: compute_descriptors(SCREEN[n][0], blocks=None).values for n in names}

    # ---- Q2: plain trait table ------------------------------------------- #
    print("=== Q2: plain sequence traits (sorted best -> worst delivery) ===")
    cols = ["len", "charge", "%greasy", "%pro", "#Trp", "#acidic", "aliphatic",
            "amphipathy(muH)", "charge_seg", "hydro_run"]
    hdr = f"{'peptide':9} {'%mCh':>6} " + " ".join(f"{c:>9}" for c in cols)
    print(hdr)
    for n in sorted(names, key=lambda x: -SCREEN[x][1]):
        t = traits(SCREEN[n][0])
        print(f"{n:9} {SCREEN[n][1]:6.2f} " + " ".join(f"{t[c]:>9}" for c in cols))

    # ---- Q1: what's left after greasiness? ------------------------------- #
    print("\n=== Q1: strongest patterns AFTER removing the aliphatic signal ===")
    aliph = np.array([feats[n]["aliphatic_index"] for n in names])
    A = np.c_[aliph, np.ones(len(names))]
    coef, *_ = np.linalg.lstsq(A, log, rcond=None)
    resid = log - A @ coef  # delivery variation greasiness does NOT explain
    scan = []
    for d in sorted(feats[names[0]]):
        vals = np.array([feats[n][d] for n in names])
        if vals.std() == 0:
            continue
        # skip descriptors that are themselves basically greasiness
        if abs(spearman(vals, aliph)) >= 0.85:
            continue
        scan.append((d, spearman(vals, resid)))
    scan.sort(key=lambda r: -abs(r[1]))
    print("(descriptors highly correlated with greasiness itself are excluded)")
    for d, r in scan[:12]:
        print(f"  {d:34} residual-spearman = {r:+.2f}")

    # ---- Q3: literature cross-check (do these traits separate pVEC from fails?) --
    print("\n=== Q3: literature check — pVEC (works) vs Kang fails ===")
    pv = traits(PVEC[1])
    fails = {n: traits(s) for n, s in KANG_FAILS.items()}
    for c in ["charge", "%greasy", "aliphatic", "amphipathy(muH)", "#acidic", "charge_seg"]:
        fmean = np.mean([fails[n][c] for n in fails])
        arrow = "higher" if pv[c] > fmean else "lower" if pv[c] < fmean else "same"
        print(f"  {c:16} pVEC={pv[c]:>7}   fails_avg={fmean:>7.1f}   (pVEC {arrow})")


if __name__ == "__main__":
    main()
