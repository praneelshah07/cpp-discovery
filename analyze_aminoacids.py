"""Amino-acid-level analysis: which residues associate with good vs bad delivery.

Two views, and we only trust a residue when BOTH agree:
  A. quantitative screen (6 peptides): rank-correlation of each AA's fraction
     with %mCherry.
  B. strong-vs-rest (11 peptides incl. Kang literature): difference in mean AA
     fraction between the 3 strong deliverers and the 8 weak/failed ones.

Also rolls residues up into biochemical GROUPS, which are far more stable than
single amino acids at this sample size. Everything is lead-generation (small n).
"""

from __future__ import annotations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
GROUPS = {
    "branched-aliphatic (L/I/V)": "LIV",
    "small-aliphatic (A/M)": "AM",
    "basic (K/R/H)": "KRH",
    "acidic (D/E)": "DE",
    "aromatic (F/W/Y)": "FWY",
    "polar (S/T/N/Q)": "STNQ",
    "special (G/P/C)": "GPC",
}

# %mCherry screen
SCREEN = {
    "pVEC-R6A": ("LLIILARRIRKQAHAHSK", 15.138),
    "FUS1":     ("IALVWSFRMLRHKP", 8.519),
    "SAG1":     ("GCAAALGYWGLREQSWAQLG", 1.502),
    "gAUT":     ("AQEEFQGVGMVKLKSAFR", 1.024),
    "ClWOX":    ("TNVYNWFQNRRARTKRK", 0.832),
    "MAR1":     ("PPRPPWPPRPPPAPPPSRPP", 0.832),
}
# strong deliverers (protein into algae) vs the rest (weak screen + Kang fails)
STRONG = {"pVEC-R6A": "LLIILARRIRKQAHAHSK", "FUS1": "IALVWSFRMLRHKP",
          "pVEC": "LLIILRRRIRKQAHAHSK"}
REST = {"SAG1": "GCAAALGYWGLREQSWAQLG", "gAUT": "AQEEFQGVGMVKLKSAFR",
        "ClWOX": "TNVYNWFQNRRARTKRK", "MAR1": "PPRPPWPPRPPPAPPPSRPP",
        "R9": "RRRRRRRRR", "TAT": "YGRKKRRQRRR",
        "Transportan": "GWTLNSAGYLLGKINLKALAALAKKIL", "Penetratin": "RQIKIWFQNRRMKWKK"}


def frac(seq, chars):
    return sum(c in chars for c in seq) / len(seq)


def spearman(x, y):
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if xr.std() == 0 or yr.std() == 0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def analyze(units: dict[str, str], label: str):
    names = list(SCREEN)
    log = np.array([np.log10(SCREEN[n][1]) for n in names])
    strong_seqs = list(STRONG.values())
    rest_seqs = list(REST.values())

    rows = []
    for name, chars in units.items():
        # A: correlation with %mCherry over the 6-peptide screen
        sp = spearman(np.array([frac(SCREEN[n][0], chars) for n in names]), log)
        # B: strong vs rest mean fraction (11 peptides)
        s_mean = np.mean([frac(s, chars) for s in strong_seqs])
        r_mean = np.mean([frac(s, chars) for s in rest_seqs])
        diff = s_mean - r_mean
        agree = (np.sign(sp) == np.sign(diff)) and abs(sp) >= 0.5 and abs(diff) >= 0.02
        rows.append((name, sp, s_mean * 100, r_mean * 100, diff * 100, agree))

    rows.sort(key=lambda r: -(r[1] + np.sign(r[4]) * min(abs(r[4]) / 10, 1)))
    print(f"\n=== {label} ===")
    print(f"{'unit':28} {'screen_corr':>11} {'strong%':>8} {'rest%':>7} {'diff':>7} {'agree':>6}")
    for name, sp, sm, rm, df, ag in rows:
        print(f"{name:28} {sp:11.2f} {sm:8.1f} {rm:7.1f} {df:+7.1f} {'  yes' if ag else '   -'}")


def main():
    analyze(GROUPS, "Biochemical GROUPS (more trustworthy at small n)")
    analyze({a: a for a in AA}, "Individual amino acids")
    print("\nscreen_corr: rank-corr of the unit's fraction with %mCherry over the 6 screen "
          "peptides (+1 good, -1 bad).\nstrong%/rest%: mean fraction in the 3 strong deliverers "
          "vs the 8 weak/failed.\nagree = both views point the same way (a lead worth trusting).")


if __name__ == "__main__":
    main()
