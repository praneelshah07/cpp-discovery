# POSEIDON model — honest findings (proper training run)

A rigorous, leakage-free evaluation of what a POSEIDON-trained model can and
cannot tell us, done *before* building the UI so we understand the logic and its
limits. All numbers are held-out (no data leakage).

## Methodology fixes vs. the first quick pass
- **No leakage:** 132 sequences have repeat measurements (up to 30×), so random
  CV would leak. Peptide-level CV uses unique sequences; measurement-level CV
  uses **GroupKFold by sequence**.
- **Regression, not just binarize:** target = `log1p(uptake)` (uptake spans
  0–1.1M, median 59 — extreme skew).
- **Condition-aware:** experimental conditions (cargo one-hots, cell line,
  concentration, time, temperature) added as features.
- **Honest importance:** permutation importance (unbiased) instead of RF Gini.

## Result 1 — how good is the sequence→uptake model, really?

**Peptide-level (design-relevant), 5-fold held-out on 338 unique peptides:**
- Spearman **0.56**, R² **0.27**.

So the model captures **real but modest** signal — it explains ~27% of the
variance in median log-uptake. Good enough to *rank* candidates coarsely
(better-vs-worse), **not** to predict uptake precisely. (This is consistent with
the earlier 0.81 ROC-AUC, which was a median split of the same signal — AUC 0.8
≈ Spearman ~0.55.)

## Result 2 — the actual "logic" (permutation importance + direction)

| Feature | Importance | Direction vs uptake | Interpretable? |
|---|---|---|---|
| `Z_z4` (QSAR axis) | **highest** | +0.34 | ❌ abstract electronic axis |
| `aromaticity`, `Z_z2` | med | +0.25 / +0.27 | ✅ more aromatic/bulk → higher |
| `turn_fraction`, `frac_group_polar_uncharged` | med | −0.26 / −0.20 | ✅ less turn/polar → higher |
| `frac_K`, `charge_pH7.4_*` | low | +0.12 / **+0.09** | ✅ but **charge is only weakly positive** |

**Honest reading:** the robust interpretable signals are **higher aromaticity /
amphipathicity and less turn/polar structure**. Net **charge matters only
weakly** here — which *tempers* the earlier "add K/R" story. And the single most
important feature (`Z_z4`) has **no clean biological meaning**, so a chunk of the
model's power is not mechanistically interpretable. We should not oversell a
mechanism.

## Result 3 — conditions carry as much signal as sequence

Measurement-level, GroupKFold (no leakage):
- sequence-only: Spearman **0.38**
- sequence + conditions: Spearman **0.47**

The same peptide gives **different** uptake depending on cargo, cell line,
concentration, and time — conditions add as much predictive power as the
sequence itself. Sequence alone is an incomplete predictor even *within*
POSEIDON.

## Result 4 — the relevance gaps for THIS lab's goal

- **mCherry:** only **3 of 825** canonical rows use mCherry cargo. POSEIDON is
  overwhelmingly **small fluorophores** (FITC, fluorescein, TAMRA). Predicting
  small-dye uptake is a *different problem* than a ~27 kDa mCherry-fusion protein.
- **Organism:** POSEIDON is **mammalian** cell lines (HeLa, NIH-3T3, MCF-7, …).
  **Zero algae.** Transfer to algae is unvalidated.
- Given conditions dominate (Result 3), algae is an entirely unseen "condition."

## Update — Stage 1: CPP-vs-non-CPP classifier (CPPsite3 + UniProt)

Added CPPsite3 (2,814 unique canonical CPPs; `data/raw/cppsite3_*`) as a large
positive set, with **length-matched random SwissProt fragments** as negatives
(UniProt human pool, `data/raw/uniprot_human_pool.fasta`).

- **2,547 positives vs 2,547 negatives**, 5-fold CV: **ROC-AUC 0.95–0.96**
  (LightGBM 0.960, RF 0.953), MCC ~0.80. Trustworthy.
- Sanity: pVEC / TAT / R9 / Penetratin → P(CPP)=1.00; random → 0.09.
- Logic (what separates CPP from random): high Trp/aromatic + cationic character
  (and the abstract `Z_z4` axis), low aliphatic/polar (Val/Asn/Gln) content.
- Saved model: `data/processed/cpp_classifier.pkl`.
- **Caveat:** this answers "is it a CPP *at all*" — it rates *all* real CPPs
  (pVEC, pVEC-R6A, ClWOX-to-come) near 1.0, so it will **not** discriminate
  between good CPPs. Fine discrimination needs the potency layer (POSEIDON,
  modest) and, decisively, the lab's own ClWOX/algae data.

## Two-stage (→ three-stage) screening procedure now in place

1. **Is it a CPP?** — CPPsite3 classifier, AUC 0.95 (trustworthy).
2. **How potent?** — POSEIDON regressor, Spearman ~0.56 (coarse, mammalian/dye).
3. **Does it work in algae as an mCherry fusion?** — lab's ClWOX/algae data
   (missing; the decisive layer).

## Bottom line

The platform is sound and the POSEIDON signal is real but modest (R²≈0.27),
partly non-interpretable, and heavily condition-dependent — for **small-molecule
cargo in mammalian cells**. For this lab's goal (**mCherry fusion → algae**),
POSEIDON is at best a weak prior. The decisive data is the lab's own ClWOX /
algae results, which the platform is designed to fold in iteratively.
