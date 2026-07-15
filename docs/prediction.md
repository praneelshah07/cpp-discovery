# `cpp_ai.prediction` — Phase 5: machine learning for CPP likelihood

Learns a **calibrated, uncertainty-aware** model of how CPP-like a peptide is,
comparing several algorithms under honest cross-validation. Built and validated
against a synthetic dataset; swaps to real labeled data at the same interface.

> Every prediction is a `PredictionResult` — a calibrated probability with
> confidence and uncertainty — **never a bare score**.

## The model zoo (`MODEL_REGISTRY`)

| Name | Estimator | Availability |
|------|-----------|--------------|
| `random_forest` | RandomForestClassifier | always |
| `svm` | scaled RBF SVC | always |
| `mlp` | scaled MLP ("simple neural network") | always |
| `logistic_regression` | scaled logistic (linear baseline) | always |
| `xgboost` | XGBClassifier | optional (`ml` extra) |
| `lightgbm` | LGBMClassifier | optional (`ml` extra) |

Each model is a factory returning a fresh estimator with fixed `random_state`.
XGBoost/LightGBM are registered only if they import cleanly, so a missing or
broken install (e.g. an XGBoost **libomp** conflict on macOS) just removes that
model rather than breaking anything. *(To enable XGBoost on macOS: `brew install
libomp`, or ensure the venv's OpenMP isn't shadowed by an old Anaconda one.)*

## Features (`FeatureBuilder`)

Assembles a matrix from **Phase 2 descriptors and/or Phase 3 embeddings**
(embeddings optional — set an `embedding_service` to include them). Feature
scaling is deliberately left to the model pipelines so it is re-fit per CV fold
and never leaks across folds. Requires canonical sequences.

## Evaluation

- `cross_validate(factory, X, y)` — stratified k-fold, returns per-fold held-out
  metrics: **ROC-AUC, PR-AUC, accuracy, F1, MCC, Brier**.
- `compare_models(names, X, y)` — cross-validates several models side by side.
- `calibration_curve_data(y, p)` — reliability-curve points + Brier score for
  checking that probabilities mean what they say (plotted in Phase 9).

## Calibration & uncertainty (`CPPClassifier`)

- **Calibration:** the model is wrapped in `CalibratedClassifierCV` (isotonic or
  Platt), so a reported 0.7 really means ~70% — essential for comparable
  rankings and multi-objective optimization (Phase 7).
- **Uncertainty on every prediction:**
  - `predictive_entropy` — total uncertainty in [0, 1] (model-agnostic).
  - `epistemic_std` — disagreement (std of P(CPP)) among ensemble members, for
    ensemble models (`random_forest`, and `xgboost`/`lightgbm` when available);
    high values flag peptides where the model is extrapolating. `None` for
    non-ensemble models.

## Usage

```python
from cpp_ai.prediction import (make_synthetic_cpp_dataset, FeatureBuilder,
                                compare_models, CPPClassifier)

peptides, y = make_synthetic_cpp_dataset(n_per_class=150)      # <- swap for real data
fb = FeatureBuilder(descriptor_blocks=["charge","composition","zscales"])
X = fb.fit_transform(peptides)

for name, rep in compare_models(["random_forest","svm","mlp"], X, y).items():
    print(name, rep.summary()["roc_auc"])

clf = CPPClassifier("random_forest", feature_builder=fb).fit(peptides, y)
for r in clf.predict(candidates):
    print(r.sequence, r.probability, r.confidence, r.predictive_entropy, r.epistemic_std)
```

To use embeddings as inputs, pass `FeatureBuilder(embedding_service=...)`.

## Synthetic data — and the seam for real data

`make_synthetic_cpp_dataset()` produces a **fake** balanced CPP/non-CPP set
(cationic vs acidic profiles that *overlap*, plus label noise) so the task is
learnable but non-trivial — models reach ROC-AUC ~0.8, and calibration and
uncertainty are genuinely exercised. **It carries no biological meaning.**

Real data plugs in at the identical interface: a `list[Peptide]` and an
`np.ndarray[int]` of labels. See the project README / handoff for how to obtain
CPPsite 2.0 and POSEIDON, and note that CPPsite is positive-only — a negative
set (or POSEIDON's quantitative uptake, for regression) is required to train.

## Testing
`tests/prediction/` (35 tests) covers entropy/result derivation, the model zoo,
the synthetic generator (balance, reproducibility, noise), the feature builder
(sources, shapes, stable column order, embeddings, canonical guard), evaluation
(CV, comparison, binary guards, calibration, learnable-signal check), and the
classifier (calibrated probabilities, epistemic std for RF vs `None` otherwise,
fit/predict guards).
