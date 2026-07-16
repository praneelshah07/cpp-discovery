"""Hemolysis prior — a *trained* Box-3 (cell-survival) predictor.

Replaces the hand-tuned GRAVY lysis heuristic (which misranks melittin) with a
model trained on real hemolysis data: **HemoPI2** (Raghava lab; sequences +
HC50 + hemolytic/non-hemolytic label, derived from DBAASP/Hemolytik). The
negatives are *poorly-hemolytic real peptides*, so the model learns the **line
between high and low membrane toxicity** among genuine peptides — not "peptide vs
random."

Honest naming and scope: this is a **membrane-disruption prior**, not ground
truth. It is trained on **human-red-blood-cell** hemolysis, so it is a
*cross-kingdom borrowed prior* for the target alga's membrane — better than a
heuristic, but a prior. The goal is an unbiased toxicity axis to set *beside*
uptake, so a genuinely low-toxicity/high-uptake peptide is kept and a
high-toxicity one is flagged — not to purge any family.

`hemolysis_prior(seq)` returns P(membrane-disruptive) in [0, 1]. If the
trained model is unavailable it falls back to the heuristic
:func:`cpp_ai.scoring.safety.membrane_lysis_risk`.

**Scope caveat — it is a *hemolysis* prior, not all membrane toxicity.** It
predicts the RBC-hemolytic phenotype only, so it is **blind to non-hemolytic
membrane damage** — e.g. KLA/(KLAKLAK)ₙ peptides are mitochondrially toxic /
pro-apoptotic yet score near-zero here. See docs/benchmark.md: hemolysis is one
piece of evidence for membrane disruption, not the whole picture.

Retrain with::

    python -m cpp_ai.scoring.disruption
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from ..core.types import is_canonical_sequence
from ..descriptors import compute_descriptors

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
_MODEL_PATH = _ROOT / "data" / "models" / "hemolysis_disruption.pkl"
_DATA_DIR = _ROOT / "data" / "raw" / "hemolysis"

# Interpretable, hemolysis-relevant descriptor wishlist. Only those actually
# emitted by the descriptor stack are used (the fitted feature list is stored in
# the model so scoring stays consistent with training).
_FEATURE_WISHLIST = (
    "length",
    "charge_pH7.4_Lehninger",
    "frac_group_cationic", "frac_group_hydrophobic", "frac_group_aromatic",
    "frac_group_polar_uncharged",
    "frac_R", "frac_K", "frac_L", "frac_W", "frac_F", "frac_G", "frac_A", "frac_I", "frac_V",
    "gravy_kyte_doolittle", "hydrophobic_moment_alpha", "aromaticity", "aliphatic_index",
    "boman_index", "instability_index_biopython", "helix_fraction",
    "longest_hydrophobic_run", "charge_segregation", "aggregation_peak_window_hydrophobicity",
)

_cache: dict[str, Any] = {}


def _available_features() -> list[str]:
    d = compute_descriptors("LLIILRRRIRKQAHAHSK").values
    return [f for f in _FEATURE_WISHLIST if f in d]


def _featurize(sequence: str, features: list[str]) -> list[float]:
    d = compute_descriptors(sequence).values
    return [float(d[f]) for f in features]


def _load_model() -> dict[str, Any] | None:
    if "model" in _cache:
        return _cache["model"]  # type: ignore[no-any-return]
    if not _MODEL_PATH.exists():
        logger.warning("Hemolysis model not found at %s; using heuristic fallback.", _MODEL_PATH)
        _cache["model"] = None
        return None
    try:
        with open(_MODEL_PATH, "rb") as fh:
            _cache["model"] = pickle.load(fh)
    except Exception as exc:  # e.g. scikit-learn version mismatch on the pickle
        logger.warning(
            "Failed to load hemolysis model (%s: %s); using heuristic fallback. "
            "This usually means a scikit-learn version mismatch — pin the version "
            "or retrain with `python -m cpp_ai.scoring.disruption`.",
            type(exc).__name__, exc,
        )
        _cache["model"] = None
    return _cache["model"]  # type: ignore[no-any-return]


def hemolysis_prior(sequence: str) -> float:
    """Trained P(hemolytic) in [0, 1]; heuristic fallback if no model."""
    if not is_canonical_sequence(sequence):
        return 0.0
    bundle = _load_model()
    if bundle is None:
        from .safety import membrane_lysis_risk
        return membrane_lysis_risk(sequence)
    x = _featurize(sequence, bundle["features"])
    proba = bundle["model"].predict_proba([x])[0][1]
    return float(proba)


def is_trained_model_available() -> bool:
    return _load_model() is not None


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    features = _available_features()

    def load(name: str) -> tuple[Any, Any]:
        df = pd.read_csv(_DATA_DIR / f"{name}.csv")
        df["SEQUENCE"] = df["SEQUENCE"].str.strip().str.upper()
        df = df[df["SEQUENCE"].apply(is_canonical_sequence)]
        x = np.asarray([_featurize(s, features) for s in df["SEQUENCE"]], dtype=np.float64)
        return x, df["label"].to_numpy()

    x_cv, y_cv = load("cross_val_dataset")
    x_ind, y_ind = load("independent_dataset")

    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
    auc_cv = cross_val_score(base, x_cv, y_cv, cv=5, scoring="roc_auc").mean()

    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
    model.fit(x_cv, y_cv)
    auc_ind = roc_auc_score(y_ind, model.predict_proba(x_ind)[:, 1])

    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_PATH, "wb") as fh:
        pickle.dump({"model": model, "features": features}, fh)
    _cache.pop("model", None)

    size_kb = _MODEL_PATH.stat().st_size / 1024
    print(f"features: {len(features)} | train n={len(y_cv)} | test n={len(y_ind)}")
    print(f"5-fold CV ROC-AUC: {auc_cv:.3f} | independent ROC-AUC: {auc_ind:.3f}")
    print(f"saved {_MODEL_PATH} ({size_kb:.0f} KB)")
    print("\nsanity (want lytic HIGH, gentle LOW):")
    for n, s in [
        ("Melittin (lytic)", "GIGAVLKVLTTGLPALISWIKRKRQQ"),
        ("Brevinin-2R", "KLKNFAKGVAQSLLNKASCKLSGQC"),
        ("Mastoparan", "INLKALAALAKKIL"),
        ("MAP", "KLALKLALKALKAALKLA"),
        ("pVEC (gentle)", "LLIILRRRIRKQAHAHSK"),
        ("pVEC-R6A", "LLIILARRIRKQAHAHSK"),
        ("TAT", "YGRKKRRQRRR"),
    ]:
        print(f"  {n:20s} {hemolysis_prior(s):.2f}")


if __name__ == "__main__":
    train()
