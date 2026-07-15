# `cpp_ai.scoring` — interpretable multi-axis scoring

Built in response to an expert review of the app: the original single similarity
number "looked more biologically authoritative than it really is." This module
**decomposes it into separate, honestly-labelled axes** and fixes three specific
flaws.

## Fixes to the physicochemical similarity (`blocks.py`, `physchem.py`)

| Problem (review) | Fix |
|---|---|
| Blocks not equally weighted (composition ~27 dims, amphipathicity 2); charge repeated ~10× | **Biological blocks** with curated, non-redundant features — one similarity per concept |
| Cosine ignores magnitude (an extreme peptide with the same *direction* scores high) | Per-block **Gaussian on standardized distance** (`exp(−d²/2σ²)`) — magnitude matters |
| `(cos+1)/2` is a misleading scale | **Library-calibrated percentile** ("top X%") |
| One opaque number | `PhyschemProfile` exposes **every block score** |

Blocks: `charge`, `hydrophobicity`, `amphipathicity` (up-weighted 1.5×),
`structure`, `composition`, `aggregation`. Weights are user-adjustable.

## Multi-axis evidence (`evidence.py`)

`EvidenceScorer.profile(anchor)` returns an `EvidenceProfile` per candidate with
each axis **separate** — never blended into one authoritative score:

- **Anchor resemblance:** block-wise physchem (+percentile), local-motif
  (Smith-Waterman), global identity, optional ESM-2 "protein-context".
- **CPP plausibility:** the trained classifier's `cpp_probability` (kept
  distinct from uptake strength, which we can't predict).
- **Applicability domain:** nearest-neighbor distance to known CPPs → a
  `high`/`medium`/`low` confidence flag, so out-of-distribution scores aren't
  trusted blindly.
- **Safety (graded):** `charge_risk` + `safety_factor` (see below), toxicity
  flag, lytic risk.
- **Evidence quality:** experimental (CPPsite3) vs computational.

`shortlist_score` is provided only as a transparent convenience ordering (mean
of available resemblance/plausibility axes × safety factor); the components are
all exposed.

## Graded safety penalties (`safety.py`)

Replaces the hard "+4…+7 or drop" gate — which discarded gently-charged (+3)
candidates — with a smooth risk function:
`+3 → small penalty · +4…+7 → preferred · +8 → moderate · +10 → strong`, plus a
membrane-lytic bump. It **down-weights** rather than excludes, and is exposed as
`charge_risk`. This is a heuristic prior, **not** a hemolysis prediction.

## In the app

The **"Evidence profile (multi-axis)"** mode (now the default) shows the full
per-axis table with a graded "Max charge risk" slider and CSV/FASTA export.

## Still to come (from the review)
Scaffold-vs-discovery modes (positional W6/T14 weighting for ClWOX homologs),
motif features (RW alternation, helical-wheel face), family-split re-validation
of the classifier AUC, and — once algae data exists — an active-learning loop.

## Testing
`tests/scoring/` (22 tests): block metadata & feature dedup, block similarity
(self=1, magnitude sensitivity, weight override, calibration), the evidence
scorer (all axes present, classifier/embedding optional, sorting, lytic
handling), and graded safety (window, low-charge leniency, monotonic high-charge
risk, lytic bump).
