# Scoring redesign — deliverables (design phase, no runtime change yet)

Per the lead-review directive: the benchmark exposed two failures — (1) the model
is **composition-driven** (scrambled pVEC ≈ native pVEC) and (2) **hemolysis is
being mistaken for general membrane toxicity** (KLA ranks #1). This document is
the required design phase. **No runtime ranking has been changed.** Implementation
follows only after these six deliverables.

Guiding rule throughout: optimize against **benchmark-wide objectives and
biological consistency**, never against individual peptides.

---

## Deliverable 1 — Current vs. proposed equations

### Current (`usable_delivery`, unchanged this turn)
```
usable_delivery = surface_adsorption(q) × insertion_fit × (1 − hemolysis)² × fusion_confidence

surface_adsorption(q) = max(0.08, σ(1.2·(q−2.5)) · σ(−1.0·(q−8.5)))          # q = net charge only
insertion_fit         = 0.5·min(1, µH/0.5) + 0.2·helix_fraction + 0.3·exp(−q_gravy²/(2·1.5²))
hemolysis             = trained HemoPI2 P(hemolytic)
fusion_confidence     = min over modifications {none 1.0, terminal 0.9, label 0.8,
                                                functional 0.4, noncanonical 0.15}
```
**Problem, quantified (Deliverable 2/4):** `surface_adsorption` is 100 %
composition-invariant; `insertion_fit` is only 50 % order-sensitive (µH), because
`helix_fraction` and whole-sequence `gravy` are composition-only. Net: the ranking
core is ~75 % composition-driven.

### Proposed (to implement *after* this design is accepted)
```
rank_score = surface_interaction_prior
           × membrane_interaction_capacity          # renamed insertion_fit; "necessary, not sufficient"
           × (1 − hemolysis_prior)²                 # renamed disruption; RBC-hemolysis ONLY
           × cytotoxicity_factor                    # from curated cytotoxicity_prior annotation
           × fusion_transferability                 # renamed fusion_confidence

# shown as SEPARATE advisory axes, not folded into rank_score:
cargo_delivery_evidence      # curated experimental record (not a sequence model)
applicability_confidence     # nearest-neighbour distance to known CPPs
```
- **surface_interaction_prior** = charge bell (kept — total charge legitimately
  drives adsorption) **× a charge-segregation modifier** (order-sensitive), so a
  clustered cationic patch and a dispersed one no longer score identically.
- **membrane_interaction_capacity** = majority **order-sensitive**: sliding-window
  / max-local hydrophobic moment + hydrophobic clustering, with whole-sequence
  hydrophobicity down-weighted and `helix_fraction` **removed** (adds no order
  information — see audit).
- **cytotoxicity_factor** ∈ {known-cytotoxic/organelle-disruptive ≈ 0.2,
  strongly-membrane-lytic ≈ 0.3, non-lytic-translocator = 1.0, conventional-CPP =
  1.0, insufficient-evidence = 1.0}, from the curated schema (Deliverable 3). **This
  is how KLA is demoted — by scaffold-level literature evidence, not by hacking the
  sequence model or the hemolysis equation.**
- `rank_score` remains a **hypothesis-prioritization score, not an uptake
  probability.** No algae-specific cell-wall term (insufficient evidence).

---

## Deliverable 2 — Descriptor audit (composition-only vs. order-sensitive)

Empirical: each descriptor computed on native pVEC vs. 12 independent scrambles;
CV = std/|mean| across scrambles (composition-only ⇒ CV = 0).

| Descriptor | CV% under scramble | Class | Used in |
|---|---|---|---|
| `charge_pH7.4_Lehninger` (net charge) | **0.0** | composition-only | surface |
| `frac_group_cationic` | 0.0 | composition-only | — |
| `frac_group_hydrophobic` | 0.0 | composition-only | — |
| `gravy_kyte_doolittle` | 0.0 | composition-only | **insertion** |
| `aliphatic_index` | 0.0 | composition-only | — |
| `aromaticity` | 0.0 | composition-only | — |
| `helix_fraction` | **0.0** | **composition-only** | **insertion** ← removes no order info |
| `boman_index` | 0.0 | composition-only | — |
| `hydrophobic_moment_alpha` (µH) | **33.7** | order-sensitive | insertion (only order term) |
| `charge_segregation` | **93.5** | order-sensitive | — (propose: add to surface) |
| `longest_hydrophobic_run` | 31.7 | order-sensitive | — (propose: add to MIC) |
| `longest_basic_run` | 23.7 | order-sensitive | — (propose: add) |
| `aggregation_peak_window_hydrophobicity` | 32.4 | order-sensitive | — |
| `cationic_aromatic_contacts` | 0.0* | order-sensitive* | — (*0 only because pVEC has no aromatics) |

**Conclusion:** the two ranking-driving terms use **net charge (0 % order)** and
**insertion (50 % order, and even that halved by composition-only helix/gravy)**.
The library already contains strong order-sensitive descriptors
(`charge_segregation`, `longest_*_run`, aggregation window) that are **currently
unused by the ranking** — the fix is to *use* them, not invent new ones.

---

## Deliverable 3 — Cytotoxicity annotation schema (curated, not an equation)

A separate **advisory evidence axis**, curated from the literature/database —
**not** inferred from descriptors (that is exactly the mistake that let KLA
through). Controlled vocabulary:

| `cytotoxicity_class` | Meaning | Example scaffolds | factor |
|---|---|---|---|
| `cytotoxic_organelle` | known cytotoxic / organelle (mitochondrial) disruptive | KLA/(KLAKLAK)ₙ, D-(KLAKLAK)₂ | ~0.2 |
| `membrane_lytic` | known strongly membrane-lytic / pore-forming | melittin, mastoparan, magainin, brevinin, MAP | ~0.3 |
| `nonlytic_translocator` | antimicrobial but reported **non-lytic** translocator | Buforin II, BR2 | 1.0 |
| `conventional_cpp` | conventional CPP, no notable cytotoxicity at working dose | pVEC, penetratin, TAT, R9 | 1.0 |
| `insufficient_evidence` | no curated record | (default for library) | 1.0 |

Schema (mirrors the evidence ledger): `CytotoxAnnotation(scaffold, sequence?,
cytotoxicity_class, citation{doi/url}, note)`, stored as a committed JSON, keyed by
peptide/scaffold, with `insufficient_evidence` as the honest default. **Crucially:
non-lytic translocators (Buforin) are NOT penalized like melittin/KLA** — the
class distinguishes them explicitly. Seed = the benchmark families with citations.

---

## Deliverable 4 — Benchmark report (current model)

`python -m cpp_ai.benchmark` (17-peptide panel). Metrics now emitted every run:

- **good-vs-disruptive separation AUC: 0.63** (0.5 = random) — weak.
- **native-over-scramble margin: pVEC +24.0, TAT +16.4, Penetratin −4.1** — the
  Penetratin margin is **negative** (scrambles beat native): the model is
  order-insensitive.
- **key ranks (of 17):** KLA **#1**, Buforin II #6, pVEC #2, Penetratin #4, TAT #5,
  MPG #14, Brevinin #9, MAP #16, Melittin #17.

New benchmark requirements (F) — every future change must report: good-vs-disruptive
AUC; native-minus-scramble margins across **multiple independent scrambles**; ranks
of KLA / melittin / MAP / brevinin / Buforin II / pVEC / penetratin / TAT / MPG;
and family-held-out performance where possible. **Acceptance targets for the
revised model:** AUC ↑ meaningfully (goal ≥ 0.8), all sequence-dependent CPPs show
a **positive** native-over-scramble margin, KLA demoted out of the top tier (via
cytotoxicity annotation), and Buforin II *not* penalized like melittin.

---

## Deliverable 5 — Proposed ranking equation, term by term

`rank_score = surface_interaction_prior × membrane_interaction_capacity × (1 − hemolysis_prior)² × cytotoxicity_factor × fusion_transferability`

| Term | Biology it represents | Why it belongs / justification |
|---|---|---|
| **surface_interaction_prior** | electrostatic adsorption to the anionic algal envelope | universal + algae-confirmed; adding charge-segregation makes it partly order-sensitive (local patch matters, per colloid electrostatics) |
| **membrane_interaction_capacity** | ability to engage/traverse the bilayer — **necessary, not sufficient** | amphipathic *patterning* (µH, local µH, clustering) is the order-sensitive physics of insertion; must not by itself imply delivery |
| **(1 − hemolysis_prior)²** | RBC-hemolytic membrane damage (one phenotype) | trained on real data (HemoPI2); squared per prior review; explicitly *not* general toxicity |
| **cytotoxicity_factor** | curated non-hemolytic toxicity (organelle/lytic) | evidence-based demotion of KLA-type scaffolds that hemolysis misses; distinguishes non-lytic translocators |
| **fusion_transferability** | can the naked cloned sequence reproduce the tested activity | encodability chemistry (strong); unchanged in spirit |

Kept as **separate advisory axes** (not multiplied in, to avoid silent reweighting
from sparse/curated data): **cargo_delivery_evidence** (Deliverable E schema),
**applicability_confidence**. `rank_score` stays a prioritization score, not a
probability.

### Cargo-delivery evidence axis (E) — curated, separate from the sequence model
Controlled vocabulary (evidence record only): `intact_protein_fusion` >
`protein_noncovalent` > `small_molecule_only` > `free_peptide_only` >
`modification_dependent` > `none`. Stored/curated like the ledger; **never inferred
from descriptors.** Displayed alongside the ranking so the user sees, e.g., that
pVEC has `protein_noncovalent` (algae) while most library hits have `none`.

---

## Deliverable 6 — Assumptions still speculative for *algae*

1. **The charge sweet spot (+4…+6)** is calibrated on the three cationic anchors
   and general electrostatics; the algal (Auxenochlorella) optimum is unverified.
2. **hemolysis_prior transfers to algal membranes** — trained on human RBC; algal
   plasma/organelle membranes differ in lipid composition and sterols.
3. **cytotoxicity annotations transfer across organisms** — most are
   mammalian/bacterial; organelle toxicity (KLA) may or may not apply to algae.
4. **membrane_interaction_capacity ≈ traversal** — patterning predicts membrane
   engagement, but the actual internalization *mechanism* (pore vs. endocytosis vs.
   direct translocation vs. trapped) is not resolved and is largely empirical.
5. **The cell wall is treated as context, not a score** — deliberately, for lack of
   Auxenochlorella-specific permeability data.
6. **Cargo (mCherry) effects** beyond the advisory charge note (fold retention,
   junction, orientation) are empirical, not modeled.
7. **The benchmark's own class labels** (good vs. disruptive) are coarse and partly
   mammalian (transportan/R9 borderline); AUC targets are internal-consistency
   goals, not validated algal ground truth.

---

## Ablation & sensitivity (G) — current model now; revised model after implementation

Measured on the current model (evidence for the redesign):
- **Remove the only order feature (µH):** insertion becomes helix+gravy = 100 %
  composition ⇒ native-over-scramble margin → ~0 for every CPP. Confirms µH is the
  sole order signal and is too weak/outweighed.
- **Composition features dominate:** with surface = pure charge and 50 % of
  insertion composition-only, ~75 % of the ranking is scramble-invariant — hence
  scrambled-pVEC #3 and Penetratin's negative margin.

To be produced **after** implementing the revised model (per G): performance
without sequence-order features / without hemolysis prior / without cytotoxicity
annotations / without cargo evidence / without composition features; plus ±10/20/30 %
weight perturbations with top-candidate and benchmark-metric stability.

---

## Implementation order (only after this design is accepted)

1. Rename in code + docs: `disruption → hemolysis_prior`, `insertion_fit →
   membrane_interaction_capacity` (+ "necessary not sufficient" docstring).
2. Strengthen order-sensitivity: add `charge_segregation` to surface; rebuild MIC
   on µH + local µH + clustering, drop `helix_fraction`, down-weight whole-seq gravy.
3. Add curated `cytotoxicity_prior` + `cargo_delivery_evidence` axes (JSON, seeded).
4. Re-run the benchmark; accept only if the Deliverable-4 targets improve **without**
   peptide-specific tuning; run the full ablation/sensitivity (G).
