# Scoring architecture — the five biological processes (frozen)

This document **freezes the architecture, not the equations.** The scoring model
is defined as five sequential biological processes a recombinant CPP–mCherry
fusion must clear to deliver folded cargo into a living alga. Each process is a
*box*; each box may hold several descriptors; each box's predictor may be a
heuristic, a literature-derived rule, or a trained model — decided **after** the
biology is settled, not before.

The causal chain:

```
1. Surface interaction   → does the fusion adsorb to the (anionic) algal envelope?
2. Membrane traversal    → can it cross the wall + bilayer?
3. Cell survival         → does the cell live (non-lytic, selective entry)?
4. Cargo compatibility   → does it still work fused to folded ~27 kDa mCherry?
5. Experimental transferability → cloneable, expressible, species/condition-honest?
```

Delivery requires **all five**; the boxes are gates. Score them multiplicatively,
but read them separately.

## Two rules baked into the architecture

1. **The boxes are coupled, not independent.** The same amphipathicity drives
   boxes 2 *and* 3; the same charge drives boxes 1 *and* 3; box 4 (cargo) **feeds
   back** into boxes 1–2 (the complex's charge and size differ from the free
   peptide's). It is a **chain with feedback**, not a clean pipeline. Practical
   consequence: **a descriptor lives in exactly one box** — never count the same
   biophysics in two boxes.
2. **Predictor type is chosen per box, later.** A box may be a heuristic today and
   a trained model tomorrow. Freezing the box does not freeze how it is computed.

---

## Box 1 — Surface interaction

**Biology:** the first step is adsorption of the fusion to the net-negative algal
envelope (wall + membrane; carboxyl/sulfate/phosphate groups). No adsorption →
no uptake. Driven by net charge, **charge distribution/clustering**, **Arg vs Lys**
(guanidinium binds anionic surfaces harder), and some hydrophobic/aromatic
wall-binding. *Ionic strength and buffer are experimental conditions, not
peptide properties — they live in Box 5.*

- **Current score:** `scoring.surface.surface_adsorption` (net charge only).
- **Evidence grade:** direction **strong** (universal + algae-confirmed); breadth
  **partial** (charge-only).
- **Predictor now → later:** literature rule (broaden to charge distribution +
  Arg/Lys + local **charge density**) → possibly trained on algal binding data.
- **Refinement (see Box 4):** driven by the CPP's **local cationic patch**, not the
  complex net charge. Cargo charge is a *fusion-level advisory* (solubility /
  orientation / long-range approach), **not** a subtraction from the CPP's charge.

## Box 2 — Membrane traversal

**Biology:** crossing the wall mesh and partitioning through/across the lipid
bilayer. Amphipathicity (µH), helix propensity (often membrane-induced), and
moderate hydrophobicity. Insertion is a **required positive step** — without it
nothing crosses — it simply must not be allowed to *dominate*.

- **Current score:** `scoring.insertion.insertion_fit` (µH + helix + moderate
  hydrophobicity; aromatics neutral; charge excluded). The cell wall is a
  first-order barrier surfaced as `CELL_WALL_CONTEXT` (constant for fixed cargo).
- **Evidence grade:** universal biophysics **strong**; algae-specific wall
  **species-dependent** (see caveat).
- **Predictor now → later:** literature rule → trained on algal uptake data.

## Box 3 — Cell survival (selectivity)

**Biology:** productive delivery means the cell **lives**. This is the process
that separates a CPP from a lytic AMP — the same amphipathic/hydrophobic features
that aid traversal also enable membrane disruption. Entry that kills the cell is
not delivery.

- **Current score:** `scoring.safety.membrane_lysis_risk` (GRAVY/µH heuristic) —
  the **weakest link**; misranks the canonical hemolytic peptide (melittin).
- **Evidence grade:** concept **strong**; implementation **weak**.
- **Predictor now → later:** **trained membrane-disruption prior** — a hemolysis
  model (HemoPI/DBAASP/Hemolytik) is a *cross-kingdom borrowed prior* (human RBC
  ≠ algal membrane), so name it a **membrane-disruption prior**, not "lysis
  prediction." The ideal eventual predictor is trained on **algal toxicity data**
  (the lab's own viability assays).

## Box 4 — Cargo compatibility

**Biology:** the cargo is not the peptide — it is **CPP + folded ~27 kDa mCherry**.
Does the CPP still function when tethered to a large, folded, **anionic** protein
that must remain folded/fluorescent after transit? Most CPP literature tests
small cargo (FITC, ~400 Da); a folded protein is much harder. This box was nearly
empty and is arguably **more important than previously credited**.

- **Current score:** only `screening.modification.fusion_confidence` (encodability)
  touches the fusion; cargo charge/size/fold are **not modeled**.
- **Evidence grade:** **thin** — the highest-value under-modeled process.
- **mCherry charge (estimate):** pI ≈ 5.6, net charge **≈ −6 at pH 7**
  (variant-/tag-/Met-/pKa-model-dependent — an estimate, not a measurement).

- **⚠ Do NOT subtract net charges (a corrected mistake).** An earlier version of
  this doc inferred that because `complex charge ≈ CPP + cargo`, the free CPP must
  be **~+10 to +12** to keep the complex in the +4…+6 window. **That is wrong**, for
  three independent reasons:
  1. **Point-charge fallacy.** Adsorption is governed by the **local charge density
     of the region that approaches the surface** (the CPP's dense cationic patch),
     plus orientation — *not* the complex's net charge. mCherry's −6 is smeared
     over a ~4 nm, 236-residue globule and, on the far side of the complex and
     beyond the Debye length, does not neutralize the local CPP–surface contact.
  2. **It contradicts the toxicity ceiling.** R9 is **+9 and algae-toxic**; pushing
     toward +10…+12 drives candidates into the membrane-destabilizing regime the
     model already penalizes.
  3. **It contradicts the data.** pVEC (**+6**) delivered 6–150 kDa protein into
     *Chlamydomonas* (Kang 2017; non-covalently). If +6 works, a +10…+12 requirement
     is falsified.

- **Correct model — two levels:**
  - *Peptide-level* (drives Boxes 1–2): **local charge / cationic-patch density**,
    Arg/Lys, amphipathicity, patterning, lysis. The CPP's own **+4…+6** sweet spot
    is unchanged.
  - *Fusion-level* (this box, **advisory not a reweight**): total fusion net charge
    (→ solubility, orientation, long-range approach, nonspecific adsorption), CPP
    local charge **density**, junction charge, CPP solvent accessibility, linker
    length/flexibility, orientation, aggregation, and — decisively — **empirical
    protein-cargo delivery evidence**.
- **Honest statement the tool should make:** *the fusion is near-neutral overall
  but retains a localized cationic CPP terminus; the effect of cargo charge on
  algal binding cannot be inferred by subtracting net charges — it is a
  fusion-level uncertainty best resolved empirically.*
- **Also in this box (mostly empirical):** cargo must stay folded/fluorescent
  after delivery; free-terminus requirements; N- vs C-terminal fusion; steric
  occlusion of the CPP by the cargo.
- **Predictor now → later:** peptide-level local-patch rule (Box 1) + fusion-level
  **advisory** signals now → empirical/trained on fusion-delivery data later.

## Box 5 — Experimental transferability

**Biology:** can this actually be built and will it transfer to *your* system?
Genetic encodability, expression/solubility of the fusion, and honesty about
cross-kingdom/cross-species and cross-condition extrapolation. Experimental
conditions (concentration, ionic strength, temperature, incubation) live here.

- **Current score:** `fusion_confidence` (encodability) + applicability-domain
  confidence (`_ad_confidence`); dose is unmodeled.
- **Evidence grade:** encodability **strong**; transfer honesty **partial**.
- **Predictor now → later:** literature rule + explicit uncertainty.

---

## Species caveat (important)

The lab's target organism is **Auxenochlorella**, but essentially all curated
"direct algal" evidence in the ledger (pVEC, Kang 2017, Suresh 2013) is
**Chlamydomonas reinhardtii** — a *different* cell wall (glycoprotein-rich vs.
Auxenochlorella's tougher, more resistant wall). So even the "direct algal"
grade is partly a **cross-species borrow**, and Box 1/2 priors should be
validated for the actual target species. *The Auxenochlorella wall composition
should be verified explicitly before trusting wall assumptions.*

## How the current `usable_delivery` maps onto the boxes

`usable_delivery = surface_adsorption × insertion_fit × (1 − lysis)² × fusion_confidence`

| Box | Current term | Status |
|---|---|---|
| 1 Surface interaction | `surface_adsorption` | partial (net charge only; broaden to local charge density/distribution) |
| 2 Membrane traversal | `insertion_fit` | partial (wall as context) |
| 3 Cell survival | `(1 − lysis)²` heuristic | **weak** — replace with a trained membrane-disruption prior |
| 4 Cargo compatibility | (only `fusion_confidence`) | **near-empty** — add complex-charge, fold/fluorescence |
| 5 Experimental transferability | `fusion_confidence` + AD confidence | partial |

**Sequencing rule:** define the process (this doc), *then* choose each box's
predictor type. Do not optimize equations before the boxes are agreed. Highest-
value next work, in order: (1) Box 3 trained membrane-disruption prior; (2) Box 4
fusion-level *advisory* signals (charge density, fusion-charge mismatch flag,
fold retention) + Box 1 broadening to local charge density; and — decisively —
growing the **algae/Auxenochlorella evidence ledger**, since Boxes 2–4 are
ultimately empirical.
