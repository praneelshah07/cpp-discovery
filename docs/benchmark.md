# Benchmark validation — known peptides (`cpp_ai.benchmark`)

Per external review: **stop adding descriptors, start validating.** This panel of
well-characterized peptides is a fixed gate — every scoring change is checked
against biology instead of tweaked blindly. Run `python -m cpp_ai.benchmark`;
regression-tracked by `tests/test_benchmark.py`.

The question: *does the model rank good CPPs above disruptive peptides and
controls for productive delivery?*

## Current answer: no — separation AUC ≈ 0.63 (0.5 = random)

```
rank  peptide          class        surf  ins  disr  USABLE
  1   KLA (KLAKLAK)2   disruptive   0.91 1.00 0.03   85.9   ← #1 false positive
  2   pVEC             good_cpp     0.91 0.86 0.05   71.4
  3   scrambled-pVEC   control      0.92 0.81 0.11   59.1   ← control ranked #3
  4   Penetratin       good_cpp     0.81 0.60 0.01   47.9
  6   Buforin II       good_cpp     0.91 0.82 0.36   30.6
  9   Brevinin-2R      disruptive   0.92 0.89 0.66    9.5
 16   MAP              disruptive   0.92 0.94 0.94    0.3
 17   Melittin         disruptive   0.92 0.86 0.97    0.1
```
Mean usable: **good 27.7 · disruptive 20.8 · control 21.0** — barely separated.

## Three failing biological processes (diagnosed, not guessed)

1. **Composition-invariance (Boxes 1–2).** `scrambled-pVEC` ranks #3: shuffling
   the sequence keeps composition/charge/hydrophobicity, and Surface + Insertion
   score it like real pVEC. These axes are **composition-driven, not
   structure/pattern-driven** — the amphipathic-patterning signal (hydrophobic
   moment) is too weak relative to the composition terms. This is also why the
   filtered shortlist's scores are compressed near ~90.

2. **Hemolysis ≠ membrane toxicity (Box 3).** `KLA` ranks #1 with disruption
   0.03 — the HemoPI2 prior honestly reports "not hemolytic," but KLA is
   mitochondrially toxic / pro-apoptotic. The prior is **blind to non-hemolytic
   membrane toxicity**. HemoPI2 is a better prior than GRAVY, but it is *one*
   phenotype, not all membrane damage.

3. **No productive-delivery term (missing box).** KLA *inserts* perfectly (1.00)
   and is rewarded; nothing represents "the cell survives and cargo is delivered"
   vs. "the peptide enters and destroys the cell." The model still optimizes
   **membrane interaction**, not **productive delivery** — a strongly-binding,
   strongly-inserting, moderately-disruptive peptide beats a gentler true
   deliverer.

## What NOT to do

Do **not** tweak weights to make KLA rank lower — that would be fitting to one
example. The failures above are *process* gaps. The honest reading: **the ranking
is not yet trustworthy for candidate selection**, and the model's outputs should
be treated as hypotheses requiring wet-lab validation.

## Where this points

- The composition-invariance and "membrane-interaction-not-delivery" failures are
  the same gap your reviewer named: a missing **internalization-mechanism /
  productive-delivery** process. But it should not be hand-built from more
  descriptors — it is fundamentally **empirical** (which peptides deliver intact
  cargo into a living cell is measured, not computed).
- Therefore the highest-value work remains **growing the algae/Auxenochlorella
  evidence ledger** with real delivery + toxicity outcomes, and using *this
  benchmark* (plus the ledger validation) as the gate for any change.
