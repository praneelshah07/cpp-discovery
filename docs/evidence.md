# Evidence ledger & structure–activity trends (`cpp_ai.evidence`)

The platform's **ground-truth memory**: a curated, literature-sourced record of
*what actually worked, and where* — the empirical counterweight to the
similarity recommender. Where the bulk databases are blind (POSEIDON = mammalian
small-dye uptake only; CPPsite3 = positives only), the ledger deliberately
records **outcomes across contexts**, so the scoring logic can be iterated
against real results instead of assumptions.

## Why it exists

The best CPP depends on the context. The seed makes this concrete with cited
data:

| Peptide | Mammalian | Algae (Chlamydomonas) |
|---|---|---|
| **R9** | success — ~20× TAT uptake (Futaki 2001) | **fail** — penetrates but does not deliver protein (Kang 2017) |
| **pVEC** | partial — below TAT/transportan for protein cargo (Säälik 2004) | **success** — the *only* CPP that delivered protein; 6–150 kDa (Suresh 2013, Kang 2017) |

Kang 2017 (read from the PDF) is emphatic: for **protein** cargo in
*Chlamydomonas*, **only pVEC delivered** — R9, Transportan, TAT and Penetratin
all failed, even though they translocate. Those are recorded as algae-protein
**negatives**, which is what lets the SAR/lysis logic separate a gentle
translocator from a membrane-lytic amphipath. The tested pVEC was
C-terminally **amidated** and is dose-toxic (~45% viability at 40 µM) — a
reminder that the naked sequence ≠ the material actually tested.

A generic "add arginine" prior is *right* for mammalian cells and *wrong* for the
lab's algae goal. Only context-stratified evidence exposes that.

## Curation policy (important)

- **Verified sources only.** Every `EvidenceEntry` carries a `Citation` with a
  resolvable DOI/URL. Numbers are transcribed from the paper; qualitative
  rankings are stored as text in `uptake_metric`, never invented as fake
  percentages.
- **Record what was reported.** Every quantitative/condition field is nullable;
  a blank means the paper didn't report it, not zero.
- **Immutable facts.** Entries are `frozen`; corrections make a new entry and the
  ledger JSON is versioned in git.
- **Content-addressed peptides.** The same sequence from two papers resolves to
  one `peptide_id` with two observations — the aggregation the SAR layer needs.
- `outcome` is a normalized curator call: `success | partial | fail | toxic |
  inconclusive`. `toxic` is separate from `fail` on purpose — a peptide can enter
  cells yet be unusable because it kills them.

## Layout

- `schema.py` — `EvidenceEntry`, `Citation`, controlled vocabularies.
- `store.py` — `EvidenceLedger`: load/save the canonical JSON
  (`data/curated/cpp_evidence_ledger.json`), queries, `to_dataframe()`.
- `seed.py` — the curated seed and its citations. Regenerate the JSON with
  `python -m cpp_ai.evidence.seed`.
- `sar.py` — `analyze_context` / `analyze_all`: contrast winner vs non-winner
  descriptors within an organism.

## SAR trends (from the seed)

`analyze_all(EvidenceLedger.load())` currently reports:

- **Mammalian winners** separate on **arginine content / cationic fraction /
  Boman index** — charge is king.
- **Algae winners** separate on **amphipathicity (helical µH) + hydrophobicity**
  and *lower* aromaticity — pure-cationic R9 falls to the non-winners.

⚠ **Read these as descriptive contrasts, not statistical inference.** The seed is
small (algae: 2 winners vs 4 others). Every `TrendReport` prints its sample
sizes and flags `is_underpowered`. The value grows as the ledger does.

## How this feeds the scoring logic

The trends are a concrete, citable directive for context-specific scoring: for
the algae goal, weight amphipathicity / hydrophobic moment **up** and treat pure
cationic charge as a weaker positive (and a toxicity risk), rather than
importing the mammalian "more arginine" prior. The ledger also becomes a **test
set**: a logic change is an improvement only if it ranks the known
context-winners above the known losers.

## Next entries to add (highest value)

- **True negatives / toxicity:** the lab's thesis data (pVEC/9R toxic to
  *Chlamydomonas* at 5–10 µM) and published non-CPP controls — the seed is
  currently all working CPPs, so winners-vs-losers is really success-vs-partial.
- **More algae protein-cargo studies** (the lab's regime), including the
  pVEC-R6A CRISPR/RNP delivery work.
- **Plant CPP data** to strengthen the non-mammalian, non-algae middle ground.
