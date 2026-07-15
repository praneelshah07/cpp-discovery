"""Preset importers for known CPP databases: CPPsite3 and POSEIDON.

These are thin configurations of :class:`CSVImporter` with the column layout
each database publishes. Real exports vary in exact header spelling, so the
column mapping can be overridden at construction without touching this file —
the presets encode the *documented* defaults.

References
----------
* CPPsite 2.0/3.0: Agrawal et al., a manually-curated database of experimentally
  validated cell-penetrating peptides. Tabular exports carry a peptide ID and a
  one-letter sequence per record.
* POSEIDON: a curated CPP dataset pairing sequences with uptake/experimental
  annotations.

Because header spellings differ between releases, mappings are resolved
case-/whitespace-insensitively (see :meth:`ColumnMapping.resolve`).
"""

from __future__ import annotations

from .base import IMPORTER_REGISTRY, ColumnMapping
from .tabular import CSVImporter


class CPPsite3Importer(CSVImporter):
    """Importer preset for CPPsite3 tabular exports."""

    dataset_name = "CPPsite3"

    @classmethod
    def default_column_mapping(cls) -> ColumnMapping:
        # 'CPPsite ID' and 'Sequence' are the canonical CPPsite export headers.
        return ColumnMapping(sequence="Sequence", identifier="CPPsite ID")

    def __init__(self, column_mapping: ColumnMapping | None = None) -> None:
        super().__init__(column_mapping, dataset_name=self.dataset_name)


class PoseidonImporter(CSVImporter):
    """Importer preset for POSEIDON tabular exports.

    Matches the schema of POSEIDON's ``curated_data/cargo_encoded.csv``: the
    sequence is in ``peptide_sequence`` and the (quantitative uptake) label in
    ``target``, with experimental conditions and one-hot cargo columns preserved
    as metadata. ``peptide_name`` is used as the record identifier.
    """

    dataset_name = "POSEIDON"

    @classmethod
    def default_column_mapping(cls) -> ColumnMapping:
        return ColumnMapping(sequence="peptide_sequence", identifier="peptide_name")

    def __init__(self, column_mapping: ColumnMapping | None = None) -> None:
        super().__init__(column_mapping, dataset_name=self.dataset_name)


IMPORTER_REGISTRY.register("CPPsite3", CPPsite3Importer)
IMPORTER_REGISTRY.register("POSEIDON", PoseidonImporter)
