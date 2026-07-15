"""Exception hierarchy for cpp_ai.

A single rooted hierarchy (:class:`CppAiError`) lets callers catch every
library-raised error with one ``except`` while still distinguishing specific
failure modes. No control-flow logic lives here; these are pure signalling
types.
"""

from __future__ import annotations


class CppAiError(Exception):
    """Base class for all errors raised by cpp_ai."""


class ValidationError(CppAiError):
    """A value failed a domain/biological validation rule.

    Examples: a peptide sequence containing non-amino-acid characters, or a
    composite-similarity weight vector that does not sum to 1.
    """


class ProvenanceError(CppAiError):
    """Raised when required provenance information is missing or inconsistent.

    Preserving where every record came from is a hard requirement of the
    platform, so provenance problems are errors rather than warnings.
    """


class RegistryError(CppAiError):
    """Base class for plugin-registry problems."""


class DuplicateComponentError(RegistryError):
    """A component name was registered twice in the same registry."""


class UnknownComponentError(RegistryError):
    """A component name was requested but is not registered."""
