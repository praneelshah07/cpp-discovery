"""A generic, type-safe plugin registry.

The platform's design goal "every algorithm should be replaceable" is realized
through registries: each family of interchangeable components (descriptors,
similarity metrics, models, mutation operators, objectives) gets its own
``Registry`` instance. Components register themselves by name, and higher
layers select them via configuration rather than imports. This keeps modules
decoupled and makes experiments reproducible from a config file alone.

Example
-------
>>> descriptors: Registry[Callable[[str], float]] = Registry("descriptor")
>>> @descriptors.register("net_charge")
... def net_charge(seq: str) -> float:
...     return float(seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"))
>>> descriptors.get("net_charge")("KRDE")
0.0
"""

from __future__ import annotations

import logging
from typing import Callable, Generic, Iterator, TypeVar

from .exceptions import DuplicateComponentError, UnknownComponentError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """A named collection of interchangeable components keyed by string.

    Parameters
    ----------
    kind:
        Human-readable name of the component family (used in error messages
        and logs), e.g. ``"similarity metric"``.
    """

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._components: dict[str, T] = {}

    @property
    def kind(self) -> str:
        return self._kind

    def register(
        self, name: str, component: T | None = None, *, overwrite: bool = False
    ) -> T | Callable[[T], T]:
        """Register ``component`` under ``name``.

        Usable either directly (``reg.register("x", obj)``) or as a decorator
        (``@reg.register("x")``). Duplicate names raise
        :class:`DuplicateComponentError` unless ``overwrite=True`` — silent
        shadowing of a scientific component would be a reproducibility hazard.
        """
        if not name:
            raise ValueError("Component name must be a non-empty string.")

        def _do_register(obj: T) -> T:
            if name in self._components and not overwrite:
                raise DuplicateComponentError(
                    f"A {self._kind} named {name!r} is already registered. "
                    f"Pass overwrite=True to replace it intentionally."
                )
            self._components[name] = obj
            logger.debug("Registered %s %r", self._kind, name)
            return obj

        # Decorator form: register("name") returns a decorator.
        if component is None:
            return _do_register
        # Direct form: register("name", obj) returns the object.
        return _do_register(component)

    def get(self, name: str) -> T:
        """Return the component registered under ``name``.

        Raises :class:`UnknownComponentError` with the available names listed,
        which turns config typos into an immediately actionable message.
        """
        try:
            return self._components[name]
        except KeyError:
            available = ", ".join(sorted(self._components)) or "(none)"
            raise UnknownComponentError(
                f"No {self._kind} named {name!r}. Available: {available}."
            ) from None

    def unregister(self, name: str) -> None:
        """Remove a registered component.

        Raises :class:`UnknownComponentError` if it was not registered. Useful
        for test teardown and for hot-swapping experimental components.
        """
        try:
            del self._components[name]
        except KeyError:
            raise UnknownComponentError(
                f"Cannot unregister unknown {self._kind} {name!r}."
            ) from None
        logger.debug("Unregistered %s %r", self._kind, name)

    def names(self) -> tuple[str, ...]:
        """Return all registered names, sorted for deterministic output."""
        return tuple(sorted(self._components))

    def __contains__(self, name: object) -> bool:
        return name in self._components

    def __len__(self) -> int:
        return len(self._components)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, components={self.names()!r})"
