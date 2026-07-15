"""Tests for cpp_ai.core.registry."""

from __future__ import annotations

from typing import Callable

import pytest

from cpp_ai.core.exceptions import DuplicateComponentError, UnknownComponentError
from cpp_ai.core.registry import Registry


def test_direct_registration_and_get() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    assert reg.get("a") == 1


def test_decorator_registration() -> None:
    reg: Registry[Callable[[str], int]] = Registry("func")

    @reg.register("length")
    def _length(s: str) -> int:
        return len(s)

    assert reg.get("length")("abcd") == 4


def test_duplicate_registration_raises() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    with pytest.raises(DuplicateComponentError):
        reg.register("a", 2)


def test_overwrite_allows_replacement() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    reg.register("a", 2, overwrite=True)
    assert reg.get("a") == 2


def test_unknown_component_lists_available() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("alpha", 1)
    reg.register("beta", 2)
    with pytest.raises(UnknownComponentError) as exc:
        reg.get("gamma")
    assert "alpha" in str(exc.value) and "beta" in str(exc.value)


def test_empty_name_rejected() -> None:
    reg: Registry[int] = Registry("thing")
    with pytest.raises(ValueError):
        reg.register("", 1)


def test_names_are_sorted() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("z", 1)
    reg.register("a", 2)
    assert reg.names() == ("a", "z")


def test_unregister_removes_component() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    reg.unregister("a")
    assert "a" not in reg
    with pytest.raises(UnknownComponentError):
        reg.get("a")


def test_unregister_unknown_raises() -> None:
    reg: Registry[int] = Registry("thing")
    with pytest.raises(UnknownComponentError):
        reg.unregister("missing")


def test_contains_len_iter() -> None:
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)
    reg.register("b", 2)
    assert "a" in reg
    assert len(reg) == 2
    assert list(reg) == ["a", "b"]
