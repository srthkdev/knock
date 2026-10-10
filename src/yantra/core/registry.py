"""Registries: what exists. The world builder lists these; anything absent is a missing capability."""
from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, what: str):
        self.what = what
        self._items: dict[str, T] = {}

    def register(self, name: str | None = None):
        """Decorator: @COMPONENTS.register() uses the class's `kind`/`name` attribute."""
        def deco(obj):
            key = name or getattr(obj, "kind", None) or getattr(obj, "name", None)
            if not key:
                raise ValueError(f"cannot infer a {self.what} name for {obj}")
            if key in self._items and self._items[key] is not obj:
                raise ValueError(f"{self.what} {key!r} already registered")
            self._items[key] = obj
            return obj
        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"missing {self.what}: {name!r} (available: {sorted(self._items)})") from None

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def names(self) -> list[str]:
        return sorted(self._items)

    def items(self):
        return self._items.items()


COMPONENTS: Registry = Registry("component")
TOOLS: Registry = Registry("tool")
TASKS: Registry = Registry("task")
PACKS: Registry = Registry("pack")
