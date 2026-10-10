"""Components and worlds.

A Component is one physical thing (a cube, a conveyor, a pipe). It declares its parameters, knows
how to describe itself publicly (visible parameters only), and carries its hidden truth. How a
component is simulated is the business of its pack: tools call into component methods, so the core
never needs to know what a cube or a pipe is.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from .params import ParamSet


class Component:
    """Base class. Subclasses set `kind` and `params`, and add their own simulation methods."""

    kind: ClassVar[str] = "component"
    params: ClassVar[ParamSet] = ParamSet()
    doc: ClassVar[str] = ""

    def __init__(self, id: str, visible: dict, hidden: dict, pose: dict | None = None):
        self.id = id
        self.visible, self.hidden = self.params.validate(visible, hidden)
        self.pose = dict(pose or {})

    @classmethod
    def sample(cls, id: str, rng: np.random.Generator, pose: dict | None = None, **fixed) -> "Component":
        vis, hid = cls.params.sample(rng, fixed)
        return cls(id, vis, hid, pose)

    def public(self) -> dict:
        """What an agent may know: identity, kind, pose and visible parameters. Never hidden ones."""
        return {"id": self.id, "kind": self.kind, "pose": self.pose, **self.visible}

    def with_hidden(self, **changes) -> "Component":
        """A copy with some hidden parameters changed (used by counterfactual checks)."""
        hid = {**self.hidden, **changes}
        return type(self)(self.id, dict(self.visible), hid, dict(self.pose))

    def to_dict(self) -> dict:
        return {"kind": self.kind, "id": self.id, "visible": self.visible, "hidden": self.hidden, "pose": self.pose}

    def __repr__(self):
        return f"{type(self).__name__}({self.id!r}, visible={self.visible}, hidden={self.hidden})"


@dataclass
class World:
    """A set of components plus world-level settings (gravity, table size, seed...)."""

    components: list[Component]
    settings: dict[str, Any] = field(default_factory=dict)
    seed: int = 0

    def __post_init__(self):
        ids = [c.id for c in self.components]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate component ids: {ids}")

    def get(self, id: str) -> Component:
        for c in self.components:
            if c.id == id:
                return c
        raise KeyError(f"no component {id!r}")

    def ids(self, kind: str | None = None) -> list[str]:
        return [c.id for c in self.components if kind is None or c.kind == kind]

    def public(self) -> dict:
        return {"components": [c.public() for c in self.components], "settings": self.settings}

    # mutation (games: building, removing, reconfiguring)
    def add(self, component: Component) -> Component:
        if component.id in self.ids():
            raise ValueError(f"component id {component.id!r} already exists")
        self.components.append(component)
        return component

    def remove(self, id: str) -> Component:
        c = self.get(id)
        self.components.remove(c)
        return c

    def next_id(self, prefix: str) -> str:
        n = 1
        while f"{prefix}{n}" in self.ids():
            n += 1
        return f"{prefix}{n}"

    def replace(self, component: Component) -> "World":
        """A copy of the world with one component swapped (counterfactuals, oracle search)."""
        comps = [component if c.id == component.id else c for c in self.components]
        return World(comps, copy.deepcopy(self.settings), self.seed)

    def to_dict(self) -> dict:
        return {"seed": self.seed, "settings": self.settings, "components": [c.to_dict() for c in self.components]}

    @staticmethod
    def from_dict(d: dict) -> "World":
        from .registry import COMPONENTS
        comps = [COMPONENTS.get(c["kind"])(c["id"], c["visible"], c["hidden"], c.get("pose")) for c in d["components"]]
        return World(comps, d.get("settings", {}), d.get("seed", 0))
