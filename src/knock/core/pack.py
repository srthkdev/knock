"""Packs and scenarios.

A Pack bundles component types, tools and task kinds for one family of worlds (tabletop, factory,
...) and knows how to generate random scenarios. A Scenario is a world plus a task; it is what gets
saved, validated, played and benchmarked.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import ClassVar

from .component import Component, World
from .registry import COMPONENTS, TASKS, TOOLS
from .task import Task
from .tool import Tool


@dataclass
class Scenario:
    id: str
    pack: str
    world: World
    task: Task
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "pack": self.pack, "world": self.world.to_dict(), "task": self.task.to_dict(),
                "notes": self.notes}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1, default=str)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(d["id"], d["pack"], World.from_dict(d["world"]), Task.from_dict(d["task"]), d.get("notes", {}))

    @staticmethod
    def from_json(s: str) -> "Scenario":
        return Scenario.from_dict(json.loads(s))

    def public(self) -> dict:
        return {"id": self.id, "pack": self.pack, "world": self.world.public(), "task": self.task.public()}


class Pack:
    name: ClassVar[str]
    description: ClassVar[str] = ""
    components: ClassVar[list[type[Component]]] = []
    tools: ClassVar[list[type[Tool]]] = []
    tasks: ClassVar[list[type[Task]]] = []

    def generate(self, seed: int, **options) -> Scenario:
        raise NotImplementedError

    @classmethod
    def register_all(cls) -> None:
        """Register this pack's components, tools and task kinds (idempotent)."""
        for c in cls.components:
            COMPONENTS.register()(c)
        for t in cls.tools:
            TOOLS.register()(t)
        for t in cls.tasks:
            TASKS.register()(t)

    @classmethod
    def catalogue(cls) -> dict:
        """What this pack offers, for world builders and docs."""
        return {
            "pack": cls.name,
            "description": cls.description,
            "components": {c.kind: {"doc": c.doc, "params": c.params.describe()} for c in cls.components},
            "tools": {t.name: t.spec() for t in cls.tools},
            "tasks": [t.kind for t in cls.tasks],
        }
