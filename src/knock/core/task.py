"""Tasks: a goal over the world, a budget, the allowed tools and how success is scored.

The goal is a predicate on the act's outcome (computed by the simulator), plus a continuous loss
so a generic oracle can search for a solution. Packs subclass Task; nothing here knows physics.
"""
from __future__ import annotations

from typing import Any, ClassVar

from .component import World
from .tool import ToolCall


class Task:
    kind: ClassVar[str] = "task"

    def __init__(self, goal: str, budget: int, probes: list[str], act: str, params: dict | None = None):
        self.goal = goal
        self.budget = budget
        self.probes = list(probes)
        self.act = act
        self.params = dict(params or {})

    # ------------------------------------------------------------------ to implement per task kind
    def success(self, world: World, call: ToolCall, outcome: dict) -> bool:
        raise NotImplementedError

    def loss(self, world: World, call: ToolCall, outcome: dict) -> float:
        """Non-negative; 0 means perfect. Used by the oracle to search the act's arguments."""
        raise NotImplementedError

    def answer_targets(self, world: World) -> list[str] | None:
        """Targets the act must be applied to (None = any). Lets scoring report 'right object'."""
        return None

    def diagnostics(self, world: World, call: ToolCall, outcome: dict) -> dict[str, Any]:
        ans = self.answer_targets(world)
        return {} if ans is None else {"correct_target": call.target in ans}

    # ------------------------------------------------------------------ description
    def public(self) -> dict:
        return {"goal": self.goal, "probe_budget": self.budget, "probes": self.probes, "act": self.act,
                **{k: v for k, v in self.params.items() if not k.startswith("_")}}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "goal": self.goal, "budget": self.budget, "probes": self.probes, "act": self.act,
                "params": self.params}

    @staticmethod
    def from_dict(d: dict) -> "Task":
        from .registry import TASKS
        cls = TASKS.get(d["kind"])
        return cls(d["goal"], d["budget"], d["probes"], d["act"], d.get("params"))
