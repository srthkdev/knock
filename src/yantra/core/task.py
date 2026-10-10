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

    def __init__(self, goal: str, budget: int | dict[str, float], probes: list[str], act: str | None,
                 params: dict | None = None):
        self.goal = goal
        # an int is the classic probe budget; a dict names several budgets (turns, sim_seconds, money...)
        self.budgets = {"probes": float(budget)} if isinstance(budget, (int, float)) else {k: float(v) for k, v in budget.items()}
        self.probes = list(probes)  # every non-terminal tool the agent may use
        self.act = act  # terminal tool, or None for games that end by a finishing tool or exhausted budgets
        self.params = dict(params or {})

    @property
    def budget(self) -> float:
        return self.budgets.get("probes", 0.0)

    # ------------------------------------------------------------------ to implement per task kind
    def success(self, world: World, call: ToolCall, outcome: dict) -> bool:
        raise NotImplementedError

    def loss(self, world: World, call: ToolCall, outcome: dict) -> float:
        """Non-negative; 0 means perfect. Used by the oracle to search the act's arguments."""
        raise NotImplementedError

    def final(self, world: World, episode) -> dict:
        """Score when the episode ends without an act (finish tool or budgets exhausted).
        Must return at least {"success": bool, "score": float}."""
        raise NotImplementedError(f"{self.kind} has no act and no final scoring")

    def answer_targets(self, world: World) -> list[str] | None:
        """Targets the act must be applied to (None = any). Lets scoring report 'right object'."""
        return None

    def diagnostics(self, world: World, call: ToolCall, outcome: dict) -> dict[str, Any]:
        ans = self.answer_targets(world)
        return {} if ans is None else {"correct_target": call.target in ans}

    # ------------------------------------------------------------------ description
    def public(self) -> dict:
        return {"goal": self.goal, "budgets": self.budgets, "probe_budget": self.budget, "tools": self.probes, "act": self.act,
                **{k: v for k, v in self.params.items() if not k.startswith("_")}}

    def to_dict(self) -> dict:
        return {"kind": self.kind, "goal": self.goal, "budget": self.budgets, "probes": self.probes, "act": self.act,
                "params": self.params}

    @staticmethod
    def from_dict(d: dict) -> "Task":
        from .registry import TASKS
        cls = TASKS.get(d["kind"])
        return cls(d["goal"], d["budget"], d["probes"], d["act"], d.get("params"))
