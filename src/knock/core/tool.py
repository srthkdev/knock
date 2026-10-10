"""Tools: everything an agent can do, behind one interface.

A probe (push, tap, run a test batch...) and an act (flick, commit a factory layout...) are both
Tools. Each declares typed arguments (a pydantic model), which component kinds it can target, its
cost, and optionally a Prediction the agent must make before the tool runs. Prediction scoring is
generic: the tool says which observed quantity each predicted field corresponds to.

Agents never see tool code, only `spec()`: name, description, argument schema, prediction schema.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel

from .component import Component, World


class NoArgs(BaseModel):
    model_config = {"extra": "forbid"}


@dataclass
class Attachment:
    kind: str  # "image" | "audio" | "video"
    path: str
    label: str = ""


@dataclass
class Observation:
    """What a tool reports back. `data` is shown to the agent; `truth` is for scoring only."""
    data: dict[str, Any]
    attachments: list[Attachment] = field(default_factory=list)
    truth: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    world: World
    step: int
    out_dir: Path | None
    history: list  # previous (ToolCall, Observation) pairs

    def seed(self, *key) -> int:
        """Deterministic seed from (world seed, key...). Use stable keys (tool, target, repeat) so
        results do not depend on the order in which an agent probes things."""
        h = hashlib.sha256(repr((self.world.seed, *key)).encode()).digest()
        return int.from_bytes(h[:8], "little") >> 1

    def rng(self, *key) -> np.random.Generator:
        return np.random.default_rng(self.seed(*key))

    def repeat(self, tool: str, target: str | None) -> int:
        """How many times this tool was already applied to this target in the episode."""
        return sum(1 for c, _ in self.history if c.tool == tool and c.target == target)


class Tool:
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]] = NoArgs
    Prediction: ClassVar[type[BaseModel] | None] = None
    prediction_required: ClassVar[bool] = False
    # predicted field -> observed field it is scored against, and the error unit
    prediction_targets: ClassVar[dict[str, tuple[str, str]]] = {}
    targets: ClassVar[tuple[str, ...]] = ()  # component kinds; empty = tool takes no target
    cost: ClassVar[int] = 1
    terminal: ClassVar[bool] = False  # an act: ends the episode and is scored by the task

    # ------------------------------------------------------------------ behaviour
    def run(self, ctx: ToolContext, target: Component | None, args: BaseModel) -> Observation:
        raise NotImplementedError

    def outcome(self, world: World, target: Component | None, args: BaseModel) -> dict:
        """Terminal tools: the noise-free result, used by the oracle and the task's scoring."""
        raise NotImplementedError(f"{self.name} is not a terminal tool")

    def score_prediction(self, prediction: BaseModel, obs: Observation) -> dict[str, dict]:
        """Absolute error per predicted field, against the observed value (or the truth for acts)."""
        out = {}
        for pf, (of, unit) in self.prediction_targets.items():
            pred = getattr(prediction, pf, None)
            actual = obs.truth.get(of, obs.data.get(of))
            if pred is None or actual is None:
                continue
            out[pf] = {"predicted": float(pred), "actual": float(actual), "error": abs(float(pred) - float(actual)),
                       "unit": unit}
        return out

    # ------------------------------------------------------------------ description
    @classmethod
    def spec(cls) -> dict:
        return {
            "name": cls.name,
            "description": cls.description,
            "targets": list(cls.targets),
            "cost": cls.cost,
            "terminal": cls.terminal,
            "args": cls.Args.model_json_schema(),
            "prediction": cls.Prediction.model_json_schema() if cls.Prediction else None,
            "prediction_required": cls.prediction_required,
        }

    @classmethod
    def arg_bounds(cls) -> dict[str, tuple[float, float]]:
        """Numeric argument ranges from the Args schema (ge/le); used by generic search."""
        out = {}
        for name, f in cls.Args.model_fields.items():
            lo = hi = None
            for m in f.metadata:
                lo = getattr(m, "ge", lo)
                hi = getattr(m, "le", hi)
            if lo is not None and hi is not None:
                out[name] = (float(lo), float(hi))
        return out


@dataclass
class ToolCall:
    tool: str
    target: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {"tool": self.tool, "target": self.target, "args": self.args, "prediction": self.prediction,
                "reasoning": self.reasoning}
