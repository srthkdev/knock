"""Parameter specs: what a component exposes (visible) or hides (hidden), and how to sample it.

A component declares its parameters once; generators, the world builder, the fairness check and
inference baselines all read the same declaration instead of re-encoding ranges in their own code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, kw_only=True)
class Param:
    """Base class. `unit` and `doc` are shown to builders (and to agents for visible params)."""
    unit: str = ""
    doc: str = ""

    def sample(self, rng: np.random.Generator) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def validate(self, value: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> dict:
        return {"type": type(self).__name__, "unit": self.unit, "doc": self.doc}


@dataclass(frozen=True)
class Uniform(Param):
    lo: float = 0.0
    hi: float = 1.0

    def sample(self, rng):
        return float(rng.uniform(self.lo, self.hi))

    def validate(self, value):
        v = float(value)
        if not self.lo <= v <= self.hi:
            raise ValueError(f"{v} outside [{self.lo}, {self.hi}]")
        return v

    def describe(self):
        return {**super().describe(), "lo": self.lo, "hi": self.hi}


@dataclass(frozen=True)
class LogUniform(Uniform):
    def sample(self, rng):
        return float(np.exp(rng.uniform(np.log(self.lo), np.log(self.hi))))


@dataclass(frozen=True)
class Choice(Param):
    options: Sequence[Any] = ()
    weights: Sequence[float] | None = None

    def sample(self, rng):
        p = None if self.weights is None else np.asarray(self.weights, float) / np.sum(self.weights)
        return self.options[int(rng.choice(len(self.options), p=p))]

    def validate(self, value):
        if value not in self.options:
            raise ValueError(f"{value!r} not in {list(self.options)}")
        return value

    def describe(self):
        return {**super().describe(), "options": list(self.options)}


@dataclass(frozen=True)
class Flag(Choice):
    options: Sequence[Any] = (False, True)


@dataclass(frozen=True)
class ParamSet:
    """Named parameters split into what an agent may see and what it must discover."""
    visible: dict[str, Param] = field(default_factory=dict)
    hidden: dict[str, Param] = field(default_factory=dict)

    def sample(self, rng, fixed: dict | None = None) -> tuple[dict, dict]:
        fixed = fixed or {}
        vis = {k: fixed[k] if k in fixed else p.sample(rng) for k, p in self.visible.items()}
        hid = {k: fixed[k] if k in fixed else p.sample(rng) for k, p in self.hidden.items()}
        return vis, hid

    def validate(self, visible: dict, hidden: dict) -> tuple[dict, dict]:
        unknown = (set(visible) - set(self.visible)) | (set(hidden) - set(self.hidden))
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        missing = (set(self.visible) - set(visible)) | (set(self.hidden) - set(hidden))
        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        return ({k: self.visible[k].validate(v) for k, v in visible.items()},
                {k: self.hidden[k].validate(v) for k, v in hidden.items()})

    def describe(self) -> dict:
        return {"visible": {k: p.describe() for k, p in self.visible.items()},
                "hidden": {k: p.describe() for k, p in self.hidden.items()}}
