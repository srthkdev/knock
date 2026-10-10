"""Generic oracle: solve any task by searching the act's arguments with the real simulator.

The oracle sees the hidden truth (it calls the act tool's noise-free `outcome` on the true world)
and minimises the task's loss. It needs no per-task code, so every generated scenario can be
checked for solvability the same way.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .component import World
from .registry import TOOLS
from .task import Task
from .tool import Tool, ToolCall


@dataclass
class OracleResult:
    call: ToolCall
    outcome: dict
    loss: float
    success: bool
    evaluations: int


def _grid(lo: float, hi: float, n: int) -> np.ndarray:
    if lo > 0 and hi / lo > 50:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def _choices(tool: type[Tool]) -> dict[str, list]:
    """Discrete arguments (enum or bool) to enumerate."""
    out = {}
    for name, prop in tool.Args.model_json_schema().get("properties", {}).items():
        if "enum" in prop:
            out[name] = list(prop["enum"])
        elif prop.get("type") == "boolean":
            out[name] = [False, True]
    return out


def solve(world: World, task: Task, act: Tool | None = None, grid: int = 48, refine_iters: int = 40) -> OracleResult:
    act = act or TOOLS.get(task.act)()
    targets = task.answer_targets(world)
    if targets is None:
        targets = [c.id for c in world.components if c.kind in act.targets] if act.targets else [None]
    bounds = act.arg_bounds()
    discrete = _choices(type(act))
    evals = 0

    def evaluate(target, args: dict):
        nonlocal evals
        evals += 1
        comp = world.get(target) if target is not None else None
        a = act.Args.model_validate(args)
        out = act.outcome(world, comp, a)
        call = ToolCall(act.name, target, args)
        return task.loss(world, call, out), out, call

    best = None
    combos = [dict(zip(discrete, v)) for v in itertools.product(*discrete.values())] or [{}]
    for target in targets:
        for fixed in combos:
            point = {k: (lo * hi) ** 0.5 if lo > 0 else (lo + hi) / 2 for k, (lo, hi) in bounds.items()}
            # coarse grid on each continuous argument, then golden-section refinement, coordinate-wise
            for _ in range(2 if len(bounds) > 1 else 1):
                for k, (lo, hi) in bounds.items():
                    xs = _grid(lo, hi, grid)
                    losses = [evaluate(target, {**fixed, **point, k: float(x)})[0] for x in xs]
                    i = int(np.argmin(losses))
                    a, b = xs[max(0, i - 1)], xs[min(len(xs) - 1, i + 1)]
                    g = (math.sqrt(5) - 1) / 2
                    c, d = b - g * (b - a), a + g * (b - a)
                    fc = evaluate(target, {**fixed, **point, k: float(c)})[0]
                    fd = evaluate(target, {**fixed, **point, k: float(d)})[0]
                    for _ in range(refine_iters):
                        if fc < fd:
                            b, d, fd = d, c, fc
                            c = b - g * (b - a)
                            fc = evaluate(target, {**fixed, **point, k: float(c)})[0]
                        else:
                            a, c, fc = c, d, fd
                            d = a + g * (b - a)
                            fd = evaluate(target, {**fixed, **point, k: float(d)})[0]
                    cand = [(losses[i], float(xs[i])), (fc, float(c)), (fd, float(d))]
                    point[k] = min(cand)[1]
            loss, out, call = evaluate(target, {**fixed, **point})
            if best is None or loss < best[0]:
                best = (loss, out, call)
    loss, out, call = best
    return OracleResult(call, out, float(loss), bool(task.success(world, call, out)), evals)
