"""Oracle: solves a level using the hidden ground truth. Proves levels are solvable."""
from __future__ import annotations

from . import world
from .level import FLICK_RANGE, Level, generate


def solve(level: Level) -> dict:
    o = level.obj(level.answer_id())
    impulse, achieved = world.flick_for_distance(o, level.task.target_x, *FLICK_RANGE)
    ok = abs(achieved - level.task.target_x) <= level.task.tol
    return {"object": o.id, "impulse_Ns": round(impulse, 4), "predicted_x": round(achieved, 4), "solvable": ok}


def valid_level(seed: int, **kw) -> Level:
    """Generate a level and keep re-rolling until the oracle can solve it."""
    for k in range(50):
        lv = generate(seed * 100 + k, **kw)
        sol = solve(lv)
        if sol["solvable"]:
            lv.id = f"L{seed:05d}"
            lv.notes["oracle"] = sol
            return lv
    raise RuntimeError(f"no solvable level for seed {seed}")
