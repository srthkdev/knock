"""Level description: visible layout, hidden physical properties, and the task.

Everything under `Obj.hidden` is ground truth that agents never see directly.
`public_view` is the only level data handed to an agent.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .materials import CONTENTS, MATERIALS, WALL

LANE_SPACING = 0.35  # metres between object lanes (y axis)
FLICK_RANGE = (0.02, 6.0)  # allowed flick impulse, N*s
PUSH_RANGE = (0.2, 20.0)  # allowed push force, N
PROBES = ("push", "lift", "tap", "shake")
PROPERTIES = {
    "hollow": "is hollow",
    "solid": "is solid",
    "rattles": "has something loose inside that rattles",
    "heaviest": "is the heaviest",
    "lightest": "is the lightest",
}


@dataclass
class Hidden:
    material: str
    hollow: bool
    contents: str  # key of CONTENTS; "empty" for solids
    friction: float  # sliding friction against the table


@dataclass
class Obj:
    id: str
    half: float  # half edge length of the cube (m)
    color: str
    y: float  # lane position; all objects start at x = 0
    hidden: Hidden

    @property
    def mass(self) -> float:
        h, p = self.hidden, MATERIALS[self.hidden.material]
        edge = 2 * self.half
        if not h.hollow:
            return edge ** 3 * p["density"]
        inner = edge - 2 * WALL
        shell = (edge ** 3 - inner ** 3) * p["density"]
        c = CONTENTS[h.contents]
        return shell + c["mass"] * c.get("count", 1)


@dataclass
class Task:
    kind: str  # "slide" or "find_and_slide"
    target_x: float  # stop the object with its centre in [target_x - tol, target_x + tol]
    tol: float
    budget: int  # number of probes allowed before the single flick
    object_id: str | None = None  # slide: which object to flick
    property: str | None = None  # find_and_slide: which property identifies the object


@dataclass
class Level:
    id: str
    seed: int
    objects: list[Obj]
    task: Task
    notes: dict = field(default_factory=dict)

    def obj(self, oid: str) -> Obj:
        return next(o for o in self.objects if o.id == oid)

    def answer_id(self) -> str:
        """The object the task refers to (ground truth)."""
        if self.task.kind == "slide":
            return self.task.object_id
        prop = self.task.property
        objs = self.objects
        if prop == "hollow":
            return next(o.id for o in objs if o.hidden.hollow)
        if prop == "solid":
            return next(o.id for o in objs if not o.hidden.hollow)
        if prop == "rattles":
            return next(o.id for o in objs if o.hidden.contents in ("marble", "beads"))
        if prop == "heaviest":
            return max(objs, key=lambda o: o.mass).id
        if prop == "lightest":
            return min(objs, key=lambda o: o.mass).id
        raise ValueError(prop)

    # ---------------------------------------------------------------- io
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_json(s: str) -> "Level":
        d = json.loads(s)
        objs = [Obj(**{**o, "hidden": Hidden(**o["hidden"])}) for o in d["objects"]]
        return Level(d["id"], d["seed"], objs, Task(**d["task"]), d.get("notes", {}))


def public_view(level: Level) -> dict:
    """What the agent is told. Contains no hidden properties."""
    t = level.task
    if t.kind == "slide":
        goal = f"Flick object {t.object_id} so it stops with its centre between x = {t.target_x - t.tol:.3f} m and x = {t.target_x + t.tol:.3f} m."
    else:
        goal = (f"Exactly one object {PROPERTIES[t.property]}. Find it and flick it so it stops with its centre "
                f"between x = {t.target_x - t.tol:.3f} m and x = {t.target_x + t.tol:.3f} m.")
    return {
        "level_id": level.id,
        "goal": goal,
        "probe_budget": t.budget,
        "objects": [
            {"id": o.id, "shape": "cube", "edge_m": round(2 * o.half, 3), "color": o.color,
             "start": [0.0, round(o.y, 3)]}
            for o in level.objects
        ],
        "target_band_x": [round(t.target_x - t.tol, 3), round(t.target_x + t.tol, 3)],
        "flick_impulse_range_Ns": list(FLICK_RANGE),
        "push_force_range_N": list(PUSH_RANGE),
    }


# -------------------------------------------------------------------- generation
COLORS = ["#c8553d", "#2f6690", "#e0a458", "#5b8e7d", "#8d6a9f", "#d8d4c7"]


def _random_hidden(rng, hollow=None, contents=None, material=None):
    hollow = bool(rng.random() < 0.5) if hollow is None else hollow
    material = material or str(rng.choice(list(MATERIALS)))
    if not hollow:
        contents = "empty"
    elif contents is None:
        contents = str(rng.choice(["empty", "foam", "weight"]))
    return Hidden(material=material, hollow=hollow, contents=contents,
                  friction=round(float(rng.uniform(0.15, 0.6)), 3))


def generate(seed: int, kind: str | None = None, prop: str | None = None, n_objects: int = 3) -> Level:
    """Random level. Objects in one level look identical (same size and colour)."""
    rng = np.random.default_rng(seed)
    kind = kind or str(rng.choice(["slide", "find_and_slide"]))
    half = round(float(rng.uniform(0.04, 0.07)), 3)
    color = str(rng.choice(COLORS))
    ids = "ABCDEF"[:n_objects]
    ys = [(i - (n_objects - 1) / 2) * LANE_SPACING for i in range(n_objects)]

    if kind == "slide":
        hidden = [_random_hidden(rng) for _ in ids]
        task = Task("slide", target_x=0.0, tol=0.0, budget=int(rng.integers(1, 4)),
                    object_id=str(rng.choice(list(ids))))
    else:
        prop = prop or str(rng.choice(list(PROPERTIES)))
        hidden = _hidden_for_property(prop, n_objects, rng, half)
        task = Task("find_and_slide", target_x=0.0, tol=0.0, budget=n_objects + 1, property=prop)

    objs = [Obj(id=i, half=half, color=color, y=y, hidden=h) for i, y, h in zip(ids, ys, hidden)]
    task.target_x = round(float(rng.uniform(0.35, 1.0)), 3)
    task.tol = round(float(rng.uniform(0.04, 0.07)), 3)
    return Level(id=f"L{seed:05d}", seed=seed, objects=objs, task=task)


def _hidden_for_property(prop, n, rng, half):
    for _ in range(1000):
        if prop in ("hollow", "solid"):
            odd = prop == "hollow"
            hs = [_random_hidden(rng, hollow=not odd) for _ in range(n)]
            k = int(rng.integers(n))
            hs[k] = _random_hidden(rng, hollow=odd)
            # contents of hollow objects must not rattle (that would be a different cue)
            for h in hs:
                if h.hollow and h.contents in ("marble", "beads"):
                    h.contents = "empty"
            return hs
        if prop == "rattles":
            hs = [_random_hidden(rng, hollow=True, contents=str(rng.choice(["empty", "foam", "weight"])))
                  for _ in range(n)]
            k = int(rng.integers(n))
            hs[k] = _random_hidden(rng, hollow=True, contents=str(rng.choice(["marble", "beads"])))
            return hs
        if prop in ("heaviest", "lightest"):
            hs = [_random_hidden(rng) for _ in range(n)]
            masses = sorted(_mass(h, half) for h in hs)
            # identifiable: the extreme object must differ from the next by at least 25%
            if prop == "heaviest" and masses[-1] > 1.25 * masses[-2]:
                return hs
            if prop == "lightest" and masses[1] > 1.25 * masses[0]:
                return hs
    raise RuntimeError(f"could not generate level for {prop}")


def _mass(h: Hidden, half: float = 0.055) -> float:
    return Obj("x", half, "", 0.0, h).mass
