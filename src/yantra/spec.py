"""LevelSpec: the structured form an LLM fills in to describe a cube-lab level.

The spec mirrors `yantra.level.Level` but only allows values from the catalogue
(materials, contents, colours, size / friction / target ranges). Pydantic validators
enforce those limits; `LevelSpec.to_level` / `LevelSpec.from_level` convert both ways.
`strict_json_schema()` exports a schema accepted by OpenAI structured outputs (and so
by `codex exec --output-schema`): every field required, no additional properties.
"""
from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .level import LANE_SPACING, Hidden, Level, Obj, Task
from .materials import CONTENTS, MATERIALS

# ---------------------------------------------------------------- catalogue
COLORS = {  # name -> hex; the same palette as the random generator
    "red": "#c8553d",
    "blue": "#2f6690",
    "ochre": "#e0a458",
    "green": "#5b8e7d",
    "purple": "#8d6a9f",
    "cream": "#d8d4c7",
}
HEX_TO_COLOR = {v: k for k, v in COLORS.items()}
EDGE_RANGE = (0.08, 0.14)  # cube edge length, m (half edge 0.04-0.07)
FRICTION_RANGE = (0.15, 0.6)  # sliding friction against the table
TARGET_RANGE = (0.1, 2.5)  # target band centre, m
TOL_RANGE = (0.02, 0.15)  # target band half width, m
BUDGET_RANGE = (0, 10)
N_OBJECTS = (2, 6)
IDS = ("A", "B", "C", "D", "E", "F")
RATTLING = ("marble", "beads")

ObjectId = Literal["A", "B", "C", "D", "E", "F"]
ColorName = Literal["red", "blue", "ochre", "green", "purple", "cream"]
Material = Literal["metal", "glass", "plastic", "wood"]
Contents = Literal["empty", "marble", "beads", "foam", "weight"]
Property = Literal["hollow", "solid", "rattles", "heaviest", "lightest"]

assert set(Material.__args__) == set(MATERIALS)
assert set(Contents.__args__) == set(CONTENTS)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectSpec(_Strict):
    id: ObjectId = Field(description="object label; objects are labelled A, B, C... in order")
    edge_m: float = Field(ge=EDGE_RANGE[0], le=EDGE_RANGE[1], description="visible cube edge length in metres")
    color: ColorName = Field(description="visible colour")
    material: Material = Field(description="hidden: what the cube is made of")
    hollow: bool = Field(description="hidden: hollow shell (3 mm wall) or solid block")
    contents: Contents = Field(description="hidden: what is inside a hollow cube; must be 'empty' for solid cubes")
    friction: float = Field(ge=FRICTION_RANGE[0], le=FRICTION_RANGE[1],
                            description="hidden: sliding friction coefficient against the table")

    @model_validator(mode="after")
    def _solid_has_no_contents(self):
        if not self.hollow and self.contents != "empty":
            raise ValueError(f"object {self.id} is solid, so its contents must be 'empty' (got {self.contents!r})")
        return self


class TaskSpec(_Strict):
    kind: Literal["slide", "find_and_slide"] = Field(
        description="slide: flick a named object; find_and_slide: find the one object with a hidden property, then flick it")
    object_id: ObjectId | None = Field(description="slide only: which object to flick; null for find_and_slide")
    property: Property | None = Field(description="find_and_slide only: the identifying property; null for slide")
    target_x: float = Field(ge=TARGET_RANGE[0], le=TARGET_RANGE[1], description="centre of the target band, m")
    tol: float = Field(ge=TOL_RANGE[0], le=TOL_RANGE[1], description="half width of the target band, m")
    budget: int = Field(ge=BUDGET_RANGE[0], le=BUDGET_RANGE[1], description="probes allowed before the single flick")

    @model_validator(mode="after")
    def _kind_fields(self):
        if self.kind == "slide":
            if self.object_id is None or self.property is not None:
                raise ValueError("a slide task needs object_id set and property null")
        else:
            if self.property is None or self.object_id is not None:
                raise ValueError("a find_and_slide task needs property set and object_id null")
        return self


class LevelSpec(_Strict):
    title: str = Field(description="short human title for the level")
    description: str = Field(description="one or two sentences: the idea of the level and how a careful agent solves it")
    objects: list[ObjectSpec] = Field(min_length=N_OBJECTS[0], max_length=N_OBJECTS[1])
    task: TaskSpec

    @model_validator(mode="after")
    def _consistent(self):
        ids = [o.id for o in self.objects]
        want = list(IDS[: len(ids)])
        if ids != want:
            raise ValueError(f"objects must be labelled {want} in order (got {ids})")
        if self.task.object_id is not None and self.task.object_id not in ids:
            raise ValueError(f"task.object_id {self.task.object_id!r} is not one of the objects {ids}")
        if self.task.target_x - self.task.tol <= 0:
            raise ValueError("the target band must lie entirely at x > 0")
        return self

    # ------------------------------------------------------------ conversion
    def to_level(self, level_id: str = "custom", seed: int = 0, notes: dict | None = None) -> Level:
        n = len(self.objects)
        ys = [(i - (n - 1) / 2) * LANE_SPACING for i in range(n)]
        objs = [Obj(id=o.id, half=round(o.edge_m / 2, 4), color=COLORS[o.color], y=y,
                    hidden=Hidden(material=o.material, hollow=o.hollow, contents=o.contents, friction=o.friction))
                for o, y in zip(self.objects, ys)]
        t = self.task
        task = Task(kind=t.kind, target_x=t.target_x, tol=t.tol, budget=t.budget,
                    object_id=t.object_id, property=t.property)
        lv_notes = {"title": self.title, "description": self.description, **(notes or {})}
        return Level(id=level_id, seed=seed, objects=objs, task=task, notes=lv_notes)

    @classmethod
    def from_level(cls, level: Level) -> "LevelSpec":
        objs = []
        for o in level.objects:
            if o.color not in HEX_TO_COLOR:
                raise ValueError(f"colour {o.color!r} of object {o.id} is not in the catalogue")
            h = o.hidden
            objs.append(dict(id=o.id, edge_m=round(2 * o.half, 4), color=HEX_TO_COLOR[o.color], material=h.material,
                             hollow=h.hollow, contents=h.contents, friction=h.friction))
        t = level.task
        return cls.model_validate(dict(
            title=level.notes.get("title", level.id), description=level.notes.get("description", ""),
            objects=objs,
            task=dict(kind=t.kind, object_id=t.object_id, property=t.property, target_x=t.target_x,
                      tol=t.tol, budget=t.budget)))


# ---------------------------------------------------------------- schema export
_DROP = {"title", "default"}


def strict_json_schema() -> dict:
    """JSON schema for LevelSpec in the strict subset used by OpenAI structured outputs.

    Every object lists all its properties as required and sets additionalProperties
    false; nullable fields are `anyOf [..., null]`. Pydantic-only keys are removed.
    """
    schema = copy.deepcopy(LevelSpec.model_json_schema())

    def walk(node):
        if isinstance(node, dict):
            for k in list(node):
                if k in _DROP and not isinstance(node[k], dict):
                    del node[k]
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            # a $ref may not carry sibling keywords in strict mode
            if "$ref" in node:
                for k in list(node):
                    if k != "$ref":
                        del node[k]
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema
