"""Tabletop pack: identical-looking cubes on a table with hidden mass, friction, material and contents.

This is Knock v0.2's world expressed with knock.core. The physics is the validated MuJoCo code in
knock.world and the sound engine in knock.sound; this module only adapts it to Components, Tools
and Tasks. Legacy levels (knock.level.Level) convert losslessly with `scenario_from_level`.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from .. import sound, world as sim
from ..core import (Attachment, Choice, Component, Flag, Observation, Pack, ParamSet, Scenario, Task, Tool,
                    ToolCall, Uniform, World)
from ..level import COLORS, FLICK_RANGE, LANE_SPACING, PROPERTIES, PUSH_RANGE, Hidden, Level, Obj
from ..materials import CONTENTS, MATERIALS

PUSH_SECONDS = 0.4

RULES = f"""\
Objects are cubes resting on a flat table. They start at x = 0 in separate lanes and only move along +x.
Cubes in a level may look identical, but their hidden material, mass, inside (hollow or solid, loose or
fixed contents) and friction against the table can differ. Probes cost 1 from the budget each, and objects
are put back at x = 0 afterwards. Gravity is 9.81 m/s^2; friction behaves roughly like Coulomb friction.
Predict before you act: pushes and the flick require a numeric prediction of their result. Predictions are
scored separately from success; they measure how well you understand this world."""


# ---------------------------------------------------------------------- component
class Cube(Component):
    kind = "cube"
    doc = "A cube resting on the table. May be hollow (thin walls) and may contain loose or fixed contents."
    params = ParamSet(
        visible={"edge_m": Uniform(0.08, 0.14, unit="m", doc="edge length"),
                 "color": Choice(options=tuple(COLORS), doc="paint colour (says nothing about the inside)")},
        hidden={"material": Choice(options=tuple(MATERIALS), doc="shell material"),
                "hollow": Flag(doc="thin-walled shell instead of a solid block"),
                "contents": Choice(options=tuple(CONTENTS), doc="what is inside a hollow cube"),
                "friction": Uniform(0.15, 0.6, doc="sliding friction against the table")},
    )

    def obj(self) -> Obj:
        """Legacy simulation object used by knock.world."""
        h = self.hidden
        return Obj(self.id, self.visible["edge_m"] / 2, self.visible["color"], self.pose.get("y", 0.0),
                   Hidden(h["material"], bool(h["hollow"]), h["contents"] if h["hollow"] else "empty", h["friction"]))

    @property
    def mass(self) -> float:
        return self.obj().mass


# ---------------------------------------------------------------------- tools
def _sound_seed(ctx, probe: str, target: Cube) -> int:
    """Same seeds as Knock v0.2 so sounds match the legacy benchmark exactly."""
    from ..env import probe_seed
    return probe_seed(_LegacySeed(ctx.world.seed), target.id, probe, ctx.repeat(probe, target.id))


class _LegacySeed:
    def __init__(self, seed):
        self.seed = seed


class PushArgs(BaseModel):
    model_config = {"extra": "forbid"}
    force_N: float = Field(ge=PUSH_RANGE[0], le=PUSH_RANGE[1], description="constant horizontal force along +x")


class PushPrediction(BaseModel):
    moved_mm: float = Field(ge=0, description="how far you expect the cube to move, in mm")


class Push(Tool):
    name = "push"
    description = (f"Push a cube with a constant horizontal force along +x for {PUSH_SECONDS} s, then release it. "
                   "Reports how far it moved in mm (about 1 mm noise). Static friction can stop it from moving.")
    Args = PushArgs
    Prediction = PushPrediction
    prediction_required = True
    prediction_targets = {"moved_mm": ("moved_mm", "mm")}
    targets = ("cube",)

    def run(self, ctx, target, args):
        k = ctx.repeat(self.name, target.id)
        disp = sim.simulate_push(target.obj(), args.force_N, PUSH_SECONDS, seed=k)
        mm = max(0.0, disp * 1000 + ctx.rng(self.name, target.id, k).normal(0, 1.0))
        return Observation({"force_N": round(args.force_N, 3), "moved_mm": round(mm, 1)})


class Lift(Tool):
    name = "lift"
    description = "Lift a cube onto a scale. Reports its weight in grams (about 2% noise)."
    targets = ("cube",)

    def run(self, ctx, target, args):
        return Observation({"weight_g": sim.weigh(target.obj(), ctx.rng(self.name, target.id, ctx.repeat(self.name, target.id)))})


class _SoundProbe(Tool):
    targets = ("cube",)
    simulate: ClassVar = None

    def run(self, ctx, target, args):
        x, feats = type(self).simulate(target.obj(), seed=_sound_seed(ctx, self.name, target))
        atts = []
        if ctx.out_dir is not None:
            stem = ctx.out_dir / f"{ctx.step:02d}_{self.name}_{target.id}"
            sound.write_wav(x, stem.with_suffix(".wav"), gain=73.0)
            sound.save_spectrogram(x, stem.with_suffix(".png"), f"{self.name} {target.id}")
            atts = [Attachment("image", str(stem.with_suffix(".png")), f"spectrogram of {self.name} {target.id}"),
                    Attachment("audio", str(stem.with_suffix(".wav")), f"{self.name} {target.id}")]
        return Observation({"sound": feats}, atts)


class Tap(_SoundProbe):
    name = "tap"
    description = ("Strike the side of a cube with a small steel ball. Returns sound features (pitch, ring time, "
                   "brightness, loudness, distinct hits) and a spectrogram image.")
    simulate = staticmethod(sim.simulate_tap)


class Shake(_SoundProbe):
    name = "shake"
    description = "Pick a cube up and shake it. Returns sound features and a spectrogram image."
    simulate = staticmethod(sim.simulate_shake)


class FlickArgs(BaseModel):
    model_config = {"extra": "forbid"}
    impulse_Ns: float = Field(ge=FLICK_RANGE[0], le=FLICK_RANGE[1], description="instantaneous strike at the centre, along +x")


class FlickPrediction(BaseModel):
    stop_x_m: float = Field(description="x position (metres) where you expect the cube's centre to stop")


class Flick(Tool):
    name = "flick"
    description = "Your one decisive strike on a cube. Ends the episode; scored by where the cube stops."
    Args = FlickArgs
    Prediction = FlickPrediction
    prediction_required = True
    prediction_targets = {"stop_x_m": ("final_x", "m")}
    targets = ("cube",)
    terminal = True

    def outcome(self, world, target, args):
        return {"final_x": round(sim.simulate_flick(target.obj(), args.impulse_Ns), 5)}


# ---------------------------------------------------------------------- tasks
class _BandTask(Task):
    def _band_error(self, outcome) -> float:
        return outcome["final_x"] - self.params["target_x"]

    def loss(self, world, call, outcome):
        wrong = 0.0 if call.target in self.answer_targets(world) else 100.0
        return abs(self._band_error(outcome)) + wrong

    def success(self, world, call, outcome):
        return call.target in self.answer_targets(world) and abs(self._band_error(outcome)) <= self.params["tol"]

    def diagnostics(self, world, call, outcome):
        return {**super().diagnostics(world, call, outcome), "error_m": round(self._band_error(outcome), 4)}


class SlideTask(_BandTask):
    kind = "tabletop_slide"

    def answer_targets(self, world):
        return [self.params["object"]]


class FindAndSlideTask(_BandTask):
    kind = "tabletop_find_and_slide"

    def answer_targets(self, world):
        prop = self.params["property"]
        cubes = [c for c in world.components if c.kind == "cube"]
        if prop == "hollow":
            return [c.id for c in cubes if c.hidden["hollow"]]
        if prop == "solid":
            return [c.id for c in cubes if not c.hidden["hollow"]]
        if prop == "rattles":
            return [c.id for c in cubes if c.hidden["hollow"] and c.hidden["contents"] in ("marble", "beads")]
        if prop == "heaviest":
            return [max(cubes, key=lambda c: c.mass).id]
        if prop == "lightest":
            return [min(cubes, key=lambda c: c.mass).id]
        raise ValueError(prop)


# ---------------------------------------------------------------------- pack
class TabletopPack(Pack):
    name = "tabletop"
    description = "Identical-looking cubes hide mass, friction, material and contents; probe, then one flick."
    rules = RULES
    components = [Cube]
    tools = [Push, Lift, Tap, Shake, Flick]
    tasks = [SlideTask, FindAndSlideTask]

    def generate(self, seed: int, **options) -> Scenario:
        from ..oracle import valid_level
        return scenario_from_level(valid_level(seed, **options))


TabletopPack.register_all()


def scenario_from_level(level: Level) -> Scenario:
    """Convert a Knock v0.2 level (same physics, same sound seeds)."""
    comps = []
    for o in level.objects:
        h = o.hidden
        comps.append(Cube(o.id, {"edge_m": round(2 * o.half, 4), "color": o.color},
                          {"material": h.material, "hollow": h.hollow, "contents": h.contents, "friction": h.friction},
                          pose={"y": o.y, "x": 0.0}))
    t = level.task
    band = [round(t.target_x - t.tol, 3), round(t.target_x + t.tol, 3)]
    common = {"target_x": t.target_x, "tol": t.tol, "target_band_x": band, "rules": RULES}
    probes = ["push", "lift", "tap", "shake"]
    if t.kind == "slide":
        goal = f"Flick cube {t.object_id} so its centre stops between x = {band[0]} m and x = {band[1]} m."
        task = SlideTask(goal, t.budget, probes, "flick", {**common, "object": t.object_id})
    else:
        goal = (f"Exactly one cube {PROPERTIES[t.property]}. Find it and flick it so its centre stops between "
                f"x = {band[0]} m and x = {band[1]} m.")
        task = FindAndSlideTask(goal, t.budget, probes, "flick", {**common, "property": t.property})
    return Scenario(level.id, TabletopPack.name, World(comps, {"lane_spacing_m": LANE_SPACING}, seed=level.seed),
                    task, notes=dict(level.notes))
