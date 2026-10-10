"""Physical Factorio: build a factory on a grid, run it on MuJoCo, fix what jams, then hand it over.

Each turn the agent may `build` (place / remove / tune parts with a small build language), `run` the
factory for a while (costs line time; a prediction of the result is required), or `finish`. The
plant's physics is hidden: belt grip, pusher speed and delay, load-cell noise, and what the inlets
actually send. Final score: a production run of the agent's factory.
"""
from __future__ import annotations

import json
import re
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from ..core import Component, LogUniform, NoArgs, Observation, Pack, ParamSet, Scenario, Task, Tool, Uniform, World
from ..core.tool import Attachment, ToolContext
from . import factory_sim as fs

PRICES = {"belt": 20, "scale": 400, "pusher": 300, "bin": 60}
LIMITS = {"speed": (0.2, 2.5), "reject_below_g": (0, 3000), "reject_above_g": (0, 3000), "offset_mm": (0, 800),
          "stroke_ms": (50, 800)}
PARAMS = {"belt": ("speed",), "scale": ("speed", "reject_below_g", "reject_above_g"), "pusher": ("offset_mm", "stroke_ms"),
          "bin": ()}
DEFAULTS = {"belt": {"speed": 1.0}, "scale": {"speed": 1.0}, "pusher": {"offset_mm": 100, "stroke_ms": 250}, "bin": {}}
PRODUCTION_S = 90.0

RULES = f"""A factory floor of square cells, {fs.C} m each; x grows east, y grows north. Boxes ({fs.BOX * 100:.0f} cm cubes)
arrive from inlets (I) and must end up in the shipping bin (G). Map symbols: > < ^ v belts (arrow = flow),
W scale, P pusher, R reject bin, # wall, . empty.
Parts (price): belt ${PRICES['belt']} (speed m/s); scale ${PRICES['scale']} (a belt section with a load cell; boxes
whose reading is below reject_below_g or above reject_above_g are flagged); pusher ${PRICES['pusher']} (stands
beside a belt and pushes across it in its direction; fires when a flagged box is offset_mm before the paddle
line, stays out stroke_ms); reject bin ${PRICES['bin']}. Removing a part refunds half its price.
Belts carry boxes by friction: boxes need time to speed up or turn, so fast turns, merges and drop-offs can
throw boxes. Rails are fitted automatically on belt edges nothing feeds into, except the side a pusher pushes
through. A box falling to the floor is a spill. A box that sits still on a belt for 3 s is a jam. Bins
stand on platform scales: runs report each bin's count and total weight. A good box weighs about
{fs.NOMINAL_KG:.2f} kg; other boxes may be underfilled or double-packed.
Build language (one command per line; X or Y may be a range like 2..5):
  belt X Y DIR speed=<m/s>          DIR is E, W, N or S
  scale X Y DIR speed=<m/s> reject_below_g=<g> reject_above_g=<g>
  pusher X Y DIR offset_mm=<mm> stroke_ms=<ms>
  bin X Y
  set X Y key=value ...           retune a placed part
  remove X Y"""


class Plant(Component):
    kind = "plant"
    doc = "The factory floor. Hidden: belt grip, pusher dynamics, scale noise, and what the inlets send."
    params = ParamSet(
        visible={},
        hidden={"belt_mu": Uniform(0.3, 0.8, doc="box-belt traction"),
                "pusher_speed": Uniform(0.7, 1.8, unit="m/s", doc="paddle speed"),
                "pusher_lag": Uniform(0.02, 0.12, unit="s", doc="valve delay"),
                "scale_noise_g": LogUniform(2.0, 12.0, unit="g", doc="load-cell noise at 0.5 s dwell")},
    )


def _layout(world: World) -> fs.Layout:
    return fs.Layout.from_dict(world.settings["_layout"])


def _hidden(world: World) -> fs.Hidden:
    p = world.get("plant").hidden
    return fs.Hidden(belt_mu=p["belt_mu"], pusher_speed=p["pusher_speed"], pusher_lag=p["pusher_lag"],
                     scale_noise_g=p["scale_noise_g"], mixes=world.settings["_mixes"])


# ---------------------------------------------------------------------- build language
_RANGE = re.compile(r"^(-?\d+)(?:\.\.(-?\d+))?$")


def _span(tok: str) -> list[int]:
    m = _RANGE.match(tok)
    if not m:
        raise ValueError(f"bad coordinate {tok!r}")
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    return list(range(a, b + 1)) if a <= b else list(range(a, b - 1, -1))


def _kv(tokens: list[str], allowed: tuple[str, ...]) -> dict:
    out = {}
    for t in tokens:
        if "=" not in t:
            raise ValueError(f"expected key=value, got {t!r}")
        k, v = t.split("=", 1)
        if k not in allowed:
            raise ValueError(f"unknown setting {k!r} (allowed: {', '.join(allowed) or 'none'})")
        lo, hi = LIMITS[k]
        v = float(v)
        if not lo <= v <= hi:
            raise ValueError(f"{k}={v:g} outside {lo}..{hi}")
        out[k] = v
    return out


def apply_ops(lay: fs.Layout, ops: str, money: float) -> tuple[list[str], float]:
    """Apply build commands in order. Returns per-line results and money spent (negative = refund).
    Lines that fail change nothing and cost nothing."""
    log, spent = [], 0.0
    for raw in ops.strip().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        tok = line.split()
        cmd = tok[0].lower()
        try:
            if cmd in PRICES:
                need = 3 if cmd != "bin" else 3
                if len(tok) < need:
                    raise ValueError("usage: " + ("bin X Y" if cmd == "bin" else f"{cmd} X Y DIR key=value ..."))
                xs, ys = _span(tok[1]), _span(tok[2])
                d = "E" if cmd == "bin" else (tok[3].upper() if len(tok) > 3 else "")
                if d not in fs.DIRS:
                    raise ValueError(f"direction must be E, W, N or S, got {d!r}")
                prm = {**DEFAULTS[cmd], **_kv(tok[4 if cmd != "bin" else 3:], PARAMS[cmd])}
                cells = [(x, y) for x in xs for y in ys]
                for c in cells:
                    if not (0 <= c[0] < lay.w and 0 <= c[1] < lay.h):
                        raise ValueError(f"cell {c} is off the floor ({lay.w}x{lay.h})")
                    if c in lay.walls:
                        raise ValueError(f"cell {c} is a wall")
                    if c in lay.parts:
                        raise ValueError(f"cell {c} is taken by a {lay.parts[c].kind} (remove it first)")
                cost = PRICES[cmd] * len(cells)
                if spent + cost > money + 1e-9:
                    raise ValueError(f"not enough money: needs ${cost}, ${money - spent:.0f} left")
                for c in cells:
                    lay.parts[c] = fs.Part(cmd, c[0], c[1], d, dict(prm), name="reject" if cmd == "bin" else "")
                spent += cost
                log.append(f"ok: {line} (${cost})")
            elif cmd in ("remove", "set"):
                if len(tok) < 3:
                    raise ValueError(f"usage: {cmd} X Y" + (" key=value ..." if cmd == "set" else ""))
                cells = [(x, y) for x in _span(tok[1]) for y in _span(tok[2])]
                for c in cells:
                    p = lay.parts.get(c)
                    if p is None:
                        raise ValueError(f"nothing at {c}")
                    if p.fixed:
                        raise ValueError(f"the {p.kind} at {c} is part of the plant and cannot be changed")
                if cmd == "remove":
                    refund = sum(PRICES[lay.parts[c].kind] / 2 for c in cells)
                    for c in cells:
                        del lay.parts[c]
                    spent -= refund
                    log.append(f"ok: {line} (refund ${refund:.0f})")
                else:
                    for c in cells:
                        p = lay.parts[c]
                        p.params.update(_kv(tok[3:], PARAMS[p.kind]))
                    log.append(f"ok: {line}")
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except ValueError as e:
            log.append(f"FAILED: {line} -> {e}")
    return log, spent


def describe(lay: fs.Layout) -> dict:
    parts = [f"{p.kind}{'(' + p.name + ')' if p.name else ''} at {p.x},{p.y} dir {p.dir}"
             + (f" {json.dumps(p.params)}" if p.params else "") + (" [plant]" if p.fixed else "")
             for p in sorted(lay.parts.values(), key=lambda p: (p.y, p.x))]
    return {"map": lay.ascii(), "parts": parts}


# ---------------------------------------------------------------------- tools
class BuildArgs(BaseModel):
    model_config = {"extra": "forbid"}
    ops: str = Field(description="build commands, one per line (see the build language in the rules)")


class Build(Tool):
    name = "build"
    description = "Place, remove or retune parts. Costs money (see prices); costs no line time."
    Args = BuildArgs
    cost = {}

    def costs(self, args: BuildArgs | None = None) -> dict[str, float]:
        return {}  # money is checked and charged inside run (failed lines cost nothing)

    def run(self, ctx: ToolContext, target, args: BuildArgs) -> Observation:
        lay = _layout(ctx.world)
        money = ctx.world.settings["money"]
        log, spent = apply_ops(lay, args.ops, money)
        ctx.world.settings["money"] = money - spent
        ctx.world.settings["_layout"] = lay.to_dict()
        ctx.world.settings["state"] = {"money_left": round(money - spent), **describe(lay)}
        return Observation({"results": log, "money_left": round(money - spent), **describe(lay)})


class RunArgs(BaseModel):
    model_config = {"extra": "forbid"}
    seconds: float = Field(ge=10, le=90, description="how long to run the factory (s of line time)")


class RunFactory(Tool):
    name = "run"
    description = ("Run the factory from empty for `seconds`. Reports what reached each bin (count and total weight), "
                   "spills and jams by cell, every scale reading, pusher fires. Costs that much line time.")
    Args = RunArgs

    class Prediction(BaseModel):
        model_config = {"extra": "forbid"}
        shipped: float = Field(description="boxes that will land in the shipping bin during this run")
        spilled: float = Field(description="boxes that will fall to the floor")

    prediction_required = True
    prediction_targets = {"shipped": ("shipped", "boxes"), "spilled": ("spilled", "boxes")}
    cost = {"sim_seconds": 1.0}

    def costs(self, args: RunArgs | None = None) -> dict[str, float]:
        return {"sim_seconds": float(args.seconds) if args else 10.0}

    def run(self, ctx: ToolContext, target, args: RunArgs) -> Observation:
        lay = _layout(ctx.world)
        r = fs.run_factory(lay, _hidden(ctx.world), args.seconds, ctx.seed("run", ctx.step),
                           record_fps=12.0 if ctx.out_dir else 0.0)
        data = r.public()
        data["shipped"] = data["bins"].get("ship", {}).get("count", 0)
        data["spilled"] = sum(r.spills.values())
        att = save_replay(ctx.out_dir, f"run{ctx.step:02d}", lay, r, f"run at turn {ctx.step}")
        return Observation(data, attachments=[att] if att else [], truth=r.truth())


class Finish(Tool):
    name = "finish"
    description = f"Hand the factory over: a {PRODUCTION_S:.0f} s production run is scored. Ends the episode."
    Args = NoArgs
    cost = {}
    finishes = True

    def costs(self, args=None):
        return {}

    def run(self, ctx, target, args) -> Observation:
        return Observation({"handed_over": True})


def save_replay(out_dir, name: str, lay: fs.Layout, r: fs.RunReport, label: str) -> Attachment | None:
    if out_dir is None or not r.frames:
        return None
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps({"layout": lay.to_dict(), "cell": fs.C, "belt_z": fs.BZ, "box": fs.BOX,
                             "frames": r.frames, "report": r.public(), "truth": r.truth()},
                            separators=(",", ":"), default=float))
    return Attachment("trajectory", str(p), label)


# ---------------------------------------------------------------------- task
class BuildFactoryTask(Task):
    kind = "build_factory"

    def production(self, world: World, out_dir=None) -> tuple[fs.RunReport, fs.Layout]:
        lay = _layout(world)
        r = fs.run_factory(lay, _hidden(world), PRODUCTION_S, world.seed * 7919 + 17, record_fps=12.0 if out_dir else 0.0)
        return r, lay

    def checks(self, t: dict) -> dict:
        p = self.params
        return {"throughput": t["good_per_min"] >= p["target_good_per_min"],
                "quality": t["bad_shipped"] <= p["max_bad_shipped"],
                "no_spills": t["spilled"] <= p["max_spilled"],
                "no_jams": t["jammed"] == 0}

    def final(self, world: World, episode) -> dict:
        r, lay = self.production(world, episode.out_dir)
        t = r.truth()
        checks = self.checks(t)
        att = save_replay(episode.out_dir, "production", lay, r, "production run")
        score = t["good_shipped"] - 5 * t["bad_shipped"] - 2 * t["spilled"] - 3 * t["jammed"]
        return {"success": all(checks.values()), "score": score, "checks": checks, "production": t,
                "production_report": r.public(), "replay": att.path if att else None,
                "money_left": round(world.settings["money"]), "layout": lay.to_dict()}


# ---------------------------------------------------------------------- levels
def _I(x, y, d, speed, feed, name="A"):
    return fs.Part("inlet", x, y, d, {"speed": speed, "feed_per_min": feed}, True, name)


def _bin(x, y, name="ship"):
    return fs.Part("bin", x, y, "E", {}, True, name)


MAPS = {
    "first_line": dict(
        title="First line", w=12, h=8,
        brief="Get every box from the inlet to the shipping bin. Walls are in the way.",
        parts=[_I(0, 1, "E", 1.0, 40), _bin(11, 6)],
        walls=[(4, y) for y in range(0, 5)] + [(8, y) for y in range(3, 8)],
        mixes={"A": [["good", 1.0, 1.0, 0.01]]}, target=0.85, max_bad=0, money=1500),
    "quality_gate": dict(
        title="Quality gate", w=12, h=8,
        brief="Some boxes leave filling underfilled. Ship the good ones, keep the bad ones out of the shipping bin.",
        parts=[_I(0, 2, "E", 1.0, 40), _bin(11, 5)],
        walls=[(6, y) for y in range(0, 4)],
        mixes={"A": [["good", 0.7, 1.0, 0.012], ["underfilled", 0.3, 0.9, 0.015]]}, target=0.8, max_bad=0, money=2500),
    "two_lines": dict(
        title="Two lines, one dock", w=14, h=9,
        brief="Two filling lines feed one shipping dock. Line B sometimes double-packs. Merge both lines and ship only good boxes.",
        parts=[_I(0, 1, "E", 1.0, 30, "A"), _I(0, 7, "E", 1.0, 30, "B"), _bin(13, 4)],
        walls=[(5, y) for y in range(2, 7)],
        mixes={"A": [["good", 1.0, 1.0, 0.012]], "B": [["good", 0.75, 1.0, 0.012], ["double", 0.25, 1.9, 0.03]]},
        target=0.8, max_bad=0, money=3000),
    "rush_hour": dict(
        title="Rush hour", w=14, h=9,
        brief="A fast filler: boxes arrive at 1.8 m/s, 70 per minute. Get them around the walls to shipping without throwing any.",
        parts=[_I(0, 1, "E", 1.8, 70), _bin(13, 7)],
        walls=[(4, y) for y in range(0, 6)] + [(9, y) for y in range(3, 9)],
        mixes={"A": [["good", 1.0, 1.0, 0.01]]}, target=0.85, max_bad=0, money=1500, turns=10, sim_seconds=240),
    "tight_spec": dict(
        title="Tight spec", w=12, h=8,
        brief="A fast filler with a drifting nozzle: some boxes are only slightly underfilled. Nothing under spec may ship.",
        parts=[_I(0, 2, "E", 1.4, 70), _bin(11, 5)],
        walls=[(6, y) for y in range(0, 4)],
        mixes={"A": [["good", 0.7, 1.0, 0.006], ["underfilled", 0.3, 0.95, 0.006]]}, target=0.7, max_bad=0,
        money=2500, turns=10, sim_seconds=240),
    "mixed_merge": dict(
        title="Mixed merge", w=14, h=9,
        brief="Two fast lines into one dock. Line A sometimes underfills, line B sometimes double-packs. Only good boxes may ship.",
        parts=[_I(0, 1, "E", 1.4, 45, "A"), _I(0, 7, "E", 1.4, 45, "B"), _bin(13, 4)],
        walls=[(5, y) for y in range(2, 7)],
        mixes={"A": [["good", 0.8, 1.0, 0.01], ["underfilled", 0.2, 0.93, 0.01]],
               "B": [["good", 0.8, 1.0, 0.01], ["double", 0.2, 1.9, 0.03]]},
        target=0.75, max_bad=0, money=3500, turns=10, sim_seconds=240),
}


GOAL = ("{brief} Score: a {seconds:.0f} s production run must ship at least {target:g} good boxes per minute to "
        "the shipping bin, with at most {max_bad} bad boxes shipped, no spills and no jams. You have ${money} for parts.")


class FactoryPack(Pack):
    name = "factory"
    description = "Physical Factorio: build, run and fix a factory on a grid; MuJoCo decides what jams."
    rules = RULES
    components = [Plant]
    tools = [Build, RunFactory, Finish]
    tasks = [BuildFactoryTask]

    def generate(self, seed: int, map: str = "first_line", **options) -> Scenario:
        spec = MAPS[map]
        rng = np.random.default_rng(seed)
        lay = fs.Layout(spec["w"], spec["h"], [fs.Part.from_dict(p.to_dict()) for p in spec["parts"]], spec["walls"])
        feed_good = sum(p.params["feed_per_min"] * sum(s for k, s, *_ in spec["mixes"][p.name] if k == "good")
                        for p in lay.parts.values() if p.kind == "inlet")
        target = float(round(spec["target"] * feed_good))
        world = World([Plant.sample("plant", rng)], seed=seed,
                      settings={"_layout": lay.to_dict(), "money": float(spec["money"]), "_mixes": spec["mixes"],
                                "state": {"money_left": spec["money"], **describe(lay)}})
        params = {"map": map, "title": spec["title"], "target_good_per_min": target, "max_bad_shipped": spec["max_bad"],
                  "max_spilled": 0, "rules": RULES}
        goal = GOAL.format(brief=spec["brief"], seconds=PRODUCTION_S, target=target, max_bad=spec["max_bad"],
                           money=spec["money"])
        budgets = {"turns": spec.get("turns", 14), "sim_seconds": spec.get("sim_seconds", 360)}
        task = BuildFactoryTask(goal, budgets, ["build", "run", "finish"], None, params)
        return Scenario(f"F-{map}-{seed}", self.name, world, task)


FactoryPack.register_all()
