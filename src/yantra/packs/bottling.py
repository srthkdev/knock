"""Bottling line commissioning: a multi-turn factory game on real physics.

The agent is the commissioning engineer of a bottling line it has never seen. It may run test
batches (each costs line time), send a batch to the lab, and then must start production with one
setting. Every test batch is a MuJoCo run: belt traction, a load-cell checkweigher and a pneumatic
reject pusher. What is hidden is what a real engineer would not know: how grippy the belt is, how
fast and how late the pusher is, how noisy the load cell is, and how bad this lot of bottles is.

Real trade-offs, none of them scripted:
- faster belt + higher feed = more bottles per minute, but less time on the scale (noisier weights)
- bottles closer than the platform length corrupt each other's weight
- the pusher must fire early enough to meet a fast bottle, but a long stroke tips its neighbours
- a tight threshold rejects good bottles, a loose one ships underfilled ones
"""
from __future__ import annotations

import json
from typing import ClassVar

import numpy as np
from pydantic import BaseModel, Field

from ..core import Component, LogUniform, Observation, Pack, ParamSet, Scenario, Task, Tool, ToolCall, Uniform, World
from ..core.tool import Attachment, ToolContext
from . import bottling_sim as sim

RULES = f"""You are commissioning a bottling line. Bottles are fed onto a conveyor, cross a checkweigher
platform (x = {sim.WEIGH[0]}..{sim.WEIGH[1]} m along the line, a load cell under a belt section), and a
pneumatic pusher at x = {sim.PUSH_X} m knocks flagged bottles off into a reject chute. The line is {sim.LINE_LEN} m long.
Empty bottles weigh about {sim.GLASS_G:.0f} g (glass varies a little bottle to bottle); a full bottle holds
{sim.FULL_G:.0f} g of product. A bottle is DEFECTIVE if it holds less than {sim.SPEC_FILL:.0%} of nominal fill.
The checkweigher reports the gross weight. It flags a bottle when its reading is below `reject_below_g`.
The pusher fires when a flagged bottle is `pusher_offset_mm` before the pusher (encoder tracking), and
stays extended for `stroke_ms`. Bottle pitch on the belt = belt_speed * 60 / feed_per_min.
Everything else (belt grip, pusher speed and valve delay, load-cell noise, how many bottles are
underfilled and by how much) you must find out from test runs. Each test run costs line time
(sim_seconds). Lab audits tell you the true fill of every bottle in your most recent test run."""


class Settings(BaseModel):
    model_config = {"extra": "forbid"}
    belt_speed: float = Field(ge=0.2, le=2.0, description="conveyor speed, m/s")
    feed_per_min: float = Field(ge=20, le=150, description="bottles released onto the belt per minute")
    reject_below_g: float = Field(ge=550, le=700, description="checkweigher: flag bottles whose reading is below this (g)")
    pusher_offset_mm: float = Field(ge=0, le=300, description="fire when a flagged bottle is this far before the pusher (mm)")
    stroke_ms: float = Field(ge=40, le=500, description="how long the pusher stays extended (ms)")


class TestArgs(Settings):
    n_bottles: int = Field(ge=6, le=30, description="bottles in this test batch")


class Line(Component):
    kind = "bottling_line"
    doc = "Conveyor + checkweigher + reject pusher. Hidden: belt grip, pusher speed/delay, load-cell noise."
    params = ParamSet(
        visible={},
        hidden={"belt_mu": Uniform(0.25, 0.8, doc="belt-bottle traction coefficient"),
                "pusher_speed": Uniform(0.6, 1.8, unit="m/s", doc="paddle speed"),
                "pusher_lag": Uniform(0.02, 0.12, unit="s", doc="valve + solenoid delay"),
                "sensor_noise_g": LogUniform(0.8, 4.0, unit="g", doc="load-cell noise at 0.5 s dwell")},
    )

    def sim(self) -> sim.LineHidden:
        return sim.LineHidden(**self.hidden)


class Lot(Component):
    kind = "bottle_lot"
    doc = "The bottles coming down the line. Hidden: share underfilled, how underfilled, glass spread."
    params = ParamSet(
        visible={},
        hidden={"underfill_rate": Uniform(0.1, 0.4, doc="fraction of bottles underfilled"),
                "underfill_level": Uniform(0.92, 0.962, doc="fill fraction of underfilled bottles"),
                "glass_sd_g": Uniform(1.5, 5.0, unit="g", doc="spread of empty-bottle weight")},
    )

    def sim(self) -> sim.LotHidden:
        return sim.LotHidden(**self.hidden)


def _components(world: World) -> tuple[Line, Lot]:
    return world.get("line"), world.get("lot")


def _settings(args: BaseModel) -> sim.Settings:
    return sim.Settings(**{k: getattr(args, k) for k in Settings.model_fields})


def batch_seconds(s: Settings, n: int) -> float:
    return n * 60.0 / s.feed_per_min + sim.LINE_LEN / s.belt_speed + 3.0


def _rates(r: sim.RunResult, s: Settings) -> dict:
    """Steady-state rates: share of the batch times the feed rate (no warm-up/drain dilution)."""
    return {"shipped_per_min": round(r.shipped / r.n * s.feed_per_min, 1),
            "good_per_min": round(r.good_shipped / r.n * s.feed_per_min, 1)}


def _save_trajectory(ctx_dir, name: str, r: sim.RunResult, s: Settings, label: str) -> Attachment | None:
    if ctx_dir is None or not r.frames:
        return None
    p = ctx_dir / f"{name}.json"
    p.write_text(json.dumps({
        "line": {"length": sim.LINE_LEN, "belt_z": sim.BELT_Z, "weigh": list(sim.WEIGH), "push_x": sim.PUSH_X,
                 "bottle_r": sim.R, "bottle_h": sim.H},
        "settings": s.model_dump(), "frames": r.frames,
        "readings": {str(k): round(v, 1) for k, v in r.readings.items()},
        "fate": {str(k): v for k, v in r.fate.items()}, "defective": r.defective, "fill_g": r.fill_g,
    }, separators=(",", ":")))
    return Attachment("trajectory", str(p), label)


class TestRun(Tool):
    name = "test_run"
    description = ("Run a test batch at these settings. Returns per-bottle checkweigher readings and what happened to "
                   "each bottle (shipped / rejected / tipped), plus line rates. Costs line time: "
                   "n_bottles * 60 / feed_per_min + 5 / belt_speed + 3 sim_seconds.")
    Args = TestArgs

    class Prediction(BaseModel):
        model_config = {"extra": "forbid"}
        rejected: float = Field(description="how many bottles of this batch will end up in the reject chute")
        tipped: float = Field(description="how many will tip over or jam")

    prediction_required = True
    prediction_targets = {"rejected": ("rejected", "bottles"), "tipped": ("tipped_or_jammed", "bottles")}
    cost = {"sim_seconds": 1.0}
    record_fps: ClassVar[float] = 15.0

    def costs(self, args: TestArgs | None = None) -> dict[str, float]:
        if args is None:  # cheapest possible batch, for availability checks
            return {"sim_seconds": 6 * 60 / 150 + sim.LINE_LEN / 2.0 + 3.0}
        return {"sim_seconds": round(batch_seconds(args, args.n_bottles), 1)}

    def run(self, ctx: ToolContext, target, args: TestArgs) -> Observation:
        line, lot = _components(ctx.world)
        r = sim.run_line(line.sim(), lot.sim(), _settings(args), args.n_bottles, ctx.seed("test", ctx.step),
                         record_fps=self.record_fps if ctx.out_dir else 0.0)
        ctx.world.settings["last_test"] = {"fill_g": r.fill_g, "readings": r.readings}
        bottles = [{"bottle": b, "reading_g": round(r.readings[b], 1) if b in r.readings else None,
                    "flagged": b in r.flagged, "fate": r.fate[b]} for b in range(r.n)]
        data = {"batch": ctx.step, "settings": args.model_dump(), "pitch_m": round(args.belt_speed * 60 / args.feed_per_min, 3),
                "bottles": bottles, "rejected": r.rejected, "tipped_or_jammed": r.tipped, "shipped": r.shipped,
                "flagged": len(r.flagged), **{k: v for k, v in _rates(r, args).items() if k == "shipped_per_min"}}
        att = _save_trajectory(ctx.out_dir, f"run{ctx.step:02d}", r, args, f"test batch {ctx.step}")
        return Observation(data, attachments=[att] if att else [], truth={**r.truth(), **_rates(r, args)})


class LabAudit(Tool):
    name = "lab_audit"
    description = ("Send every bottle of your most recent test batch to the lab. Returns the true product fill of each "
                   "bottle (g of product, nominal 330) next to its checkweigher reading.")
    cost = {"audits": 1.0}

    def run(self, ctx: ToolContext, target, args) -> Observation:
        last = ctx.world.settings.get("last_test")
        if not last:
            return Observation({"error": "no test batch to audit yet"})
        rows = [{"bottle": b, "fill_g": f, "reading_g": round(last["readings"][b], 1) if b in last["readings"] else None}
                for b, f in enumerate(last["fill_g"])]
        return Observation({"bottles": rows, "spec_min_fill_g": round(sim.FULL_G * sim.SPEC_FILL, 1)})


class StartProduction(Tool):
    name = "start_production"
    description = ("Commit: run a production shift of 40 bottles at these settings. Scored on good bottles per minute, "
                   "defects shipped, good bottles rejected and bottles tipped. Ends the episode.")
    Args = Settings
    terminal = True
    n_bottles: ClassVar[int] = 40

    class Prediction(BaseModel):
        model_config = {"extra": "forbid"}
        good_per_min: float = Field(description="good (in-spec) bottles shipped per minute")
        defects_shipped: float = Field(description="underfilled bottles that will reach customers")

    prediction_required = True
    prediction_targets = {"good_per_min": ("good_per_min", "bottles/min"), "defects_shipped": ("defects_shipped", "bottles")}

    def outcome(self, world: World, target, args: Settings, record_dir=None) -> dict:
        line, lot = _components(world)
        r = sim.run_line(line.sim(), lot.sim(), _settings(args), self.n_bottles, world.seed * 7919 + 1,
                         record_fps=15.0 if record_dir else 0.0)
        out = {**r.truth(), **_rates(r, args), "shipped": r.shipped, "rejected": r.rejected, "bottles": r.n}
        att = _save_trajectory(record_dir, "production", r, args, "production shift")
        if att:
            out["trajectory"] = att.path
        return out


class CommissionTask(Task):
    kind = "commission_line"

    def _ok(self, o: dict) -> dict:
        p = self.params
        return {"throughput": o["good_per_min"] >= p["target_good_per_min"],
                "quality": o["defects_shipped"] <= p["max_defects_shipped"],
                "yield": o["good_rejected"] <= p["max_good_rejected"],
                "no_tips": o["tipped"] == 0}

    def success(self, world, call, outcome) -> bool:
        return all(self._ok(outcome).values())

    def loss(self, world, call, outcome) -> float:
        p = self.params
        return (max(0.0, p["target_good_per_min"] - outcome["good_per_min"]) / 10
                + max(0.0, outcome["defects_shipped"] - p["max_defects_shipped"])
                + max(0.0, outcome["good_rejected"] - p["max_good_rejected"]) + outcome["tipped"])

    def diagnostics(self, world, call, outcome) -> dict:
        return {"checks": self._ok(outcome)}


GOAL = ("Commission this bottling line, then start production. Production (40 bottles) must ship at least "
        "{target_good_per_min:g} good bottles per minute, ship at most {max_defects_shipped} underfilled bottles, "
        "reject at most {max_good_rejected} good bottles, and tip none.")


class BottlingPack(Pack):
    name = "bottling"
    description = "Commission a bottling line (conveyor, checkweigher, reject pusher) over many turns, then run production."
    rules = RULES
    components = [Line, Lot]
    tools = [TestRun, LabAudit, StartProduction]
    tasks = [CommissionTask]

    def generate(self, seed: int, target_good_per_min: float = 45, **options) -> Scenario:
        rng = np.random.default_rng(seed)
        world = World([Line.sample("line", rng), Lot.sample("lot", rng)], settings={}, seed=seed)
        params = {"target_good_per_min": float(target_good_per_min), "max_defects_shipped": 0, "max_good_rejected": 1,
                  "rules": RULES}
        task = CommissionTask(GOAL.format(**params), {"turns": 8, "sim_seconds": 240, "audits": 2},
                              ["test_run", "lab_audit"], "start_production", params)
        return Scenario(f"B{seed:05d}", self.name, world, task)


BottlingPack.register_all()
