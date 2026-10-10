"""Contract tests for knock.core using a tiny analytic pack (no MuJoCo)."""
import json
import math

import numpy as np
import pytest
from pydantic import BaseModel, Field

from knock.core import (COMPONENTS, TOOLS, Choice, Component, Episode, EventLog, Observation, Pack, ParamSet,
                        Scenario, Task, Tool, ToolCall, Uniform, World, action_schema, parse_action, run_episode,
                        solve)

G = 9.81


# ---------------------------------------------------------------------- toy pack
class Block(Component):
    kind = "toy_block"
    doc = "a block sliding on a table (analytic Coulomb friction)"
    params = ParamSet(visible={"color": Choice(options=("red", "blue"))},
                      hidden={"mass": Uniform(0.1, 2.0, unit="kg"), "mu": Uniform(0.1, 0.8)})

    def push_distance(self, force, seconds=0.4):
        m, mu = self.hidden["mass"], self.hidden["mu"]
        a = max(0.0, force / m - mu * G)
        v = a * seconds
        return a * seconds ** 2 / 2 + (v * v / (2 * mu * G) if a > 0 else 0.0)

    def flick_distance(self, impulse):
        v = impulse / self.hidden["mass"]
        return v * v / (2 * self.hidden["mu"] * G)


class PushArgs(BaseModel):
    model_config = {"extra": "forbid"}
    force_N: float = Field(ge=0.1, le=30, description="push force")


class PushPrediction(BaseModel):
    moved_mm: float = Field(description="how far you expect it to move")


class Push(Tool):
    name = "toy_push"
    description = "push with a constant force for 0.4 s"
    Args = PushArgs
    Prediction = PushPrediction
    prediction_targets = {"moved_mm": ("moved_mm", "mm")}
    targets = ("toy_block",)

    def run(self, ctx, target, args):
        noise = ctx.rng("toy_push", target.id, ctx.repeat("toy_push", target.id)).normal(0, 1.0)
        mm = max(0.0, target.push_distance(args.force_N) * 1000 + noise)
        return Observation({"moved_mm": round(mm, 2)})


class Weigh(Tool):
    name = "toy_weigh"
    description = "read the mass in grams"
    targets = ("toy_block",)

    def run(self, ctx, target, args):
        return Observation({"mass_g": round(target.hidden["mass"] * 1000, 1)})


class FlickArgs(BaseModel):
    model_config = {"extra": "forbid"}
    impulse_Ns: float = Field(ge=0.01, le=8, description="strike impulse")


class FlickPrediction(BaseModel):
    stop_x_m: float


class Flick(Tool):
    name = "toy_flick"
    description = "one strike; ends the episode"
    Args = FlickArgs
    Prediction = FlickPrediction
    prediction_required = True
    prediction_targets = {"stop_x_m": ("final_x", "m")}
    targets = ("toy_block",)
    terminal = True

    def outcome(self, world, target, args):
        return {"final_x": target.flick_distance(args.impulse_Ns)}


class SlideTo(Task):
    kind = "toy_slide_to"

    def answer_targets(self, world):
        return [self.params["object"]]

    def loss(self, world, call, outcome):
        return abs(outcome["final_x"] - self.params["target_x"]) + (0 if call.target == self.params["object"] else 100)

    def success(self, world, call, outcome):
        return call.target == self.params["object"] and abs(outcome["final_x"] - self.params["target_x"]) <= self.params["tol"]


class ToyPack(Pack):
    name = "toy"
    components = [Block]
    tools = [Push, Weigh, Flick]
    tasks = [SlideTo]

    def generate(self, seed, **options):
        rng = np.random.default_rng(seed)
        world = World([Block.sample(i, rng, pose={"lane": k}) for k, i in enumerate("AB")], seed=seed)
        task = SlideTo("slide B into the band", budget=2, probes=["toy_push", "toy_weigh"], act="toy_flick",
                       params={"object": "B", "target_x": 0.6, "tol": 0.03})
        return Scenario(f"toy{seed}", self.name, world, task)


ToyPack.register_all()


def scenario(seed=1):
    return ToyPack().generate(seed)


# ---------------------------------------------------------------------- params and components
def test_params_sample_validate_and_reject():
    ps = Block.params
    vis, hid = ps.sample(np.random.default_rng(0))
    assert set(hid) == {"mass", "mu"} and 0.1 <= hid["mass"] <= 2.0
    with pytest.raises(ValueError):
        ps.validate(vis, {**hid, "mass": 5.0})
    with pytest.raises(ValueError):
        ps.validate(vis, {"mass": 1.0})
    with pytest.raises(ValueError):
        ps.validate({**vis, "secret": 1}, hid)


def test_public_view_never_contains_hidden_values():
    sc = scenario()
    text = json.dumps(sc.public())
    for c in sc.world.components:
        for v in c.hidden.values():
            assert str(v) not in text


def test_world_roundtrip_and_duplicate_ids():
    sc = scenario()
    again = Scenario.from_json(sc.to_json())
    assert again.world.to_dict() == sc.world.to_dict() and again.task.to_dict() == sc.task.to_dict()
    with pytest.raises(ValueError):
        World([sc.world.get("A"), sc.world.get("A")])


def test_registry_reports_missing_capabilities():
    with pytest.raises(KeyError, match="missing component"):
        COMPONENTS.get("conveyor")
    assert "toy_push" in TOOLS


# ---------------------------------------------------------------------- tools and episodes
def test_tool_spec_and_bounds():
    spec = Push.spec()
    assert spec["prediction"]["properties"]["moved_mm"] and spec["targets"] == ["toy_block"]
    assert Push.arg_bounds() == {"force_N": (0.1, 30.0)}


def test_episode_budget_validation_and_prediction_scoring(tmp_path):
    sc = scenario()
    ep = Episode(sc.world, sc.task, tmp_path)
    assert "error" in ep.call(ToolCall("toy_push", "Z", {"force_N": 3})).data          # unknown target
    assert "error" in ep.call(ToolCall("toy_push", "B", {"force_N": 300})).data        # out of range
    assert "error" in ep.call(ToolCall("toy_jump", "B")).data                          # no such tool
    assert ep.used == 0
    b = sc.world.get("B")
    true_mm = b.push_distance(3) * 1000
    obs = ep.call(ToolCall("toy_push", "B", {"force_N": 3}, prediction={"moved_mm": true_mm + 10}))
    assert abs(obs.data["prediction_error"]["moved_mm"]["error"] - 10) < 5
    ep.call(ToolCall("toy_weigh", "B"))
    assert ep.budget_left == 0
    assert "budget exhausted" in ep.call(ToolCall("toy_weigh", "A")).data["error"]
    assert "requires a prediction" in ep.call(ToolCall("toy_flick", "B", {"impulse_Ns": 1})).data["error"]
    J = b.hidden["mass"] * math.sqrt(2 * b.hidden["mu"] * G * 0.6)
    ep.call(ToolCall("toy_flick", "B", {"impulse_Ns": J}, prediction={"stop_x_m": 0.5}))
    r = ep.result
    assert r["success"] and r["correct_target"] and abs(r["prediction"]["stop_x_m"]["error"] - 0.1) < 1e-6
    assert len(r["probe_predictions"]) == 1
    kinds = [e.kind for e in ep.events.events]
    assert kinds[0] == "episode_start" and kinds[-2:] == ["outcome", "episode_end"]
    assert ep.save() and json.loads((tmp_path / "trace.json").read_text())["result"]["success"]
    with pytest.raises(RuntimeError):
        ep.call(ToolCall("toy_weigh", "A"))


def test_probe_noise_does_not_depend_on_probe_order():
    sc = scenario()
    e1 = Episode(sc.world, sc.task)
    e1.call(ToolCall("toy_weigh", "A"))
    a = e1.call(ToolCall("toy_push", "B", {"force_N": 3})).data["moved_mm"]
    e2 = Episode(sc.world, sc.task)
    b = e2.call(ToolCall("toy_push", "B", {"force_N": 3})).data["moved_mm"]
    assert a == b


def test_events_stream_to_subscribers():
    sc = scenario()
    seen = []
    log = EventLog()
    log.subscribe(lambda ev: seen.append(ev.kind))
    ep = Episode(sc.world, sc.task, events=log)
    ep.call(ToolCall("toy_weigh", "A"))
    assert seen == ["episode_start", "tool_call", "observation"]


# ---------------------------------------------------------------------- agent-facing schema
def test_action_schema_is_strict_and_parses_back():
    tools = [Push, Weigh, Flick]
    s = action_schema(tools, ["A", "B"])
    assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
    for sub in ("args", "prediction"):
        assert s["properties"][sub]["additionalProperties"] is False
        assert set(s["properties"][sub]["required"]) == set(s["properties"][sub]["properties"])
    reply = {"tool": "toy_push", "target": "B", "args": {"force_N": 3, "impulse_Ns": None},
             "prediction": {"moved_mm": 120, "stop_x_m": None}, "reasoning": "measure friction"}
    call = parse_action(reply, tools)
    assert call.args == {"force_N": 3} and call.prediction == {"moved_mm": 120}
    weigh = parse_action({"tool": "toy_weigh", "target": "A", "args": {"force_N": 5}, "prediction": {}}, tools)
    assert weigh.args == {} and weigh.prediction is None


# ---------------------------------------------------------------------- oracle and agents
def test_generic_oracle_solves_without_task_specific_code():
    for seed in range(5):
        sc = scenario(seed)
        res = solve(sc.world, sc.task)
        assert res.success and res.call.target == "B", (seed, res)


class ScriptedToy:
    name = "toy-scripted"

    def act(self, view):
        done = [h["call"]["tool"] for h in view.history]
        if "toy_weigh" not in done:
            return ToolCall("toy_weigh", "B")
        if "toy_push" not in done:
            return ToolCall("toy_push", "B", {"force_N": 20}, prediction={"moved_mm": 500})
        m = next(h["observation"]["mass_g"] for h in view.history if "mass_g" in h["observation"]) / 1000
        mm = next(h["observation"]["moved_mm"] for h in view.history if "moved_mm" in h["observation"])
        lo, hi = 0.05, 1.0  # solve the push formula for mu by bisection
        for _ in range(60):
            mu = (lo + hi) / 2
            blk = Block("x", {"color": "red"}, {"mass": m, "mu": mu})
            lo, hi = (mu, hi) if blk.push_distance(20) * 1000 > mm else (lo, mu)
        J = m * math.sqrt(2 * mu * G * view.task["target_x"])
        return ToolCall("toy_flick", "B", {"impulse_Ns": J}, prediction={"stop_x_m": view.task["target_x"]})


class Stubborn:
    name = "toy-stubborn"

    def act(self, view):
        return ToolCall("toy_dance", "B")


def test_run_episode_drives_any_agent():
    sc = scenario(3)
    r = run_episode(ScriptedToy(), Episode(sc.world, sc.task))
    assert r["success"], r
    r2 = run_episode(Stubborn(), Episode(sc.world, sc.task))
    assert not r2["success"] and "error" in r2
