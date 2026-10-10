"""Factory pack: build language, physics sanity, no hidden-info leaks, reference solvability."""
import json
from pathlib import Path

import yantra.packs  # noqa: F401
from yantra.agents.llm import build_prompt
from yantra.core import Episode, Scenario, ToolCall
from yantra.packs import FactoryPack
from yantra.packs import factory_sim as fs
from yantra.packs.factory import apply_ops


def line(speed=1.0, below=0.0):
    parts = [fs.Part("inlet", 0, 1, "E", {"speed": speed, "feed_per_min": 40}, True, "A"),
             fs.Part("bin", 6, 1, "E", {}, True, "ship")]
    lay = fs.Layout(8, 4, parts)
    ops = f"belt 1..2 1 E speed={speed}\nscale 3 1 E speed={speed}" + (f" reject_below_g={below}" if below else "") + \
          f"\nbelt 4..5 1 E speed={speed}"
    log, spent = apply_ops(lay, ops, 1e9)
    assert all(l.startswith("ok") for l in log), log
    return lay


def test_build_language_validates_and_charges():
    lay = fs.Layout(6, 4, [fs.Part("inlet", 0, 1, "E", {"speed": 1, "feed_per_min": 30}, True, "A")], [(3, 1)])
    log, spent = apply_ops(lay, "belt 1..2 1 E speed=1.0\nbelt 3 1 E\nscale 0 1 E\nbelt 4 1 Q\nbin 4 2", 1000)
    assert log[0].startswith("ok") and spent == 40 + 60
    assert "wall" in log[1] and "taken" in log[2] and "direction" in log[3] and log[4].startswith("ok")
    log, spent = apply_ops(lay, "remove 0 1\nremove 1 1", 1000)
    assert "cannot be changed" in log[0] and spent == -10


def test_load_cell_reads_true_weight():
    lay = line()
    hid = fs.Hidden(scale_noise_g=1e-6, mixes={"A": [["good", 0.5, 1.0, 1e-6], ["double", 0.5, 1.9, 1e-6]]})
    r = fs.run_factory(lay, hid, 20, seed=3)
    assert r.scales["3,1"] and all(abs(x - 1000) < 3 or abs(x - 1900) < 3 for x in r.scales["3,1"])


def test_deterministic_and_conserves_items():
    lay = line(1.2)
    hid = fs.Hidden(mixes={"A": [["good", 1.0, 1.0, 0.01]]})
    a, b = fs.run_factory(lay, hid, 30, seed=7), fs.run_factory(lay, hid, 30, seed=7)
    assert a.public() == b.public()
    total = sum(v["count"] for v in a.bins.values()) + sum(a.spills.values()) + a.in_transit
    assert total == sum(a.emitted.values())


def test_prompt_hides_hidden_physics():
    sc = FactoryPack().generate(1, map="quality_gate")
    prompt = build_prompt(Episode(sc.world, sc.task).view(), [])
    for secret in ("belt_mu", "pusher_lag", "pusher_speed", "scale_noise", "_mixes", "_layout"):
        assert secret not in prompt
    for v in sc.world.get("plant").hidden.values():
        assert f"{v:.4f}" not in prompt


def test_generated_levels_are_solved_by_their_reference():
    files = sorted(Path("levels/factory").glob("F-gen*.json"))[:2]
    for f in files:
        sc = Scenario.from_json(f.read_text())
        ep = Episode(sc.world, sc.task)
        ep.world.settings["money"] = 1e9
        ep.call(ToolCall("build", args={"ops": sc.notes["reference"]["ops"]}))
        ep.call(ToolCall("finish"))
        assert ep.result["production"]["bad_shipped"] == 0 and ep.result["production"]["spilled"] == 0
        assert ep.result["production"]["good_per_min"] >= sc.task.params["target_good_per_min"]
