"""The tabletop pack on knock.core must match Knock v0.2 behaviour."""
import json

import numpy as np

import knock.packs  # noqa: F401  (registers the pack)
from knock.agents.tabletop_bayes import TabletopBayesAgent
from knock.core import Episode, Scenario, ToolCall, action_schema, run_episode, solve
from knock.env import Episode as LegacyEpisode
from knock.oracle import valid_level
from knock.packs.tabletop import Cube, Flick, Push, TabletopPack, scenario_from_level


def test_conversion_roundtrip_and_no_leak():
    lv = valid_level(3)
    sc = scenario_from_level(lv)
    again = Scenario.from_json(sc.to_json())
    assert again.world.to_dict() == sc.world.to_dict()
    for c, o in zip(sc.world.components, lv.objects):
        assert abs(c.mass - o.mass) < 1e-12
    text = json.dumps(sc.public()).lower()
    for key in ("material", "hollow", "contents", "friction", "oracle"):
        assert f'"{key}":' not in text  # no hidden key appears as a field
    for c in sc.world.components:
        assert str(c.hidden["friction"]) not in text  # and no hidden value leaks into any text


def test_sounds_match_legacy_episode(tmp_path):
    lv = valid_level(5)
    sc = scenario_from_level(lv)
    new = Episode(sc.world, sc.task).call(ToolCall("tap", "B")).data["sound"]
    old_ep = LegacyEpisode(lv, tmp_path, render=False)
    old_ep.reset()
    old = old_ep.step({"type": "probe", "probe": "tap", "object": "B"})["sound"]
    assert new == old


def test_push_and_flick_require_and_score_predictions():
    sc = scenario_from_level(valid_level(2))
    ep = Episode(sc.world, sc.task)
    assert "requires a prediction" in ep.call(ToolCall("push", "A", {"force_N": 3})).data["error"]
    obs = ep.call(ToolCall("push", "A", {"force_N": 3}, prediction={"moved_mm": 50}))
    assert "prediction_error" in obs.data and ep.used == 1
    oracle = solve(sc.world, sc.task)
    ep.call(ToolCall("flick", oracle.call.target, oracle.call.args, prediction={"stop_x_m": sc.task.params["target_x"]}))
    assert ep.result["success"] and ep.result["prediction"]["stop_x_m"]["error"] < sc.task.params["tol"]


def test_generic_oracle_agrees_with_legacy_oracle():
    for seed in (0, 1, 4):
        lv = valid_level(seed)
        res = solve(*(lambda s: (s.world, s.task))(scenario_from_level(lv)))
        assert res.success and res.call.target == lv.notes["oracle"]["object"]


def test_scripted_bayes_agent_plays_through_core():
    sc = scenario_from_level(valid_level(9))
    r = run_episode(TabletopBayesAgent("scripted"), Episode(sc.world, sc.task))
    assert "success" in r and r["probes_used"] <= sc.task.budget
    assert r["prediction"]["stop_x_m"]["error"] >= 0


def test_llm_schema_covers_pack_tools():
    s = action_schema(TabletopPack.tools, ["A", "B", "C"])
    assert set(s["properties"]["tool"]["enum"]) == {"push", "lift", "tap", "shake", "flick"}
    assert {"force_N", "impulse_Ns"} <= set(s["properties"]["args"]["properties"])
    assert {"moved_mm", "stop_x_m"} <= set(s["properties"]["prediction"]["properties"])
    assert Push.prediction_required and Flick.prediction_required and Cube.kind == "cube"


def test_llm_prompt_is_generic_and_leak_free():
    from knock.agents.llm import build_prompt
    sc = scenario_from_level(valid_level(11))
    ep = Episode(sc.world, sc.task)
    ep.call(ToolCall("lift", "A"))
    prompt = build_prompt(ep.view(), [])
    for name in ("push", "lift", "tap", "shake", "flick", "REQUIRED", "DECISIVE ACT"):
        assert name in prompt
    for c in sc.world.components:
        assert str(c.hidden["friction"]) not in prompt
        assert f'"material": "{c.hidden["material"]}"' not in prompt
