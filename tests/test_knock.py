import json

import mujoco
import numpy as np
import pytest

from knock import world
from knock.agents.baselines import HOLLOW_PITCH_HZ
from knock.env import Episode
from knock.level import Hidden, Obj, generate, public_view
from knock.materials import MATERIALS
from knock.oracle import solve, valid_level

G = 9.81


def cube(material="wood", hollow=False, contents="empty", mu=0.3, half=0.05):
    return Obj("A", half, "#c8553d", 0.0, Hidden(material, hollow, contents, mu))


@pytest.mark.parametrize("hollow,contents", [(False, "empty"), (True, "empty"), (True, "weight"), (True, "marble")])
def test_simulated_mass_matches_reported_mass(hollow, contents):
    o = cube("metal", hollow, contents)
    m = mujoco.MjModel.from_xml_string(world.object_scene(o, np.random.default_rng(0)))
    assert sum(m.body_mass[1:]) == pytest.approx(o.mass, rel=1e-6)


@pytest.mark.parametrize("hollow", [False, True])
@pytest.mark.parametrize("material", list(MATERIALS))
def test_flick_matches_coulomb_friction(material, hollow):
    o = cube(material, hollow, mu=0.4)
    v = 0.8
    d = world.simulate_flick(o, v * o.mass)
    assert d == pytest.approx(v ** 2 / (2 * 0.4 * G), rel=0.02)


def test_push_below_friction_does_not_move():
    o = cube(mu=0.5)
    assert world.simulate_push(o, 0.8 * 0.5 * o.mass * G) < 1e-3
    assert world.simulate_push(o, 1.5 * 0.5 * o.mass * G) > 0.05


def test_physics_is_deterministic():
    o = cube("plastic", True, "beads")
    assert world.simulate_flick(o, 0.3) == world.simulate_flick(o, 0.3)
    a, fa = world.simulate_shake(o, seed=3)
    b, fb = world.simulate_shake(o, seed=3)
    assert np.array_equal(a, b) and fa == fb


def test_tap_pitch_separates_hollow_from_solid():
    rng = np.random.default_rng(1)
    for i in range(24):
        hollow = bool(i % 2)
        o = cube(str(rng.choice(list(MATERIALS))), hollow, half=float(rng.uniform(0.04, 0.07)))
        _, f = world.simulate_tap(o, seed=i)
        assert (f["dominant_pitch_hz"] > HOLLOW_PITCH_HZ) == hollow


def test_public_view_leaks_no_hidden_properties():
    for seed in range(20):
        lv = generate(seed)
        text = json.dumps(public_view(lv)).lower()
        for word in ["hidden", "hollow\":", "material", "friction\":", "contents", "oracle", "mass"]:
            assert word not in text, (seed, word)
        for o in lv.objects:
            assert str(o.hidden.friction) not in text


def test_levels_round_trip_and_are_solvable():
    for seed in range(5):
        lv = valid_level(seed)
        again = type(lv).from_json(lv.to_json())
        assert again == lv
        assert solve(again)["solvable"]


def test_find_levels_have_exactly_one_answer():
    for seed in range(40):
        lv = generate(seed, kind="find_and_slide")
        p = lv.task.property
        if p in ("hollow", "solid"):
            n = sum(o.hidden.hollow == (p == "hollow") for o in lv.objects)
            assert n == 1
        if p == "rattles":
            assert sum(o.hidden.contents in ("marble", "beads") for o in lv.objects) == 1


def test_episode_budget_and_scoring(tmp_path):
    lv = valid_level(2)
    ep = Episode(lv, tmp_path, render=False)
    ep.reset()
    for _ in range(lv.task.budget):
        assert "error" not in ep.step({"type": "probe", "probe": "lift", "object": "A"})
    assert "error" in ep.step({"type": "probe", "probe": "lift", "object": "A"})
    oracle = lv.notes["oracle"]
    res = ep.step({"type": "flick", "object": oracle["object"], "impulse_Ns": oracle["impulse_Ns"]})
    assert res["success"]


def test_sound_probe_does_not_depend_on_probe_order(tmp_path):
    lv = valid_level(5)
    a = Episode(lv, tmp_path / "a", render=False)
    a.reset()
    a.step({"type": "probe", "probe": "lift", "object": "A"})
    ra = a.step({"type": "probe", "probe": "tap", "object": "B"})
    b = Episode(lv, tmp_path / "b", render=False)
    b.reset()
    rb = b.step({"type": "probe", "probe": "tap", "object": "B"})
    assert ra["sound"] == rb["sound"]
