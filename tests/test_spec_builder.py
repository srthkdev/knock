"""LevelSpec validation, spec <-> Level conversion, and world-builder checks (no LLM calls)."""
import copy
import json

import pytest
from pydantic import ValidationError

from knock.builder import (build_level, builder_prompt, check_fair, check_identifiable, check_level,
                           check_solvable, check_unique, slugify)
from knock.level import Level
from knock.oracle import valid_level
from knock.spec import LevelSpec, strict_json_schema


def obj(i, material="plastic", hollow=True, contents="empty", friction=0.3, edge=0.1, color="blue"):
    return dict(id=i, edge_m=edge, color=color, material=material, hollow=hollow, contents=contents, friction=friction)


def spec_dict(objects=None, **task):
    t = dict(kind="find_and_slide", object_id=None, property="rattles", target_x=0.8, tol=0.05, budget=4) | task
    objects = objects or [obj("A"), obj("B", contents="marble"), obj("C", contents="weight")]
    return dict(title="test", description="a test level", objects=objects, task=t)


GOOD = spec_dict()


def level_of(d, seed=7):
    return LevelSpec.model_validate(d).to_level("test", seed)


# ---------------------------------------------------------------- spec validation
def test_good_spec_validates():
    LevelSpec.model_validate(GOOD)


@pytest.mark.parametrize("path,value", [
    (("objects", 0, "material"), "gold"),
    (("objects", 0, "contents"), "water"),
    (("objects", 0, "color"), "#123456"),
    (("objects", 0, "friction"), 0.9),
    (("objects", 0, "friction"), 0.05),
    (("objects", 0, "edge_m"), 0.3),
    (("objects", 0, "id"), "Z"),
    (("task", "property"), "bounciest"),
    (("task", "kind"), "throw"),
    (("task", "target_x"), 5.0),
    (("task", "tol"), 0.5),
    (("task", "budget"), 50),
])
def test_rejects_out_of_catalogue_values(path, value):
    d = copy.deepcopy(GOOD)
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(ValidationError):
        LevelSpec.model_validate(d)


@pytest.mark.parametrize("bad", [
    lambda d: d["objects"][0].update(hollow=False, contents="marble"),  # solid with contents
    lambda d: d["objects"].__setitem__(1, obj("C")),  # ids not A, B, C in order
    lambda d: d.update(objects=d["objects"][:1]),  # too few objects
    lambda d: d.update(objects=[obj(i) for i in "ABCDEF"] + [obj("A")]),  # too many objects
    lambda d: d["task"].update(object_id="A"),  # find task with object_id
    lambda d: d["task"].update(kind="slide", property=None, object_id=None),  # slide without object
    lambda d: d["task"].update(kind="slide", property=None, object_id="E"),  # slide object not present
    lambda d: d["task"].update(target_x=0.1, tol=0.15),  # band reaches x <= 0
    lambda d: d.update(extra_field=1),  # additional properties
    lambda d: d["objects"][0].update(sparkle=True),
])
def test_rejects_inconsistent_specs(bad):
    d = copy.deepcopy(GOOD)
    bad(d)
    with pytest.raises(ValidationError):
        LevelSpec.model_validate(d)


def test_strict_schema_shape():
    s = strict_json_schema()
    json.dumps(s)

    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for v in node.values():
                yield from objects(v)
        elif isinstance(node, list):
            for v in node:
                yield from objects(v)

    objs = list(objects(s))
    assert len(objs) == 3  # LevelSpec, ObjectSpec, TaskSpec
    for o in objs:
        assert o["additionalProperties"] is False
        assert set(o["required"]) == set(o["properties"])
    assert "default" not in json.dumps(s)
    assert s["$defs"]["ObjectSpec"]["properties"]["material"]["enum"] == ["metal", "glass", "plastic", "wood"]


# ---------------------------------------------------------------- conversion
def test_spec_level_round_trip():
    spec = LevelSpec.model_validate(GOOD)
    lv = spec.to_level("rt", 42)
    assert [o.id for o in lv.objects] == ["A", "B", "C"]
    assert lv.objects[1].hidden.contents == "marble" and lv.objects[0].half == pytest.approx(0.05)
    assert LevelSpec.from_level(lv) == spec
    lv2 = Level.from_json(lv.to_json())
    assert LevelSpec.from_level(lv2) == spec


@pytest.mark.parametrize("seed", range(6))
def test_generated_levels_convert_to_spec(seed):
    lv = valid_level(seed)
    back = LevelSpec.from_level(lv).to_level(lv.id, lv.seed)
    assert [(o.id, o.half, o.color, o.y, o.hidden) for o in back.objects] == \
           [(o.id, o.half, o.color, o.y, o.hidden) for o in lv.objects]
    assert back.task == lv.task


# ---------------------------------------------------------------- validation checks
def test_good_rattle_level_passes_all_checks():
    checks, sol = check_level(level_of(GOOD))
    assert all(c.passed for c in checks), checks
    assert sol["object"] == "B" and sol["solvable"]


def test_unique_fails_with_two_rattlers():
    d = spec_dict([obj("A", contents="beads"), obj("B", contents="marble"), obj("C")])
    assert not check_unique(level_of(d)).passed
    assert not all(c.passed for c in check_level(level_of(d))[0])


def test_unique_fails_with_no_hollow():
    d = spec_dict([obj("A", hollow=False), obj("B", hollow=False)], property="hollow")
    assert not check_unique(level_of(d)).passed


def test_fair_fails_when_objects_look_different():
    d = spec_dict([obj("A"), obj("B", contents="marble", color="red"), obj("C")])
    assert not check_fair(level_of(d)).passed
    d = spec_dict([obj("A"), obj("B", contents="marble", edge=0.12), obj("C")])
    assert not check_fair(level_of(d)).passed


def test_fair_allows_different_looks_in_slide_tasks():
    d = spec_dict([obj("A", color="red"), obj("B", edge=0.12)], kind="slide", property=None, object_id="A")
    assert check_fair(level_of(d)).passed


def test_solvable_fails_for_far_target_with_heavy_block():
    d = spec_dict([obj("A", "metal", hollow=False, friction=0.6, edge=0.14), obj("B")],
                  kind="slide", property=None, object_id="A", target_x=2.0, tol=0.05, budget=1)
    c, _ = check_solvable(level_of(d))
    assert not c.passed and "cannot be reached" in c.detail


def test_identifiable_budget_too_small():
    d = spec_dict(budget=1)  # three objects need two shakes
    c = check_identifiable(level_of(d))
    assert not c.passed and "budget" in c.detail


def test_identifiable_quiet_rattle_fails(monkeypatch):
    # a muffled rattle (first shake only ~7 dB over the floor) must not count as identifiable;
    # the shake readings are stubbed so the check is tested independently of audio details
    import numpy as np
    from knock import world

    def quiet_shake(o, seed=0):
        feats = {"ring_time_s": 0.0, "distinct_hits": 0, "brightness_hz": 500, "dominant_pitch_hz": 300,
                 "loudness_db_over_floor": 7.0 if o.hidden.contents == "marble" else 5.0}
        return np.zeros(10), feats

    monkeypatch.setattr(world, "simulate_shake", quiet_shake)
    d = spec_dict([obj("A", "metal", edge=0.14), obj("B", "metal", contents="marble", edge=0.14),
                   obj("C", "metal", contents="weight", edge=0.14)])
    c = check_identifiable(level_of(d, seed=0))
    assert not c.passed and "quiet" in c.detail


def test_identifiable_mass_gap():
    near = spec_dict([obj("A", "wood", hollow=False), obj("B", "wood", hollow=False, edge=0.1),
                      obj("C", "plastic", contents="weight")], property="heaviest", budget=4)
    # all same edge: wood solid 0.65 kg vs plastic shell + weight ~0.45 kg vs tie -> no unique heaviest
    assert not check_unique(level_of(near)).passed
    gap_small = spec_dict([obj("A", "wood", hollow=False), obj("B", "plastic", contents="weight"),
                           obj("C", "glass", contents="weight")], property="heaviest", budget=4)
    c = check_identifiable(level_of(gap_small))
    assert not c.passed and "%" in c.detail
    clear = spec_dict([obj("A", "wood", hollow=False), obj("B", "metal", hollow=False), obj("C")],
                      property="heaviest", budget=3)
    assert check_identifiable(level_of(clear)).passed
    lightest = spec_dict([obj("A", "wood", hollow=False), obj("B", "metal", hollow=False), obj("C")],
                         property="lightest", budget=2)
    c = check_identifiable(level_of(lightest))
    assert not c.passed and "budget" in c.detail  # three lifts needed


def test_identifiable_hollow_by_tap():
    d = spec_dict([obj("A", "wood", hollow=False), obj("B", "glass"), obj("C", "plastic", hollow=False)],
                  property="hollow", budget=3)
    checks, _ = check_level(level_of(d))
    assert all(c.passed for c in checks), checks


# ---------------------------------------------------------------- build loop (fake proposer)
def test_build_loop_feeds_back_failures(tmp_path):
    bad = spec_dict(budget=1)
    replies = [None, {"nonsense": 1}, bad, GOOD]
    prompts = []

    def propose(text, schema, k):
        prompts.append(text)
        return replies[k - 1], 0.0, []

    level, rep = build_level("three boxes, one rattles", out_dir=tmp_path, log_dir=tmp_path / "logs",
                             run_baselines=False, propose=propose, log=lambda *a: None)
    assert rep.passed and len(rep.rounds) == 4
    assert [r.passed for r in rep.rounds] == [False, False, False, True]
    assert "budget 1 is too small" in prompts[3] and "REJECTED" in prompts[3]
    assert "schema:" in prompts[2]
    saved = Level.from_json((tmp_path / f"{slugify('three boxes, one rattles')}.json").read_text())
    assert saved.answer_id() == "B" and saved.notes["oracle"]["solvable"]
    assert json.loads((tmp_path / f"{rep.slug}.report.json").read_text())["passed"]


def test_build_gives_up_after_max_rounds(tmp_path):
    level, rep = build_level("x", out_dir=tmp_path, log_dir=tmp_path / "logs", max_rounds=2, run_baselines=False,
                             propose=lambda t, s, k: (spec_dict(budget=0), 0.0, []), log=lambda *a: None)
    assert not rep.passed and len(rep.rounds) == 2 and rep.level_path is None
    assert (tmp_path / "x.report.json").exists() and not (tmp_path / "x.json").exists()


def test_builder_prompt_mentions_catalogue():
    p = builder_prompt("anything", [])
    for word in ("metal", "marble", "beads", "friction", "heaviest", "identical", "15"):
        assert word in p
