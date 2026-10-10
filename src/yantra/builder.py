"""World builder: turn a natural-language request into a validated cube-lab level.

    level, report = build_level("three identical boxes, one has a marble, far target")

A Codex CLI call proposes a `LevelSpec` (forced into its strict JSON schema). The proposal is
converted to a `Level` and checked:
    solvable      the oracle can land the answer object in the band with a flick in range
    unique        find tasks: exactly one object has the property
    identifiable  the property can be revealed with the allowed probes within the budget
    fair          find tasks: all objects look identical (same size and colour)
Failures are fed back to the model, which revises the level (up to `max_rounds` calls).
A passing level also gets a difficulty report from the none / random / scripted baselines.
"""
from __future__ import annotations

import json
import re
import tempfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import ValidationError

from . import world
from .agents.codex_cli import run_codex
from .env import Episode, probe_seed
from .level import FLICK_RANGE, PROPERTIES, Level
from .materials import CONTENTS, MATERIALS, WALL
from .oracle import solve
from .spec import (BUDGET_RANGE, COLORS, EDGE_RANGE, FRICTION_RANGE, N_OBJECTS, RATTLING, TARGET_RANGE,
                   TOL_RANGE, LevelSpec, strict_json_schema)

G = 9.81
MASS_GAP = 1.25  # heaviest / lightest must differ from the runner-up by at least 25%
RATTLE_DB = 15.0  # first shake of the rattling object must be at least this loud (dB over floor)
RANDOM_SEEDS = 3  # the random baseline is run with this many seeds


# ================================================================ checks
@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _mass_str(o) -> str:
    return f"{o.mass:.3f} kg"


def check_unique(level: Level) -> Check:
    t = level.task
    if t.kind == "slide":
        return Check("unique", True, f"slide task names object {t.object_id}")
    objs, prop = level.objects, t.property
    if prop in ("hollow", "solid", "rattles"):
        has = {"hollow": lambda o: o.hidden.hollow, "solid": lambda o: not o.hidden.hollow,
               "rattles": lambda o: o.hidden.contents in RATTLING}[prop]
        ids = [o.id for o in objs if has(o)]
        ok = len(ids) == 1
        return Check("unique", ok, f"objects with property {prop!r}: {ids or 'none'}"
                     + ("" if ok else " (need exactly one)"))
    masses = sorted(((o.mass, o.id) for o in objs), reverse=prop == "heaviest")
    ok = masses[0][0] != masses[1][0]
    return Check("unique", ok, f"{prop}: {masses[0][1]} ({masses[0][0]:.3f} kg)"
                 + ("" if ok else f" ties with {masses[1][1]}"))


def check_fair(level: Level) -> Check:
    looks = {(o.half, o.color) for o in level.objects}
    if level.task.kind == "slide":
        return Check("fair", True, "slide task: the object is named, looks may differ")
    ok = len(looks) == 1
    return Check("fair", ok, "all objects look identical" if ok else
                 f"find task but objects differ in size/colour: {sorted(looks)} (they must all look the same)")


def check_identifiable(level: Level) -> Check:
    t = level.task
    n = len(level.objects)
    if t.kind == "slide":
        return Check("identifiable", True, "slide task: nothing to identify")
    prop = t.property
    if prop in ("hollow", "solid"):
        from .agents.baselines import sounds_hollow
        need = n - 1  # tap all but one; the last is known by elimination
        wrong = []
        for o in level.objects:
            _, f = world.simulate_tap(o, seed=probe_seed(level, o.id, "tap", 0))
            if sounds_hollow(f) != o.hidden.hollow:
                wrong.append(f"{o.id} ({f['dominant_pitch_hz']} Hz, {f['loudness_db_over_floor']} dB)")
        if wrong:
            return Check("identifiable", False, f"tap misleads for {wrong} (hollow cubes ring loudly below ~4 kHz; "
                         f"solid cubes give a high tick or a quiet click); try other materials or sizes")
        msg = f"tap separates hollow from solid; needs {need} tap(s), budget {t.budget}"
    elif prop == "rattles":
        need = n - 1
        loud = {}
        for o in level.objects:
            _, f = world.simulate_shake(o, seed=probe_seed(level, o.id, "shake", 0))
            loud[o.id] = f["loudness_db_over_floor"]
        r = next((o.id for o in level.objects if o.hidden.contents in RATTLING), None)
        if r is None:
            return Check("identifiable", False, "no rattling object")
        others = {k: v for k, v in loud.items() if k != r}
        if loud[r] < RATTLE_DB:
            return Check("identifiable", False,
                         f"the rattle of {r} is too quiet: {loud[r]} dB over floor on its first shake (need >= "
                         f"{RATTLE_DB}). Plastic or wood boxes and smaller boxes rattle louder; "
                         f"big metal/glass boxes muffle a marble")
        if others and max(others.values()) >= loud[r]:
            return Check("identifiable", False, f"shake loudness {loud}: another object is as loud as the "
                         f"rattler {r} (foam can thud); make the decoys quieter (empty/weight)")
        msg = f"shake loudness (dB) {loud}; needs {need} shake(s), budget {t.budget}"
    else:  # heaviest / lightest
        need = n  # weigh every object
        ms = sorted((o.mass for o in level.objects), reverse=prop == "heaviest")
        ratio = ms[0] / ms[1] if prop == "heaviest" else ms[1] / ms[0]
        if ratio < MASS_GAP:
            return Check("identifiable", False,
                         f"{prop} object differs from the runner-up by only {100 * (ratio - 1):.0f}% "
                         f"(masses {[round(m, 3) for m in ms]} kg); need >= {100 * (MASS_GAP - 1):.0f}% for lift to tell")
        msg = f"lift separates by {100 * (ratio - 1):.0f}%; needs {need} lift(s), budget {t.budget}"
    if t.budget < need:
        return Check("identifiable", False, f"budget {t.budget} is too small: identifying '{prop}' needs {need} probes")
    return Check("identifiable", True, msg)


def check_solvable(level: Level) -> tuple[Check, dict | None]:
    try:
        sol = solve(level)
    except StopIteration:
        return Check("solvable", False, "no object has the property, nothing to solve"), None
    t = level.task
    o = level.obj(sol["object"])
    if sol["solvable"]:
        return Check("solvable", True, f"oracle flicks {o.id} with {sol['impulse_Ns']} N*s -> "
                     f"x = {sol['predicted_x']} m (band {t.target_x - t.tol:.3f}-{t.target_x + t.tol:.3f})"), sol
    lo, hi = world.simulate_flick(o, FLICK_RANGE[0]), world.simulate_flick(o, FLICK_RANGE[1])
    return Check("solvable", False,
                 f"object {o.id} (mass {_mass_str(o)}, friction {o.hidden.friction}) travels {lo:.3f}-{hi:.3f} m "
                 f"for flick impulses {FLICK_RANGE[0]}-{FLICK_RANGE[1]} N*s, so the band "
                 f"{t.target_x - t.tol:.3f}-{t.target_x + t.tol:.3f} m cannot be reached. Distance is about "
                 f"(J/m)^2 / (2*friction*9.81): make it lighter / lower friction, or move the target"), sol


def check_level(level: Level) -> tuple[list[Check], dict | None]:
    """All validation checks; returns (checks, oracle solution or None)."""
    checks = [check_unique(level), check_fair(level)]
    sol = None
    if checks[0].passed:
        checks.append(check_identifiable(level))
        solv, sol = check_solvable(level)
        checks.append(solv)
    else:
        checks.append(Check("identifiable", False, "skipped: property is not unique"))
        checks.append(Check("solvable", False, "skipped: property is not unique"))
    return checks, sol


# ================================================================ difficulty
def difficulty(level: Level) -> dict:
    """Run the baselines (no rendering) and give a qualitative label."""
    from .agents.baselines import BaselineAgent

    def run(policy, seed=0):
        with tempfile.TemporaryDirectory(prefix="yantra-build-") as tmp:
            ep = Episode(level, tmp, render=False)
            BaselineAgent(policy, seed=seed).run(ep)
            r = ep.result or {}
        return {k: r.get(k) for k in ("success", "correct_object", "landed_in_band", "error_m", "probes_used")}

    none, scripted = run("none"), run("scripted")
    randoms = [run("random", seed=s) for s in range(RANDOM_SEEDS)]
    rnd_rate = sum(bool(r["success"]) for r in randoms) / len(randoms)
    if none["success"]:
        label = "easy"  # a blind flick works: probing is not needed
    elif not scripted["success"]:
        label = "hard"  # even the sensible scripted baseline fails
    elif rnd_rate > 0:
        label = "moderate"  # targeted probing works, random probing sometimes does
    else:
        label = "tricky"  # only targeted probing works
    return {"none": none, "random": randoms, "random_success_rate": round(rnd_rate, 3),
            "scripted": scripted, "label": label}


# ================================================================ prompt
def _catalogue() -> str:
    mats = []
    for name, p in MATERIALS.items():
        ms = []
        for e in (EDGE_RANGE[0], 0.11, EDGE_RANGE[1]):
            solid = e ** 3 * p["density"]
            hollow = (e ** 3 - (e - 2 * WALL) ** 3) * p["density"]
            ms.append(f"edge {e}: solid {solid:.3f} kg, hollow-empty {hollow:.3f} kg")
        mats.append(f"  - {name}: density {p['density']} kg/m^3; " + "; ".join(ms))
    conts = {"empty": "nothing inside", "marble": "one loose 5 g marble: RATTLES when shaken",
             "beads": "eight loose 0.8 g beads: RATTLES when shaken",
             "foam": "a loose 4 g foam ball: soft thud, does not rattle (can still be a little loud)",
             "weight": "a 250 g weight glued to the floor: silent, makes a hollow box heavy"}
    assert set(conts) == set(CONTENTS)
    return f"""Catalogue (you may only use these values)
- 2 to 6 cube objects labelled A, B, C, ... in order. They start side by side at x = 0, in separate lanes.
- Visible: edge_m in [{EDGE_RANGE[0]}, {EDGE_RANGE[1]}] m; color one of {list(COLORS)}.
- Hidden material (sound and density):
{chr(10).join(mats)}
- Hidden hollow: hollow cubes are a {int(WALL * 1000)} mm shell; mass = shell + contents. Solid cubes have contents "empty".
- Hidden contents (hollow cubes only):
{chr(10).join(f'  - {k}: {v}' for k, v in conts.items())}
- Hidden friction against the table in [{FRICTION_RANGE[0]}, {FRICTION_RANGE[1]}].
- Task kind "slide": flick the named object_id (property null). Kind "find_and_slide": exactly one object has
  the property (object_id null). Properties: {', '.join(f'{k} ({v})' for k, v in PROPERTIES.items())}.
- target_x (band centre) in [{TARGET_RANGE[0]}, {TARGET_RANGE[1]}] m, tol (band half width) in [{TOL_RANGE[0]}, {TOL_RANGE[1]}] m,
  probe budget in [{BUDGET_RANGE[0]}, {BUDGET_RANGE[1]}]. Number of objects in [{N_OBJECTS[0]}, {N_OBJECTS[1]}]."""


RULES = f"""Physics
- Flick: one instantaneous impulse J along +x, J in [{FLICK_RANGE[0]}, {FLICK_RANGE[1]}] N*s. The object slides
  d = (J/m)^2 / (2 * friction * 9.81) metres (Coulomb friction; simulation matches within ~2%).
  So the farthest a mass m can go is (6/m)^2 / (2*friction*9.81): heavy metal blocks cannot reach far targets.
- Probes (each costs 1 from the budget): lift (weight, 2% noise), push (distance under a known force),
  tap (pitch: hollow cubes ring high, solid cubes low), shake (loose marble/beads rattle).

Fairness rules (the level is rejected unless all hold)
- find_and_slide: EXACTLY one object has the property, and all objects look identical (same edge_m and color).
- Identifiable within the budget:
  - hollow / solid: tap must tell them apart; budget >= number of objects - 1.
  - rattles: the rattler's first shake must be >= {RATTLE_DB} dB over the noise floor and louder than every other
    object's shake. Plastic or wood shells, and smaller boxes, rattle loudest; big metal/glass boxes muffle it.
    Decoys should be hollow too (else tap gives it away) with empty, foam or weight contents.
    budget >= number of objects - 1.
  - heaviest / lightest: the extreme mass must differ from the runner-up by >= {int(100 * (MASS_GAP - 1))}%;
    budget >= number of objects.
- Solvable: the answer object must be able to stop inside the band with some J in the flick range.
- Interesting is good: decoys that differ in other hidden ways (friction, material, a glued weight) make
  agents measure rather than guess, as long as the rules above hold. Make the budget a little more than the
  minimum (e.g. +1 or +2) so an agent can also measure mass/friction before flicking, unless asked otherwise."""


def builder_prompt(request: str, attempts: list[dict]) -> str:
    fb = ""
    if attempts:
        last = attempts[-1]
        prev = json.dumps(last["spec"], indent=1) if last.get("spec") is not None else "(no valid JSON)"
        fails = "\n".join(f"- {f}" for f in last["failures"])
        older = ""
        if len(attempts) > 1:
            older = "\nEarlier attempts also failed with:\n" + "\n".join(
                f"- round {a['round']}: {'; '.join(a['failures'])}" for a in attempts[:-1])
        fb = f"""
Your previous proposal (round {last['round']}):
{prev}

It was REJECTED by the validator:
{fails}
{older}
Revise the level so every check passes while staying true to the request. Change only what is needed.
"""
    return f"""You are the world builder for Knock, a MuJoCo physics benchmark. Identical-looking cubes hide mass,
friction, hollow/solid, material and contents; an AI agent probes them within a budget, then gets ONE flick
that must stop the right object inside a target band on the x axis.

Design one level that fulfils the user's request. Do NOT run commands, read files, or browse.
Compute masses and flick distances yourself from the formulas, then answer only with the JSON level spec.

{_catalogue()}

{RULES}

User request: {request!r}
{fb}
Reply with the JSON spec: title, description (the idea and how a careful agent solves it), objects, task."""


# ================================================================ build loop
@dataclass
class Round:
    round: int
    seconds: float
    spec: dict | None
    checks: list[Check] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    tainted_items: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.spec is not None and not self.failures


@dataclass
class BuildReport:
    prompt: str
    slug: str
    model: str
    effort: str
    passed: bool = False
    rounds: list[Round] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    oracle: dict | None = None
    difficulty: dict | None = None
    level_path: str | None = None
    report_path: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for r, rd in zip(self.rounds, d["rounds"]):
            rd["passed"] = r.passed
        return d


def slugify(text: str, max_words: int = 7) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:max_words]
    return "-".join(words) or "level"


def build_level(prompt: str, model: str = "gpt-6.1-sol", effort: str = "medium", max_rounds: int = 4,
                out_dir: str | Path = "levels/custom", slug: str | None = None, run_baselines: bool = True,
                log_dir: str | Path = "runs/build", propose=None, log=print) -> tuple[Level | None, BuildReport]:
    """Iterate LLM proposal -> validation -> feedback until the level passes or rounds run out.

    `propose(prompt_text, schema, round_no) -> (reply_dict_or_None, seconds, tainted_items)` can replace the
    Codex call (used by tests). Saves `<out_dir>/<slug>.json` (if it passes) and `<slug>.report.json`;
    raw Codex prompts and event streams go to `<log_dir>/<slug>/`.
    Returns (level or None, report); the level is the last valid proposal even if checks failed.
    """
    slug = slug or slugify(prompt)
    seed = zlib.crc32(slug.encode()) % 100_000
    out = Path(out_dir)
    log_dir = Path(log_dir) / slug
    schema = strict_json_schema()
    report = BuildReport(prompt=prompt, slug=slug, model=model, effort=effort)
    attempts: list[dict] = []
    level = None

    if propose is None:
        def propose(text, schema, k):
            res = run_codex(text, schema, model=model, effort=effort, log_stem=log_dir / f"round{k}")
            return res.reply, res.seconds, res.tainted_items

    for k in range(1, max_rounds + 1):
        reply, secs, tainted = propose(builder_prompt(prompt, attempts), schema, k)
        rnd = Round(round=k, seconds=round(secs, 1), spec=reply, tainted_items=list(tainted))
        report.rounds.append(rnd)
        if reply is None:
            rnd.failures = ["no parseable JSON reply"]
        else:
            try:
                spec = LevelSpec.model_validate(reply)
            except ValidationError as e:
                rnd.failures = [f"schema: {'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]
            else:
                level = spec.to_level(slug, seed, notes={"prompt": prompt})
                rnd.checks, sol = check_level(level)
                rnd.failures = [f"{c.name}: {c.detail}" for c in rnd.checks if not c.passed]
                report.checks, report.oracle = rnd.checks, sol
        log(f"  round {k}: {'PASS' if rnd.passed else 'fail'} ({rnd.seconds:.0f}s)"
            + "".join(f"\n    - {f}" for f in rnd.failures))
        if rnd.passed:
            report.passed = True
            break
        attempts.append({"round": k, "spec": reply, "failures": rnd.failures})

    out.mkdir(parents=True, exist_ok=True)
    if report.passed and level is not None:
        if run_baselines:
            report.difficulty = difficulty(level)
        level.notes.update(oracle=report.oracle, rounds=len(report.rounds),
                           difficulty=(report.difficulty or {}).get("label"), builder_model=f"{model}/{effort}")
        path = out / f"{slug}.json"
        path.write_text(level.to_json())
        report.level_path = str(path)
    rp = out / f"{slug}.report.json"
    report.report_path = str(rp)
    rp.write_text(json.dumps(report.to_dict(), indent=1, default=str))
    return level, report
