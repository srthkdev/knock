"""Episode logic: probes within a budget, then one flick that is scored.

The agent-facing API is two calls:
    obs = ep.reset()
    obs = ep.step({"type": "probe", "probe": "push", "object": "A", "force_N": 3.0})
    obs = ep.step({"type": "flick", "object": "B", "impulse_Ns": 0.8})   # ends the episode
Observations hold only public information and probe readings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import sound, world
from .level import FLICK_RANGE, PROBES, PUSH_RANGE, Level, public_view

PUSH_SECONDS = 0.4

RULES = f"""\
World rules
- Objects are cubes resting on a flat table. They start at x = 0 in separate lanes and only move along +x.
- Objects in a level look identical, but their hidden material, mass, inside (hollow or solid,
  loose or fixed contents) and friction against the table can all differ.
- Probes (each costs 1 from the probe budget; objects are put back at x = 0 afterwards):
  - push(object, force_N): constant horizontal force along +x for {PUSH_SECONDS} s, then released.
    Reports how far it moved in mm (about 1 mm noise). Force range {PUSH_RANGE[0]}-{PUSH_RANGE[1]} N.
  - lift(object): reports its weight in grams (about 2% noise).
  - tap(object): strikes its side with a small steel ball. Returns sound features and a spectrogram image.
  - shake(object): picks it up and shakes it. Returns sound features and a spectrogram image.
- flick(object, impulse_Ns): a single instantaneous strike at the object's centre along +x.
  Impulse range {FLICK_RANGE[0]}-{FLICK_RANGE[1]} N*s. You get exactly one flick and it ends the episode.
- Gravity is 9.81 m/s^2. Friction behaves roughly like Coulomb friction.
"""


def probe_seed(level: Level, oid: str, probe: str, repeat: int) -> int:
    """Sound randomness depends only on (level, object, probe, repeat), not on probe order."""
    return level.seed * 1000 + "ABCDEF".index(oid) * 100 + PROBES.index(probe) * 10 + repeat


class Episode:
    def __init__(self, level: Level, out_dir: str | Path, render: bool = True):
        self.level = level
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.render = render
        self.used = 0
        self.history: list[dict] = []
        self.done = False
        self.result: dict | None = None

    # ------------------------------------------------------------ api
    def reset(self) -> dict:
        obs = {"rules": RULES, "level": public_view(self.level), "probes_left": self.level.task.budget}
        if self.render:
            from .render import render_level
            obs["image"] = str(render_level(self.level, self.out / "scene.png"))
        self.history.append({"event": "reset", "obs": obs})
        return obs

    def step(self, action: dict) -> dict:
        if self.done:
            raise RuntimeError("episode is over")
        try:
            obs = self._step(action)
        except (KeyError, ValueError, TypeError) as e:
            obs = {"error": f"invalid action: {e}", "probes_left": self.level.task.budget - self.used}
        self.history.append({"event": "step", "action": action, "obs": obs})
        return obs

    # ------------------------------------------------------------ internals
    def _rng(self):
        return np.random.default_rng(self.level.seed * 1000 + len(self.history))

    def _count(self, probe, oid):
        """How many times this probe was already applied to this object."""
        return sum(1 for h in self.history if h.get("event") == "step"
                   and h["action"].get("probe") == probe and h["obs"].get("object") == oid)

    def _obj(self, action):
        oid = str(action["object"]).strip().upper()
        if oid not in {o.id for o in self.level.objects}:
            raise ValueError(f"unknown object {action['object']!r}")
        return self.level.obj(oid)

    def _step(self, a: dict) -> dict:
        kind = a.get("type")
        if kind == "flick":
            return self._flick(self._obj(a), float(a["impulse_Ns"]))
        if kind != "probe":
            raise ValueError("type must be 'probe' or 'flick'")
        probe = a["probe"]
        if probe not in PROBES:
            raise ValueError(f"probe must be one of {PROBES}")
        if self.used >= self.level.task.budget:
            return {"error": "probe budget exhausted; you must flick now", "probes_left": 0}
        o = self._obj(a)
        self.used += 1
        rng = self._rng()
        n = len(self.history)
        obs: dict = {"probe": probe, "object": o.id}
        if probe == "push":
            f = float(np.clip(float(a["force_N"]), *PUSH_RANGE))
            disp = world.simulate_push(o, f, PUSH_SECONDS, seed=n)
            mm = max(0.0, disp * 1000 + rng.normal(0, 1.0))
            obs.update(force_N=round(f, 3), moved_mm=round(mm, 1))
        elif probe == "lift":
            obs.update(weight_g=world.weigh(o, rng))
        else:
            sim = world.simulate_tap if probe == "tap" else world.simulate_shake
            x, feats = sim(o, seed=probe_seed(self.level, o.id, probe, self._count(probe, o.id)))
            obs.update(sound=feats)
            stem = self.out / f"{n:02d}_{probe}_{o.id}"
            sound.write_wav(x, stem.with_suffix(".wav"), gain=73.0)
            sound.save_spectrogram(x, stem.with_suffix(".png"), f"{probe} {o.id}")
            obs.update(spectrogram=str(stem.with_suffix(".png")), audio=str(stem.with_suffix(".wav")))
        obs["probes_left"] = self.level.task.budget - self.used
        return obs

    def _flick(self, o, impulse):
        impulse = float(np.clip(impulse, *FLICK_RANGE))
        moved = world.simulate_flick(o, impulse)
        t = self.level.task
        correct = o.id == self.level.answer_id()
        landed = abs(moved - t.target_x) <= t.tol
        self.done = True
        self.result = {
            "success": bool(correct and landed),
            "correct_object": bool(correct),
            "landed_in_band": bool(landed),
            "final_x": round(moved, 4),
            "error_m": round(moved - t.target_x, 4),
            "impulse_Ns": round(impulse, 4),
            "probes_used": self.used,
        }
        if self.render:
            from .render import render_level
            pos = {p.id: 0.0 for p in self.level.objects} | {o.id: moved}
            self.result["image"] = str(render_level(self.level, self.out / "final.png", positions=pos))
        return {"flicked": o.id, **self.result}

    def save_trace(self, extra: dict | None = None):
        trace = {"level": json.loads(self.level.to_json()), "history": self.history,
                 "result": self.result, **(extra or {})}
        (self.out / "trace.json").write_text(json.dumps(trace, indent=1, default=str))
