"""Bayesian physics baselines for the tabletop pack, as core agents (AgentView -> ToolCall).

Same inference as yantra.agents.baselines (prior samples re-weighted by each reading with Coulomb
formulas), but driven only by what the agent is allowed to see. Policies differ in probe choice:
    none      no probes
    random    random probes
    scripted  identify by elimination, then lift + push the answer
Every push and the flick carry the agent's own posterior-mean prediction.
"""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np

from ..core import AgentView, ToolCall
from .baselines import G, Belief, choose_impulse, choose_object, flick_distance, push_distance
from ..level import PUSH_RANGE


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


class TabletopBayesAgent:
    def __init__(self, policy: str = "scripted", seed: int = 0):
        assert policy in ("none", "random", "scripted")
        self.policy = policy
        self.seed = seed
        self.name = f"tabletop-bayes-{policy}"

    # ------------------------------------------------------------------ state from the view
    def reset(self, view: AgentView):
        base = self.seed + _stable_seed(view.task["goal"] + str([c["id"] for c in view.world["components"]]))
        self.rng = np.random.default_rng(base)
        half = view.world["components"][0]["edge_m"] / 2
        self._belief_seeds = {c["id"]: int(self.rng.integers(1 << 30)) for c in view.world["components"]}
        self._half = half

    def _beliefs(self, view):
        beliefs = {i: Belief(self._half, np.random.default_rng(s)) for i, s in self._belief_seeds.items()}
        for h in view.history:
            obs = h["observation"]
            if "error" in obs or h["call"]["tool"] == "flick":
                continue
            beliefs[h["call"]["target"]].update({"probe": h["call"]["tool"], **obs})
        return beliefs

    @staticmethod
    def _level(view):
        t = view.task
        kind = "slide" if "object" in t else "find_and_slide"
        task = SimpleNamespace(kind=kind, property=t.get("property"), object_id=t.get("object"),
                               target_x=t["target_x"], tol=t["tol"], budget=t["probe_budget"])
        return SimpleNamespace(task=task, objects=[SimpleNamespace(id=c["id"]) for c in view.world["components"]])

    # ------------------------------------------------------------------ decisions
    def act(self, view: AgentView) -> ToolCall:
        level = self._level(view)
        beliefs = self._beliefs(view)
        probe = self._next_probe(view, level, beliefs) if view.budget_left > 0 else None
        if probe is not None:
            if probe.tool == "push":
                b = beliefs[probe.target]
                probe.prediction = {"moved_mm": max(0.0, float((b.w * push_distance(probe.args["force_N"], b.m, b.mu) * 1000).sum()))}
            return probe
        target = choose_object(level, beliefs, self.rng)
        J = choose_impulse(level, beliefs[target])
        b = beliefs[target]
        return ToolCall("flick", target, {"impulse_Ns": J},
                        prediction={"stop_x_m": float((b.w * flick_distance(J, b.m, b.mu)).sum())})

    def _next_probe(self, view, level, beliefs):
        ids = [o.id for o in level.objects]
        if self.policy == "none":
            return None
        if self.policy == "random":
            p = str(self.rng.choice(["push", "lift", "tap", "shake"]))
            args = {"force_N": float(self.rng.uniform(*PUSH_RANGE))} if p == "push" else {}
            return ToolCall(p, str(self.rng.choice(ids)), args)
        return self._scripted(view, level, beliefs)

    def _scripted(self, view, level, beliefs):
        t = level.task
        done = [(h["call"]["tool"], h["call"]["target"]) for h in view.history if "error" not in h["observation"]]
        left = view.budget_left
        ids = [o.id for o in level.objects]
        target = t.object_id
        if t.kind == "find_and_slide":
            if t.property in ("hollow", "solid", "rattles"):
                probe = "shake" if t.property == "rattles" else "tap"
                target = self._identified(level, beliefs)
                untested = [i for i in ids if (probe, i) not in done]
                if target is None and len(untested) > 1:
                    return ToolCall(probe, untested[0])
                target = target or choose_object(level, beliefs, None)
            else:
                unweighed = [i for i in ids if ("lift", i) not in done]
                if unweighed:
                    return ToolCall("lift", unweighed[0])
                target = choose_object(level, beliefs, np.random.default_rng(0))
        if ("lift", target) not in done and left >= 2:
            return ToolCall("lift", target)
        if ("push", target) not in done:
            m = beliefs[target].mean_mass()
            return ToolCall("push", target, {"force_N": float(np.clip(0.7 * m * G, *PUSH_RANGE))})
        return None

    @staticmethod
    def _identified(level, beliefs):
        t = level.task
        key = "rattle" if t.property == "rattles" else "hollow"
        for o in level.objects:
            p = beliefs[o.id].prob(getattr(beliefs[o.id], key))
            if t.property == "solid":
                p = 1 - p
            if p > 0.9:
                return o.id
        return None
