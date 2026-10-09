"""Non-AI baselines built on a Bayesian physics estimator.

Every baseline keeps a cloud of prior samples (mass, friction, hollow, contents) per object,
re-weights them with each probe reading using textbook Coulomb-friction formulas, and
finally picks the object and the flick impulse that maximise the chance of success.
They differ only in which probes they choose:
    none      no probes at all
    random    random probes
    scripted  a sensible fixed strategy (identify by elimination, then lift + push)
"""
from __future__ import annotations

import numpy as np

from ..env import PUSH_SECONDS
from ..level import FLICK_RANGE, PUSH_RANGE, Hidden, Obj, _random_hidden

G = 9.81
N_SAMPLES = 4000
HOLLOW_PITCH_HZ = 250
RATTLE = ("marble", "beads")


def is_rattle(feats):
    return feats["distinct_hits"] >= 3 or feats["loudness_db_over_floor"] > 28


# ---------------------------------------------------------------- physics formulas
def push_distance(F, m, mu, T=PUSH_SECONDS):
    a = F / m - mu * G
    a = np.maximum(a, 0.0)
    v1 = a * T
    return a * T ** 2 / 2 + v1 ** 2 / (2 * mu * G)


def flick_distance(J, m, mu):
    v = J / m
    return v ** 2 / (2 * mu * G)


# ---------------------------------------------------------------- belief
class Belief:
    def __init__(self, half: float, rng):
        hs = [_random_hidden(rng) for _ in range(N_SAMPLES)]
        self.m = np.array([Obj("x", half, "#000000", 0, h).mass for h in hs])
        self.mu = np.array([h.friction for h in hs])
        self.hollow = np.array([h.hollow for h in hs])
        self.rattle = np.array([h.contents in RATTLE for h in hs])
        self.logw = np.zeros(N_SAMPLES)
        # rattling contents are only possible in hollow objects; the prior sampler never
        # draws them, so seed a fraction explicitly
        idx = rng.choice(np.where(self.hollow)[0], size=N_SAMPLES // 8, replace=False)
        self.rattle[idx] = True

    @property
    def w(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def update(self, obs):
        p = obs["probe"]
        if p == "lift":
            g = self.m * 1000
            self.logw += -0.5 * ((obs["weight_g"] - g) / (0.02 * g)) ** 2
        elif p == "push":
            s = push_distance(obs["force_N"], self.m, self.mu) * 1000
            sd = np.sqrt(1.0 + (0.12 * s) ** 2)
            if obs["moved_mm"] < 1.5:
                self.logw += np.where(s < 2.0, 0.0, -0.5 * (s / sd) ** 2)
            else:
                self.logw += -0.5 * ((obs["moved_mm"] - s) / sd) ** 2
        elif p == "tap":
            hollow = obs["sound"]["dominant_pitch_hz"] > HOLLOW_PITCH_HZ
            self.logw += np.log(np.where(self.hollow == hollow, 0.97, 0.03))
        elif p == "shake":
            r = is_rattle(obs["sound"])
            self.logw += np.log(np.where(self.rattle == r, 0.92, 0.08))

    def prob(self, mask):
        return float((self.w * mask).sum())

    def mean_mass(self):
        return float((self.w * self.m).sum())


# ---------------------------------------------------------------- decision
def choose_object(level, beliefs, rng):
    t = level.task
    ids = [o.id for o in level.objects]
    if t.kind == "slide":
        return t.object_id
    if t.property in ("hollow", "solid", "rattles"):
        key = {"hollow": "hollow", "solid": "hollow", "rattles": "rattle"}[t.property]
        p = np.array([beliefs[i].prob(getattr(beliefs[i], key)) for i in ids])
        if t.property == "solid":
            p = 1 - p
        # exactly one object has the property: P(i) ∝ p_i * prod_{j != i} (1 - p_j)
        score = [p[i] * np.prod(np.delete(1 - p, i)) for i in range(len(ids))]
        return ids[int(np.argmax(score))]
    # heaviest / lightest by Monte Carlo over posterior masses
    draws = np.stack([rng.choice(beliefs[i].m, 2000, p=beliefs[i].w) for i in ids])
    best = draws.argmax(0) if t.property == "heaviest" else draws.argmin(0)
    return ids[int(np.bincount(best, minlength=len(ids)).argmax())]


def choose_impulse(level, belief):
    t = level.task
    w = belief.w
    J = np.geomspace(*FLICK_RANGE, 400)
    d = flick_distance(J[:, None], belief.m[None], belief.mu[None])
    p_hit = ((np.abs(d - t.target_x) <= t.tol) * w[None]).sum(1)
    return float(J[int(np.argmax(p_hit))])


# ---------------------------------------------------------------- agents
class BaselineAgent:
    def __init__(self, policy: str = "scripted", seed: int = 0):
        assert policy in ("none", "random", "scripted")
        self.policy = policy
        self.seed = seed

    name = property(lambda self: f"baseline-{self.policy}")

    def run(self, ep) -> dict:
        level = ep.level
        rng = np.random.default_rng(self.seed + level.seed)
        obs = ep.reset()
        half = level.objects[0].half
        beliefs = {o.id: Belief(half, np.random.default_rng(rng.integers(1 << 30))) for o in level.objects}
        while obs.get("probes_left", 0) > 0:
            action = self._next_probe(level, beliefs, ep, rng)
            if action is None:
                break
            obs = ep.step(action)
            if "error" not in obs:
                beliefs[obs["object"]].update(obs)
        target = choose_object(level, beliefs, rng)
        J = choose_impulse(level, beliefs[target])
        return ep.step({"type": "flick", "object": target, "impulse_Ns": J})

    # ------------------------------------------------------------ probe policies
    def _next_probe(self, level, beliefs, ep, rng):
        ids = [o.id for o in level.objects]
        if self.policy == "none":
            return None
        if self.policy == "random":
            p = str(rng.choice(["push", "lift", "tap", "shake"]))
            return {"type": "probe", "probe": p, "object": str(rng.choice(ids)),
                    "force_N": float(rng.uniform(*PUSH_RANGE))}
        return self._scripted(level, beliefs, ep)

    def _scripted(self, level, beliefs, ep):
        t = level.task
        done = [(h["action"]["probe"], h["action"]["object"]) for h in ep.history if h["event"] == "step"]
        left = t.budget - ep.used
        ids = [o.id for o in level.objects]

        target = t.object_id
        if t.kind == "find_and_slide":
            if t.property in ("hollow", "solid", "rattles"):
                probe = "shake" if t.property == "rattles" else "tap"
                target = self._identified(level, beliefs)
                untested = [i for i in ids if (probe, i) not in done]
                # elimination: after n-1 tests the last object is known without testing it
                if target is None and len(untested) > 1:
                    return {"type": "probe", "probe": probe, "object": untested[0]}
                target = target or choose_object(level, beliefs, None)
            else:
                unweighed = [i for i in ids if ("lift", i) not in done]
                if unweighed:
                    return {"type": "probe", "probe": "lift", "object": unweighed[0]}
                target = choose_object(level, beliefs, np.random.default_rng(0))
        if ("lift", target) not in done and left >= 2:
            return {"type": "probe", "probe": "lift", "object": target}
        if ("push", target) not in done:
            m = beliefs[target].mean_mass()
            return {"type": "probe", "probe": "push", "object": target,
                    "force_N": float(np.clip(0.7 * m * G, *PUSH_RANGE))}
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
