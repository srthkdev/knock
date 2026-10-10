"""Procedural factory maps, each proven solvable before it is kept.

A map is a floor with walls, one or two inlets on the west edge and a shipping bin on the east side,
plus what the inlets send. The generator then builds a reference factory itself (route by
Dijkstra with a turn penalty, a quality station on a straight run after any merge) and searches its
speeds and pusher timing with the TRUE hidden physics. The map is kept only if that reference passes
a production run; the agent's throughput target is a fraction of the reference's.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from . import factory_sim as fs
from .factory import FactoryPack, apply_ops

DIR_OF = {v: k for k, v in fs.DIRS.items()}
TIERS = {
    1: dict(defects=False, inlets=1, speed=(1.0, 1.6), feed=(40, 60), walls=(2, 3), turns=10, sim_seconds=240,
            noise=(2, 6), underfill=(0.9, 0.94)),
    2: dict(defects=True, inlets=1, speed=(1.2, 1.8), feed=(45, 65), walls=(1, 3), turns=10, sim_seconds=240,
            noise=(3, 12), underfill=(0.93, 0.965)),
    3: dict(defects=True, inlets=2, speed=(1.2, 1.8), feed=(30, 45), walls=(1, 3), turns=10, sim_seconds=240,
            noise=(3, 12), underfill=(0.92, 0.96)),
}
TARGET = 0.9      # agents must reach 90% of the reference build's good items per minute
MONEY_SLACK = 1.3  # and may spend at most 1.3x what the reference build cost


@dataclass
class Station:
    scale: tuple
    target: tuple
    pusher: tuple
    push_dir: str
    bins: list


def _route(free, start, start_dir, goals: dict, w, h):
    """Dijkstra over (cell, heading); goals maps cell -> required final heading (None = any).
    Cost 1 per cell, +2.5 per turn. Returns [(cell, dir)] or None."""
    pq = [(0.0, start, start_dir, ())]
    seen = {}
    while pq:
        cost, c, d, path = heapq.heappop(pq)
        if seen.get((c, d), 1e9) <= cost:
            continue
        seen[(c, d)] = cost
        for nd, (dx, dy) in fs.DIRS.items():
            if (-dx, -dy) == fs.DIRS[d]:
                continue
            # the belt at c points nd; next cell is c + nd
            n = (c[0] + dx, c[1] + dy)
            step = path + ((c, nd),)
            if n in goals and (goals[n] is None or goals[n] == nd):
                return list(step)
            if not (0 <= n[0] < w and 0 <= n[1] < h) or n not in free or n in {p for p, _ in step}:
                continue
            heapq.heappush(pq, (cost + 1 + (2.5 if nd != d else 0), n, nd, step))
    return None


def _station(path, free, used, w, h):
    """First straight run [scale, gap, target, next] with a pusher beside the target and two bins across
    (the gap gives the pusher time to fire), falling back to [scale, target, next]."""
    for gap in (1, 0):
        r = _station_gap(path, free, used, w, h, gap)
        if r:
            return r
    return None


def _station_gap(path, free, used, w, h, gap):
    for k in range(len(path) - 2 - gap):
        run = path[k:k + 3 + gap]
        if len({d for _, d in run}) != 1:
            continue
        (a, da), (b, _), (c, _) = run[0], run[1 + gap], run[2 + gap]
        ux, uy = fs.DIRS[da]
        for px, py in ((-uy, ux), (uy, -ux)):
            pcell = (b[0] + px, b[1] + py)
            bins = [(b[0] - px, b[1] - py), (c[0] - px, c[1] - py)]
            cells = [pcell, *bins]
            if all(0 <= x < w and 0 <= y < h and (x, y) in free and (x, y) not in used for x, y in cells):
                return k, Station(a, b, pcell, DIR_OF[(-px, -py)], bins)
    return None


def random_spec(seed: int, tier: int) -> dict | None:
    rng = np.random.default_rng(seed)
    t = TIERS[tier]
    w, h = int(rng.integers(12, 16)), int(rng.integers(8, 11))
    walls = set()
    for _ in range(int(rng.integers(*t["walls"], endpoint=True))):
        x = int(rng.integers(3, w - 3))
        y0, L = int(rng.integers(0, h - 3)), int(rng.integers(3, h - 1))
        walls |= {(x, y) for y in range(y0, min(h, y0 + L))}
        if rng.random() < 0.3:  # an L-shaped wall
            yy = min(h - 1, y0 + L - 1)
            walls |= {(xx, yy) for xx in range(x, min(w - 2, x + int(rng.integers(2, 4))))}
    ys = sorted(rng.choice(np.arange(1, h - 1), size=t["inlets"], replace=False).tolist())
    if t["inlets"] == 2 and ys[1] - ys[0] < 3:
        return None
    speed = round(float(rng.uniform(*t["speed"])), 1)
    parts = [fs.Part("inlet", 0, y, "E", {"speed": speed, "feed_per_min": int(rng.integers(*t["feed"]))}, True, "AB"[k])
             for k, y in enumerate(ys)]
    by = int(rng.integers(1, h - 1))
    parts.append(fs.Part("bin", w - 1, by, "E", {}, True, "ship"))
    walls -= {(0, y) for y in ys} | {(1, y) for y in ys} | {(w - 1, by), (w - 2, by)}
    mixes = {}
    for p in parts:
        if p.kind != "inlet":
            continue
        if not t["defects"]:
            mixes[p.name] = [["good", 1.0, 1.0, 0.01]]
        else:
            share = round(float(rng.uniform(0.15, 0.35)), 2)
            if rng.random() < 0.5:
                mixes[p.name] = [["good", 1 - share, 1.0, 0.008],
                                 ["underfilled", share, round(float(rng.uniform(*t["underfill"])), 3), 0.008]]
            else:
                mixes[p.name] = [["good", 1 - share, 1.0, 0.008], ["double", share, round(float(rng.uniform(1.8, 2.0)), 2), 0.03]]
    sub = np.random.default_rng(seed + 991)
    hidden = {"belt_mu": float(sub.uniform(0.3, 0.8)), "pusher_speed": float(sub.uniform(0.7, 1.8)),
              "pusher_lag": float(sub.uniform(0.02, 0.12)), "scale_noise_g": float(np.exp(sub.uniform(np.log(t["noise"][0]), np.log(t["noise"][1]))))}
    item = str(rng.choice(["box", "box", "bob", "blub"] + (["spot"] if tier >= 2 else [])))
    if item != "box":  # toys weigh less than packed boxes
        for m in mixes.values():
            for e in m:
                e[2] = round(e[2] * 0.4, 3)
                e[3] = round(e[3] * 0.4, 4)
    kinds = sorted({k for m in mixes.values() for k, *_ in m if k != "good"})
    # defects are not named: the agent has to find out from weights what "out of spec" means here
    brief = "Get every box to the shipping bin." if not kinds else "Some boxes leave filling out of spec. Ship only good ones."
    if t["inlets"] == 2:
        brief = "Two lines, one dock. " + brief
    toy = {"bob": "This line makes rubber ducks (Bob): 18 cm long, 9 cm tall, about 400 g. ",
           "blub": "This line makes toy fish (Blub): 22 cm long, lying on their side, about 400 g. ",
           "spot": "This line makes toy cows (Spot): 20 cm long and 20 cm tall on four legs, about 400 g. Tall toys tip. "}
    brief = toy.get(item, "") + brief.replace("boxes", "items" if item != "box" else "boxes")
    return {"title": f"Plant {seed}", "w": w, "h": h, "walls": sorted(walls), "parts": [p.to_dict() for p in parts],
            "mixes": mixes, "hidden": hidden, "brief": brief, "tier": tier, "max_bad": 0,
            "item": item, "money": 0, "turns": t["turns"], "sim_seconds": t["sim_seconds"], "target": TARGET}


def reference_ops(spec: dict, v: float, thr: tuple, off: float, stroke: float) -> str | None:
    lay = fs.Layout(spec["w"], spec["h"], [fs.Part.from_dict(p) for p in spec["parts"]], spec["walls"])
    w, h = lay.w, lay.h
    free = {(x, y) for x in range(w) for y in range(h)} - lay.walls - set(lay.parts)
    ship = next(p for p in lay.parts.values() if p.kind == "bin")
    inlets = [p for p in lay.parts.values() if p.kind == "inlet"]
    path = _route(free, (1, inlets[0].y), "E", {(ship.x, ship.y): None}, w, h)
    if path is None:
        return None
    used = {c for c, _ in path}
    paths = [path]
    join_k = 0
    for p in inlets[1:]:
        goals = {c: None for c, _ in path[2:]}
        b = _route(free - used, (1, p.y), "E", goals, w, h)
        if b is None:
            return None
        nxt = (b[-1][0][0] + fs.DIRS[b[-1][1]][0], b[-1][0][1] + fs.DIRS[b[-1][1]][1])
        k = next(i for i, (c, _) in enumerate(path) if c == nxt)
        if fs.DIRS[path[k][1]] == tuple(-x for x in fs.DIRS[b[-1][1]]):
            return None  # would feed head-on
        join_k = max(join_k, k + 1)
        used |= {c for c, _ in b}
        paths.append(b)
    defects = any(k != "good" for m in spec["mixes"].values() for k, *_ in m)
    stations = []  # (station, thresholds)
    if defects:
        found = _station(path[join_k:], free, used, w, h)
        if found is not None:
            stations = [(found[1], thr)]
        else:  # one station per line, before the merge
            taken = set(used)
            for pth, inl in zip(paths, inlets):
                if all(k == "good" for k, *_ in spec["mixes"][inl.name]):
                    continue
                stop = join_k - 1 if pth is path else len(pth)
                f = _station(pth[:stop], free, taken, w, h)
                if f is None:
                    return None
                taken |= {f[1].pusher, *f[1].bins}
                stations.append((f[1], line_thresholds(spec["mixes"][inl.name], thr)))
    scale_at = {st.scale: th for st, th in stations}
    ops = []
    for pth in paths:
        for c, d in pth:
            if c in scale_at:
                lo, hi = scale_at[c]
                kv = (f" reject_below_g={lo:.0f}" if lo else "") + (f" reject_above_g={hi:.0f}" if hi else "")
                ops.append(f"scale {c[0]} {c[1]} {d} speed={v}{kv}")
            else:
                ops.append(f"belt {c[0]} {c[1]} {d} speed={v}")
    for st, _ in stations:
        ops.append(f"pusher {st.pusher[0]} {st.pusher[1]} {st.push_dir} offset_mm={off:.0f} stroke_ms={stroke:.0f}")
        ops += [f"bin {x} {y}" for x, y in st.bins]
    return "\n".join(ops)


def line_thresholds(mix, shifted) -> tuple:
    """Thresholds for one line: keep only the sides this line's defects need (shifted = global search value)."""
    kinds = {k for k, *_ in mix}
    return (shifted[0] if "underfilled" in kinds else 0.0, shifted[1] if "double" in kinds else 0.0)


def thresholds(spec: dict) -> tuple:
    lo = hi = 0.0
    for m in spec["mixes"].values():
        good = next(e[2] for e in m if e[0] == "good")
        for k, _, mu, _ in m:
            if k == "underfilled":
                lo = max(lo, 1000 * (good + mu) / 2)
            if k == "double":
                hi = 1000 * (good + mu) / 2 if not hi else min(hi, 1000 * (good + mu) / 2)
    return lo, hi


def solve_spec(spec: dict, seed: int) -> tuple[dict, dict] | None:
    """Search the reference build with the true physics; return (best ops+settings, production truth) if it passes."""
    from ..core import Episode, ToolCall
    hid = fs.Hidden(**spec["hidden"], mixes=spec["mixes"], item=spec.get("item", "box"))
    thr = thresholds(spec)
    defects = any(thr)
    best = None

    def sc(t):
        return t["good_shipped"] - 5 * t["bad_shipped"] - 2 * t["spilled"] - 3 * t["jammed"]

    shifts = (-12, 0, 12) if defects else (0,)
    grid = itertools.product((0.8, 1.1, 1.4, 1.7), range(50, 701, 100) if defects else (0,), (250, 450) if defects else (0,), shifts)
    for v, off, stroke, sh in grid:
        th = (thr[0] + sh if thr[0] else 0, thr[1] + sh if thr[1] else 0)
        ops = reference_ops(spec, v, th, off, stroke)
        if ops is None:
            return None
        lay = fs.Layout(spec["w"], spec["h"], [fs.Part.from_dict(p) for p in spec["parts"]], spec["walls"])
        apply_ops(lay, ops, 1e9)
        t = fs.run_factory(lay, hid, 45, seed=123).truth()
        if best is None or sc(t) > best[0]:
            best = (sc(t), ops, v, off, stroke, th)
    scen = FactoryPack().generate(seed, map=f"gen{seed}", spec={**spec, "target_abs": 0})
    ep = Episode(scen.world, scen.task)
    ep.world.settings["money"] = 1e9
    ep.call(ToolCall("build", args={"ops": best[1]}))
    ep.call(ToolCall("finish"))
    prod = ep.result["production"]
    ok = prod["bad_shipped"] == 0 and prod["spilled"] == 0 and prod["jammed"] == 0 and prod["good_per_min"] > 0
    cost = 1e9 - ep.world.settings["money"]
    return ({"ops": best[1], "speed": best[2], "offset_mm": best[3], "stroke_ms": best[4], "thresholds": best[5],
             "cost": cost}, prod) if ok else None
