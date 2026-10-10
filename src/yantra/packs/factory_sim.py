"""Physical factory on a grid: the agent places parts, MuJoCo runs the floor.

Parts sit on 0.5 m cells. Belts (and scales, which are belts that weigh) carry boxes in their
direction with Coulomb-capped traction, so turns, merges and speed changes behave like real
conveyors: fast boxes overshoot a turn into the rail, crowd at merges, slide off open ends. Rails are
added automatically on every belt edge that nothing feeds into. Pushers knock boxes flagged by a
scale sideways off a belt, into whatever is there (a bin, another belt, or the floor). Bins count
what lands in them and weigh it (bins stand on platform scales).

Everything is deterministic given a seed. A run returns what a plant engineer would see (bin counts
and weights, scale readings, spills and jams by cell) plus the truth used for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import math

import mujoco
import numpy as np

G = 9.81
DT = 0.004
C = 0.5                      # cell size (m)
BZ = 0.40                    # belt top height (m)
BOX = 0.14                   # box edge (m)
NOMINAL_KG = 1.00            # a good box
DIRS = {"E": (1, 0), "N": (0, 1), "W": (-1, 0), "S": (0, -1)}
CONVEY = ("belt", "scale", "inlet")
MAX_ITEMS = 48


@dataclass
class Part:
    kind: str                # belt | scale | pusher | bin | inlet
    x: int
    y: int
    dir: str = "E"           # flow direction (belt/scale/inlet) or push direction (pusher)
    params: dict = field(default_factory=dict)
    fixed: bool = False      # placed by the level, cannot be removed
    name: str = ""           # bins: "ship" / "reject"; inlets: "A", "B"...

    def to_dict(self) -> dict:
        return {"kind": self.kind, "x": self.x, "y": self.y, "dir": self.dir, "params": dict(self.params),
                "fixed": self.fixed, "name": self.name}

    @staticmethod
    def from_dict(d: dict) -> "Part":
        return Part(d["kind"], d["x"], d["y"], d.get("dir", "E"), dict(d.get("params", {})), d.get("fixed", False),
                    d.get("name", ""))


@dataclass
class Hidden:
    belt_mu: float = 0.5           # box-belt traction
    pusher_speed: float = 1.2      # paddle speed (m/s)
    pusher_lag: float = 0.05       # valve delay (s)
    scale_noise_g: float = 4.0     # load-cell noise at 0.5 s dwell (g)
    # per inlet: list of (kind, share, mass_kg mean, sd)
    mixes: dict = field(default_factory=dict)
    item: str = "box"              # box, or a toy from assets/items.json (spot, bob, blub)


_ITEMS: dict | None = None


def item_shape(item: str) -> dict:
    """Collision hull and resting geometry of an item. Boxes are analytic; toys are convex hulls."""
    global _ITEMS
    if item == "box":
        return {"rest": BOX / 2, "half": BOX / 2, "clear": BOX * 1.3, "floor": BOX}
    if _ITEMS is None:
        import json
        from pathlib import Path
        _ITEMS = json.loads((Path(__file__).parent / "assets" / "items.json").read_text())
    it = _ITEMS[item]
    return {"rest": it["rest"], "half": it["half"], "clear": 2.2 * it["half"], "floor": it["rest"] + 0.05,
            "geoms": it["geoms"], "inertia": it["inertia"]}


def _item_body(i: int, item: str) -> str:
    """One item body. Toys: primitives plus an explicit inertial (see scripts/make_items.py)."""
    head = f'<body name="it{i}" pos="{-5 - i * 0.3} -5 0.1"><freejoint/>'
    common = 'friction="0 0 0" solref="0.008 1"'
    if item == "box":
        return head + f'<geom name="itg{i}" type="box" size="{BOX / 2} {BOX / 2} {BOX / 2}" mass="1" {common}/></body>'
    shp = item_shape(item)
    ix, iy, iz = shp["inertia"]
    out = head + f'<inertial pos="0 0 0" mass="1" diaginertia="{ix} {iy} {iz}"/>'
    for k, g in enumerate(shp["geoms"]):
        where = (f'fromto="{" ".join(str(v) for v in g["fromto"])}"' if "fromto" in g
                 else f'pos="{" ".join(str(v) for v in g["pos"])}"')
        name = f' name="itg{i}"' if k == 0 else ""
        out += f'<geom{name} type="{g["type"]}" size="{" ".join(str(v) for v in g["size"])}" {where} {common}/>'
    return out + "</body>"


def cell_center(x: int, y: int) -> tuple[float, float]:
    return (x + 0.5) * C, (y + 0.5) * C


class Layout:
    def __init__(self, w: int, h: int, parts: list[Part], walls: list[tuple[int, int]] | None = None):
        self.w, self.h = w, h
        self.walls = {tuple(c) for c in (walls or [])}
        self.parts: dict[tuple[int, int], Part] = {}
        for p in parts:
            self.parts[(p.x, p.y)] = p

    def copy(self) -> "Layout":
        return Layout(self.w, self.h, [Part.from_dict(p.to_dict()) for p in self.parts.values()], list(self.walls))

    def to_dict(self) -> dict:
        return {"w": self.w, "h": self.h, "walls": sorted(self.walls), "parts": [p.to_dict() for p in self.parts.values()]}

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        return Layout(d["w"], d["h"], [Part.from_dict(p) for p in d["parts"]], [tuple(c) for c in d.get("walls", [])])

    def conveys(self, c) -> Part | None:
        p = self.parts.get(c)
        return p if p and p.kind in CONVEY else None

    def feeds_into(self, src: tuple[int, int], dst: tuple[int, int]) -> bool:
        p = self.conveys(src)
        if not p:
            return False
        dx, dy = DIRS[p.dir]
        return (src[0] + dx, src[1] + dy) == dst

    def ascii(self) -> str:
        sym = {"E": ">", "W": "<", "N": "^", "S": "v"}
        rows = []
        for y in range(self.h - 1, -1, -1):
            row = []
            for x in range(self.w):
                p = self.parts.get((x, y))
                if (x, y) in self.walls:
                    row.append("#")
                elif p is None:
                    row.append(".")
                elif p.kind == "belt":
                    row.append(sym[p.dir])
                elif p.kind == "scale":
                    row.append("W")
                elif p.kind == "pusher":
                    row.append("P")
                elif p.kind == "inlet":
                    row.append("I")
                elif p.kind == "bin":
                    row.append("G" if p.name == "ship" else "R")
            rows.append(f"{y:2d} " + " ".join(row))
        rows.append("   " + " ".join(str(x % 10) for x in range(self.w)))
        return "\n".join(rows)


# ---------------------------------------------------------------------- scene
def _scene(lay: Layout, item: str = "box") -> tuple[str, list]:
    g = []
    rails = []
    for (x, y), p in lay.parts.items():
        cx, cy = cell_center(x, y)
        if p.kind in CONVEY:
            slab = f'type="box" size="{C / 2} {C / 2} 0.02" friction="0 0 0"'
            if p.kind == "scale":  # its own static body: the load cell reads that body's contact force
                g.append(f'<body name="sc_{x}_{y}" pos="{cx} {cy} {BZ - 0.02}"><geom name="cv_{x}_{y}" {slab}/></body>')
            else:
                g.append(f'<geom name="cv_{x}_{y}" pos="{cx} {cy} {BZ - 0.02}" {slab}/>')
            dx, dy = DIRS[p.dir]
            for ex, ey in DIRS.values():
                if (ex, ey) == (dx, dy):
                    continue  # front edge open
                nb = (x + ex, y + ey)
                if lay.feeds_into(nb, (x, y)):
                    continue
                q = lay.parts.get(nb)
                if q and q.kind == "pusher" and DIRS[q.dir] == (-ex, -ey):
                    continue  # the paddle comes in from here
                if any(pp.kind == "pusher" and (pp.x + DIRS[pp.dir][0], pp.y + DIRS[pp.dir][1]) == (x, y)
                       and DIRS[pp.dir] == (ex, ey) for pp in lay.parts.values()):
                    continue  # pushed boxes leave through here
                rx, ry = cx + ex * (C / 2 - 0.01), cy + ey * (C / 2 - 0.01)
                sx, sy = (0.01, C / 2) if ex else (C / 2, 0.01)
                rails.append((x, y, ex, ey))
                g.append(f'<geom type="box" pos="{rx} {ry} {BZ + 0.04}" size="{sx} {sy} 0.04" friction="0.15 0 0"/>')
        elif p.kind == "bin":
            g.append(f'<geom name="bin_{x}_{y}" type="box" pos="{cx} {cy} 0.05" size="{C / 2} {C / 2} 0.05" friction="0.8 0 0"/>')
            for ex, ey in DIRS.values():
                rx, ry = cx + ex * (C / 2 - 0.01), cy + ey * (C / 2 - 0.01)
                sx, sy = (0.01, C / 2) if ex else (C / 2, 0.01)
                g.append(f'<geom type="box" pos="{rx} {ry} {0.1 + 0.1}" size="{sx} {sy} 0.1"/>')
    for (x, y) in lay.walls:
        cx, cy = cell_center(x, y)
        g.append(f'<geom type="box" pos="{cx} {cy} 0.6" size="{C / 2 - 0.02} {C / 2 - 0.02} 0.6"/>')
    pushers = []
    for (x, y), p in lay.parts.items():
        if p.kind != "pusher":
            continue
        dx, dy = DIRS[p.dir]
        cx, cy = cell_center(x, y)
        rest = (cx + dx * (C / 2 - 0.06), cy + dy * (C / 2 - 0.06))
        sx, sy = (0.03, 0.12) if dx else (0.12, 0.03)
        pushers.append(p)
        g.append(f'<body name="push_{x}_{y}" mocap="true" pos="{rest[0]} {rest[1]} {BZ + 0.07}">'
                 f'<geom type="box" size="{sx} {sy} 0.05" friction="0.2 0 0" solref="0.008 1"/></body>')
    items = "".join(_item_body(i, item) for i in range(MAX_ITEMS))
    xml = f"""<mujoco><option timestep="{DT}" cone="elliptic" impratio="10"/>
    <worldbody><geom name="floor" type="plane" size="40 40 .1" friction="0.8 0 0"/>
    {''.join(g)}{items}</worldbody></mujoco>"""
    return xml, rails


@dataclass
class RunReport:
    seconds: float
    emitted: dict            # inlet name -> count
    bins: dict               # bin name -> {"count", "weight_kg", "kinds": {kind: n}}
    spills: dict             # "x,y" -> count  (boxes on the floor, nearest cell)
    jams: dict               # "x,y" -> count  (boxes stuck >3 s on a belt)
    blocked_inlet: int       # boxes the inlet could not release (no room)
    in_transit: int
    scales: dict             # "x,y" -> list of readings (g)
    pusher_fires: dict       # "x,y" -> count
    frames: list = field(default_factory=list)

    def public(self) -> dict:
        return {"seconds": round(self.seconds, 1), "emitted": self.emitted,
                "bins": {k: {"count": v["count"], "weight_kg": round(v["weight_kg"], 3)} for k, v in self.bins.items()},
                "spills_by_cell": self.spills, "jams_by_cell": self.jams, "blocked_at_inlet": self.blocked_inlet,
                "in_transit": self.in_transit,
                "scale_readings_g": {k: [round(r) for r in v] for k, v in self.scales.items()},
                "pusher_fires": self.pusher_fires}

    def truth(self) -> dict:
        ship = self.bins.get("ship", {"kinds": {}})["kinds"]
        good = ship.get("good", 0)
        bad = sum(v for k, v in ship.items() if k != "good")
        rejected_good = sum(b["kinds"].get("good", 0) for n, b in self.bins.items() if n != "ship")
        return {"good_shipped": good, "bad_shipped": bad, "good_rejected": rejected_good,
                "good_per_min": round(good / self.seconds * 60, 1), "spilled": sum(self.spills.values()),
                "jammed": sum(self.jams.values())}


class Rng:
    """mulberry32 + Box-Muller: tiny, and reproduced exactly by the browser port (web/src/sim/factory.ts)."""

    def __init__(self, seed: int):
        self.a = seed & 0xFFFFFFFF

    def random(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    def normal(self, mu: float, sd: float) -> float:
        u = 1.0 - self.random()
        v = self.random()
        return mu + sd * math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def run_factory(lay: Layout, hid: Hidden, seconds: float, seed: int, record_fps: float = 0.0) -> RunReport:
    """Keep in step with runFactory in web/src/sim/factory.ts (same order of operations)."""
    rng = Rng(seed)
    shp = item_shape(hid.item)
    xml, _ = _scene(lay, hid.item)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    it_bid = [m.body(f"it{i}").id for i in range(MAX_ITEMS)]
    q0 = m.jnt_qposadr[m.body_jntadr[it_bid[0]]]
    v0 = m.jnt_dofadr[m.body_jntadr[it_bid[0]]]
    parts = list(lay.parts.values())
    conv = {(p.x, p.y): p for p in parts if p.kind in CONVEY}
    scales = [p for p in parts if p.kind == "scale"]
    scale_bid = {(p.x, p.y): m.body(f"sc_{p.x}_{p.y}").id for p in scales}
    bins = {(p.x, p.y): p for p in parts if p.kind == "bin"}
    inlets = [p for p in parts if p.kind == "inlet"]
    pushers = [p for p in parts if p.kind == "pusher"]
    push_mocap = [int(m.body_mocapid[m.body(f"push_{p.x}_{p.y}").id]) for p in pushers]
    push_rest = [[float(v) for v in d.mocap_pos[i]] for i in push_mocap]
    inertia0 = [float(v) for v in m.body_inertia[it_bid[0]]]  # at mass 1

    active = [False] * MAX_ITEMS
    kind = [""] * MAX_ITEMS
    mass = [1.0] * MAX_ITEMS
    flagged = [False] * MAX_ITEMS
    pushed = [False] * MAX_ITEMS
    still = [0.0] * MAX_ITEMS
    jam_logged = [False] * MAX_ITEMS
    weighing: list = [None] * MAX_ITEMS     # (scale cell, samples) while a box is fully on a scale
    free_ids = list(range(MAX_ITEMS))
    report = RunReport(seconds, {p.name: 0 for p in inlets},
                       {p.name: {"count": 0, "weight_kg": 0.0, "kinds": {}} for p in bins.values()},
                       {}, {}, 0, 0, {f"{p.x},{p.y}": [] for p in scales}, {f"{p.x},{p.y}": 0 for p in pushers})
    next_emit = [0.3 + 0.1 * k for k in range(len(inlets))]
    fire_at: list[list[float]] = [[] for _ in pushers]
    extend_until = [-1.0] * len(pushers)
    every = int(round(1 / (record_fps * DT))) if record_fps else 0
    rest_z = BZ + shp["rest"]
    half_in = C / 2 - shp["half"] - 0.01
    load = {c: 0.0 for c in scale_bid}

    def park(i):
        active[i] = False
        d.qpos[q0 + 7 * i:q0 + 7 * i + 7] = [-5 - i * 0.3, -5, 0.1, 1, 0, 0, 0]
        d.qvel[v0 + 6 * i:v0 + 6 * i + 6] = 0
        d.xfrc_applied[it_bid[i], :] = 0
        weighing[i] = None
        free_ids.append(i)

    nsteps = int(seconds / DT)
    for step in range(nsteps):
        t = step * DT
        # ---------------- inlets release boxes at their feed rate when there is room
        for k, p in enumerate(inlets):
            if t < next_emit[k]:
                continue
            next_emit[k] += 60.0 / float(p.params.get("feed_per_min", 30))
            cx, cy = cell_center(p.x, p.y)
            dx, dy = DIRS[p.dir]
            sx, sy = cx - dx * 0.12, cy - dy * 0.12
            room = True
            for i in range(MAX_ITEMS):
                if active[i]:
                    ex, ey = d.xpos[it_bid[i]][0] - sx, d.xpos[it_bid[i]][1] - sy
                    if math.sqrt(ex * ex + ey * ey) <= shp["clear"]:
                        room = False
                        break
            if not free_ids or not room:
                report.blocked_inlet += 1
                continue
            i = free_ids.pop(0)
            mix = hid.mixes.get(p.name, [["good", 1.0, NOMINAL_KG, 0.01]])
            total = sum(float(e[1]) for e in mix)
            u, acc, pick = rng.random() * total, 0.0, mix[-1]
            for e in mix:
                acc += float(e[1])
                if u < acc:
                    pick = e
                    break
            kind[i] = pick[0]
            mass[i] = max(0.2, rng.normal(float(pick[2]), float(pick[3])))
            m.body_mass[it_bid[i]] = mass[i]
            m.body_inertia[it_bid[i]] = [mass[i] * v for v in inertia0]
            active[i] = True
            flagged[i] = pushed[i] = jam_logged[i] = False
            still[i] = 0.0
            weighing[i] = None
            d.qpos[q0 + 7 * i:q0 + 7 * i + 7] = [sx, sy, rest_z + 0.001, 1, 0, 0, 0]
            d.qvel[v0 + 6 * i:v0 + 6 * i + 6] = 0
            report.emitted[p.name] += 1
        # ---------------- belts: Coulomb-capped traction on boxes resting on a conveying cell
        for i in range(MAX_ITEMS):
            if not active[i]:
                continue
            x, y, z = d.xpos[it_bid[i]]
            c = (math.floor(x / C), math.floor(y / C))
            p = conv.get(c)
            fx = fy = 0.0
            if p is not None and abs(z - rest_z) < 0.02:
                ux, uy = DIRS[p.dir]
                sp = float(p.params.get("speed", 1.0))
                wx = ux * sp - d.qvel[v0 + 6 * i]
                wy = uy * sp - d.qvel[v0 + 6 * i + 1]
                fx, fy = mass[i] * wx / DT, mass[i] * wy / DT
                cap = hid.belt_mu * mass[i] * G
                n = math.sqrt(fx * fx + fy * fy)
                if n > cap:
                    fx, fy = fx * cap / n, fy * cap / n
            # the belt grips the base, not the centre of mass: same force plus torque r x F, r = (0, 0, -rest)
            h = shp["rest"]
            d.xfrc_applied[it_bid[i], 0] = fx
            d.xfrc_applied[it_bid[i], 1] = fy
            d.xfrc_applied[it_bid[i], 3] = h * fy
            d.xfrc_applied[it_bid[i], 4] = -h * fx
        # ---------------- scales: sample the load cell while a box is fully on the platform
        for i in range(MAX_ITEMS):
            if not active[i]:
                continue
            x, y, z = d.xpos[it_bid[i]]
            c = (math.floor(x / C), math.floor(y / C))
            on = None
            if c in scale_bid and abs(z - rest_z) < 0.02:
                cx, cy = cell_center(*c)
                if abs(x - cx) < half_in and abs(y - cy) < half_in:
                    on = c
            w = weighing[i]
            if on is not None and (w is None or w[0] == on):
                if w is None:
                    w = weighing[i] = (on, [])
                w[1].append(load[on])
            elif w is not None:  # left the platform: report the mean load plus load-cell noise
                weighing[i] = None
                n = len(w[1])
                dwell = n * DT
                reading = sum(w[1]) / n / G * 1000 + rng.normal(0.0, hid.scale_noise_g * math.sqrt(0.5 / max(dwell, 1e-3)))
                report.scales[f"{w[0][0]},{w[0][1]}"].append(reading)
                prm = lay.parts[w[0]].params
                if reading < float(prm.get("reject_below_g", -1e9)) or reading > float(prm.get("reject_above_g", 1e9)):
                    flagged[i] = True
        # ---------------- pushers: fire when a flagged box is `offset_mm` before the paddle line
        for j, p in enumerate(pushers):
            dx, dy = DIRS[p.dir]
            tc = (p.x + dx, p.y + dy)
            tp = conv.get(tc)
            if tp is not None:
                ux, uy = DIRS[tp.dir]
                tcx, tcy = cell_center(*tc)
                off = float(p.params.get("offset_mm", 50)) / 1000
                for i in range(MAX_ITEMS):
                    if not (active[i] and flagged[i] and not pushed[i]):
                        continue
                    x, y = d.xpos[it_bid[i]][0], d.xpos[it_bid[i]][1]
                    along = (x - tcx) * ux + (y - tcy) * uy
                    across = (x - tcx) * dx + (y - tcy) * dy
                    if abs(across) < C / 2 and along > -C / 2 - 0.3 and along >= -off:
                        pushed[i] = True
                        fire_at[j].append(t + hid.pusher_lag)
            if fire_at[j] and t >= fire_at[j][0]:
                fire_at[j].pop(0)
                extend_until[j] = t + float(p.params.get("stroke_ms", 200)) / 1000
                report.pusher_fires[f"{p.x},{p.y}"] += 1
            ext = C * 1.2 if t < extend_until[j] else 0.0
            rest = push_rest[j]
            mp = d.mocap_pos[push_mocap[j]]
            tx, ty = rest[0] + dx * ext, rest[1] + dy * ext
            ddx, ddy = tx - mp[0], ty - mp[1]
            dist = math.sqrt(ddx * ddx + ddy * ddy)
            step_len = hid.pusher_speed * DT
            if dist <= step_len:
                mp[0], mp[1] = tx, ty
            else:
                mp[0], mp[1] = mp[0] + ddx / dist * step_len, mp[1] + ddy / dist * step_len
        mujoco.mj_step(m, d)
        if scale_bid:
            mujoco.mj_rnePostConstraint(m, d)
            for c, b in scale_bid.items():
                load[c] = abs(float(d.cfrc_ext[b][5]))
        # ---------------- bookkeeping: bins, spills, jams
        for i in range(MAX_ITEMS):
            if not active[i]:
                continue
            x, y, z = d.xpos[it_bid[i]]
            c = (math.floor(x / C), math.floor(y / C))
            if z < BZ - 0.12:
                if c in bins and z < 0.35:
                    b = report.bins[bins[c].name]
                    b["count"] += 1
                    b["weight_kg"] += mass[i]
                    b["kinds"][kind[i]] = b["kinds"].get(kind[i], 0) + 1
                    park(i)
                elif z < shp["floor"]:
                    key = f"{c[0]},{c[1]}"
                    report.spills[key] = report.spills.get(key, 0) + 1
                    park(i)
                continue
            vx, vy = d.qvel[v0 + 6 * i], d.qvel[v0 + 6 * i + 1]
            still[i] = still[i] + DT if math.sqrt(vx * vx + vy * vy) < 0.02 else 0.0
            if still[i] > 3.0 and not jam_logged[i]:
                jam_logged[i] = True
                key = f"{c[0]},{c[1]}"
                report.jams[key] = report.jams.get(key, 0) + 1
        if every and step % every == 0:
            report.frames.append({
                "t": round(t, 2),
                "items": [[i, kind[i] != "good", *[round(float(v), 3) for v in d.xpos[it_bid[i]]],
                           *[round(float(v), 3) for v in d.xquat[it_bid[i]]]] for i in range(MAX_ITEMS) if active[i]],
                "push": [round(math.dist(d.mocap_pos[push_mocap[j]][:2], push_rest[j][:2]), 3) for j in range(len(pushers))],
            })
    report.in_transit = sum(active)
    return report
