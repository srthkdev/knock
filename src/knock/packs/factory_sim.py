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
def _scene(lay: Layout) -> tuple[str, list]:
    g = []
    rails = []
    for (x, y), p in lay.parts.items():
        cx, cy = cell_center(x, y)
        if p.kind in CONVEY:
            g.append(f'<geom name="cv_{x}_{y}" type="box" pos="{cx} {cy} {BZ - 0.02}" size="{C / 2} {C / 2} 0.02" friction="0 0 0"/>')
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
    items = "".join(
        f'<body name="it{i}" pos="{-5 - i * 0.3} -5 0.1"><freejoint/>'
        f'<geom name="itg{i}" type="box" size="{BOX / 2} {BOX / 2} {BOX / 2}" mass="1" friction="0 0 0" solref="0.008 1"/></body>'
        for i in range(MAX_ITEMS))
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


def run_factory(lay: Layout, hid: Hidden, seconds: float, seed: int, record_fps: float = 0.0) -> RunReport:
    rng = np.random.default_rng(seed)
    xml, _ = _scene(lay)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    it_bid = np.array([m.body(f"it{i}").id for i in range(MAX_ITEMS)])
    it_geom = np.array([m.geom(f"itg{i}").id for i in range(MAX_ITEMS)])
    geom_item = -np.ones(m.ngeom, int)
    geom_item[it_geom] = np.arange(MAX_ITEMS)
    q0 = m.jnt_qposadr[m.body_jntadr[it_bid[0]]]
    v0 = m.jnt_dofadr[m.body_jntadr[it_bid[0]]]
    conv_cells = [c for c, p in lay.parts.items() if p.kind in CONVEY]
    geom_conv = -np.ones(m.ngeom, int)
    for k, (x, y) in enumerate(conv_cells):
        geom_conv[m.geom(f"cv_{x}_{y}").id] = k
    conv_dir = np.array([DIRS[lay.parts[c].dir] for c in conv_cells], float).reshape(-1, 2)
    conv_speed = np.array([float(lay.parts[c].params.get("speed", 1.0)) for c in conv_cells])
    cell_index = {c: k for k, c in enumerate(conv_cells)}
    scale_cells = [c for c in conv_cells if lay.parts[c].kind == "scale"]
    scale_gid = {m.geom(f"cv_{x}_{y}").id: (x, y) for (x, y) in scale_cells}
    bins = {c: p for c, p in lay.parts.items() if p.kind == "bin"}
    inlets = [p for p in lay.parts.values() if p.kind == "inlet"]
    pushers = [p for p in lay.parts.values() if p.kind == "pusher"]
    push_mocap = [m.body_mocapid[m.body(f"push_{p.x}_{p.y}").id] for p in pushers]
    push_rest = [d.mocap_pos[i].copy() for i in push_mocap]

    # item state
    active = np.zeros(MAX_ITEMS, bool)
    kind = [""] * MAX_ITEMS
    mass = np.ones(MAX_ITEMS)
    flagged = np.zeros(MAX_ITEMS, bool)
    pushed = np.zeros(MAX_ITEMS, bool)
    still = np.zeros(MAX_ITEMS)
    jam_logged = np.zeros(MAX_ITEMS, bool)
    samples: dict[tuple[int, int], list[float]] = {}
    on_scale = -np.ones(MAX_ITEMS, int)  # index into scale_cells
    report = RunReport(seconds, {p.name: 0 for p in inlets}, {p.name: {"count": 0, "weight_kg": 0.0, "kinds": {}} for p in bins.values()},
                       {}, {}, 0, 0, {f"{x},{y}": [] for (x, y) in scale_cells}, {f"{p.x},{p.y}": 0 for p in pushers})
    next_emit = {p.name: 0.3 + 0.1 * k for k, p in enumerate(inlets)}
    fire_at = [[] for _ in pushers]
    extend_until = [-1.0] * len(pushers)
    f6 = np.zeros(6)
    every = int(round(1 / (record_fps * DT))) if record_fps else 0
    free_ids = list(range(MAX_ITEMS))

    def park(i):
        active[i] = False
        d.qpos[q0 + 7 * i:q0 + 7 * i + 7] = [-5 - i * 0.3, -5, 0.1, 1, 0, 0, 0]
        d.qvel[v0 + 6 * i:v0 + 6 * i + 6] = 0
        free_ids.append(i)

    def sample_kind(inlet: Part):
        mix = hid.mixes.get(inlet.name, [["good", 1.0, NOMINAL_KG, 0.01]])
        shares = np.array([m_[1] for m_ in mix], float)
        k = rng.choice(len(mix), p=shares / shares.sum())
        name, _, mu, sd = mix[k]
        return name, float(max(0.2, rng.normal(mu, sd)))

    nsteps = int(seconds / DT)
    for step in range(nsteps):
        t = step * DT
        # ---------------- inlets
        for p in inlets:
            if t >= next_emit[p.name]:
                cx, cy = cell_center(p.x, p.y)
                dx, dy = DIRS[p.dir]
                sx, sy = cx - dx * 0.12, cy - dy * 0.12
                pos = d.xpos[it_bid[active]][:, :2] if active.any() else np.zeros((0, 2))
                if free_ids and (len(pos) == 0 or np.min(np.hypot(pos[:, 0] - sx, pos[:, 1] - sy)) > BOX * 1.3):
                    i = free_ids.pop(0)
                    kind[i], mass[i] = sample_kind(p)
                    m.body_mass[it_bid[i]] = mass[i]
                    m.body_inertia[it_bid[i]] = mass[i] * BOX ** 2 / 6
                    active[i] = True
                    flagged[i] = pushed[i] = jam_logged[i] = False
                    still[i] = 0
                    d.qpos[q0 + 7 * i:q0 + 7 * i + 7] = [sx, sy, BZ + BOX / 2 + 0.001, 1, 0, 0, 0]
                    d.qvel[v0 + 6 * i:v0 + 6 * i + 6] = 0
                    report.emitted[p.name] += 1
                else:
                    report.blocked_inlet += 1
                next_emit[p.name] += 60.0 / float(p.params.get("feed_per_min", 30))
        # ---------------- contacts: which belt each box rides, scale loads
        nc = d.ncon
        support = -np.ones(MAX_ITEMS, int)
        scale_load: dict[tuple[int, int], float] = {}
        scale_touch: dict[int, tuple[int, int]] = {}
        if nc:
            g1 = d.contact.geom1[:nc] if hasattr(d.contact, "geom1") else np.array([d.contact[k].geom1 for k in range(nc)])
            g2 = d.contact.geom2[:nc] if hasattr(d.contact, "geom2") else np.array([d.contact[k].geom2 for k in range(nc)])
            for a, b in ((g1, g2), (g2, g1)):
                sel = (geom_item[a] >= 0) & (geom_conv[b] >= 0)
                support[geom_item[a][sel]] = geom_conv[b][sel]
            for k in np.flatnonzero(np.isin(g1, list(scale_gid)) | np.isin(g2, list(scale_gid))):
                gi, gs = (g1[k], g2[k]) if geom_item[g1[k]] >= 0 else (g2[k], g1[k])
                if geom_item[gi] < 0 or gs not in scale_gid:
                    continue
                mujoco.mj_contactForce(m, d, int(k), f6)
                sc = scale_gid[gs]
                scale_load[sc] = scale_load.get(sc, 0.0) + abs(f6[0])
                scale_touch[int(geom_item[gi])] = sc
        # ---------------- belt traction (2D, Coulomb capped)
        qv = d.qvel[v0:v0 + 6 * MAX_ITEMS].reshape(MAX_ITEMS, 6)
        xy = d.xpos[it_bid][:, :2]
        d.xfrc_applied[it_bid, :] = 0
        for i in np.flatnonzero(active & (support >= 0)):
            # the cell under the box's center governs (if it conveys), else the belt it touches
            c = (int(xy[i, 0] // C), int(xy[i, 1] // C))
            k = cell_index.get(c, support[i])
            want = conv_dir[k] * conv_speed[k] - qv[i, :2]
            f = mass[i] * want / DT
            cap = hid.belt_mu * mass[i] * G
            n = np.hypot(*f)
            if n > cap:
                f *= cap / n
            d.xfrc_applied[it_bid[i], :2] = f
        # ---------------- scales: weigh boxes fully on the platform
        for i in np.flatnonzero(active):
            sc = scale_touch.get(int(i))
            if sc is not None:
                cx, cy = cell_center(*sc)
                if abs(xy[i, 0] - cx) < C / 2 - BOX / 2 - 0.01 and abs(xy[i, 1] - cy) < C / 2 - BOX / 2 - 0.01:
                    samples.setdefault((int(i), sc), []).append(scale_load.get(sc, 0.0))
                    on_scale[i] = scale_cells.index(sc)
                    continue
            if on_scale[i] >= 0:
                sc = scale_cells[on_scale[i]]
                cx, cy = cell_center(*sc)
                if max(abs(xy[i, 0] - cx), abs(xy[i, 1] - cy)) > C / 2 - BOX / 2:  # left the platform
                    s = samples.pop((int(i), sc), [])
                    on_scale[i] = -1
                    if s:
                        dwell = len(s) * DT
                        reading = float(np.mean(s) / G * 1000 + rng.normal(0, hid.scale_noise_g * np.sqrt(0.5 / max(dwell, 1e-3))))
                        report.scales[f"{sc[0]},{sc[1]}"].append(reading)
                        prm = lay.parts[sc].params
                        if reading < float(prm.get("reject_below_g", -1e9)) or reading > float(prm.get("reject_above_g", 1e9)):
                            flagged[i] = True
        # ---------------- pushers: fire when a flagged box is `offset_mm` before the paddle line
        for j, p in enumerate(pushers):
            dx, dy = DIRS[p.dir]
            tc = (p.x + dx, p.y + dy)
            tp = lay.conveys(tc)
            if tp is None:
                continue
            ux, uy = DIRS[tp.dir]
            tcx, tcy = cell_center(*tc)
            off = float(p.params.get("offset_mm", 50)) / 1000
            for i in np.flatnonzero(active & flagged & ~pushed):
                along = (xy[i, 0] - tcx) * ux + (xy[i, 1] - tcy) * uy
                across = (xy[i, 0] - tcx) * dx + (xy[i, 1] - tcy) * dy
                if abs(across) < C / 2 and -C / 2 - 0.3 < along and along >= -off:
                    pushed[i] = True
                    fire_at[j].append(t + hid.pusher_lag)
            if fire_at[j] and t >= fire_at[j][0]:
                fire_at[j].pop(0)
                extend_until[j] = t + float(p.params.get("stroke_ms", 200)) / 1000
                report.pusher_fires[f"{p.x},{p.y}"] += 1
            rest = push_rest[j]
            cur = d.mocap_pos[push_mocap[j]]
            target = rest + np.array([dx, dy, 0]) * (C * 1.2 if t < extend_until[j] else 0.0)
            delta = target - cur
            step_len = hid.pusher_speed * DT
            nrm = np.linalg.norm(delta)
            d.mocap_pos[push_mocap[j]] = target if nrm <= step_len else cur + delta / nrm * step_len
        mujoco.mj_step(m, d)
        # ---------------- bookkeeping: bins, spills, jams
        pos = d.xpos[it_bid]
        for i in np.flatnonzero(active):
            x, y, z = pos[i]
            c = (int(np.floor(x / C)), int(np.floor(y / C)))
            if z < BZ - 0.12:
                if c in bins and z < 0.35:
                    b = report.bins[bins[c].name]
                    b["count"] += 1
                    b["weight_kg"] += mass[i]
                    b["kinds"][kind[i]] = b["kinds"].get(kind[i], 0) + 1
                    park(i)
                elif z < BOX:
                    key = f"{c[0]},{c[1]}"
                    report.spills[key] = report.spills.get(key, 0) + 1
                    park(i)
                continue
            sp = np.hypot(*d.qvel[v0 + 6 * i:v0 + 6 * i + 2])
            still[i] = still[i] + DT if sp < 0.02 else 0.0
            if still[i] > 3.0 and not jam_logged[i]:
                jam_logged[i] = True
                key = f"{c[0]},{c[1]}"
                report.jams[key] = report.jams.get(key, 0) + 1
        if every and step % every == 0:
            report.frames.append({
                "t": round(t, 2),
                "items": [[int(i), kind[i] != "good", *[round(float(v), 3) for v in d.xpos[it_bid[i]]],
                           *[round(float(v), 3) for v in d.xquat[it_bid[i]]]] for i in np.flatnonzero(active)],
                "push": [round(float(np.linalg.norm(d.mocap_pos[push_mocap[j]] - push_rest[j])), 3) for j in range(len(pushers))],
            })
    report.in_transit = int(active.sum())
    return report
