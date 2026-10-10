"""Bottling line simulator (MuJoCo): conveyor, checkweigher, encoder-tracked reject pusher.

Geometry is visible to agents. Hidden: belt traction, pusher valve lag and stroke speed, load-cell
noise, and the bottle lot (underfill rate and level, glass weight spread). Every run is deterministic
given its seed and can record a trajectory for the viewer.

Physics notes
- Belt: Coulomb-capped traction toward belt speed on bottles touching belt geoms (validated: slip
  distance matches v^2 / (2 mu g)).
- Checkweigher: mean normal force while a bottle is fully on the platform / g, plus load-cell noise
  that shrinks with dwell time (sigma * sqrt(0.5 s / dwell)): faster belts weigh less precisely.
- Pusher: fires `lag_s` after the tracked bottle crosses (push_x - offset); the paddle moves at the
  hidden stroke speed. Wrong offsets miss the bottle or hit its neighbours; long strokes tip bottles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

G = 9.81
DT = 0.002
R, H = 0.032, 0.24           # bottle radius, height (m)
GLASS_G = 300.0              # nominal empty bottle (g)
FULL_G = 330.0               # liquid when full (g)
BELT_Z = 0.80
LINE_LEN = 5.0
WEIGH = (2.0, 2.4)
PUSH_X = 3.6
FEED_X = 0.1
SPEC_FILL = 0.97            # a bottle under 97% of nominal fill is a defect (label claim)


@dataclass
class LineHidden:
    belt_mu: float = 0.5           # traction between belt and bottles
    pusher_speed: float = 1.2      # paddle speed (m/s), set by air pressure
    pusher_lag: float = 0.05       # valve + solenoid latency (s)
    sensor_noise_g: float = 1.5    # load-cell noise at 0.5 s dwell (g)


@dataclass
class LotHidden:
    underfill_rate: float = 0.25   # fraction of bottles underfilled
    underfill_level: float = 0.93  # fill fraction of underfilled bottles
    glass_sd_g: float = 3.0        # spread of empty-bottle weight (g)


@dataclass
class Settings:
    belt_speed: float = 0.5        # m/s
    feed_per_min: float = 40.0     # bottles released per minute
    reject_below_g: float = 600.0  # checkweigher threshold
    pusher_offset_mm: float = 60.0 # fire when a flagged bottle is this far before the pusher
    stroke_ms: float = 180.0       # how long the paddle stays extended


@dataclass
class RunResult:
    n: int
    seconds: float
    shipped: int
    rejected: int
    tipped: int
    good_shipped: int
    defects_shipped: int
    good_rejected: int
    defects_rejected: int
    readings: dict[int, float]
    flagged: list[int]
    fate: dict[int, str]
    defective: list[int]
    fill_g: list[float]
    frames: list = field(default_factory=list)

    def kpis(self) -> dict:
        minutes = self.seconds / 60
        return {"bottles": self.n, "seconds": round(self.seconds, 1), "shipped": self.shipped, "rejected": self.rejected,
                "tipped_or_jammed": self.tipped, "shipped_per_min": round(self.shipped / minutes, 1)}

    def truth(self) -> dict:
        minutes = self.seconds / 60
        return {"good_shipped": self.good_shipped, "defects_shipped": self.defects_shipped,
                "good_rejected": self.good_rejected, "defects_rejected": self.defects_rejected,
                "good_per_min": round(self.good_shipped / minutes, 2), "tipped": self.tipped}


def _scene(fills, glass):
    bottles = []
    for i, (f, g) in enumerate(zip(fills, glass)):
        liq_h = max(0.01, 0.20 * f)
        bottles.append(
            f'<body name="b{i}" pos="{-2 - i * 0.2} {i * 0.2} 0.2"><freejoint/>'
            f'<geom name="g{i}" type="cylinder" size="{R} {H / 2}" mass="{g / 1000}" friction="0 0 0" solref="0.004 1"/>'
            f'<geom type="cylinder" size="{R * 0.9} {liq_h / 2}" pos="0 0 {-H / 2 + 0.008 + liq_h / 2}" mass="{FULL_G * f / 1000}"'
            f' contype="0" conaffinity="0"/></body>')
    return f"""<mujoco><option timestep="{DT}" cone="elliptic" impratio="10"/>
    <worldbody>
      <geom name="floor" type="plane" size="20 20 .1"/>
      <geom name="belt_a" type="box" pos="{WEIGH[0] / 2} 0 {BELT_Z - 0.01}" size="{WEIGH[0] / 2} 0.05 0.01" friction="0 0 0"/>
      <geom name="weigh" type="box" pos="{sum(WEIGH) / 2} 0 {BELT_Z - 0.01}" size="{(WEIGH[1] - WEIGH[0]) / 2 - 0.001} 0.05 0.01" friction="0 0 0"/>
      <geom name="belt_b" type="box" pos="{(WEIGH[1] + LINE_LEN) / 2} 0 {BELT_Z - 0.01}" size="{(LINE_LEN - WEIGH[1]) / 2} 0.05 0.01" friction="0 0 0"/>
      <geom name="rail_l" type="box" pos="{LINE_LEN / 2} 0.05 {BELT_Z + 0.03}" size="{LINE_LEN / 2} 0.003 0.03" friction="0.2 0 0"/>
      <geom name="rail_r1" type="box" pos="{(PUSH_X - 0.2) / 2} -0.05 {BELT_Z + 0.03}" size="{(PUSH_X - 0.2) / 2} 0.003 0.03" friction="0.2 0 0"/>
      <geom name="rail_r2" type="box" pos="{(PUSH_X + 0.25 + LINE_LEN) / 2} -0.05 {BELT_Z + 0.03}" size="{(LINE_LEN - PUSH_X - 0.25) / 2} 0.003 0.03" friction="0.2 0 0"/>
      <geom name="reject_table" type="box" pos="{PUSH_X + 0.25} -0.30 {BELT_Z - 0.01}" size="0.45 0.22 0.01" friction="0.7 0 0"/>
      <geom name="reject_wall" type="box" pos="{PUSH_X + 0.25} -0.53 {BELT_Z + 0.05}" size="0.45 0.01 0.08"/>
      <geom name="end_table" type="box" pos="{LINE_LEN + 0.6} 0 {BELT_Z - 0.01}" size="0.6 0.3 0.01" friction="0.8 0 0"/>
      <body name="pusher" mocap="true" pos="{PUSH_X} 0.13 {BELT_Z + 0.09}">
        <geom type="box" size="0.06 0.03 0.05" friction="0.3 0 0" solref="0.004 1"/></body>
      {''.join(bottles)}
    </worldbody></mujoco>"""


def sample_lot(lot: LotHidden, n: int, rng: np.random.Generator):
    defective = rng.random(n) < lot.underfill_rate
    fills = np.where(defective, lot.underfill_level + rng.normal(0, 0.005, n), 1.0 + rng.normal(0, 0.003, n))
    glass = GLASS_G + rng.normal(0, lot.glass_sd_g, n)
    return fills.clip(0.5, 1.02), glass, defective


def run_line(line: LineHidden, lot: LotHidden, s: Settings, n: int, seed: int, record_fps: float = 0.0) -> RunResult:
    rng = np.random.default_rng(seed)
    fills, glass, defective = sample_lot(lot, n, rng)
    m = mujoco.MjModel.from_xml_string(_scene(fills, glass))
    d = mujoco.MjData(m)
    belt_geoms = {m.geom(g).id for g in ("belt_a", "weigh", "belt_b")}
    weigh_id = m.geom("weigh").id
    bid = np.array([m.body(f"b{i}").id for i in range(n)])
    qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in bid]
    dof = [m.jnt_dofadr[m.body_jntadr[b]] for b in bid]
    body_of_geom = {m.geom(f"g{i}").id: i for i in range(n)}
    mass = (glass + FULL_G * fills) / 1000

    spacing_s = 60.0 / s.feed_per_min
    released = np.zeros(n, bool)
    samples: list[list[float]] = [[] for _ in range(n)]
    readings: dict[int, float] = {}
    flagged: list[int] = []
    queue: list[int] = []
    fire_at: list[float] = []
    pusher_until = -1.0
    fate: dict[int, str] = {}
    tip_where: dict[int, tuple] = {}
    f6 = np.zeros(6)
    frames = []
    every = int(1 / (record_fps * DT)) if record_fps else 0
    T = n * spacing_s + LINE_LEN / max(s.belt_speed, 0.05) + 3.0
    for k in range(int(T / DT)):
        t = k * DT
        j = int(t / spacing_s)
        if j < n and not released[j]:
            released[j] = True
            d.qpos[qadr[j]:qadr[j] + 7] = [FEED_X, 0, BELT_Z + H / 2 + 0.0005, 1, 0, 0, 0]
            d.qvel[dof[j]:dof[j] + 6] = 0
        N = np.zeros(n)
        Nw = np.zeros(n)  # per bottle, only to know who touches the platform; the load cell sees the sum
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            other = g2 if g1 in belt_geoms else g1 if g2 in belt_geoms else None
            if other is None or other not in body_of_geom:
                continue
            b = body_of_geom[other]
            mujoco.mj_contactForce(m, d, i, f6)
            N[b] += abs(f6[0])
            if weigh_id in (g1, g2):
                Nw[b] += abs(f6[0])
        for b in np.flatnonzero(released):
            if N[b] > 0:
                want = mass[b] * (s.belt_speed - d.qvel[dof[b]]) / DT
                d.xfrc_applied[bid[b], 0] = float(np.clip(want, -line.belt_mu * N[b], line.belt_mu * N[b]))
            else:
                d.xfrc_applied[bid[b], 0] = 0.0
            xb = d.xpos[bid[b]][0]
            if Nw[b] > 0 and WEIGH[0] + R < xb < WEIGH[1] - R:
                samples[b].append(Nw.sum())  # a load cell weighs everything on it: pitch must exceed the platform
            elif samples[b] and b not in readings and xb >= WEIGH[1] - R:
                dwell = len(samples[b]) * DT
                noise = line.sensor_noise_g * np.sqrt(0.5 / max(dwell, 1e-3))
                readings[b] = float(np.mean(samples[b]) / G * 1000 + rng.normal(0, noise))
                if readings[b] < s.reject_below_g:
                    flagged.append(int(b))
                    queue.append(int(b))
        # encoder tracking: when the next flagged bottle crosses (PUSH_X - offset), fire after the valve lag
        if queue and d.xpos[bid[queue[0]]][0] >= PUSH_X - s.pusher_offset_mm / 1000:
            queue.pop(0)
            fire_at.append(t + line.pusher_lag)
        if fire_at and t >= fire_at[0]:
            fire_at.pop(0)
            pusher_until = t + s.stroke_ms / 1000
        y = d.mocap_pos[0, 1]
        target = -0.09 if t < pusher_until else 0.13
        step = line.pusher_speed * DT
        d.mocap_pos[0, 1] = y + np.clip(target - y, -step, step)
        mujoco.mj_step(m, d)
        if every and k % every == 0:
            frames.append({"t": round(t, 3), "pusher": round(float(d.mocap_pos[0, 1]), 4), "checked": len(readings),
                           "rejected": sum(f == "rejected" for f in fate.values()),
                           "b": [[round(float(v), 4) for v in (*d.xpos[bid[b]], *d.xquat[bid[b]])] if released[b] else None
                                 for b in range(n)]})
        for b in np.flatnonzero(released):
            if b in fate:
                continue
            p = d.xpos[bid[b]]
            up = d.xmat[bid[b]].reshape(3, 3)[:, 2]
            if p[1] < -0.10:
                fate[int(b)] = "rejected"  # in the chute, standing or not
            elif up[2] < 0.7 and abs(p[1]) < 0.045:  # down on the line itself: a jam
                fate[int(b)] = "tipped"
                tip_where[int(b)] = (round(t, 2), round(float(p[0]), 2), round(float(p[1]), 2))
            elif p[0] > LINE_LEN - 0.08:
                fate[int(b)] = "shipped"  # handed to the next machine: park it out of the way
                d.qpos[qadr[b]:qadr[b] + 7] = [LINE_LEN + 3 + 0.2 * b, 3.0, 0.2, 1, 0, 0, 0]
                d.qvel[dof[b]:dof[b] + 6] = 0
        if len(fate) == n and k * DT > n * spacing_s:
            T = t
            break
    run_line.tip_where = tip_where
    for b in range(n):
        fate.setdefault(b, "tipped")  # never arrived: stuck or fell
    defect = set(np.flatnonzero(fills < SPEC_FILL).tolist())
    shipped = [b for b, f in fate.items() if f == "shipped"]
    rejected = [b for b, f in fate.items() if f == "rejected"]
    return RunResult(
        n=n, seconds=float(T), shipped=len(shipped), rejected=len(rejected),
        tipped=sum(f == "tipped" for f in fate.values()),
        good_shipped=sum(b not in defect for b in shipped), defects_shipped=sum(b in defect for b in shipped),
        good_rejected=sum(b not in defect for b in rejected), defects_rejected=sum(b in defect for b in rejected),
        readings={int(k): v for k, v in readings.items()}, flagged=flagged, fate=fate, defective=[int(b) for b in sorted(defect)],
        fill_g=[round(float(FULL_G * f), 1) for f in fills], frames=frames,
    )
