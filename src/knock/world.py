"""MuJoCo world: builds scenes from a Level and runs probes and the final flick."""
from __future__ import annotations

import numpy as np
import mujoco

from . import sound
from .level import Level, Obj
from .materials import CONTENTS, MATERIALS, WALL

DT = 0.0005
G = 9.81
OPTION = f'<option timestep="{DT}" cone="elliptic" impratio="10"/>'


# ------------------------------------------------------------------ scene xml
def _walls(o: Obj) -> str:
    """Six non-overlapping wall geoms of a hollow cube (in body frame).

    Top/bottom span the full face, x-walls sit between them, y-walls fill the rest,
    so the summed volume equals the exact shell volume used by `Obj.mass`, and only
    the bottom wall touches the table.
    """
    h, mu, rho = o.half, o.hidden.friction, MATERIALS[o.hidden.material]["density"]
    inner = h - WALL
    sizes = {2: [h, h, WALL / 2], 0: [WALL / 2, h, inner], 1: [inner, WALL / 2, inner]}
    out = []
    for axis in (2, 0, 1):
        for sign in (-1, 1):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = sign * (h - WALL / 2)
            size = sizes[axis]
            out.append(f'<geom type="box" pos="{pos[0]} {pos[1]} {pos[2]}" size="{size[0]} {size[1]} {size[2]}" '
                       f'density="{rho}" friction="{mu} 0.005 0.0001" solref="0.002 1" rgba="{_rgba(o.color)}"/>')
    if o.hidden.contents == "weight":
        w = CONTENTS["weight"]["mass"]
        out.append(f'<geom type="box" pos="0 0 {-h + WALL + 0.006}" size="{h * 0.5} {h * 0.5} 0.006" mass="{w}" '
                   f'contype="0" conaffinity="0" rgba="0.3 0.3 0.3 1" group="3"/>')
    return "\n".join(out)


def _shell(o: Obj) -> str:
    if o.hidden.hollow:
        return _walls(o)
    rho, mu = MATERIALS[o.hidden.material]["density"], o.hidden.friction
    return (f'<geom type="box" size="{o.half} {o.half} {o.half}" density="{rho}" '
            f'friction="{mu} 0.005 0.0001" solref="0.002 1" rgba="{_rgba(o.color)}"/>')


def _loose(o: Obj, origin, rng) -> str:
    """Free bodies for loose contents, placed inside the box."""
    c = o.hidden.contents
    if not o.hidden.hollow or c not in ("marble", "beads", "foam"):
        return ""
    spec = CONTENTS[c]
    inner = o.half - WALL - spec["radius"] - 0.001
    out = []
    for i in range(spec["count"]):
        p = np.array(origin) + rng.uniform(-inner, inner, 3) * [1, 1, 0.3]
        out.append(f'<body name="{o.id}_c{i}" pos="{p[0]} {p[1]} {p[2]}"><freejoint/>'
                   f'<geom type="sphere" size="{spec["radius"]}" mass="{spec["mass"]}" solref="{spec["solref"]}" '
                   f'group="3"/></body>')
    return "\n".join(out)


def _rgba(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return f"{r:.3f} {g:.3f} {b:.3f} 1"


def object_scene(o: Obj, rng, mocap=False, striker=False, z=None) -> str:
    """Scene with a single object on the table (or held, for shaking)."""
    z = o.half + 0.0002 if z is None else z
    joint = "" if mocap else "<freejoint/>"
    mocap_attr = 'mocap="true"' if mocap else ""
    strike = ""
    if striker:
        strike = (f'<body name="striker" pos="{-o.half - 0.03} 0 {z + o.half * rng.uniform(-0.4, 0.4)}"><freejoint/>'
                  f'<geom type="sphere" size="0.008" mass="0.01" solref="0.002 1" contype="2" conaffinity="1"/></body>')
    return f"""<mujoco>{OPTION}
      <worldbody>
        <geom name="table" type="plane" size="3 3 .1" friction="0 0 0" solref="0.004 1"/>
        <body name="obj" pos="0 0 {z}" {mocap_attr}>{joint}{_shell(o)}</body>
        {_loose(o, (0, 0, z), rng)}
        {strike}
      </worldbody></mujoco>"""


# ------------------------------------------------------------------ simulation helpers
def _settle(m, d, seconds=0.3):
    for _ in range(int(seconds / DT)):
        mujoco.mj_step(m, d)


def _obj_qpos_adr(m):
    return m.jnt_qposadr[m.body("obj").jntadr[0]]


def _obj_dof_adr(m):
    return m.jnt_dofadr[m.body("obj").jntadr[0]]


def _run_until_rest(m, d, max_s=8.0, force=None, force_s=0.0, record=None, record_every=None):
    """Step until the object stops; optionally apply a horizontal force first.

    If `record` is a list, (time, pos, quat) of the object is appended every `record_every` steps.
    """
    bid = m.body("obj").id
    dof = _obj_dof_adr(m)
    still = 0
    for s in range(int(max_s / DT)):
        t = s * DT
        d.xfrc_applied[bid, :3] = (force, 0, 0) if (force is not None and t < force_s) else (0, 0, 0)
        mujoco.mj_step(m, d)
        if record is not None and s % record_every == 0:
            record.append((t, d.xpos[bid].copy(), d.xquat[bid].copy()))
        moving = np.linalg.norm(d.qvel[dof:dof + 3]) > 2e-3
        still = 0 if (moving or t < force_s) else still + 1
        if still * DT > 0.15:
            break
    return float(d.xpos[bid][0])


def simulate_flick(o: Obj, impulse: float, seed: int = 0, record=None, fps=30) -> float:
    """Strike the object's centre with `impulse` (N*s) along +x; return where it stops."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(object_scene(o, rng))
    d = mujoco.MjData(m)
    _settle(m, d)
    x0 = float(d.xpos[m.body("obj").id][0])
    struck_mass = m.body_subtreemass[m.body("obj").id]
    dof = _obj_dof_adr(m)
    d.qvel[dof] = impulse / struck_mass
    return _run_until_rest(m, d, record=record, record_every=int(1 / (fps * DT))) - x0


def simulate_push(o: Obj, force: float, seconds: float = 0.4, seed: int = 0, record=None, fps=30) -> float:
    """Apply a constant horizontal force for `seconds`; return total displacement (m)."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(object_scene(o, rng))
    d = mujoco.MjData(m)
    _settle(m, d)
    x0 = float(d.xpos[m.body("obj").id][0])
    return _run_until_rest(m, d, force=force, force_s=seconds, record=record,
                           record_every=int(1 / (fps * DT))) - x0


def _contact_events(m, d, steps, body_names, drive=None, ignore=("table",)):
    """Step physics; return impact events (t, impulse, contact_time) on the given bodies."""
    f6 = np.zeros(6)
    ids = {m.body(b).id for b in body_names}
    ignore_ids = {m.geom(g).id for g in ignore}
    taus = _contact_taus(m)
    active, events = {}, []
    for s in range(steps):
        t = s * DT
        if drive is not None:
            d.mocap_pos[0] = drive(t)
        mujoco.mj_step(m, d)
        seen = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in ignore_ids or c.geom2 in ignore_ids:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 not in ids and b2 not in ids:
                continue
            if b1 in ids and b2 in ids:
                continue  # internal geoms of the same object
            key = (min(b1, b2), max(b1, b2))
            mujoco.mj_contactForce(m, d, i, f6)
            seen.add(key)
            if key not in active:
                active[key] = [t, 0.0, max(taus.get(b1, 0), taus.get(b2, 0))]
            if t - active[key][0] < 0.005:
                active[key][1] += abs(f6[0]) * DT
        for key in list(active):
            if key not in seen:
                st, J, tau = active.pop(key)
                if J > 1e-7:
                    events.append((st, J, tau))
    events += [(st, J, tau) for st, J, tau in active.values() if J > 1e-7]
    return sorted(events)


def _contact_taus(m):
    """Per-body contact time; soft contents (foam) override the shell material."""
    out = {}
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        out[b] = 0.0
        if "_c" in name and m.geom_solref[m.body_geomadr[b]][0] > 0.01:
            out[b] = CONTENTS["foam"]["tau"]
    return out


CLIP_S = 1.4


def simulate_tap(o: Obj, seed: int = 0):
    """Flick a small striker into the object's side; return (audio, features)."""
    rng = np.random.default_rng(seed)
    m = mujoco.MjModel.from_xml_string(object_scene(o, rng, striker=True))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    d.qvel[m.jnt_dofadr[m.body("striker").jntadr[0]]] = rng.uniform(0.8, 1.6)
    events = _contact_events(m, d, int(CLIP_S / DT), ["obj"])
    return _render_audio(o, events, rng, rustle=False)


def simulate_shake(o: Obj, seed: int = 0):
    """Pick the object up and shake it; return (audio, features)."""
    rng = np.random.default_rng(seed)
    z = 0.5
    m = mujoco.MjModel.from_xml_string(object_scene(o, rng, mocap=True, z=z))
    d = mujoco.MjData(m)
    f_hz, amp = rng.uniform(3.0, 5.0), rng.uniform(0.02, 0.04)
    drive = lambda t: (amp * np.sin(2 * np.pi * f_hz * t), 0, z + 0.4 * amp * np.sin(4 * np.pi * f_hz * t))
    events = _contact_events(m, d, int(CLIP_S / DT), ["obj"], drive=drive)
    return _render_audio(o, events, rng, rustle=True)


def _render_audio(o: Obj, events, rng, rustle):
    tau0 = MATERIALS[o.hidden.material]["tau"]
    events = [(t, J, tau if tau > 0 else tau0) for t, J, tau in events]
    modes = sound.modes_for(o.hidden.material, 2 * o.half, o.hidden.hollow, rng, wall=WALL)
    x = sound.synthesize(events, *modes, rng, CLIP_S)
    x = sound.room_and_mic(x, rng, rustle=rustle)
    return x, sound.features(x)


def weigh(o: Obj, rng) -> float:
    """Lift probe: scale reading in grams with 2% noise."""
    return round(o.mass * 1000 * (1 + rng.normal(0, 0.02)), 1)


# ------------------------------------------------------------------ oracle helpers
def flick_for_distance(o: Obj, distance: float, lo=0.02, hi=6.0, iters=30) -> tuple[float, float]:
    """Bisection on impulse so the object travels `distance`; returns (impulse, achieved)."""
    d_lo, d_hi = simulate_flick(o, lo), simulate_flick(o, hi)
    if not d_lo <= distance <= d_hi:
        return (lo if distance < d_lo else hi), (d_lo if distance < d_lo else d_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        dm = simulate_flick(o, mid)
        if dm < distance:
            lo = mid
        else:
            hi = mid
        if abs(dm - distance) < 1e-3:
            return mid, dm
    return mid, dm
