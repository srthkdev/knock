"""Feasibility prototype: bottles on a conveyor, a checkweigher platform, a pneumatic reject pusher.

Bottle: glass shell (cylinder, contact geom) + liquid (inner cylinder, mass only, height = fill).
Conveyor traction: Coulomb-capped force toward belt speed for bottles touching belt geoms.
Checkweigher: a belt section; reading = mean normal force on it during the bottle's dwell / g.
Pusher: mocap piston on the side; fires when a flagged bottle is predicted to be in front of it.
"""
import time

import mujoco
import numpy as np

G = 9.81
DT = 0.002
R, H = 0.032, 0.24           # bottle radius, height (m)
GLASS = 0.30                 # empty bottle mass (kg)
FULL = 0.33                  # liquid mass when full (kg)
BELT_Z = 0.80
LINE_LEN = 5.0
WEIGH = (2.0, 2.4)           # checkweigher platform x range
PUSH_X = 3.6                 # pusher position


def scene(n_bottles, fills):
    bottles = []
    for i in range(n_bottles):
        f = fills[i]
        liq_h = max(0.01, 0.20 * f)
        bottles.append(f"""
        <body name="b{i}" pos="{-1 - i * 0.5} 0 {BELT_Z + H / 2 + 0.0005}"><freejoint/>
          <geom name="g{i}" type="cylinder" size="{R} {H / 2}" mass="{GLASS}" friction="0 0 0" rgba=".55 .75 .6 .55"/>
          <geom type="cylinder" size="{R * 0.9} {liq_h / 2}" pos="0 0 {-H / 2 + 0.008 + liq_h / 2}" mass="{FULL * f}"
                contype="0" conaffinity="0" rgba=".85 .55 .2 .8"/>
        </body>""")
    return f"""<mujoco><option timestep="{DT}" cone="elliptic" impratio="10"/>
    <visual><headlight ambient=".35 .35 .35" diffuse=".5 .5 .5"/><quality shadowsize="4096" offsamples="8"/>
      <global offwidth="1280" offheight="720"/></visual>
    <asset>
      <texture type="skybox" builtin="gradient" rgb1=".85 .87 .9" rgb2=".55 .6 .66" width="256" height="256"/>
      <texture name="fl" type="2d" builtin="checker" rgb1=".42 .44 .46" rgb2=".38 .40 .42" width="512" height="512"/>
      <material name="floor" texture="fl" texrepeat="20 20" reflectance=".1"/>
      <material name="belt" rgba=".12 .12 .13 1" specular=".2"/>
      <material name="steel" rgba=".72 .74 .78 1" specular=".9" shininess=".8" reflectance=".25"/>
    </asset>
    <worldbody>
      <light pos="2 -1.5 4" dir="0 .3 -1" diffuse=".8 .8 .8" castshadow="true"/>
      <geom name="floor" type="plane" size="10 10 .1" material="floor"/>
      <geom type="box" pos="{LINE_LEN / 2} 0 {BELT_Z / 2 - 0.02}" size="{LINE_LEN / 2} 0.08 {BELT_Z / 2 - 0.03}" material="steel" contype="0" conaffinity="0"/>
      <geom name="belt_a" type="box" pos="{WEIGH[0] / 2} 0 {BELT_Z - 0.01}" size="{WEIGH[0] / 2} 0.05 0.01" friction="0 0 0" material="belt"/>
      <geom name="weigh" type="box" pos="{sum(WEIGH) / 2} 0 {BELT_Z - 0.01}" size="{(WEIGH[1] - WEIGH[0]) / 2 - 0.001} 0.05 0.01" friction="0 0 0" rgba=".3 .3 .8 1"/>
      <geom name="belt_b" type="box" pos="{(WEIGH[1] + LINE_LEN) / 2} 0 {BELT_Z - 0.01}" size="{(LINE_LEN - WEIGH[1]) / 2} 0.05 0.01" friction="0 0 0" material="belt"/>
      <geom name="rail_l" type="box" pos="{LINE_LEN / 2} 0.05 {BELT_Z + 0.03}" size="{LINE_LEN / 2} 0.003 0.03" friction="0.2 0 0"/>
      <geom name="rail_r1" type="box" pos="{PUSH_X / 2 - 0.1} -0.05 {BELT_Z + 0.03}" size="{PUSH_X / 2 - 0.1} 0.003 0.03" friction="0.2 0 0"/>
      <geom name="rail_r2" type="box" pos="{(PUSH_X + 0.1 + LINE_LEN) / 2} -0.05 {BELT_Z + 0.03}" size="{(LINE_LEN - PUSH_X - 0.1) / 2} 0.003 0.03" friction="0.2 0 0"/>
      <geom name="reject_table" type="box" pos="{PUSH_X + 0.25} -0.30 {BELT_Z - 0.03}" size="0.45 0.22 0.01" friction="0.7 0 0" rgba=".7 .5 .3 1"/>
      <geom name="reject_wall" type="box" pos="{PUSH_X + 0.25} -0.53 {BELT_Z + 0.05}" size="0.45 0.01 0.08" rgba=".7 .5 .3 1"/>
      <body name="pusher" mocap="true" pos="{PUSH_X} 0.12 {BELT_Z + 0.06}">
        <geom type="box" size="0.06 0.02 0.04" rgba=".8 .3 .2 1" friction="0.4 0 0"/></body>
      {''.join(bottles)}
      <camera name="cam" pos="{PUSH_X - 1.6} -2.1 1.9" xyaxes="0.8 -0.6 0 0.3 0.4 0.87"/>
    </worldbody></mujoco>"""


def run(n=12, belt_v=0.5, fills=None, reject_threshold_g=None, fire_offset=0.06, stroke_s=0.18, push_speed=1.2,
        sensor_noise_g=1.5, seconds=None, seed=0, on_frame=None, fps=30):
    rng = np.random.default_rng(seed)
    fills = fills if fills is not None else [1.0 if rng.random() > 0.3 else 0.85 for _ in range(n)]
    m = mujoco.MjModel.from_xml_string(scene(n, fills))
    d = mujoco.MjData(m)
    belt_geoms = {m.geom(g).id for g in ("belt_a", "weigh", "belt_b")}
    weigh_id = m.geom("weigh").id
    bid = [m.body(f"b{i}").id for i in range(n)]
    dof = [m.jnt_dofadr[m.body_jntadr[b]] for b in bid]
    body_of_geom = {m.geom(f"g{i}").id: i for i in range(n)}
    mu_belt = 0.5
    # feeder: release bottles one by one at 1/s by teleporting them onto the belt start
    spacing_s = 1.0
    released = [False] * n
    weigh_samples = [[] for _ in range(n)]
    readings = {}
    flagged = set()
    pusher_until = -1.0
    push_queue = []
    fate = {}
    f6 = np.zeros(6)
    T = seconds or (n * spacing_s + LINE_LEN / belt_v + 3)
    t0 = time.time()
    for s in range(int(T / DT)):
        t = s * DT
        k = int(t / spacing_s)
        if k < n and not released[k]:
            released[k] = True
            q = m.jnt_qposadr[m.body_jntadr[bid[k]]]
            d.qpos[q:q + 7] = [0.1, 0, BELT_Z + H / 2 + 0.0005, 1, 0, 0, 0]
            d.qvel[dof[k]:dof[k] + 6] = 0
        # contacts: belt normal force per bottle, weigh platform force
        N = np.zeros(n)
        Nw = np.zeros(n)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            other = g2 if g1 in belt_geoms else g1 if g2 in belt_geoms else None
            if other is None or other not in body_of_geom:
                continue
            j = body_of_geom[other]
            mujoco.mj_contactForce(m, d, i, f6)
            N[j] += abs(f6[0])
            if weigh_id in (g1, g2):
                Nw[j] += abs(f6[0])
        for j in range(n):
            if N[j] > 0:
                vx = d.qvel[dof[j]]
                want = (GLASS + FULL * fills[j]) * (belt_v - vx) / DT
                d.xfrc_applied[bid[j], 0] = float(np.clip(want, -mu_belt * N[j], mu_belt * N[j]))
            else:
                d.xfrc_applied[bid[j], 0] = 0.0
            xj = d.xpos[bid[j]][0]
            if Nw[j] > 0 and WEIGH[0] + R < xj < WEIGH[1] - R:
                weigh_samples[j].append(Nw[j])
            elif weigh_samples[j] and j not in readings and xj >= WEIGH[1] - R:
                # bottle left the platform: reading = mean force / g + sensor noise (shorter dwell = noisier)
                dwell = len(weigh_samples[j]) * DT
                noise = sensor_noise_g * np.sqrt(0.5 / max(dwell, 1e-3))
                readings[j] = float(np.mean(weigh_samples[j]) / G * 1000 + rng.normal(0, noise))
                if reject_threshold_g is not None and readings[j] < reject_threshold_g:
                    flagged.add(j)
                    push_queue.append(j)
        # encoder tracking: fire when the next flagged bottle reaches PUSH_X - fire_offset
        if push_queue and d.xpos[bid[push_queue[0]]][0] >= PUSH_X - fire_offset:
            push_queue.pop(0)
            pusher_until = t + stroke_s
        y = d.mocap_pos[0][1]
        target = -0.10 if t < pusher_until else 0.12
        step = push_speed * DT
        d.mocap_pos[0][1] = y + np.clip(target - y, -step, step)
        mujoco.mj_step(m, d)
        if on_frame is not None and s % int(1 / (fps * DT)) == 0:
            on_frame(m, d, t, flagged, readings)
        for j in range(n):
            if j in fate or not released[j]:
                continue
            p = d.xpos[bid[j]]
            up = d.xmat[bid[j]].reshape(3, 3)[:, 2]
            if up[2] < 0.7:
                fate[j] = "tipped"
            elif p[1] < -0.10:
                fate[j] = "rejected"
            elif p[0] > LINE_LEN - 0.05:
                fate[j] = "shipped"
    wall = time.time() - t0
    return fills, readings, flagged, fate, T, wall


if __name__ == "__main__":
    fills, readings, flagged, fate, T, wall = run(n=12, belt_v=0.5, reject_threshold_g=600)
    print(f"simulated {T:.1f} s in {wall:.1f} s wall ({T / wall:.1f}x real time)")
    for j in range(12):
        print(f"bottle {j:2d} fill {fills[j]:.2f} true {(0.30 + 0.33 * fills[j]) * 1000:5.0f} g  read {readings.get(j, float('nan')):6.1f} g"
              f"  flagged {str(j in flagged):5}  fate {fate.get(j)}")
