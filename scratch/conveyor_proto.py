"""Feasibility: belt traction + projectile hand-off in MuJoCo.

Belt = static box. Each step, a part touching the belt gets a horizontal traction force toward the
belt speed, capped by Coulomb friction (mu * normal force). Check: part reaches belt speed after the
analytic slip distance, then leaves the end and lands where projectile motion predicts.
"""
import mujoco, numpy as np

G = 9.81
def run(v_belt, mu, mass, belt_len=1.0, belt_h=0.5, dt=0.0005):
    xml = f"""<mujoco><option timestep="{dt}" cone="elliptic" impratio="10"/>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <geom name="belt" type="box" pos="{belt_len/2} 0 {belt_h}" size="{belt_len/2} 0.1 0.01" friction="0 0 0"/>
      <body name="part" pos="0.05 0 {belt_h+0.01+0.02+0.001}"><freejoint/>
        <geom type="box" size="0.02 0.02 0.02" mass="{mass}" friction="0 0 0"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    bid = m.body("part").id; belt = m.geom("belt").id; dof = m.jnt_dofadr[m.body_jntadr[bid]]
    f6 = np.zeros(6); reached = None; left = None
    for s in range(int(6 / dt)):
        N = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if belt in (c.geom1, c.geom2):
                mujoco.mj_contactForce(m, d, i, f6); N += abs(f6[0])
        vx = d.qvel[dof]
        if N > 0:
            # traction toward belt speed, Coulomb-capped; stiff gain so it sticks once at speed
            want = mass * (v_belt - vx) / dt
            d.xfrc_applied[bid, 0] = float(np.clip(want, -mu * N, mu * N))
            if reached is None and abs(vx - v_belt) < 1e-3: reached = d.xpos[bid][0]
        else:
            d.xfrc_applied[bid, 0] = 0.0
            if left is None and d.xpos[bid][0] > belt_len: left = (d.xpos[bid][0], d.qvel[dof], d.xpos[bid][2])
        mujoco.mj_step(m, d)
        if d.xpos[bid][2] < 0.0201 + 1e-3 and left is not None: break
    slip_theory = 0.05 + v_belt**2 / (2 * mu * G)
    # projectile from belt end (centre height belt_h+0.03 falling to 0.02)
    h = belt_h + 0.01 + 0.02 - 0.02
    land_theory = belt_len + v_belt * np.sqrt(2 * h / G)
    return reached, slip_theory, d.xpos[bid][0], land_theory

for v, mu, mass in [(0.8, 0.3, 0.2), (1.2, 0.5, 1.0), (0.5, 0.15, 0.05)]:
    r, rt, land, lt = run(v, mu, mass)
    print(f"v={v} mu={mu} m={mass}: reaches belt speed at x={r:.3f} (theory {rt:.3f}); lands x={land:.3f} (projectile theory {lt:.3f}, edge effects expected)")
