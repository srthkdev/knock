// Same procedures as knock/world.py (simulate_flick / simulate_push / _run_until_rest),
// running the identical MJCF in MuJoCo WebAssembly.
import loadMujoco from '@mujoco/mujoco'

export type Pose = [t: number, x: number, z: number, qw: number, qx: number, qy: number, qz: number]
type Mujoco = Awaited<ReturnType<typeof loadMujoco>>

let mj: Promise<Mujoco> | null = null
export const mujoco = () => (mj ??= loadMujoco())

const SETTLE_S = 0.3
const MAX_S = 8.0

function run(M: Mujoco, xml: string, setup: (m: any, d: any, dof: number) => void,
             force: number | null, forceS: number, fps: number) {
  const model = M.MjModel.from_xml_string(xml)
  const data = new M.MjData(model)
  try {
    const dt = model.opt.timestep
    const bid = model.body('obj').id
    const jnt = model.body_jntadr[bid]
    const dof = model.jnt_dofadr[jnt]
    for (let s = 0; s < Math.round(SETTLE_S / dt); s++) M.mj_step(model, data)
    const x0 = data.xpos[bid * 3]
    setup(model, data, dof)
    const every = Math.round(1 / (fps * dt))
    const traj: Pose[] = []
    let still = 0
    const steps = Math.floor(MAX_S / dt)
    const xfrc = data.xfrc_applied
    const qvel = data.qvel
    for (let s = 0; s < steps; s++) {
      const t = s * dt
      xfrc[bid * 6] = force !== null && t < forceS ? force : 0
      M.mj_step(model, data)
      if (s % every === 0) {
        const p = data.xpos, q = data.xquat
        traj.push([t, p[bid * 3] - x0, p[bid * 3 + 2], q[bid * 4], q[bid * 4 + 1], q[bid * 4 + 2], q[bid * 4 + 3]])
      }
      const v = Math.hypot(qvel[dof], qvel[dof + 1], qvel[dof + 2])
      still = v > 2e-3 || t < forceS ? 0 : still + 1
      if (still * dt > 0.15) break
    }
    return { distance: data.xpos[bid * 3] - x0, traj }
  } finally {
    data.delete()
    model.delete()
  }
}

export async function simulateFlick(xml: string, impulse: number, fps = 60) {
  const M = await mujoco()
  return run(M, xml, (m, d, dof) => {
    d.qvel[dof] = impulse / m.body_subtreemass[m.body('obj').id]
  }, null, 0, fps)
}

export async function simulatePush(xml: string, force: number, seconds: number, fps = 60) {
  const M = await mujoco()
  return run(M, xml, () => {}, force, seconds, fps)
}
