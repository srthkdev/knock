// Browser port of knock/packs/factory_sim.py: the same MJCF, the same per-step procedure and the same
// RNG, running in MuJoCo WebAssembly. Keep the two files in step (order of operations matters).
import loadMujoco from '@mujoco/mujoco'

type Mujoco = Awaited<ReturnType<typeof loadMujoco>>
let mj: Promise<Mujoco> | null = null
export const mujoco = () => (mj ??= loadMujoco())

export const G = 9.81
export const DT = 0.004
export const C = 0.5
export const BZ = 0.4
export const BOX = 0.14
export const MAX_ITEMS = 48
export const DIRS: Record<string, [number, number]> = { E: [1, 0], N: [0, 1], W: [-1, 0], S: [0, -1] }
const CONVEY = new Set(['belt', 'scale', 'inlet'])

export interface Part { kind: string; x: number; y: number; dir: string; params: Record<string, number>; fixed: boolean; name: string }
export interface Layout { w: number; h: number; walls: [number, number][]; parts: Part[] }
export interface Hidden {
  belt_mu: number; pusher_speed: number; pusher_lag: number; scale_noise_g: number
  mixes: Record<string, [string, number, number, number][]>; item?: string
}
export interface ItemGeom { type: string; size: number[]; pos?: number[]; fromto?: number[] }
export interface ItemAsset { geoms: ItemGeom[]; inertia: number[]; rest: number; half: number; size: number[] }
export type ItemAssets = Record<string, ItemAsset>

export function itemShape(item: string | undefined, assets?: ItemAssets) {
  if (!item || item === 'box') return { rest: BOX / 2, half: BOX / 2, clear: BOX * 1.3, floor: BOX }
  const it = assets![item]
  return { rest: it.rest, half: it.half, clear: 2.2 * it.half, floor: it.rest + 0.05 }
}

// Python str(float) and JS String(number) agree for these short decimals (shortest round-trip both).
const num = (v: number) => (Number.isInteger(v) ? `${v}.0` : String(v))

function itemBody(i: number, item: string, assets?: ItemAssets) {
  const head = `<body name="it${i}" pos="${-5 - i * 0.3} -5 0.1"><freejoint/>`
  const common = 'friction="0 0 0" solref="0.008 1"'
  if (item === 'box') return head + `<geom name="itg${i}" type="box" size="${BOX / 2} ${BOX / 2} ${BOX / 2}" mass="1" ${common}/></body>`
  const it = assets![item]
  let out = head + `<inertial pos="0 0 0" mass="1" diaginertia="${it.inertia.map(num).join(' ')}"/>`
  it.geoms.forEach((g, k) => {
    const where = g.fromto ? `fromto="${g.fromto.map(num).join(' ')}"` : `pos="${g.pos!.map(num).join(' ')}"`
    out += `<geom${k === 0 ? ` name="itg${i}"` : ''} type="${g.type}" size="${g.size.map(num).join(' ')}" ${where} ${common}/>`
  })
  return out + '</body>'
}
export interface Frame { t: number; items: [number, boolean, number, number, number, number, number, number, number][]; push: number[] }
export interface Report {
  seconds: number; emitted: Record<string, number>
  bins: Record<string, { count: number; weight_kg: number; kinds: Record<string, number> }>
  spills: Record<string, number>; jams: Record<string, number>; blocked_inlet: number; in_transit: number
  scales: Record<string, number[]>; pusher_fires: Record<string, number>; frames: Frame[]
}

export const cellCenter = (x: number, y: number): [number, number] => [(x + 0.5) * C, (y + 0.5) * C]
const key = (x: number, y: number) => `${x},${y}`

export class Rng {
  a: number
  constructor(seed: number) { this.a = seed >>> 0 }
  random() {
    this.a = (this.a + 0x6d2b79f5) >>> 0
    let t = this.a
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0
    t = (t ^ ((t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0)) >>> 0)) >>> 0
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  normal(mu: number, sd: number) {
    const u = 1 - this.random()
    const v = this.random()
    return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
}

function feedsInto(lay: Map<string, Part>, src: [number, number], dst: [number, number]) {
  const p = lay.get(key(...src))
  if (!p || !CONVEY.has(p.kind)) return false
  const [dx, dy] = DIRS[p.dir]
  return src[0] + dx === dst[0] && src[1] + dy === dst[1]
}

export function sceneXml(layout: Layout, item = 'box', assets?: ItemAssets) {
  const parts = new Map(layout.parts.map((p) => [key(p.x, p.y), p]))
  const g: string[] = []
  for (const p of layout.parts) {
    const { x, y } = p
    const [cx, cy] = cellCenter(x, y)
    if (CONVEY.has(p.kind)) {
      const slab = `type="box" size="${C / 2} ${C / 2} 0.02" friction="0 0 0"`
      if (p.kind === 'scale') g.push(`<body name="sc_${x}_${y}" pos="${cx} ${cy} ${BZ - 0.02}"><geom name="cv_${x}_${y}" ${slab}/></body>`)
      else g.push(`<geom name="cv_${x}_${y}" pos="${cx} ${cy} ${BZ - 0.02}" ${slab}/>`)
      const [dx, dy] = DIRS[p.dir]
      for (const [ex, ey] of Object.values(DIRS)) {
        if (ex === dx && ey === dy) continue
        const nb: [number, number] = [x + ex, y + ey]
        if (feedsInto(parts, nb, [x, y])) continue
        const q = parts.get(key(...nb))
        if (q && q.kind === 'pusher' && DIRS[q.dir][0] === -ex && DIRS[q.dir][1] === -ey) continue
        if (layout.parts.some((pp) => pp.kind === 'pusher' && pp.x + DIRS[pp.dir][0] === x && pp.y + DIRS[pp.dir][1] === y
          && DIRS[pp.dir][0] === ex && DIRS[pp.dir][1] === ey)) continue
        const rx = cx + ex * (C / 2 - 0.01), ry = cy + ey * (C / 2 - 0.01)
        const [sx, sy] = ex ? [0.01, C / 2] : [C / 2, 0.01]
        g.push(`<geom type="box" pos="${rx} ${ry} ${BZ + 0.04}" size="${sx} ${sy} 0.04" friction="0.15 0 0"/>`)
      }
    } else if (p.kind === 'bin') {
      g.push(`<geom name="bin_${x}_${y}" type="box" pos="${cx} ${cy} 0.05" size="${C / 2} ${C / 2} 0.05" friction="0.8 0 0"/>`)
      for (const [ex, ey] of Object.values(DIRS)) {
        const rx = cx + ex * (C / 2 - 0.01), ry = cy + ey * (C / 2 - 0.01)
        const [sx, sy] = ex ? [0.01, C / 2] : [C / 2, 0.01]
        g.push(`<geom type="box" pos="${rx} ${ry} ${0.1 + 0.1}" size="${sx} ${sy} 0.1"/>`)
      }
    }
  }
  for (const [wx, wy] of layout.walls) {
    const [cx, cy] = cellCenter(wx, wy)
    g.push(`<geom type="box" pos="${cx} ${cy} 0.6" size="${C / 2 - 0.02} ${C / 2 - 0.02} 0.6"/>`)
  }
  for (const p of layout.parts) {
    if (p.kind !== 'pusher') continue
    const [dx, dy] = DIRS[p.dir]
    const [cx, cy] = cellCenter(p.x, p.y)
    const rest = [cx + dx * (C / 2 - 0.06), cy + dy * (C / 2 - 0.06)]
    const [sx, sy] = dx ? [0.03, 0.12] : [0.12, 0.03]
    g.push(`<body name="push_${p.x}_${p.y}" mocap="true" pos="${rest[0]} ${rest[1]} ${BZ + 0.07}">`
      + `<geom type="box" size="${sx} ${sy} 0.05" friction="0.2 0 0" solref="0.008 1"/></body>`)
  }
  let items = ''
  for (let i = 0; i < MAX_ITEMS; i++) items += itemBody(i, item, assets)
  return `<mujoco><option timestep="${DT}" cone="elliptic" impratio="10"/>
    <worldbody><geom name="floor" type="plane" size="40 40 .1" friction="0.8 0 0"/>
    ${g.join('')}${items}</worldbody></mujoco>`
}

export async function runFactory(layout: Layout, hid: Hidden, seconds: number, seed: number, recordFps = 0,
                                 onProgress?: (f: number) => void, assets?: ItemAssets): Promise<Report> {
  const M = await mujoco()
  const rng = new Rng(seed)
  const shp = itemShape(hid.item, assets)
  const m = M.MjModel.from_xml_string(sceneXml(layout, hid.item ?? 'box', assets))
  const d = new M.MjData(m)
  try {
    const itBid = Array.from({ length: MAX_ITEMS }, (_, i) => m.body(`it${i}`).id)
    const q0 = m.jnt_qposadr[m.body_jntadr[itBid[0]]]
    const v0 = m.jnt_dofadr[m.body_jntadr[itBid[0]]]
    const conv = new Map(layout.parts.filter((p) => CONVEY.has(p.kind)).map((p) => [key(p.x, p.y), p]))
    const scales = layout.parts.filter((p) => p.kind === 'scale')
    const scaleBid = new Map(scales.map((p) => [key(p.x, p.y), m.body(`sc_${p.x}_${p.y}`).id]))
    const bins = new Map(layout.parts.filter((p) => p.kind === 'bin').map((p) => [key(p.x, p.y), p]))
    const inlets = layout.parts.filter((p) => p.kind === 'inlet')
    const pushers = layout.parts.filter((p) => p.kind === 'pusher')
    const pushMocap = pushers.map((p) => m.body_mocapid[m.body(`push_${p.x}_${p.y}`).id])
    const mocap = d.mocap_pos
    const pushRest = pushMocap.map((i) => [mocap[3 * i], mocap[3 * i + 1]])
    const inertia0 = [0, 1, 2].map((a) => m.body_inertia[3 * itBid[0] + a])

    const active = new Array(MAX_ITEMS).fill(false)
    const kind = new Array(MAX_ITEMS).fill('')
    const mass = new Array(MAX_ITEMS).fill(1)
    const flagged = new Array(MAX_ITEMS).fill(false)
    const pushed = new Array(MAX_ITEMS).fill(false)
    const still = new Array(MAX_ITEMS).fill(0)
    const jamLogged = new Array(MAX_ITEMS).fill(false)
    const weighing: ([string, number[]] | null)[] = new Array(MAX_ITEMS).fill(null)
    const freeIds = Array.from({ length: MAX_ITEMS }, (_, i) => i)
    const rep: Report = {
      seconds, emitted: Object.fromEntries(inlets.map((p) => [p.name, 0])),
      bins: Object.fromEntries([...bins.values()].map((p) => [p.name, { count: 0, weight_kg: 0, kinds: {} }])),
      spills: {}, jams: {}, blocked_inlet: 0, in_transit: 0,
      scales: Object.fromEntries(scales.map((p) => [key(p.x, p.y), []])),
      pusher_fires: Object.fromEntries(pushers.map((p) => [key(p.x, p.y), 0])), frames: [],
    }
    const nextEmit = inlets.map((_, k) => 0.3 + 0.1 * k)
    const fireAt: number[][] = pushers.map(() => [])
    const extendUntil = pushers.map(() => -1)
    const every = recordFps ? Math.round(1 / (recordFps * DT)) : 0
    const restZ = BZ + shp.rest
    const halfIn = C / 2 - shp.half - 0.01
    const load = new Map([...scaleBid.keys()].map((k) => [k, 0]))
    const xpos = d.xpos, qvel = d.qvel, qpos = d.qpos, xfrc = d.xfrc_applied, xquat = d.xquat
    const bodyMass = m.body_mass, bodyInertia = m.body_inertia

    const park = (i: number) => {
      active[i] = false
      qpos.set([-5 - i * 0.3, -5, 0.1, 1, 0, 0, 0], q0 + 7 * i)
      for (let k = 0; k < 6; k++) { qvel[v0 + 6 * i + k] = 0; xfrc[itBid[i] * 6 + k] = 0 }
      weighing[i] = null
      freeIds.push(i)
    }
    const nsteps = Math.floor(seconds / DT)
    for (let step = 0; step < nsteps; step++) {
      const t = step * DT
      for (let k = 0; k < inlets.length; k++) {
        const p = inlets[k]
        if (t < nextEmit[k]) continue
        nextEmit[k] += 60.0 / (p.params.feed_per_min ?? 30)
        const [cx, cy] = cellCenter(p.x, p.y)
        const [dx, dy] = DIRS[p.dir]
        const sx = cx - dx * 0.12, sy = cy - dy * 0.12
        let room = true
        for (let i = 0; i < MAX_ITEMS; i++) {
          if (!active[i]) continue
          const ex = xpos[3 * itBid[i]] - sx, ey = xpos[3 * itBid[i] + 1] - sy
          if (Math.sqrt(ex * ex + ey * ey) <= shp.clear) { room = false; break }
        }
        if (!freeIds.length || !room) { rep.blocked_inlet += 1; continue }
        const i = freeIds.shift()!
        const mix = hid.mixes[p.name] ?? [['good', 1, 1, 0.01]]
        const total = mix.reduce((a, e) => a + e[1], 0)
        const u = rng.random() * total
        let acc = 0, pick = mix[mix.length - 1]
        for (const e of mix) { acc += e[1]; if (u < acc) { pick = e; break } }
        kind[i] = pick[0]
        mass[i] = Math.max(0.2, rng.normal(pick[2], pick[3]))
        bodyMass[itBid[i]] = mass[i]
        for (let a = 0; a < 3; a++) bodyInertia[3 * itBid[i] + a] = mass[i] * inertia0[a]
        active[i] = true
        flagged[i] = pushed[i] = jamLogged[i] = false
        still[i] = 0
        weighing[i] = null
        qpos.set([sx, sy, restZ + 0.001, 1, 0, 0, 0], q0 + 7 * i)
        for (let a = 0; a < 6; a++) qvel[v0 + 6 * i + a] = 0
        rep.emitted[p.name] += 1
      }
      for (let i = 0; i < MAX_ITEMS; i++) {
        if (!active[i]) continue
        const b = itBid[i]
        const x = xpos[3 * b], y = xpos[3 * b + 1], z = xpos[3 * b + 2]
        const p = conv.get(key(Math.floor(x / C), Math.floor(y / C)))
        let fx = 0, fy = 0
        if (p && Math.abs(z - restZ) < 0.02) {
          const [ux, uy] = DIRS[p.dir]
          const sp = p.params.speed ?? 1
          const wx = ux * sp - qvel[v0 + 6 * i]
          const wy = uy * sp - qvel[v0 + 6 * i + 1]
          fx = mass[i] * wx / DT; fy = mass[i] * wy / DT
          const cap = hid.belt_mu * mass[i] * G
          const n = Math.sqrt(fx * fx + fy * fy)
          if (n > cap) { fx = fx * cap / n; fy = fy * cap / n }
        }
        // the belt grips the base, not the centre of mass: same force plus torque r x F, r = (0, 0, -rest)
        const h = shp.rest
        xfrc[6 * b] = fx
        xfrc[6 * b + 1] = fy
        xfrc[6 * b + 3] = h * fy
        xfrc[6 * b + 4] = -h * fx
      }
      for (let i = 0; i < MAX_ITEMS; i++) {
        if (!active[i]) continue
        const b = itBid[i]
        const x = xpos[3 * b], y = xpos[3 * b + 1], z = xpos[3 * b + 2]
        const cxI = Math.floor(x / C), cyI = Math.floor(y / C)
        const ck = key(cxI, cyI)
        let on: string | null = null
        if (scaleBid.has(ck) && Math.abs(z - restZ) < 0.02) {
          const [cx, cy] = cellCenter(cxI, cyI)
          if (Math.abs(x - cx) < halfIn && Math.abs(y - cy) < halfIn) on = ck
        }
        let w = weighing[i]
        if (on !== null && (w === null || w[0] === on)) {
          if (w === null) w = weighing[i] = [on, []]
          w[1].push(load.get(on)!)
        } else if (w !== null) {
          weighing[i] = null
          const n = w[1].length
          const dwell = n * DT
          let sum = 0
          for (const s of w[1]) sum += s
          const reading = sum / n / G * 1000 + rng.normal(0, hid.scale_noise_g * Math.sqrt(0.5 / Math.max(dwell, 1e-3)))
          rep.scales[w[0]].push(reading)
          const prm = layout.parts.find((pp) => key(pp.x, pp.y) === w![0])!.params
          if (reading < (prm.reject_below_g ?? -1e9) || reading > (prm.reject_above_g ?? 1e9)) flagged[i] = true
        }
      }
      for (let j = 0; j < pushers.length; j++) {
        const p = pushers[j]
        const [dx, dy] = DIRS[p.dir]
        const tp = conv.get(key(p.x + dx, p.y + dy))
        if (tp) {
          const [ux, uy] = DIRS[tp.dir]
          const [tcx, tcy] = cellCenter(p.x + dx, p.y + dy)
          const off = (p.params.offset_mm ?? 50) / 1000
          for (let i = 0; i < MAX_ITEMS; i++) {
            if (!(active[i] && flagged[i] && !pushed[i])) continue
            const x = xpos[3 * itBid[i]], y = xpos[3 * itBid[i] + 1]
            const along = (x - tcx) * ux + (y - tcy) * uy
            const across = (x - tcx) * dx + (y - tcy) * dy
            if (Math.abs(across) < C / 2 && along > -C / 2 - 0.3 && along >= -off) {
              pushed[i] = true
              fireAt[j].push(t + hid.pusher_lag)
            }
          }
        }
        if (fireAt[j].length && t >= fireAt[j][0]) {
          fireAt[j].shift()
          extendUntil[j] = t + (p.params.stroke_ms ?? 200) / 1000
          rep.pusher_fires[key(p.x, p.y)] += 1
        }
        const ext = t < extendUntil[j] ? C * 1.2 : 0
        const mi = pushMocap[j]
        const tx = pushRest[j][0] + dx * ext, ty = pushRest[j][1] + dy * ext
        const ddx = tx - mocap[3 * mi], ddy = ty - mocap[3 * mi + 1]
        const dist = Math.sqrt(ddx * ddx + ddy * ddy)
        const stepLen = hid.pusher_speed * DT
        if (dist <= stepLen) { mocap[3 * mi] = tx; mocap[3 * mi + 1] = ty }
        else { mocap[3 * mi] += ddx / dist * stepLen; mocap[3 * mi + 1] += ddy / dist * stepLen }
      }
      M.mj_step(m, d)
      if (scaleBid.size) {
        M.mj_rnePostConstraint(m, d)
        const cf = d.cfrc_ext
        for (const [k, b] of scaleBid) load.set(k, Math.abs(cf[6 * b + 5]))
      }
      for (let i = 0; i < MAX_ITEMS; i++) {
        if (!active[i]) continue
        const b = itBid[i]
        const x = xpos[3 * b], y = xpos[3 * b + 1], z = xpos[3 * b + 2]
        const cx = Math.floor(x / C), cy = Math.floor(y / C)
        if (z < BZ - 0.12) {
          const bin = bins.get(key(cx, cy))
          if (bin && z < 0.35) {
            const r = rep.bins[bin.name]
            r.count += 1; r.weight_kg += mass[i]; r.kinds[kind[i]] = (r.kinds[kind[i]] ?? 0) + 1
            park(i)
          } else if (z < shp.floor) {
            rep.spills[key(cx, cy)] = (rep.spills[key(cx, cy)] ?? 0) + 1
            park(i)
          }
          continue
        }
        const vx = qvel[v0 + 6 * i], vy = qvel[v0 + 6 * i + 1]
        still[i] = Math.sqrt(vx * vx + vy * vy) < 0.02 ? still[i] + DT : 0
        if (still[i] > 3.0 && !jamLogged[i]) { jamLogged[i] = true; rep.jams[key(cx, cy)] = (rep.jams[key(cx, cy)] ?? 0) + 1 }
      }
      if (every && step % every === 0) {
        const items: Frame['items'] = []
        for (let i = 0; i < MAX_ITEMS; i++) {
          if (!active[i]) continue
          const b = itBid[i]
          const r3 = (v: number) => Math.round(v * 1000) / 1000
          items.push([i, kind[i] !== 'good', r3(xpos[3 * b]), r3(xpos[3 * b + 1]), r3(xpos[3 * b + 2]),
            r3(xquat[4 * b]), r3(xquat[4 * b + 1]), r3(xquat[4 * b + 2]), r3(xquat[4 * b + 3])])
        }
        rep.frames.push({ t: Math.round(t * 100) / 100, items, push: pushers.map((_, j) => {
          const mi = pushMocap[j]
          return Math.round(Math.hypot(mocap[3 * mi] - pushRest[j][0], mocap[3 * mi + 1] - pushRest[j][1]) * 1000) / 1000
        }) })
      }
      if (onProgress && step % 2500 === 0) onProgress(step / nsteps)
    }
    rep.in_transit = active.filter(Boolean).length
    return rep
  } finally {
    d.delete()
    m.delete()
  }
}

export function truth(r: Report) {
  const ship = r.bins.ship?.kinds ?? {}
  const good = ship.good ?? 0
  const bad = Object.entries(ship).filter(([k]) => k !== 'good').reduce((a, [, v]) => a + v, 0)
  const goodRejected = Object.entries(r.bins).filter(([n]) => n !== 'ship').reduce((a, [, b]) => a + (b.kinds.good ?? 0), 0)
  return {
    good_shipped: good, bad_shipped: bad, good_rejected: goodRejected,
    good_per_min: Math.round((good / r.seconds) * 60 * 10) / 10,
    spilled: Object.values(r.spills).reduce((a, b) => a + b, 0), jammed: Object.values(r.jams).reduce((a, b) => a + b, 0),
  }
}
