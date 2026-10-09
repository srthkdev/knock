// Turns a replay (agent trace + simulator trajectories) into a scrubbable timeline.
import type { LevelBundle, Pose, Replay, ReplayStep } from '../data'
import type { Effect, Frame, ObjPose } from '../scene/World'

export type Phase = 'intro' | 'think' | 'act' | 'result'
export interface Segment { t0: number; dur: number; phase: Phase; step: number; kind: string }
export interface Cue { t: number; kind: 'audio' | 'click'; path?: string; strength?: number }

const ease = (u: number) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, u)))

function samplePose(traj: Pose[], t: number): ObjPose {
  if (!traj.length) return { x: 0, z: 0, q: [1, 0, 0, 0] }
  if (t <= traj[0][0]) return { x: traj[0][1], z: traj[0][2], q: [traj[0][3], traj[0][4], traj[0][5], traj[0][6]] }
  const last = traj[traj.length - 1]
  if (t >= last[0]) return { x: last[1], z: last[2], q: [last[3], last[4], last[5], last[6]] }
  let lo = 0, hi = traj.length - 1
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (traj[m][0] <= t) lo = m; else hi = m }
  const a = traj[lo], b = traj[hi], u = (t - a[0]) / (b[0] - a[0])
  return { x: a[1] + (b[1] - a[1]) * u, z: a[2] + (b[2] - a[2]) * u, q: [a[3], a[4], a[5], a[6]] }
}

const trajSpan = (traj?: Pose[]) => (traj && traj.length ? traj[traj.length - 1][0] - traj[0][0] : 0)

export class Timeline {
  segments: Segment[] = []
  cues: Cue[] = []
  duration = 0
  private level: LevelBundle
  private replay: Replay

  constructor(level: LevelBundle, replay: Replay, opts: { intro?: boolean; think?: boolean } = {}) {
    this.level = level
    this.replay = replay
    let t = 0
    const add = (dur: number, phase: Phase, step: number, kind: string) => {
      this.segments.push({ t0: t, dur, phase, step, kind })
      t += dur
    }
    if (opts.intro !== false) add(1.6, 'intro', -1, 'intro')
    replay.steps.forEach((s, i) => {
      const words = (s.reasoning ?? '').split(/\s+/).length
      if (opts.think !== false) add(Math.min(3.2, 1.0 + words * 0.035), 'think', i, 'think')
      const a = s.action
      if (s.reading.error) { add(1.0, 'act', i, 'error'); return }
      if (a.type === 'flick') {
        const span = Math.max(trajSpan(s.traj), trajSpan(replay.oracle_traj ?? []))
        const rate = Math.max(1, span / 4)
        this.cues.push({ t: t + 0.35, kind: 'click', strength: Math.min(1, 0.3 + (a.impulse_Ns ?? 1) / 3) })
        add(0.35 + span / rate + 0.2, 'act', i, 'flick')
        add(opts.think === false ? 0.6 : 3.0, 'result', i, 'result')
        return
      }
      switch (a.probe) {
        case 'push': {
          const span = trajSpan(s.traj)
          add(span / Math.max(1, span / 3.5) + 1.3, 'act', i, 'push')
          break
        }
        case 'lift': add(2.3, 'act', i, 'lift'); break
        case 'tap':
          if (s.audio) this.cues.push({ t: t + 0.45, kind: 'audio', path: s.audio })
          add(2.1, 'act', i, 'tap')
          break
        case 'shake':
          if (s.audio) this.cues.push({ t: t + 0.4, kind: 'audio', path: s.audio })
          add(2.3, 'act', i, 'shake')
          break
        default: add(1.0, 'act', i, 'unknown')
      }
    })
    if (opts.intro !== false && !replay.steps.some((s) => s.action.type === 'flick')) add(2.0, 'result', replay.steps.length - 1, 'result')
    this.duration = t
  }

  segmentAt(t: number): Segment {
    const segs = this.segments
    for (let i = segs.length - 1; i >= 0; i--) if (t >= segs[i].t0) return segs[i]
    return segs[0]
  }

  /** Object positions before step `i` begins (every probe resets objects to the start line). */
  private restPoses(): Record<string, ObjPose> {
    const out: Record<string, ObjPose> = {}
    for (const o of this.level.objects) out[o.id] = { x: 0, z: o.half, q: [1, 0, 0, 0] }
    return out
  }

  frame(t: number, f: Frame) {
    const seg = this.segmentAt(t)
    const u = t - seg.t0
    f.poses = this.restPoses()
    f.striker = null
    f.ghost = null
    f.ghostObj = null
    f.effects = []
    f.trail = null
    f.now = t
    if (seg.phase !== 'act' && seg.phase !== 'result') return seg
    const step: ReplayStep = this.replay.steps[seg.step]
    const o = this.level.objects.find((x) => x.id === step.action.object)
    if (!o) return seg
    const effect = (e: Omit<Effect, 'obj'>) => f.effects.push({ obj: o.id, ...e })

    switch (seg.kind) {
      case 'push': {
        const traj = step.traj ?? []
        const span = trajSpan(traj)
        const rate = Math.max(1, span / 3.5)
        const play = span / rate
        if (u < play + 0.8) {
          f.poses[o.id] = samplePose(traj, (traj[0]?.[0] ?? 0) + Math.min(u, play) * rate)
          if (u < 0.4 / rate + 0.05) effect({ kind: 'arrow', t0: seg.t0, value: step.reading.force_N })
          if (u > play) effect({ kind: 'weight', t0: seg.t0 + play, label: `moved ${step.reading.moved_mm?.toFixed(0)} mm` })
        } else {
          const end = samplePose(traj, Infinity)
          const k = ease((u - play - 0.8) / 0.5)
          f.poses[o.id] = { x: end.x * (1 - k), z: o.half, q: [1, 0, 0, 0] }
        }
        break
      }
      case 'lift': {
        const h = u < 0.6 ? ease(u / 0.6) : u < 1.8 ? 1 : 1 - ease((u - 1.8) / 0.5)
        f.poses[o.id] = { x: 0, z: o.half + 0.12 * h, q: [1, 0, 0, 0] }
        if (u > 0.5 && u < 1.9) effect({ kind: 'weight', t0: seg.t0 + 0.5, label: `${step.reading.weight_g?.toFixed(0)} g` })
        break
      }
      case 'tap': {
        const xs = u < 0.45 ? -o.half - 0.012 - 0.2 * (1 - ease(u / 0.45)) : -o.half - 0.012 - 0.07 * ease((u - 0.45) / 0.3)
        if (u < 0.75) f.striker = [xs, o.y, o.half]
        if (u > 0.45) effect({ kind: 'ripple', t0: seg.t0 + 0.45 })
        break
      }
      case 'shake': {
        const lift = 0.14
        let z = o.half, x = 0
        if (u < 0.4) z += lift * ease(u / 0.4)
        else if (u < 1.8) {
          const s = u - 0.4
          z += lift + 0.012 * Math.sin(4 * Math.PI * 4 * s)
          x = 0.03 * Math.sin(2 * Math.PI * 4 * s)
        } else z += lift * (1 - ease((u - 1.8) / 0.4))
        f.poses[o.id] = { x, z, q: [1, 0, 0, 0] }
        break
      }
      case 'flick':
      case 'result': {
        const traj = step.traj ?? []
        const oracle = this.replay.oracle_traj ?? []
        const span = Math.max(trajSpan(traj), trajSpan(oracle))
        const rate = Math.max(1, span / 4)
        const local = seg.kind === 'result' ? Infinity : Math.max(0, u - 0.35) * rate
        if (seg.kind === 'flick' && u < 0.35) f.striker = [-o.half - 0.012 - 0.14 * (1 - ease(u / 0.35)), o.y, o.half]
        f.poses[o.id] = samplePose(traj, (traj[0]?.[0] ?? 0) + local)
        if (seg.kind === 'flick' && u > 0.35) f.trail = o.id
        if (oracle.length && u > 0.35) {
          const g = samplePose(oracle, (oracle[0]?.[0] ?? 0) + local)
          f.ghost = g
          f.ghostObj = this.replay.oracle.object
          if (f.ghostObj === o.id && seg.kind === 'result' && Math.abs(g.x - f.poses[o.id].x) < 0.01) f.ghost = null
        }
        break
      }
    }
    return seg
  }
}
