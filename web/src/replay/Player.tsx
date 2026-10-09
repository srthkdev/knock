// Plays a replay: drives the 3D frame each animation frame and fires sound cues.
import { useEffect, useMemo, useRef, useState } from 'react'
import { playClick, playSound } from '../audio'
import type { LevelBundle, Replay } from '../data'
import { emptyFrame, World, type Frame } from '../scene/World'
import { Timeline, type Segment } from './timeline'

export interface PlayerState { seg: Segment; t: number; duration: number; playing: boolean }

export function usePlayer(level: LevelBundle | null, replay: Replay | null, opts: { autoplay?: boolean; loop?: boolean; speed?: number; onEnd?: () => void } = {}) {
  const timeline = useMemo(() => (level && replay ? new Timeline(level, replay) : null), [level, replay])
  const frame = useRef<Frame>(emptyFrame())
  const t = useRef(0)
  const playing = useRef(!!opts.autoplay)
  const [state, setState] = useState<PlayerState | null>(null)
  const optsRef = useRef(opts)
  optsRef.current = opts

  useEffect(() => {
    if (!timeline) return
    t.current = 0
    playing.current = !!optsRef.current.autoplay
    timeline.frame(0, frame.current)
    let last = performance.now()
    let lastUi = 0
    let raf = 0
    let prevSeg: Segment | null = null
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      if (playing.current) {
        const t0 = t.current
        t.current = Math.min(timeline.duration, t.current + dt * (optsRef.current.speed ?? 1))
        for (const c of timeline.cues) {
          if (c.t > t0 && c.t <= t.current) {
            if (c.kind === 'audio' && c.path) playSound(c.path)
            else playClick(c.strength)
          }
        }
        if (t.current >= timeline.duration) {
          if (optsRef.current.loop) t.current = 0
          else { playing.current = false; optsRef.current.onEnd?.() }
        }
      }
      const seg = timeline.frame(t.current, frame.current)
      if (seg !== prevSeg || now - lastUi > 80) {
        prevSeg = seg
        lastUi = now
        setState({ seg, t: t.current, duration: timeline.duration, playing: playing.current })
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [timeline])

  const controls = {
    play: () => { if (timeline && t.current >= timeline.duration) t.current = 0; playing.current = true },
    pause: () => { playing.current = false },
    toggle: () => { if (playing.current) controls.pause(); else controls.play() },
    seek: (s: number) => { t.current = Math.max(0, Math.min(timeline?.duration ?? 0, s)) },
  }
  return { timeline, frame, state, controls }
}

export function ReplayCanvas({ level, frame, auto }: { level: LevelBundle; frame: React.RefObject<Frame>; auto?: boolean }) {
  return <World level={level} frame={frame} auto={auto} />
}
