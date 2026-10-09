import { useEffect, useMemo, useRef, useState } from 'react'
import { playClick, playSound } from '../audio'
import { goalText, loadIndex, loadLevel, type Index, type LevelBundle, type Pose, type Replay, type ReplayStep } from '../data'
import { navigate } from '../router'
import { emptyFrame, World, type Frame } from '../scene/World'
import { simulateFlick, simulatePush } from '../sim/physics'
import { Timeline } from '../replay/timeline'
import { Pill, probeIcon, readingText, SpectrogramView, stepTitle } from '../ui/bits'

const G = 9.81
const logSlider = (u: number, lo: number, hi: number) => lo * Math.pow(hi / lo, u)
const unLog = (v: number, lo: number, hi: number) => Math.log(v / lo) / Math.log(hi / lo)
const gauss = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random())

/** Coulomb-friction push model: distance after a force F for T seconds, then release. */
function pushDistance(F: number, m: number, mu: number, T: number) {
  const a = Math.max(0, F / m - mu * G)
  const v = a * T
  return (a * T * T) / 2 + (v * v) / (2 * mu * G)
}

function notebook(level: LevelBundle, steps: ReplayStep[], obj: string) {
  const lift = steps.find((s) => s.action.object === obj && s.action.probe === 'lift')
  const push = steps.filter((s) => s.action.object === obj && s.action.probe === 'push' && !s.reading.error)
  const m = lift?.reading.weight_g ? lift.reading.weight_g / 1000 : null
  let mu: number | null = null
  let muBound: string | null = null
  const moved = push.find((p) => (p.reading.moved_mm ?? 0) > 2)
  if (m && moved) {
    let lo = 0.02, hi = 2
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2
      if (pushDistance(moved.reading.force_N!, m, mid, level.physics.push_seconds) * 1000 > moved.reading.moved_mm!) lo = mid
      else hi = mid
    }
    mu = (lo + hi) / 2
  } else if (m && push.length) {
    muBound = `μ ≥ ${(push[0].reading.force_N! / (m * G)).toFixed(2)} (it didn't move)`
  }
  const d = level.task.target_x
  const J = m && mu ? m * Math.sqrt(2 * mu * G * d) : null
  return { m, mu, muBound, J }
}

export function Play({ levelId }: { levelId: string }) {
  const [level, setLevel] = useState<LevelBundle | null>(null)
  const [index, setIndex] = useState<Index | null>(null)
  const [steps, setSteps] = useState<ReplayStep[]>([])
  const [selected, setSelected] = useState<string | null>(null)
  const [hovered, setHovered] = useState<string | null>(null)
  const [force, setForce] = useState(3)
  const [impulse, setImpulse] = useState(0.5)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ success: boolean; correct: boolean; error: number; x: number } | null>(null)
  const [assist, setAssist] = useState(false)
  const frame = useRef<Frame>(emptyFrame())
  const anim = useRef<{ tl: Timeline; t: number; hold: boolean } | null>(null)

  useEffect(() => { loadIndex().then(setIndex) }, [])
  useEffect(() => {
    setLevel(null); setSteps([]); setResult(null); setSelected(null)
    frame.current = emptyFrame()
    loadLevel(levelId).then((l) => {
      setLevel(l)
      setSelected(l.task.kind === 'slide' ? l.task.object_id : null)
    })
  }, [levelId])

  // animation loop for the most recent action
  useEffect(() => {
    let raf = 0, last = performance.now()
    const loop = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      const a = anim.current
      if (a) {
        const t0 = a.t
        a.t = Math.min(a.tl.duration, a.t + dt)
        for (const c of a.tl.cues) if (c.t > t0 && c.t <= a.t) {
          if (c.kind === 'audio' && c.path) playSound(c.path)
          else playClick(c.strength)
        }
        a.tl.frame(a.t, frame.current)
        if (a.t >= a.tl.duration && !a.hold) { anim.current = null; frame.current = emptyFrame(); setBusy(false) }
        else if (a.t >= a.tl.duration) setBusy(false)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  const left = level ? level.task.budget - steps.filter((s) => s.action.type === 'probe').length : 0
  const done = !!result

  function animate(step: ReplayStep, oracleTraj: Pose[] | null = null, hold = false) {
    if (!level) return
    const replay: Replay = {
      run: 'you', label: 'You', level: level.id, steps: [step], result: null, tainted: false,
      oracle: level.oracle ?? { object: level.answer, impulse_Ns: 0, predicted_x: 0 }, oracle_traj: oracleTraj,
    }
    anim.current = { tl: new Timeline(level, replay, { intro: false, think: false }), t: 0, hold }
    setBusy(true)
  }

  async function probe(kind: 'push' | 'lift' | 'tap' | 'shake') {
    if (!level || !selected || busy || left <= 0 || done) return
    const o = level.objects.find((x) => x.id === selected)!
    let step: ReplayStep
    if (kind === 'push') {
      const r = await simulatePush(o.mjcf, force, level.physics.push_seconds, 30)
      const mm = Math.max(0, r.distance * 1000 + gauss())
      step = { action: { type: 'probe', probe: 'push', object: o.id, force_N: force }, reading: { force_N: force, moved_mm: mm }, reasoning: null, traj: r.traj }
    } else if (kind === 'lift') {
      step = { action: { type: 'probe', probe: 'lift', object: o.id }, reading: { weight_g: o.lift_g }, reasoning: null }
    } else {
      const s = o[kind]
      step = { action: { type: 'probe', probe: kind, object: o.id }, reading: { sound: s.features }, reasoning: null, audio: s.audio }
    }
    setSteps((p) => [...p, step])
    animate(step)
  }

  async function flick() {
    if (!level || !selected || busy || done) return
    const o = level.objects.find((x) => x.id === selected)!
    const r = await simulateFlick(o.mjcf, impulse, 30)
    let ghost: Pose[] | null = null
    if (level.oracle) {
      const oo = level.objects.find((x) => x.id === level.oracle!.object)!
      ghost = (await simulateFlick(oo.mjcf, level.oracle.impulse_Ns, 30)).traj
    }
    const step: ReplayStep = { action: { type: 'flick', object: o.id, impulse_Ns: impulse }, reading: {}, reasoning: null, traj: r.traj }
    setSteps((p) => [...p, step])
    animate(step, ghost, true)
    const correct = o.id === level.answer
    const err = r.distance - level.task.target_x
    setTimeout(() => setResult({ success: correct && Math.abs(err) <= level.task.tol, correct, error: err, x: r.distance }),
      (0.35 + Math.max(...r.traj.map((p) => p[0])) / Math.max(1, Math.max(...r.traj.map((p) => p[0])) / 4)) * 1000 + 300)
  }

  const nb = useMemo(() => (level && selected ? notebook(level, steps, selected) : null), [level, steps, selected])
  const agents = index?.runs.filter((r) => r.episodes[levelId] && ['sol', 'luna', 'scripted'].includes(r.run)) ?? []
  const lastAudio = [...steps].reverse().find((s) => s.audio)

  if (!level) return <div className="grid h-full place-items-center text-muted">Loading level…</div>
  const fr = level.physics.flick_range, pr = level.physics.push_range

  return (
    <div className="relative h-full w-full">
      <div className="absolute inset-0">
        <World level={level} frame={frame} selected={selected} hovered={hovered}
          onSelect={(id) => !done && setSelected(id)} onHover={setHovered} />
      </div>

      <div className="pointer-events-none absolute left-4 top-20 max-w-[min(520px,calc(100%-2rem))] sm:left-6">
        <div className="glass pointer-events-auto rounded-2xl px-4 py-3">
          <div className="flex items-center gap-2 text-xs text-muted">
            <Pill tone="dark">You</Pill><span className="font-mono">{level.id}</span>
          </div>
          <div className="mt-1.5 text-[15px] font-medium leading-snug">{goalText(level.task)}</div>
          <div className="mt-1 text-xs text-muted">The cubes look identical. Probe them, then make one flick. Click a cube to select it.</div>
        </div>
      </div>

      <aside className="glass absolute bottom-4 right-4 top-20 flex w-[min(380px,calc(100%-2rem))] flex-col overflow-y-auto rounded-2xl">
        <div className="flex items-center justify-between border-b border-line px-4 py-3">
          <div className="text-sm font-semibold">Your lab bench</div>
          <div className="font-mono text-xs text-muted">{left} / {level.task.budget} probes left</div>
        </div>

        <div className="px-4 pt-3">
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">Object</div>
          <div className="flex gap-2">
            {level.objects.map((o) => (
              <button key={o.id} disabled={done || (level.task.kind === 'slide' && o.id !== level.task.object_id && false)}
                onClick={() => setSelected(o.id)}
                className={`size-10 rounded-xl text-sm font-bold transition ${selected === o.id ? 'bg-ink text-white' : 'bg-black/5 hover:bg-black/10'}`}>
                {o.id}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 px-4 pt-3">
          {(['lift', 'tap', 'shake'] as const).map((k) => (
            <button key={k} disabled={!selected || busy || left <= 0 || done} onClick={() => probe(k)}
              className="rounded-xl border border-line bg-white px-3 py-2 text-left text-sm font-medium transition hover:border-ink/30 disabled:opacity-40">
              <span className="mr-1.5 font-mono">{probeIcon(k)}</span>{k}
            </button>
          ))}
          <button disabled={!selected || busy || left <= 0 || done} onClick={() => probe('push')}
            className="rounded-xl border border-line bg-white px-3 py-2 text-left text-sm font-medium transition hover:border-ink/30 disabled:opacity-40">
            <span className="mr-1.5 font-mono">→</span>push {force.toFixed(1)} N
          </button>
        </div>
        <label className="flex items-center gap-2 px-4 pt-2 text-xs text-muted">
          push force
          <input type="range" min={0} max={1} step={0.001} value={unLog(force, pr[0], pr[1])}
            onChange={(e) => setForce(Math.round(logSlider(parseFloat(e.target.value), pr[0], pr[1]) * 100) / 100)} className="flex-1" />
        </label>

        <div className="mx-4 mt-3 rounded-xl bg-ink p-3 text-white">
          <div className="flex items-center justify-between text-sm font-semibold">
            <span>Flick {selected ?? '—'}</span>
            <span className="font-mono">{impulse.toFixed(3)} N·s</span>
          </div>
          <input type="range" min={0} max={1} step={0.0005} value={unLog(impulse, fr[0], fr[1])}
            onChange={(e) => setImpulse(Math.round(logSlider(parseFloat(e.target.value), fr[0], fr[1]) * 1000) / 1000)}
            className="mt-2 w-full" disabled={done} />
          <button disabled={!selected || busy || done} onClick={flick}
            className="mt-2 w-full rounded-lg bg-coral py-2 text-sm font-bold text-white transition hover:brightness-110 disabled:opacity-40">
            Flick it (one shot)
          </button>
        </div>

        <div className="px-4 pt-3">
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={assist} onChange={(e) => setAssist(e.target.checked)} /> lab notebook (does the physics arithmetic)
          </label>
          {assist && nb && (
            <div className="mt-2 rounded-xl border border-line bg-white/70 p-3 font-mono text-xs leading-relaxed">
              <div>mass m = {nb.m ? `${(nb.m * 1000).toFixed(0)} g` : 'lift it'}</div>
              <div>friction μ = {nb.mu ? nb.mu.toFixed(3) : nb.muBound ?? 'push it (enough force to move)'}</div>
              <div>stop distance d = J² / (2 m² μ g)</div>
              <div className="text-mint-deep">{nb.J ? `→ flick ≈ ${nb.J.toFixed(3)} N·s` : ''}</div>
              {nb.J && <button className="mt-1 text-[11px] underline" onClick={() => setImpulse(Math.round(nb.J! * 1000) / 1000)}>use it</button>}
            </div>
          )}
        </div>

        <ol className="space-y-1 px-4 py-3">
          {steps.map((s, i) => (
            <li key={i} className="flex items-start gap-2.5 rounded-lg px-1 py-1 text-sm">
              <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-md bg-ink font-mono text-[11px] text-white">{probeIcon(s.action.type === 'flick' ? 'flick' : s.action.probe!)}</span>
              <div><div className="font-medium">{stepTitle(s)}</div><div className="font-mono text-xs text-muted">{readingText(s)}</div></div>
            </li>
          ))}
        </ol>
        {lastAudio?.audio && <div className="px-4 pb-3"><SpectrogramView path={lastAudio.audio} className="h-[90px]" /></div>}

        {result && (
          <div className={`mx-4 mb-4 rounded-xl px-4 py-3 text-white ${result.success ? 'bg-mint-deep' : 'bg-[#b5543a]'}`}>
            <div className="text-lg font-bold">{result.success ? 'Solved!' : result.correct ? 'So close' : 'Wrong cube'}</div>
            <div className="text-sm opacity-90">
              {result.correct ? `Stopped ${Math.abs(result.error * 100).toFixed(1)} cm ${result.error < 0 ? 'short' : 'past'} the centre.` : `The answer was ${level.answer}.`}
              {level.oracle && ` Ideal flick: ${level.oracle.impulse_Ns.toFixed(3)} N·s on ${level.oracle.object}.`}
            </div>
            {agents.length > 0 && (
              <div className="mt-2 space-y-0.5 border-t border-white/20 pt-2 text-xs">
                {agents.map((r) => {
                  const e = r.episodes[levelId]
                  return (
                    <button key={r.run} className="flex w-full justify-between hover:underline" onClick={() => navigate(`/watch/${r.run}/${levelId}`)}>
                      <span>{r.label}</span>
                      <span>{e.success ? 'solved' : e.correct_object ? `missed by ${Math.abs((e.error_m ?? 0) * 100).toFixed(1)} cm` : 'wrong cube'} · watch →</span>
                    </button>
                  )
                })}
              </div>
            )}
            <div className="mt-2 flex gap-2">
              <button className="rounded-lg bg-white/20 px-3 py-1 text-xs font-semibold" onClick={() => navigate(`/play/${levelId}?r=${Date.now()}`)}>Retry</button>
              <button className="rounded-lg bg-white px-3 py-1 text-xs font-semibold text-ink" onClick={() => {
                const ids = index?.levels.map((l) => l.id) ?? []
                navigate(`/play/${ids[(ids.indexOf(levelId) + 1) % ids.length]}`)
              }}>Next level →</button>
            </div>
          </div>
        )}
      </aside>
    </div>
  )
}
