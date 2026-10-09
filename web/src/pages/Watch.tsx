import { useEffect, useState } from 'react'
import { goalText, loadIndex, loadLevel, loadReplay, type Index, type LevelBundle, type Replay } from '../data'
import { usePlayer } from '../replay/Player'
import { World } from '../scene/World'
import { Pill, probeIcon, readingText, SpectrogramView, stepTitle, Typing } from '../ui/bits'
import { navigate } from '../router'
import { muted, setMuted } from '../audio'

export function Watch({ run, levelId }: { run: string; levelId: string }) {
  const [index, setIndex] = useState<Index | null>(null)
  const [level, setLevel] = useState<LevelBundle | null>(null)
  const [replay, setReplay] = useState<Replay | null>(null)
  const [speed, setSpeed] = useState(1)
  const [autoNext, setAutoNext] = useState(true)
  const [mute, setMute] = useState(muted)

  useEffect(() => { loadIndex().then(setIndex) }, [])
  useEffect(() => {
    let alive = true
    Promise.all([loadLevel(levelId), loadReplay(run, levelId)]).then(([l, r]) => {
      if (alive) { setLevel(l); setReplay(r) }
    }).catch(() => { if (alive) { setReplay(null) } })
    return () => { alive = false }
  }, [run, levelId])

  const runInfo = index?.runs.find((r) => r.run === run)
  const ids = index?.levels.map((l) => l.id).filter((id) => runInfo?.episodes[id]) ?? []
  const pos = ids.indexOf(levelId)
  const go = (d: number) => {
    const next = ids[(pos + d + ids.length) % ids.length]
    if (next) navigate(`/watch/${run}/${next}`)
  }

  const { state, frame, controls } = usePlayer(level, replay, {
    autoplay: true, speed,
    onEnd: () => { if (autoNext) setTimeout(() => go(1), 1200) },
  })

  if (!level || !replay) {
    return <div className="grid h-full place-items-center text-muted">Loading replay…</div>
  }
  const seg = state?.seg
  const current = seg ? seg.step : -1
  const shown = replay.steps.filter((_, i) => i < current || (i === current && seg?.phase !== 'think'))
  const step = current >= 0 ? replay.steps[current] : null
  const thinkProgress = seg && seg.phase === 'think' && state ? (state.t - seg.t0) / (seg.dur * 0.8) : 1
  const probesLeft = level.task.budget - replay.steps.slice(0, current + (seg?.phase === 'think' ? 0 : 1)).filter((s) => s.action.type === 'probe' && !s.reading.error).length
  const showResult = seg?.phase === 'result' && replay.result
  const audioStep = step?.audio && seg?.phase === 'act' ? step : null

  return (
    <div className="relative h-full w-full">
      <div className="absolute inset-0">
        <World level={level} frame={frame} />
      </div>

      {/* level header */}
      <div className="pointer-events-none absolute left-4 top-20 max-w-[min(520px,calc(100%-2rem))] sm:left-6">
        <div className="glass pointer-events-auto rounded-2xl px-4 py-3">
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted">
            <Pill tone="dark">{replay.label}</Pill>
            <span className="font-mono">{level.id}</span>
            <span>·</span>
            <span>{level.task.budget} probes</span>
            {replay.tainted && <Pill tone="bad">tainted</Pill>}
          </div>
          <div className="mt-1.5 text-[15px] font-medium leading-snug">{goalText(level.task)}</div>
        </div>
      </div>

      {/* agent panel */}
      <aside className="glass absolute bottom-24 right-4 top-20 hidden w-[380px] flex-col overflow-hidden rounded-2xl md:flex">
        <div className="flex items-center justify-between border-b border-line px-4 py-3">
          <div className="text-sm font-semibold">Experiments</div>
          <div className="font-mono text-xs text-muted">{Math.max(0, probesLeft)} / {level.task.budget} probes left</div>
        </div>
        <ol className="space-y-1.5 px-4 py-3">
          {shown.map((s, i) => (
            <li key={i} className={`flex items-start gap-2.5 rounded-lg px-2 py-1.5 text-sm ${i === current ? 'bg-mint/10' : ''}`}>
              <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-md bg-ink font-mono text-[11px] text-white">{probeIcon(s.action.type === 'flick' ? 'flick' : s.action.probe!)}</span>
              <div className="min-w-0">
                <div className="font-medium">{stepTitle(s)}</div>
                <div className="font-mono text-xs text-muted">{readingText(s)}</div>
              </div>
            </li>
          ))}
          {!shown.length && <li className="px-2 text-sm text-muted">No experiments yet.</li>}
        </ol>
        <div className="mx-4 mb-3 rounded-xl border border-line bg-white/70 p-3">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted">
            {seg?.phase === 'intro' ? 'Reading the level' : 'Agent reasoning'}
          </div>
          <div className="min-h-[88px] text-[13.5px] leading-relaxed">
            {step?.reasoning ? <Typing text={step.reasoning} progress={thinkProgress} /> : <span className="text-muted">…</span>}
          </div>
        </div>
        {audioStep?.audio && (
          <div className="mx-4 mb-3">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-muted">What it heard</div>
            <SpectrogramView path={audioStep.audio} silent={(audioStep.reading.sound?.loudness_db_over_floor ?? 99) < 10} progress={state && seg ? Math.min(1, Math.max(0, (state.t - seg.t0 - 0.4) / 1.4)) : 1} className="h-[96px]" />
          </div>
        )}
        <div className="flex-1" />
        {showResult && replay.result && (
          <div className={`m-4 rounded-xl px-4 py-3 text-white ${replay.result.success ? 'bg-mint-deep' : 'bg-[#b5543a]'}`}>
            <div className="text-lg font-bold">{replay.result.success ? 'Solved' : replay.result.correct_object ? 'Missed' : 'Wrong object'}</div>
            <div className="text-sm opacity-90">
              {replay.result.correct_object
                ? `Stopped ${Math.abs(replay.result.error_m * 100).toFixed(1)} cm ${replay.result.error_m < 0 ? 'short of' : 'past'} the centre`
                : `The answer was ${replay.oracle.object}`}
              {' · '}{replay.result.probes_used} probes
            </div>
          </div>
        )}
      </aside>

      {/* transport */}
      <div className="glass absolute bottom-4 left-4 right-4 flex items-center gap-3 rounded-2xl px-3 py-2.5 sm:left-6 sm:right-6">
        <button className="rounded-lg px-2 py-1 text-sm hover:bg-black/5" onClick={() => go(-1)} aria-label="Previous level">⏮</button>
        <button className="grid size-9 place-items-center rounded-full bg-ink text-white" onClick={controls.toggle} aria-label="Play or pause">
          {state?.playing ? '❚❚' : '▶'}
        </button>
        <button className="rounded-lg px-2 py-1 text-sm hover:bg-black/5" onClick={() => go(1)} aria-label="Next level">⏭</button>
        <input type="range" min={0} max={state?.duration ?? 1} step={0.01} value={state?.t ?? 0}
          onChange={(e) => controls.seek(parseFloat(e.target.value))} className="min-w-0 flex-1" aria-label="Seek" />
        <span className="hidden w-24 text-right font-mono text-xs text-muted sm:block">
          {(state?.t ?? 0).toFixed(1)} / {(state?.duration ?? 0).toFixed(1)} s
        </span>
        <select className="rounded-lg bg-transparent px-1 py-1 text-sm" value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} aria-label="Speed">
          {[0.5, 1, 1.5, 2].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
        <label className="hidden items-center gap-1.5 text-xs text-muted sm:flex">
          <input type="checkbox" checked={autoNext} onChange={(e) => setAutoNext(e.target.checked)} /> autoplay next
        </label>
        <button className="rounded-lg px-2 py-1 text-sm hover:bg-black/5" onClick={() => { setMuted(!mute); setMute(!mute) }} aria-label="Mute">
          {mute ? '🔇' : '🔊'}
        </button>
      </div>
    </div>
  )
}
