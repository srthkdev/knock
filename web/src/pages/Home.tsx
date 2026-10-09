import { useEffect, useState } from 'react'
import { loadIndex, loadLevel, loadReplay, type Index, type LevelBundle, type Replay } from '../data'
import { usePlayer } from '../replay/Player'
import { navigate } from '../router'
import { World } from '../scene/World'
import { Pill, readingText, stepTitle } from '../ui/bits'

const HERO = { run: 'sol', level: 'L00009' }

export function Home() {
  const [index, setIndex] = useState<Index | null>(null)
  const [level, setLevel] = useState<LevelBundle | null>(null)
  const [replay, setReplay] = useState<Replay | null>(null)
  useEffect(() => {
    loadIndex().then(setIndex)
    Promise.all([loadLevel(HERO.level), loadReplay(HERO.run, HERO.level)]).then(([l, r]) => { setLevel(l); setReplay(r) })
  }, [])
  const { frame, state } = usePlayer(level, replay, { autoplay: true, loop: true })
  const seg = state?.seg
  const step = replay && seg && seg.step >= 0 ? replay.steps[seg.step] : null

  return (
    <div className="h-full overflow-y-auto">
      {/* hero */}
      <section className="relative h-[100vh] min-h-[640px] w-full">
        <div className="absolute inset-0">{level && <World level={level} frame={frame} auto framing="hero" />}</div>
        <div className="pointer-events-none absolute inset-x-0 top-0 h-56 bg-gradient-to-b from-paper via-paper/70 to-transparent" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-paper to-transparent" />
        <div className="pointer-events-none absolute inset-x-0 top-24 px-6 text-center">
          <h1 className="mx-auto max-w-4xl text-[clamp(2.2rem,5.5vw,4.4rem)] font-extrabold leading-[1.02] tracking-tight">
            Can an AI figure out<br />what&apos;s inside the box?
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-[clamp(1rem,1.6vw,1.2rem)] text-muted">
            Knock is a physics lab for AI agents. Every cube looks the same. Mass, friction, material and what&apos;s inside are hidden.
            Agents get a few experiments, then one shot.
          </p>
          <div className="pointer-events-auto mt-7 flex flex-wrap justify-center gap-3">
            <button onClick={() => navigate(`/watch/sol/${index?.levels[0]?.id ?? 'L00000'}`)} className="rounded-full bg-ink px-6 py-3 text-sm font-semibold text-white shadow-lg transition hover:-translate-y-0.5">Watch GPT-6.1 Sol play</button>
            <button onClick={() => navigate(`/play/${index?.levels[5]?.id ?? 'L00005'}`)} className="rounded-full bg-white px-6 py-3 text-sm font-semibold shadow-lg ring-1 ring-black/10 transition hover:-translate-y-0.5">Play it yourself</button>
          </div>
        </div>
        {step && (
          <div className="glass absolute bottom-8 left-4 w-[min(420px,calc(100%-2rem))] rounded-2xl px-4 py-3 text-sm sm:left-8">
            <div className="flex items-center gap-2 text-xs text-muted"><Pill tone="dark">GPT-6.1 Sol</Pill> live replay · {HERO.level}</div>
            <div className="mt-1 font-medium">{seg?.phase === 'think' ? 'thinking…' : `${stepTitle(step)} → ${readingText(step)}`}</div>
            {step.reasoning && <div className="mt-0.5 line-clamp-2 text-muted">{step.reasoning}</div>}
          </div>
        )}
      </section>

      {/* how it works */}
      <section className="mx-auto max-w-6xl px-6 py-16">
        <div className="grid gap-4 md:grid-cols-3">
          {[
            ['Probe', 'Push with a chosen force, lift to weigh, tap to hear if it rings or thuds, shake to hear what rattles. Every probe costs budget.'],
            ['Infer', 'A slow push could mean heavy or sticky. Weighing says nothing about friction. Good agents pick experiments that tell those apart.'],
            ['Flick', 'One strike. The cube has to stop inside the band. Physics is MuJoCo; sound comes from the simulated impacts themselves.'],
          ].map(([h, b], i) => (
            <div key={h} className="rounded-2xl border border-line bg-white p-6">
              <div className="font-mono text-xs text-muted">0{i + 1}</div>
              <div className="mt-1 text-xl font-bold">{h}</div>
              <p className="mt-2 text-[15px] leading-relaxed text-muted">{b}</p>
            </div>
          ))}
        </div>
      </section>

      {/* leaderboard */}
      <section id="leaderboard" className="mx-auto max-w-6xl px-6 pb-24">
        <div className="flex items-end justify-between gap-4">
          <div>
            <h2 className="text-3xl font-extrabold tracking-tight">Leaderboard</h2>
            <p className="mt-1 text-muted">Dev set, {index?.levels.length ?? 30} oracle-verified levels. Click any cell to watch the replay.</p>
          </div>
        </div>
        <div className="mt-6 overflow-x-auto rounded-2xl border border-line bg-white">
          <table className="w-full text-sm">
            <tbody>
              {index && [...index.runs].sort((a, b) => b.solved / Math.max(1, b.n) - a.solved / Math.max(1, a.n)).map((r) => (
                <tr key={r.run} className="border-b border-line last:border-0">
                  <td className="whitespace-nowrap px-4 py-3 font-semibold">{r.label}</td>
                  <td className="whitespace-nowrap px-2 py-3 text-right font-mono">{r.solved}/{r.n}</td>
                  <td className="w-24 px-3">
                    <div className="h-1.5 rounded-full bg-black/5"><div className="h-1.5 rounded-full bg-mint-deep" style={{ width: `${(100 * r.solved) / Math.max(1, r.n)}%` }} /></div>
                  </td>
                  <td className="px-3 py-3">
                    <div className="flex flex-wrap gap-1">
                      {index.levels.map((l) => {
                        const e = r.episodes[l.id]
                        if (!e) return <span key={l.id} className="size-4 rounded bg-black/5" title={`${l.id}: not run`} />
                        const cls = e.success ? 'bg-mint' : e.correct_object ? 'bg-coral/70' : 'bg-[#8a3a28]'
                        return <button key={l.id} onClick={() => navigate(`/watch/${r.run}/${l.id}`)} className={`size-4 rounded ${cls} transition hover:scale-125`}
                          title={`${l.id}: ${e.success ? 'solved' : e.correct_object ? `missed by ${Math.abs((e.error_m ?? 0) * 100).toFixed(1)} cm` : 'wrong cube'}`} />
                      })}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 flex gap-4 text-xs text-muted">
          <span className="flex items-center gap-1.5"><span className="size-3 rounded bg-mint" />solved</span>
          <span className="flex items-center gap-1.5"><span className="size-3 rounded bg-coral/70" />right cube, missed band</span>
          <span className="flex items-center gap-1.5"><span className="size-3 rounded bg-[#8a3a28]" />wrong cube</span>
        </div>
      </section>

      <footer className="border-t border-line px-6 py-8 text-center text-sm text-muted">
        Knock · physics by MuJoCo · <a className="underline" href="https://github.com/srthkdev/knock">GitHub</a>
      </footer>
    </div>
  )
}
