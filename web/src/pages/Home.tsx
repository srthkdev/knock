import { useEffect, useState } from 'react'
import { loadIndex, type Index } from '../data'
import { navigate } from '../router'
import { FactoryHero } from './FactoryWatch'

const TRACKS = [
  {
    pillar: 'Factory', title: 'Physical Factorio', status: 'Live',
    body: 'Agents build a factory on a grid (belts, scales, pushers, bins), run it on real physics, watch what spills and jams, fix it, and hand it over for a scored production run. Six maps, easy to brutal.',
    href: '/watch',
  },
  {
    pillar: 'Robot', title: 'Disassembly', status: 'In development',
    body: 'A robot arm takes apart an unknown device to recover one part intact. Hidden screws, clips and glue; too much force breaks things.',
  },
  {
    pillar: 'Game', title: 'Factory Duel', status: 'In development',
    body: 'Two models build rival lines from the same budget and compete for the same customer orders. Physics plus economics plus reading your opponent.',
  },
]

export function Home() {
  const [index, setIndex] = useState<Index | null>(null)
  const [fx, setFx] = useState<{ tasks: { id: string; title: string; runs: { run: string; label: string; success: boolean; score: number; turns: number }[] }[] } | null>(null)
  useEffect(() => {
    loadIndex().then(setIndex)
    fetch(`${import.meta.env.BASE_URL}data/factory/index.json`).then((r) => r.json()).then(setFx).catch(() => {})
  }, [])
  const fxTasks = (fx?.tasks ?? []).filter((t) => t.id !== 'bottling')
  const fxRuns = [...new Set(fxTasks.flatMap((t) => t.runs.map((r) => r.run)))]
  const fxLabel = (run: string) => fxTasks.flatMap((t) => t.runs).find((r) => r.run === run)?.label ?? run

  return (
    <div className="h-full overflow-y-auto">
      {/* hero */}
      <section className="relative h-[100vh] min-h-[640px] w-full bg-[#2a2b38]">
        <div className="absolute inset-0"><FactoryHero run="sol" task="two_lines" /></div>
        <div className="pointer-events-none absolute inset-x-0 top-0 h-72 bg-gradient-to-b from-[#1d1e27] via-[#1d1e27]/70 to-transparent" />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-paper to-transparent" />
        <div className="pointer-events-none absolute inset-x-0 top-24 px-6 text-center text-white">
          <h1 className="mx-auto max-w-4xl text-[clamp(2.2rem,5.5vw,4.4rem)] font-extrabold leading-[1.02] tracking-tight">
            Can AI run a real factory?
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-[clamp(1rem,1.6vw,1.2rem)] text-white/70">
            GPT agents build factories on real physics: they place belts, scales and pushers, run the line, see boxes spill and jam,
            fix it over many turns, then hand it over for a scored production run.
          </p>
          <div className="pointer-events-auto mt-7 flex flex-wrap justify-center gap-3">
            <button onClick={() => navigate('/watch/factory/sol/two_lines')} className="rounded-full bg-white px-6 py-3 text-sm font-semibold text-ink shadow-lg transition hover:-translate-y-0.5">Watch an agent build one</button>
            <button onClick={() => document.getElementById('tracks')?.scrollIntoView({ behavior: 'smooth' })} className="rounded-full bg-white/10 px-6 py-3 text-sm font-semibold text-white ring-1 ring-white/25 backdrop-blur transition hover:-translate-y-0.5">What's inside</button>
          </div>
        </div>
      </section>

      {/* tracks */}
      <section id="tracks" className="mx-auto max-w-6xl px-6 py-16">
        <h2 className="text-3xl font-extrabold tracking-tight">Three tracks</h2>
        <p className="mt-1 max-w-2xl text-muted">Each one is a multi-turn job with real equipment and hidden physical properties. Agents run experiments, read the results, adjust, and are scored on what actually happens.</p>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {TRACKS.map((t) => (
            <div key={t.title} className="flex flex-col rounded-2xl border border-line bg-white p-6">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs uppercase tracking-wider text-muted">{t.pillar}</span>
                <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${t.href ? 'bg-mint/15 text-mint-deep' : 'bg-black/5 text-muted'}`}>{t.status}</span>
              </div>
              <div className="mt-2 text-xl font-bold">{t.title}</div>
              <p className="mt-2 flex-1 text-[15px] leading-relaxed text-muted">{t.body}</p>
              {t.href && <button onClick={() => navigate(t.href!)} className="mt-4 self-start rounded-full bg-ink px-4 py-2 text-sm font-semibold text-white">Open</button>}
            </div>
          ))}
        </div>
      </section>

      {/* factory leaderboard */}
      {fxTasks.length > 0 && (
        <section className="mx-auto max-w-6xl px-6 pb-16">
          <h2 className="text-3xl font-extrabold tracking-tight">Factory leaderboard</h2>
          <p className="mt-1 max-w-2xl text-muted">Each cell is one full episode: the model builds, tests and fixes its factory, then a 90 s production run is scored. Click to watch every turn.</p>
          <div className="mt-6 overflow-x-auto rounded-2xl border border-line bg-white">
            <table className="w-full text-sm">
              <thead><tr className="border-b border-line text-left text-xs text-muted">
                <th className="px-4 py-3 font-medium">Model</th><th className="px-2 py-3 font-medium">Passed</th>
                {fxTasks.map((t) => <th key={t.id} className="px-2 py-3 font-medium">{t.title}</th>)}
              </tr></thead>
              <tbody>
                {fxRuns.map((run) => {
                  const cells = fxTasks.map((t) => t.runs.find((r) => r.run === run))
                  const passed = cells.filter((c) => c?.success).length
                  return (
                    <tr key={run} className="border-b border-line last:border-0">
                      <td className="whitespace-nowrap px-4 py-3 font-semibold">{fxLabel(run)}</td>
                      <td className="px-2 py-3 font-mono">{passed}/{cells.filter(Boolean).length}</td>
                      {fxTasks.map((t, k) => {
                        const c = cells[k]
                        if (!c) return <td key={t.id} className="px-2 py-3 text-muted">–</td>
                        return (
                          <td key={t.id} className="px-2 py-2">
                            <button onClick={() => navigate(`/watch/factory/${run}/${t.id}`)}
                              className={`rounded-lg px-2.5 py-1 font-mono text-xs ring-1 transition hover:scale-105 ${c.success ? 'bg-mint/15 text-mint-deep ring-mint/30' : 'bg-coral/10 text-[#a8432c] ring-coral/30'}`}>
                              {c.success ? '✓' : '✗'} {c.score} · {c.turns}t
                            </button>
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* leaderboard */}
      <section id="leaderboard" className="mx-auto max-w-6xl px-6 pb-24">
        <div className="flex items-end justify-between gap-4">
          <div>
            <div className="font-mono text-xs uppercase tracking-wider text-muted">Tutorial tier</div>
            <h2 className="text-3xl font-extrabold tracking-tight">The cube lab</h2>
            <p className="mt-1 max-w-2xl text-muted">Where Knock started: identical-looking cubes hide mass, friction and contents; agents probe, then make one flick. 30 oracle-verified levels, benchmark v0.2. Click any cell to watch the replay.</p>
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
        Knock · physics by MuJoCo · factory models by Kenney (CC0) · <a className="underline" href="https://github.com/srthkdev/knock">GitHub</a>
      </footer>
    </div>
  )
}
