import { useEffect, useRef, useState } from 'react'
import { loadLevel, type LevelBundle } from '../data'
import { navigate } from '../router'
import { emptyFrame, World, type Frame } from '../scene/World'
import { Pill } from '../ui/bits'

const EXAMPLES = [
  'three identical boxes, one has a marble inside, far target',
  'five cream cubes, exactly one is hollow, tight band',
  'heavy-looking trap: four big cubes, find the lightest',
  'icy table, very different masses, slide the heaviest just past half a metre',
  'two red boxes: one full of beads, one with silent foam, small budget',
]

interface Job {
  id: string
  status: 'running' | 'done' | 'failed'
  log: string[]
  elapsed: number
  level_id?: string
  report?: { passed: boolean; rounds: number; difficulty: { label: string } | null; checks: { name: string; passed: boolean; detail: string }[] }
  hidden?: { id: string; material: string; hollow: boolean; contents: string; friction: number; mass_g: number }[]
}

export function Build() {
  const [api, setApi] = useState<boolean | null>(null)
  const [prompt, setPrompt] = useState(EXAMPLES[0])
  const [job, setJob] = useState<Job | null>(null)
  const [level, setLevel] = useState<LevelBundle | null>(null)
  const [reveal, setReveal] = useState(false)
  const frame = useRef<Frame>(emptyFrame())

  useEffect(() => {
    fetch('/api/health').then((r) => setApi(r.ok)).catch(() => setApi(false))
  }, [])

  useEffect(() => {
    if (!job || job.status !== 'running') return
    const t = setInterval(async () => {
      const r = await fetch(`/api/build/${job.id}`)
      if (r.ok) setJob(await r.json())
    }, 1000)
    return () => clearInterval(t)
  }, [job])

  useEffect(() => {
    if (job?.status === 'done' && job.level_id) loadLevel(job.level_id).then(setLevel)
  }, [job?.status, job?.level_id])

  async function build() {
    setLevel(null); setReveal(false)
    const r = await fetch('/api/build', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt }) })
    const { id } = await r.json()
    setJob({ id, status: 'running', log: [], elapsed: 0 })
  }

  return (
    <div className="relative h-full w-full">
      <div className="absolute inset-0">
        {level ? <World level={level} frame={frame} auto /> : <div className="h-full w-full bg-[#ebe6dc]" />}
      </div>
      <div className="absolute left-4 top-20 w-[min(560px,calc(100%-2rem))] sm:left-6">
        <div className="glass rounded-2xl p-4">
          <div className="text-lg font-bold">Build a world from a prompt</div>
          <p className="mt-1 text-sm text-muted">A world-builder agent turns your request into a level. Validators check it's solvable, that exactly one cube fits, and that the probes can reveal it. Failures go back to the builder until it passes.</p>
          {api === false && (
            <div className="mt-3 rounded-xl bg-ink p-3 font-mono text-xs leading-relaxed text-white">
              Building needs a local Codex login. Run:<br />
              <span className="text-mint">uv run knock serve</span> and open http://127.0.0.1:8765/#/build
            </div>
          )}
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3}
            className="mt-3 w-full resize-none rounded-xl border border-line bg-white p-3 text-[15px] outline-none focus:border-ink/40" />
          <div className="mt-2 flex flex-wrap gap-1.5">
            {EXAMPLES.map((e) => (
              <button key={e} onClick={() => setPrompt(e)} className="rounded-full bg-black/5 px-2.5 py-1 text-xs hover:bg-black/10">{e}</button>
            ))}
          </div>
          <button disabled={!api || job?.status === 'running'} onClick={build}
            className="mt-3 w-full rounded-xl bg-ink py-2.5 text-sm font-bold text-white transition hover:brightness-125 disabled:opacity-40">
            {job?.status === 'running' ? `Building… ${job.elapsed.toFixed(0)} s` : 'Build world'}
          </button>
        </div>

        {job && (
          <div className="glass mt-3 rounded-2xl p-4">
            <div className="flex items-center justify-between">
              <div className="text-sm font-semibold">Builder log</div>
              <Pill tone={job.status === 'done' ? 'good' : job.status === 'failed' ? 'bad' : 'neutral'}>{job.status}</Pill>
            </div>
            <pre className="mt-2 max-h-40 overflow-y-auto whitespace-pre-wrap font-mono text-[11.5px] leading-relaxed text-muted">{job.log.join('\n') || 'starting…'}</pre>
            {job.report && (
              <div className="mt-2 space-y-1">
                {job.report.checks.map((c) => (
                  <div key={c.name} className="flex gap-2 text-xs">
                    <span className={c.passed ? 'text-mint-deep' : 'text-[#a8432c]'}>{c.passed ? '✓' : '✗'}</span>
                    <span className="font-semibold">{c.name}</span>
                    <span className="text-muted">{c.detail}</span>
                  </div>
                ))}
                {job.report.difficulty && <div className="pt-1 text-xs">Difficulty: <b>{job.report.difficulty.label}</b> · {job.report.rounds} round(s)</div>}
              </div>
            )}
            {job.status === 'done' && job.level_id && (
              <div className="mt-3 flex gap-2">
                <button onClick={() => navigate(`/play/${job.level_id}`)} className="flex-1 rounded-xl bg-coral py-2 text-sm font-bold text-white">Play it</button>
                <button onClick={() => setReveal(!reveal)} className="rounded-xl bg-black/5 px-3 py-2 text-sm">{reveal ? 'Hide' : 'Reveal'} hidden truth</button>
              </div>
            )}
            {reveal && job.hidden && (
              <table className="mt-2 w-full font-mono text-[11.5px]">
                <thead className="text-muted"><tr><th className="text-left">id</th><th className="text-left">material</th><th>hollow</th><th className="text-left">inside</th><th>μ</th><th className="text-right">mass</th></tr></thead>
                <tbody>{job.hidden.map((h) => (
                  <tr key={h.id}><td>{h.id}</td><td>{h.material}</td><td className="text-center">{h.hollow ? 'yes' : 'no'}</td><td>{h.contents}</td><td className="text-center">{h.friction.toFixed(2)}</td><td className="text-right">{h.mass_g.toFixed(0)} g</td></tr>
                ))}</tbody>
              </table>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
