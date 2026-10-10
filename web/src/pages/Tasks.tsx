// Task browser: every task in every environment, for Watch (agents) or Play (you).
import { useEffect, useState } from 'react'
import { loadIndex, type Index } from '../data'
import { navigate } from '../router'

export interface TaskEntry {
  env: 'factory' | 'cube'
  id: string
  title: string
  detail: string
  runs: { run: string; label: string; success?: boolean }[]
}

const ENV_INFO: Record<string, { name: string; blurb: string }> = {
  factory: { name: 'Factory', blurb: 'Physical Factorio: the agent places belts, scales, pushers and bins on a grid, runs the factory on MuJoCo physics, sees what spills and jams, fixes it, then hands it over for a scored production run.' },
  cube: { name: 'Cube lab', blurb: 'Identical-looking cubes hide mass, friction and contents. Probe, then one flick.' },
}

export function useTasks() {
  const [tasks, setTasks] = useState<TaskEntry[] | null>(null)
  useEffect(() => {
    Promise.all([
      loadIndex().catch(() => null as Index | null),
      fetch(`${import.meta.env.BASE_URL}data/factory/index.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    ]).then(([cube, factory]) => {
      const out: TaskEntry[] = []
      for (const t of factory?.tasks ?? []) out.push({ env: 'factory', ...t })
      for (const l of cube?.levels ?? []) {
        out.push({
          env: 'cube', id: l.id, title: l.kind === 'slide' ? 'Slide to target' : `Find the cube that ${l.property}`,
          detail: `${l.n_objects} cubes · ${l.budget} probes`,
          runs: (cube?.runs ?? []).filter((r) => r.episodes[l.id]).map((r) => ({ run: r.run, label: r.label, success: r.episodes[l.id].success })),
        })
      }
      setTasks(out)
    })
  }, [])
  return tasks
}

export function Tasks({ mode }: { mode: 'watch' | 'play' }) {
  const tasks = useTasks()
  const [env, setEnv] = useState<'factory' | 'cube'>('factory')
  const list = (tasks ?? []).filter((t) => t.env === env)
  const open = (t: TaskEntry, run?: string) => {
    if (mode === 'watch') navigate(`/watch/${t.env}/${run ?? t.runs[0]?.run ?? 'sol'}/${t.id}`)
    else navigate(t.env === 'cube' ? `/play/cube/${t.id}` : `/play/factory/${t.id}`)
  }
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-6xl px-6 pb-16 pt-24">
        <h1 className="text-3xl font-extrabold tracking-tight">{mode === 'watch' ? 'Watch agents work' : 'Play it yourself'}</h1>
        <p className="mt-1 text-muted">{mode === 'watch'
          ? 'Pick a task, then a model. You see every turn: the tool it used, its settings, its reasoning, what it predicted and what actually happened.'
          : 'Pick a task and take the same tools and budget the agents get.'}</p>
        <div className="mt-6 flex flex-wrap gap-2">
          {(['factory', 'cube'] as const).map((e) => (
            <button key={e} onClick={() => setEnv(e)}
              className={`rounded-full px-4 py-2 text-sm font-semibold ${env === e ? 'bg-ink text-white' : 'bg-white ring-1 ring-line hover:bg-black/5'}`}>
              {ENV_INFO[e].name}
            </button>
          ))}
          <span className="rounded-full px-4 py-2 text-sm text-muted ring-1 ring-line">Robot · soon</span>
          <span className="rounded-full px-4 py-2 text-sm text-muted ring-1 ring-line">Game · soon</span>
        </div>
        <p className="mt-3 text-sm text-muted">{ENV_INFO[env].blurb}</p>
        {!tasks && <div className="mt-8 text-muted">Loading tasks…</div>}
        {tasks && !list.length && <div className="mt-8 rounded-2xl border border-line bg-white p-6 text-muted">No tasks exported for this environment yet.</div>}
        <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {list.map((t) => (
            <div key={t.id} className="flex flex-col rounded-2xl border border-line bg-white p-4">
              <div className="font-semibold">{t.title}</div>
              <p className="mt-1 text-sm leading-snug text-muted">{t.detail}</p>
              {mode === 'watch' ? (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {t.runs.length === 0 && <span className="text-xs text-muted">no agent runs yet</span>}
                  {t.runs.map((r) => (
                    <button key={r.run} onClick={() => open(t, r.run)}
                      className={`rounded-full px-2.5 py-1 text-xs font-medium ring-1 ${r.success ? 'bg-mint/15 text-mint-deep ring-mint/30' : 'bg-coral/10 text-[#a8432c] ring-coral/30'} hover:brightness-95`}>
                      {r.label} {r.success ? '✓' : '✗'}
                    </button>
                  ))}
                </div>
              ) : (
                <button onClick={() => open(t)} className="mt-3 self-start rounded-full bg-ink px-4 py-1.5 text-sm font-semibold text-white"
                  disabled={t.env === 'factory'}>{t.env === 'factory' ? 'Play · coming next' : 'Play'}</button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
