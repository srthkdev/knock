// Play a factory map yourself: same floor, parts, prices, hidden physics and scoring as the agents.
// The MuJoCo simulation runs in your browser (web/src/sim/factory.ts, a port of the Python engine).
import { useEffect, useMemo, useRef, useState } from 'react'
import { navigate } from '../router'
import { truth, type Hidden, type Layout, type Part, type Report } from '../sim/factory'
import { Scene, type Replay } from './FactoryWatch'

const BASE = `${import.meta.env.BASE_URL}`
const PRICES: Record<string, number> = { belt: 20, scale: 400, pusher: 300, bin: 60 }
const DEFAULTS: Record<string, Record<string, number>> = {
  belt: { speed: 1.0 }, scale: { speed: 1.0, reject_below_g: 0, reject_above_g: 3000 }, pusher: { offset_mm: 100, stroke_ms: 250 }, bin: {},
}
const LIMITS: Record<string, [number, number, number]> = {
  speed: [0.2, 2.5, 0.1], reject_below_g: [0, 3000, 1], reject_above_g: [0, 3000, 1], offset_mm: [0, 800, 10], stroke_ms: [50, 800, 10],
}
const LABEL: Record<string, string> = { speed: 'speed (m/s)', reject_below_g: 'reject below (g)', reject_above_g: 'reject above (g)', offset_mm: 'fire offset (mm)', stroke_ms: 'stroke (ms)' }
const DIRN = ['E', 'S', 'W', 'N']
const ARROW: Record<string, string> = { E: '→', S: '↓', W: '←', N: '↑' }

interface PlayLevel {
  id: string; title: string; goal: string; start: Layout; hidden: Hidden; money: number; target: number; max_bad: number
  budgets: { turns: number; sim_seconds: number }; production_seed: number; production_s: number
}

let worker: Worker | null = null
let jobId = 0
let assets: Promise<any> | null = null
async function simulate(layout: Layout, hidden: Hidden, seconds: number, seed: number, onProgress: (f: number) => void): Promise<Report> {
  worker ??= new Worker(new URL('../sim/factory.worker.ts', import.meta.url), { type: 'module' })
  assets ??= fetch(`${BASE}data/factory/items.json`).then((r) => r.json())
  const items = hidden.item && hidden.item !== 'box' ? await assets : undefined
  const id = ++jobId
  return new Promise((resolve, reject) => {
    const on = (e: MessageEvent) => {
      if (e.data.id !== id) return
      if (e.data.progress !== undefined) return onProgress(e.data.progress)
      worker!.removeEventListener('message', on)
      e.data.error ? reject(new Error(e.data.error)) : resolve(e.data.report)
    }
    worker!.addEventListener('message', on)
    worker!.postMessage({ id, layout, hidden, seconds, seed, fps: 12, assets: items })
  })
}

const sumv = (o: Record<string, number>) => Object.values(o ?? {}).reduce((a, b) => a + b, 0)

export function FactoryPlay({ levelId }: { levelId: string }) {
  const [lvl, setLvl] = useState<PlayLevel | null>(null)
  const [err, setErr] = useState('')
  const [layout, setLayout] = useState<Layout | null>(null)
  const [money, setMoney] = useState(0)
  const [tool, setTool] = useState<'belt' | 'scale' | 'pusher' | 'bin' | 'erase' | 'select'>('belt')
  const [dir, setDir] = useState('E')
  const [defaults, setDefaults] = useState(DEFAULTS)
  const [sel, setSel] = useState<[number, number] | null>(null)
  const [hover, setHover] = useState<[number, number] | null>(null)
  const [seconds, setSeconds] = useState(30)
  const [simLeft, setSimLeft] = useState(0)
  const [runs, setRuns] = useState<{ seconds: number; report: Report }[]>([])
  const [replay, setReplay] = useState<Replay | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [final, setFinal] = useState<{ t: ReturnType<typeof truth>; score: number; checks: Record<string, boolean> } | null>(null)
  const [agents, setAgents] = useState<{ run: string; label: string; success: boolean; score: number }[]>([])
  const frameRef = useRef(0)

  useEffect(() => {
    fetch(`${BASE}data/factory/play/${levelId}.json`).then((r) => { if (!r.ok) throw new Error(); return r.json() }).then((d: PlayLevel) => {
      setLvl(d); setLayout(structuredClone(d.start)); setMoney(d.money); setSimLeft(d.budgets.sim_seconds)
    }).catch(() => setErr('This map is not playable yet.'))
    fetch(`${BASE}data/factory/index.json`).then((r) => r.json()).then((ix) => setAgents(ix.tasks.find((t: any) => t.id === levelId)?.runs ?? [])).catch(() => {})
  }, [levelId])

  // keyboard: R rotates, 1-6 tools
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return
      if (e.key === 'r' || e.key === 'R') setDir((d) => DIRN[(DIRN.indexOf(d) + 1) % 4])
      const t = ['belt', 'scale', 'pusher', 'bin', 'erase', 'select'][Number(e.key) - 1]
      if (t) setTool(t as any)
    }
    window.addEventListener('keydown', on)
    return () => window.removeEventListener('keydown', on)
  }, [])

  const at = (x: number, y: number) => layout?.parts.find((p) => p.x === x && p.y === y)
  const isWall = (x: number, y: number) => !!layout?.walls.some(([a, b]) => a === x && b === y)
  const canPlace = (x: number, y: number) => !!layout && !isWall(x, y) && !at(x, y) && (tool in PRICES ? money >= PRICES[tool] : false)

  const onCell = (x: number, y: number) => {
    if (!layout || busy || final) return
    const p = at(x, y)
    if (tool === 'select' || (p && tool !== 'erase')) { setSel(p ? [x, y] : null); return }
    if (tool === 'erase') {
      if (!p || p.fixed) return
      setLayout({ ...layout, parts: layout.parts.filter((q) => q !== p) })
      setMoney((m) => m + PRICES[p.kind] / 2)
      if (sel && sel[0] === x && sel[1] === y) setSel(null)
      return
    }
    if (!canPlace(x, y)) return
    const part: Part = { kind: tool, x, y, dir: tool === 'bin' ? 'E' : dir, params: { ...defaults[tool] }, fixed: false, name: tool === 'bin' ? 'reject' : '' }
    setLayout({ ...layout, parts: [...layout.parts, part] })
    setMoney((m) => m - PRICES[tool])
  }

  const selPart = sel ? at(sel[0], sel[1]) : undefined
  const setParam = (k: string, v: number) => {
    if (!layout || !selPart || selPart.fixed) return
    setLayout({ ...layout, parts: layout.parts.map((p) => (p === selPart ? { ...p, params: { ...p.params, [k]: v } } : p)) })
  }
  const rotateSel = () => {
    if (!layout || !selPart || selPart.fixed || selPart.kind === 'bin') return
    setLayout({ ...layout, parts: layout.parts.map((p) => (p === selPart ? { ...p, dir: DIRN[(DIRN.indexOf(p.dir) + 1) % 4] } : p)) })
  }

  const go = async (kind: 'test' | 'production') => {
    if (!lvl || !layout || busy) return
    const secs = kind === 'test' ? Math.min(seconds, simLeft) : lvl.production_s
    if (kind === 'test' && secs < 10) return
    setBusy(kind === 'test' ? `Running ${secs} s test…` : 'Production run (scored)…'); setProgress(0)
    const seed = kind === 'test' ? 1000 + runs.length : lvl.production_seed
    const report = await simulate(layout, lvl.hidden, secs, seed, setProgress)
    frameRef.current = 0
    setReplay({ layout, frames: report.frames, report, truth: truth(report) })
    if (kind === 'test') { setRuns((r) => [...r, { seconds: secs, report }]); setSimLeft((s) => s - secs) }
    else {
      const t = truth(report)
      const checks = { throughput: t.good_per_min >= lvl.target, quality: t.bad_shipped <= lvl.max_bad, no_spills: t.spilled === 0, no_jams: t.jammed === 0 }
      const score = t.good_shipped - 5 * t.bad_shipped - 2 * t.spilled - 3 * t.jammed
      setFinal({ t, score, checks })
      try {
        const k = `yantra.best.${lvl.id}`
        if (score > Number(localStorage.getItem(k) ?? -1e9)) localStorage.setItem(k, String(score))
      } catch { /* storage unavailable */ }
    }
    setBusy(null)
  }

  // test hook for automated checks (tools/factory_play_test.mjs)
  useEffect(() => {
    (window as any).__play = {
      place: (kind: string, x: number, y: number, d: string, params: Record<string, number> = {}) => setLayout((l) => l && ({
        ...l, parts: [...l.parts, { kind, x, y, dir: d, params: { ...DEFAULTS[kind], ...params }, fixed: false, name: kind === 'bin' ? 'reject' : '' }],
      })),
      test: () => go('test'), handover: () => go('production'),
    }
  })
  const fresh = useMemo(() => new Set<string>(), [])
  if (err) return <div className="grid h-full place-items-center bg-[#23242f] text-white/70">{err}</div>
  if (!lvl || !layout) return <div className="grid h-full place-items-center bg-[#23242f] text-white/60">Loading map…</div>
  const last = runs[runs.length - 1]?.report
  const btn = (on: boolean) => `rounded-lg px-2.5 py-1.5 text-xs font-semibold ring-1 ${on ? 'bg-white text-black ring-white' : 'bg-white/5 ring-white/15 hover:bg-white/10'}`
  return (
    <div className="relative h-full w-full bg-[#23242f] text-white">
      <Scene layout={replay && busy === null && final ? replay.layout : layout} replay={replay} frameRef={frameRef} playing speed={2} xray={!!final}
        fresh={fresh} onEnd={() => {}} onCell={onCell} hover={hover} onHover={setHover} item={lvl.hidden.item}
        hoverOk={hover ? (tool === 'erase' ? !!at(...hover) && !at(...hover)!.fixed : tool === 'select' ? true : canPlace(...hover)) : true} />

      <div className="absolute bottom-4 left-4 top-[72px] flex w-[360px] max-w-[calc(100vw-32px)] flex-col gap-3 overflow-y-auto rounded-2xl bg-[#15161d]/92 p-4 shadow-2xl ring-1 ring-white/10 backdrop-blur">
        <div>
          <button onClick={() => navigate('/play')} className="text-xs text-white/50 hover:text-white">← all maps</button>
          <div className="mt-1 text-lg font-bold">{lvl.title} <span className="text-sm font-normal text-white/50">· you play</span></div>
          <p className="mt-1 text-[12.5px] leading-snug text-white/65">{lvl.goal}</p>
        </div>
        <div className="grid grid-cols-2 gap-2 font-mono text-sm">
          <div className="rounded-lg bg-white/5 p-2"><div className="text-[10px] text-white/50">money</div>${money}</div>
          <div className="rounded-lg bg-white/5 p-2"><div className="text-[10px] text-white/50">test line time left</div>{simLeft} s</div>
        </div>
        <div>
          <div className="mb-1.5 text-[11px] uppercase tracking-wider text-white/45">Parts · keys 1–6, R rotates</div>
          <div className="flex flex-wrap gap-1.5">
            {(['belt', 'scale', 'pusher', 'bin', 'erase', 'select'] as const).map((t, k) => (
              <button key={t} className={btn(tool === t)} onClick={() => setTool(t)}>
                {k + 1} {t}{t in PRICES ? ` $${PRICES[t]}` : ''}
              </button>
            ))}
          </div>
          {tool !== 'bin' && tool !== 'erase' && tool !== 'select' && (
            <div className="mt-2 flex items-center gap-2 text-xs">
              <span className="text-white/50">direction</span>
              {DIRN.map((d) => <button key={d} className={btn(dir === d)} onClick={() => setDir(d)}>{ARROW[d]} {d}</button>)}
            </div>
          )}
          {tool in DEFAULTS && Object.keys(defaults[tool]).length > 0 && (
            <div className="mt-2 space-y-1.5 rounded-lg bg-white/[0.03] p-2">
              <div className="text-[11px] text-white/45">new {tool} settings</div>
              {Object.entries(defaults[tool]).map(([k, v]) => (
                <label key={k} className="flex items-center justify-between gap-2 text-xs">
                  <span className="text-white/60">{LABEL[k]}</span>
                  <input type="number" className="w-24 rounded bg-black/40 px-2 py-1 text-right font-mono" value={v}
                    min={LIMITS[k][0]} max={LIMITS[k][1]} step={LIMITS[k][2]}
                    onChange={(e) => setDefaults({ ...defaults, [tool]: { ...defaults[tool], [k]: Number(e.target.value) } })} />
                </label>
              ))}
            </div>
          )}
        </div>
        {selPart && (
          <div className="rounded-lg bg-white/[0.05] p-2 ring-1 ring-[#7ee0b0]/40">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold">{selPart.kind}{selPart.name ? ` (${selPart.name})` : ''} at {selPart.x},{selPart.y} {selPart.kind !== 'bin' ? ARROW[selPart.dir] : ''}</span>
              {!selPart.fixed && selPart.kind !== 'bin' && <button className={btn(false)} onClick={rotateSel}>rotate</button>}
            </div>
            {selPart.fixed ? <div className="mt-1 text-[11px] text-white/50">Part of the plant: cannot be changed.</div> : Object.entries(selPart.params).map(([k, v]) => (
              <label key={k} className="mt-1.5 flex items-center justify-between gap-2 text-xs">
                <span className="text-white/60">{LABEL[k] ?? k}</span>
                <input type="number" className="w-24 rounded bg-black/40 px-2 py-1 text-right font-mono" value={v}
                  min={LIMITS[k]?.[0]} max={LIMITS[k]?.[1]} step={LIMITS[k]?.[2]} onChange={(e) => setParam(k, Number(e.target.value))} />
              </label>
            ))}
          </div>
        )}
        <div className="space-y-2 border-t border-white/10 pt-3">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-white/60">test for</span>
            <input type="range" min={10} max={90} step={5} value={seconds} onChange={(e) => setSeconds(Number(e.target.value))} className="flex-1" />
            <span className="w-10 font-mono">{seconds}s</span>
          </div>
          <div className="flex gap-2">
            <button disabled={!!busy || !!final || simLeft < 10} onClick={() => go('test')}
              className="flex-1 rounded-full bg-white/10 px-3 py-2 text-sm font-semibold ring-1 ring-white/20 disabled:opacity-40">▶ Test run</button>
            <button disabled={!!busy || !!final} onClick={() => go('production')}
              className="flex-1 rounded-full bg-[#3fbf8a] px-3 py-2 text-sm font-semibold text-black disabled:opacity-40">Hand over</button>
          </div>
          {busy && <div className="text-xs text-white/70">{busy}<div className="mt-1 h-1 rounded bg-white/10"><div className="h-1 rounded bg-[#7ee0b0]" style={{ width: `${progress * 100}%` }} /></div></div>}
        </div>
        {last && !final && (
          <div className="rounded-lg bg-white/[0.04] p-2 font-mono text-[11.5px] text-white/75">
            <div className="mb-1 font-sans text-xs font-semibold text-white">Test {runs.length} · {runs[runs.length - 1].seconds} s</div>
            {Object.entries(last.bins).map(([k, b]) => <div key={k}>{k} bin: {b.count} · {b.weight_kg.toFixed(2)} kg</div>)}
            <div>spilled {sumv(last.spills)} · jams {sumv(last.jams)} · pusher fires {sumv(last.pusher_fires)}</div>
            {Object.entries(last.scales).map(([k, v]) => <div key={k}>scale {k}: {v.length} readings {v.length ? `${Math.round(Math.min(...v))}…${Math.round(Math.max(...v))} g` : ''}</div>)}
            {Object.entries(last.scales).map(([k, v]) => <div key={k + 'r'} className="mt-1 break-words text-[10.5px] text-white/50">{v.map((r) => Math.round(r)).join(' ')}</div>)}
          </div>
        )}
        {final && (
          <div className="rounded-xl bg-white/[0.06] p-3 ring-1 ring-white/15">
            <div className="flex items-center justify-between"><span className="font-semibold">Production run</span>
              <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${Object.values(final.checks).every(Boolean) ? 'bg-[#3fbf8a]/25 text-[#7ee0b0]' : 'bg-[#e07a5a]/25 text-[#ffb27a]'}`}>
                {Object.values(final.checks).every(Boolean) ? 'PASSED' : 'FAILED'} · score {final.score}</span></div>
            <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[12px]">
              {Object.entries(final.checks).map(([k, v]) => <div key={k} className={v ? 'text-[#7ee0b0]' : 'text-[#ffb27a]'}>{v ? '✓' : '✗'} {k.replace('_', ' ')}</div>)}
            </div>
            <div className="mt-1.5 font-mono text-[12px] text-white/70">{final.t.good_per_min} good/min (target {lvl.target}) · {final.t.bad_shipped} bad shipped · {final.t.spilled} spilled · {final.t.jammed} jammed</div>
            <div className="mt-1 text-[11px] text-white/50">X-ray on: red boxes were the bad ones.</div>
            {agents.length > 0 && (
              <div className="mt-3 space-y-1 border-t border-white/10 pt-2 text-xs">
                <div className="text-white/50">You vs the agents on this map</div>
                {[...agents, { run: 'you', label: 'You', success: Object.values(final.checks).every(Boolean), score: final.score }]
                  .sort((a, b) => b.score - a.score).map((a) => (
                    <div key={a.run} className={`flex justify-between rounded px-2 py-1 ${a.run === 'you' ? 'bg-[#3fbf8a]/15' : ''}`}>
                      <span>{a.label} {a.success ? '✓' : '✗'}</span><span className="font-mono">{a.score}</span>
                    </div>
                  ))}
              </div>
            )}
            <button className="mt-3 w-full rounded-full bg-white/10 py-1.5 text-sm" onClick={() => { setFinal(null); setReplay(null); setLayout(structuredClone(lvl.start)); setMoney(lvl.money); setRuns([]); setSimLeft(lvl.budgets.sim_seconds) }}>Start over</button>
          </div>
        )}
      </div>
    </div>
  )
}
