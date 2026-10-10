// Watch one GPT agent build a factory, turn by turn. Left: every turn (the build commands it wrote,
// its reasoning, what it predicted vs what the physics did). Centre: the factory as it stood after
// that turn, replaying that turn's MuJoCo run. 1 grid cell = 0.5 m = 2 Kenney kit units.
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Clone, ContactShadows, Environment, Lightformer, OrbitControls, useGLTF } from '@react-three/drei'
import { Bloom, EffectComposer, N8AO, SMAA, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import { navigate } from '../router'

const BASE = `${import.meta.env.BASE_URL}`
const S = 0.25 // metres per kit unit
const CELL = 2 // kit units per cell
const BELT_TOP = 0.4
const kit = (name: string) => `${BASE}assets/kenney/${name}.glb`
const ROT: Record<string, number> = { E: 0, N: Math.PI / 2, W: Math.PI, S: -Math.PI / 2 }
const VEC: Record<string, [number, number]> = { E: [1, 0], N: [0, 1], W: [-1, 0], S: [0, -1] }

interface Part { kind: string; x: number; y: number; dir: string; params: Record<string, number>; fixed: boolean; name: string }
interface Layout { w: number; h: number; walls: [number, number][]; parts: Part[] }
interface Frame { t: number; items: [number, boolean, number, number, number, number, number, number, number][]; push: number[] }
interface Replay { layout: Layout; frames: Frame[]; report: any; truth: any }
interface Turn {
  i: number; tool: string; args: any; reasoning: string; prediction: Record<string, number> | null
  obs: any; layout: Layout; replay: string | null
}
interface EpisodeData {
  id: string; run: string; label: string; model: string; title: string; goal: string; params: any
  budgets: Record<string, number>; turns: Turn[]; think_seconds: number
  result: { success: boolean; score: number; checks: Record<string, boolean>; production: any; end: string; money_left: number }
  final_layout: Layout; final_replay: string | null
}

const cellXZ = (x: number, y: number): [number, number] => [(x + 0.5) * CELL, -(y + 0.5) * CELL]

function Model({ name, position = [0, 0, 0], rotation = [0, 0, 0], scale = 1 }: {
  name: string; position?: [number, number, number]; rotation?: [number, number, number]; scale?: number | [number, number, number]
}) {
  const { scene } = useGLTF(kit(name))
  useMemo(() => scene.traverse((o) => { if ((o as THREE.Mesh).isMesh) { o.castShadow = true; o.receiveShadow = true } }), [scene])
  return <Clone object={scene} position={position} rotation={rotation} scale={scale} />
}

// sim z (m) -> kit y. Boxes resting on a belt sit on BELT_TOP; boxes in bins or on the floor sit on the floor.
function kitY(z: number) {
  const onBelt = 0.47, onFloor = 0.17, half = 0.28
  if (z >= onBelt) return BELT_TOP + half + (z - onBelt) / S
  return half + Math.max(0, (z - onFloor) / (onBelt - onFloor)) * BELT_TOP
}

function Arrow({ dir }: { dir: string }) {
  const shape = useMemo(() => {
    const s = new THREE.Shape()
    s.moveTo(0.35, 0); s.lineTo(-0.2, 0.3); s.lineTo(-0.05, 0); s.lineTo(-0.2, -0.3); s.lineTo(0.35, 0)
    return s
  }, [])
  return (
    <mesh rotation={[-Math.PI / 2, 0, ROT[dir]]} position={[0, BELT_TOP + 0.02, 0]}>
      <shapeGeometry args={[shape]} />
      <meshBasicMaterial color="#ffd166" transparent opacity={0.55} />
    </mesh>
  )
}

function Bin({ ship }: { ship: boolean }) {
  const c = ship ? '#3fbf8a' : '#e07a5a'
  const wall = (p: [number, number, number], s: [number, number, number]) => (
    <mesh castShadow receiveShadow position={p}><boxGeometry args={s} /><meshStandardMaterial color={c} roughness={0.6} /></mesh>
  )
  return (
    <group>
      <mesh receiveShadow position={[0, 0.02, 0]}><boxGeometry args={[1.9, 0.04, 1.9]} /><meshStandardMaterial color="#3a3d48" /></mesh>
      {wall([0.93, 0.35, 0], [0.08, 0.7, 1.9])}{wall([-0.93, 0.35, 0], [0.08, 0.7, 1.9])}
      {wall([0, 0.35, 0.93], [1.9, 0.7, 0.08])}{wall([0, 0.35, -0.93], [1.9, 0.7, 0.08])}
      {ship ? <Model name="box-wide" position={[0.55, 0, 0.55]} scale={0.6} /> : <Model name="warning-orange" position={[0, 0.05, 0]} scale={0.8} />}
    </group>
  )
}

function Parts({ layout, fresh }: { layout: Layout; fresh: Set<string> }) {
  return (
    <group>
      {layout.parts.map((p) => {
        const [x, z] = cellXZ(p.x, p.y)
        const key = `${p.kind}-${p.x}-${p.y}`
        const glow = fresh.has(`${p.x},${p.y}`)
        let body = null
        if (p.kind === 'belt' || p.kind === 'inlet' || p.kind === 'scale') {
          body = (
            <group rotation={[0, ROT[p.dir], 0]}>
              <Model name="conveyor-long-sides" scale={[1, 1, 2]} />
              <Arrow dir="E" />
              {p.kind === 'scale' && <Model name="scanner-low" rotation={[0, Math.PI / 2, 0]} scale={1.2} />}
              {p.kind === 'inlet' && <Model name="hopper-high-round" position={[-0.6, 0, 0]} scale={1.0} />}
            </group>
          )
        } else if (p.kind === 'pusher') {
          body = <group rotation={[0, ROT[p.dir], 0]}><Model name="piston-square" position={[-0.3, 0, 0]} rotation={[0, Math.PI / 2, 0]} scale={0.9} /></group>
        } else if (p.kind === 'bin') {
          body = <Bin ship={p.name === 'ship'} />
        }
        return (
          <group key={key} position={[x, 0, z]}>
            {body}
            {glow && <mesh position={[0, 0.03, 0]} rotation={[-Math.PI / 2, 0, 0]}><planeGeometry args={[1.95, 1.95]} /><meshBasicMaterial color="#7ee0b0" transparent opacity={0.35} /></mesh>}
          </group>
        )
      })}
      {layout.walls.map(([wx, wy]) => {
        const [x, z] = cellXZ(wx, wy)
        return (
          <group key={`w${wx},${wy}`} position={[x, 0, z]}>
            <mesh castShadow receiveShadow position={[0, 0.8, 0]}><boxGeometry args={[1.9, 1.6, 1.9]} /><meshStandardMaterial color="#8a8f9c" roughness={0.9} /></mesh>
            <mesh position={[0, 0.15, 0]}><boxGeometry args={[1.96, 0.3, 1.96]} /><meshStandardMaterial color="#f2b134" /></mesh>
          </group>
        )
      })}
    </group>
  )
}

function Floor({ layout }: { layout: Layout }) {
  return (
    <group>
      <mesh receiveShadow rotation={[-Math.PI / 2, 0, 0]} position={[layout.w, 0, -layout.h]}>
        <planeGeometry args={[layout.w * CELL + 60, layout.h * CELL + 60]} />
        <meshStandardMaterial color="#3b3e4a" roughness={0.95} />
      </mesh>
      {/* the buildable area */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[layout.w, 0.012, -layout.h]}>
        <planeGeometry args={[layout.w * CELL, layout.h * CELL]} />
        <meshStandardMaterial color="#555a6b" roughness={0.9} />
      </mesh>
      <gridHelper args={[Math.max(layout.w, layout.h) * CELL, Math.max(layout.w, layout.h), '#6b7280', '#4b5060']}
        position={[Math.max(layout.w, layout.h), 0.015, -Math.max(layout.w, layout.h)]} />
    </group>
  )
}

function Items({ replay, frameRef, xray }: { replay: Replay | null; frameRef: React.RefObject<number>; xray: boolean }) {
  const N = 48
  const refs = useRef<(THREE.Mesh | null)[]>([])
  const q = useMemo(() => new THREE.Quaternion(), [])
  const c = useMemo(() => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2), [])
  const cInv = useMemo(() => c.clone().invert(), [c])
  const good = useMemo(() => new THREE.MeshStandardMaterial({ color: '#c8955a', roughness: 0.85 }), [])
  const bad = useMemo(() => new THREE.MeshStandardMaterial({ color: '#e0523a', roughness: 0.7, emissive: '#5a1408' }), [])
  useFrame(() => {
    for (const m of refs.current) if (m) m.visible = false
    if (!replay || !replay.frames.length) return
    const f = replay.frames[Math.min(replay.frames.length - 1, Math.floor(frameRef.current))]
    for (const it of f.items) {
      const m = refs.current[it[0]]
      if (!m) continue
      m.visible = true
      m.position.set(it[2] / S, kitY(it[4]), -it[3] / S)
      q.set(it[6], it[7], it[8], it[5])
      m.quaternion.copy(c).multiply(q).multiply(cInv)
      m.material = xray && it[1] ? bad : good
    }
  })
  return (
    <group>
      {Array.from({ length: N }, (_, i) => (
        <mesh key={i} ref={(el) => { refs.current[i] = el }} castShadow receiveShadow material={good} visible={false}>
          <boxGeometry args={[0.56, 0.56, 0.56]} />
        </mesh>
      ))}
    </group>
  )
}

function Paddles({ layout, replay, frameRef }: { layout: Layout; replay: Replay | null; frameRef: React.RefObject<number> }) {
  const pushers = layout.parts.filter((p) => p.kind === 'pusher')
  const refs = useRef<(THREE.Group | null)[]>([])
  useFrame(() => {
    const f = replay?.frames[Math.min(replay.frames.length - 1, Math.floor(frameRef.current))]
    pushers.forEach((p, j) => {
      const g = refs.current[j]
      if (!g) return
      const ext = (f?.push[j] ?? 0) / S
      const [x, z] = cellXZ(p.x, p.y)
      const [dx, dy] = VEC[p.dir]
      const rest = CELL / 2 - 0.24
      g.position.set(x + dx * (rest + ext), BELT_TOP + 0.28, z - dy * (rest + ext))
    })
  })
  return (
    <group>
      {pushers.map((p, j) => (
        <group key={`${p.x},${p.y}`} ref={(el) => { refs.current[j] = el }} rotation={[0, ROT[p.dir], 0]}>
          <mesh castShadow><boxGeometry args={[0.24, 0.4, 0.96]} /><meshStandardMaterial color="#e58a2b" roughness={0.45} metalness={0.2} /></mesh>
          <mesh castShadow position={[-0.7, 0, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.06, 0.06, 1.2, 12]} /><meshStandardMaterial color="#c9ccd3" metalness={1} roughness={0.2} /></mesh>
        </group>
      ))}
    </group>
  )
}

function Clock({ replay, frameRef, playing, speed, onEnd }: {
  replay: Replay | null; frameRef: React.MutableRefObject<number>; playing: boolean; speed: number; onEnd: () => void
}) {
  const ended = useRef(false)
  useFrame((_, dt) => {
    if (!playing || !replay) return
    const n = replay.frames.length
    frameRef.current = Math.min(n - 1, frameRef.current + dt * 12 * speed)
    if (frameRef.current >= n - 1 && !ended.current) { ended.current = true; onEnd() }
    if (frameRef.current < n - 1) ended.current = false
  })
  return null
}

function Scene({ layout, replay, frameRef, playing, speed, xray, fresh, onEnd }: {
  layout: Layout; replay: Replay | null; frameRef: React.MutableRefObject<number>; playing: boolean; speed: number
  xray: boolean; fresh: Set<string>; onEnd: () => void
}) {
  const cx = layout.w, cz = -layout.h
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: [cx - 5, 19, cz + 18], fov: 42 }}
      gl={{ antialias: false, toneMapping: THREE.ACESFilmicToneMapping }}>
      <color attach="background" args={['#23242f']} />
      <fog attach="fog" args={['#23242f', 40, 80]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[cx + 10, 22, cz + 14]} intensity={2.3} castShadow shadow-mapSize={[4096, 4096]}
        shadow-camera-left={-30} shadow-camera-right={30} shadow-camera-top={30} shadow-camera-bottom={-30} shadow-bias={-0.0003} />
      <Environment resolution={256}>
        <Lightformer form="rect" intensity={2} position={[0, 10, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[40, 40, 1]} />
        <Lightformer form="rect" intensity={1.1} color="#ffd9b0" position={[-12, 4, 6]} rotation={[0, Math.PI / 2, 0]} scale={[12, 4, 1]} />
      </Environment>
      <Suspense fallback={null}>
        <Floor layout={layout} />
        <Parts layout={layout} fresh={fresh} />
        <Paddles layout={layout} replay={replay} frameRef={frameRef} />
        <Items replay={replay} frameRef={frameRef} xray={xray} />
        <ContactShadows position={[cx, 0.01, cz]} scale={[layout.w * 2 + 8, layout.h * 2 + 8]} opacity={0.3} blur={2.5} far={4} />
      </Suspense>
      <Clock replay={replay} frameRef={frameRef} playing={playing} speed={speed} onEnd={onEnd} />
      <OrbitControls makeDefault target={[cx - 4.5, 0, cz + 0.5]} maxPolarAngle={Math.PI / 2.15} minDistance={6} maxDistance={70} />
      <EffectComposer multisampling={0}>
        <N8AO aoRadius={0.6} intensity={2} distanceFalloff={1} halfRes />
        <Bloom intensity={0.25} luminanceThreshold={0.9} mipmapBlur />
        <SMAA />
        <Vignette offset={0.3} darkness={0.45} />
      </EffectComposer>
    </Canvas>
  )
}

// ---------------------------------------------------------------- data
const cache = new Map<string, Promise<any>>()
const getJSON = (p: string) => {
  if (!cache.has(p)) cache.set(p, fetch(`${BASE}data/factory/${p}`).then((r) => { if (!r.ok) throw new Error(p); return r.json() }))
  return cache.get(p)!
}

// ---------------------------------------------------------------- turn cards
const fmt = (v: any) => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(1)) : String(v))

function PredVsActual({ pred, actual, label }: { pred?: number; actual?: number; label: string }) {
  if (pred === undefined || actual === undefined) return null
  const err = Math.abs(pred - actual)
  const ok = err <= Math.max(1, 0.15 * Math.abs(actual))
  return (
    <div className="flex items-center justify-between rounded-lg bg-white/5 px-2 py-1">
      <span className="text-white/60">{label}</span>
      <span className="font-mono">predicted <b>{fmt(pred)}</b> · got <b className={ok ? 'text-[#7ee0b0]' : 'text-[#ffb27a]'}>{fmt(actual)}</b></span>
    </div>
  )
}

function TurnCard({ t, active, onClick }: { t: Turn; active: boolean; onClick: () => void }) {
  const icon = t.tool === 'build' ? '🔧' : t.tool === 'run' ? '▶' : t.tool === 'finish' ? '🏁' : '⚠'
  const title = t.tool === 'build' ? 'build' : t.tool === 'run' ? `run ${fmt(t.args.seconds)} s` : t.tool === 'finish' ? 'hand over' : t.tool
  const failed = t.tool === 'build' ? (t.obs?.results ?? []).filter((r: string) => r.startsWith('FAILED')) : []
  const o = t.obs ?? {}
  return (
    <button onClick={onClick} className={`w-full rounded-xl p-3 text-left text-[13px] ring-1 transition ${active ? 'bg-white/10 ring-[#7ee0b0]/60' : 'bg-white/[0.03] ring-white/10 hover:bg-white/[0.06]'}`}>
      <div className="flex items-center justify-between">
        <span className="font-semibold">{icon} Turn {t.i} · {title}</span>
        {t.tool === 'build' && <span className="font-mono text-xs text-white/50">${o.money_left} left</span>}
        {t.tool === 'run' && <span className="font-mono text-xs text-white/50">{o.shipped} shipped</span>}
      </div>
      {t.reasoning && <p className="mt-1.5 text-white/75 italic leading-snug">“{t.reasoning}”</p>}
      {active && t.tool === 'build' && (
        <pre className="mt-2 max-h-56 overflow-auto rounded-lg bg-black/40 p-2 font-mono text-[11.5px] leading-[1.45] text-[#cfe3ff]">{t.args.ops}</pre>
      )}
      {failed.length > 0 && <div className="mt-1.5 space-y-0.5 font-mono text-[11px] text-[#ffb27a]">{failed.map((f: string, k: number) => <div key={k}>{f}</div>)}</div>}
      {t.tool === 'run' && (
        <div className="mt-2 space-y-1 text-[12px]">
          <PredVsActual label="shipped" pred={t.prediction?.shipped} actual={o.shipped} />
          <PredVsActual label="spilled" pred={t.prediction?.spilled} actual={o.spilled} />
          {active && (
            <div className="grid grid-cols-2 gap-1 pt-1 font-mono text-[11.5px] text-white/70">
              {Object.entries(o.bins ?? {}).map(([k, b]: [string, any]) => <div key={k}>{k} bin: {b.count} · {b.weight_kg.toFixed(2)} kg</div>)}
              <div>jams: {Object.values(o.jams_by_cell ?? {}).reduce((a: number, b: any) => a + b, 0)}</div>
              <div>pusher fires: {Object.values(o.pusher_fires ?? {}).reduce((a: number, b: any) => a + b, 0)}</div>
              {Object.entries(o.scales ?? {}).map(([k, s]: [string, any]) => <div key={k} className="col-span-2">scale {k}: {s.n} readings, {s.min ?? '–'}…{s.max ?? '–'} g</div>)}
              {Object.keys(o.spills_by_cell ?? {}).length > 0 && <div className="col-span-2 text-[#ffb27a]">spills at {Object.entries(o.spills_by_cell).map(([c, n]) => `${c}×${n}`).join(', ')}</div>}
              {Object.keys(o.jams_by_cell ?? {}).length > 0 && <div className="col-span-2 text-[#ffb27a]">jams at {Object.entries(o.jams_by_cell).map(([c, n]) => `${c}×${n}`).join(', ')}</div>}
            </div>
          )}
        </div>
      )}
    </button>
  )
}

// ---------------------------------------------------------------- page
export function FactoryWatch({ run, taskId }: { run: string; taskId: string }) {
  const [ep, setEp] = useState<EpisodeData | null>(null)
  const [err, setErr] = useState('')
  const [others, setOthers] = useState<{ run: string; label: string; success: boolean }[]>([])
  const [sel, setSel] = useState(0) // 0..turns-1, turns.length = production
  const [replay, setReplay] = useState<Replay | null>(null)
  const [playing, setPlaying] = useState(true)
  const [auto, setAuto] = useState(true)
  const [speed, setSpeed] = useState(2)
  const [xray, setXray] = useState(false)
  const frameRef = useRef(0)
  const [clock, setClock] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getJSON(`${run}/${taskId}.json`).then((d) => { setEp(d); setSel(0) }).catch(() => setErr('No agent run exported for this task yet.'))
    getJSON('index.json').then((ix) => setOthers(ix.tasks.find((t: any) => t.id === taskId)?.runs ?? []))
  }, [run, taskId])

  const isFinal = ep ? sel >= ep.turns.length : false
  const turn = ep && !isFinal ? ep.turns[sel] : null
  const layout = ep ? (isFinal ? ep.final_layout ?? ep.turns[ep.turns.length - 1]?.layout : turn!.layout) : null
  const replayPath = ep ? (isFinal ? ep.final_replay : turn!.replay) : null
  const prevLayout = ep && sel > 0 ? ep.turns[Math.min(sel, ep.turns.length) - 1].layout : null
  const fresh = useMemo(() => {
    if (!layout || !turn || turn.tool !== 'build') return new Set<string>()
    const before = new Set((prevLayout?.parts ?? []).map((p) => `${p.kind}${p.x},${p.y}`))
    return new Set(layout.parts.filter((p) => !before.has(`${p.kind}${p.x},${p.y}`)).map((p) => `${p.x},${p.y}`))
  }, [layout, prevLayout, turn])

  useEffect(() => {
    frameRef.current = 0
    setReplay(null)
    if (replayPath) getJSON(replayPath).then(setReplay)
  }, [replayPath])

  // auto-advance: build/finish turns hold 3.5 s, run turns advance when their replay ends
  useEffect(() => {
    if (!ep || !auto || !playing || replayPath) return
    if (sel >= ep.turns.length) return
    const id = setTimeout(() => setSel((s) => Math.min(ep.turns.length, s + 1)), 3500 / Math.max(1, speed / 2))
    return () => clearTimeout(id)
  }, [ep, sel, auto, playing, replayPath, speed])
  const onEnd = () => { if (ep && auto && sel < ep.turns.length) setTimeout(() => setSel((s) => s + 1), 800) }

  useEffect(() => {
    const id = setInterval(() => {
      const f = replay?.frames[Math.floor(frameRef.current)]
      setClock(f ? f.t : 0)
    }, 120)
    return () => clearInterval(id)
  }, [replay])
  useEffect(() => { listRef.current?.querySelector(`[data-turn="${sel}"]`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) }, [sel])

  if (err) return <div className="grid h-full place-items-center bg-[#23242f] text-white/70">{err}</div>
  if (!ep || !layout) return <div className="grid h-full place-items-center bg-[#23242f] text-white/60">Loading episode…</div>
  const r = ep.result
  const dur = replay?.frames.length ? replay.frames[replay.frames.length - 1].t : 0
  return (
    <div className="relative h-full w-full bg-[#23242f] text-white">
      <Scene layout={layout} replay={replay} frameRef={frameRef} playing={playing} speed={speed} xray={xray} fresh={fresh} onEnd={onEnd} />

      {/* left: the agent's turns */}
      <div className="absolute bottom-4 left-4 top-[72px] flex w-[390px] max-w-[calc(100vw-32px)] flex-col rounded-2xl bg-[#15161d]/92 shadow-2xl ring-1 ring-white/10 backdrop-blur">
        <div className="border-b border-white/10 p-4">
          <button onClick={() => navigate('/watch')} className="text-xs text-white/50 hover:text-white">← all tasks</button>
          <div className="mt-1 flex items-center justify-between">
            <div className="text-lg font-bold">{ep.title}</div>
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${r.success ? 'bg-[#3fbf8a]/20 text-[#7ee0b0]' : 'bg-[#e07a5a]/20 text-[#ffb27a]'}`}>
              {r.success ? 'PASSED' : 'FAILED'} · score {r.score}
            </span>
          </div>
          <p className="mt-1 text-[12.5px] leading-snug text-white/65">{ep.goal}</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {others.map((o) => (
              <button key={o.run} onClick={() => navigate(`/watch/factory/${o.run}/${taskId}`)}
                className={`rounded-full px-2.5 py-1 text-xs ring-1 ${o.run === run ? 'bg-white text-black ring-white' : 'ring-white/20 hover:bg-white/10'}`}>
                {o.label} {o.success ? '✓' : '✗'}
              </button>
            ))}
          </div>
        </div>
        <div ref={listRef} className="flex-1 space-y-2 overflow-y-auto p-3">
          {ep.turns.map((t, k) => (
            <div key={k} data-turn={k}><TurnCard t={t} active={k === sel} onClick={() => { setSel(k); setAuto(false) }} /></div>
          ))}
          <div data-turn={ep.turns.length}>
            <button onClick={() => { setSel(ep.turns.length); setAuto(false) }}
              className={`w-full rounded-xl p-3 text-left text-[13px] ring-1 ${isFinal ? 'bg-white/10 ring-[#7ee0b0]/60' : 'bg-white/[0.03] ring-white/10'}`}>
              <div className="font-semibold">🏭 Production run (scored)</div>
              <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[12px]">
                {Object.entries(r.checks ?? {}).map(([k, v]) => <div key={k} className={v ? 'text-[#7ee0b0]' : 'text-[#ffb27a]'}>{v ? '✓' : '✗'} {k.replace('_', ' ')}</div>)}
              </div>
              {r.production && <div className="mt-1.5 font-mono text-[12px] text-white/70">
                {r.production.good_per_min} good/min (target {ep.params.target_good_per_min}) · {r.production.bad_shipped} bad shipped · {r.production.spilled} spilled · {r.production.jammed} jammed
              </div>}
            </button>
          </div>
        </div>
      </div>

      {/* right: what is on screen */}
      <div className="pointer-events-none absolute right-4 top-[72px] w-[260px] rounded-2xl bg-[#15161d]/90 p-4 shadow-2xl ring-1 ring-white/10">
        <div className="text-xs uppercase tracking-wider text-white/50">{ep.label}</div>
        <div className="mt-1 text-sm font-semibold">{isFinal ? 'Production run (scored)' : turn!.tool === 'run' ? `Turn ${turn!.i}: test run` : `Turn ${turn!.i}: ${turn!.tool}`}</div>
        <div className="mt-3 grid grid-cols-2 gap-3 font-mono">
          <div><div className="text-[11px] text-white/50">line time</div><div className="text-xl">{replay ? `${clock.toFixed(1)}s` : '–'}</div></div>
          <div><div className="text-[11px] text-white/50">of</div><div className="text-xl text-white/60">{replay ? `${dur.toFixed(0)}s` : '–'}</div></div>
          <div><div className="text-[11px] text-white/50">turns used</div><div className="text-xl">{ep.turns.length}/{ep.budgets.turns}</div></div>
          <div><div className="text-[11px] text-white/50">thinking</div><div className="text-xl">{Math.round(ep.think_seconds / 60)} min</div></div>
        </div>
        {!replay && !isFinal && turn?.tool === 'build' && <div className="mt-3 text-[12px] text-[#7ee0b0]">Green cells: parts placed this turn</div>}
      </div>

      {/* bottom: playback */}
      <div className="absolute bottom-4 left-[calc(390px+32px)] right-4 flex items-center gap-2 rounded-full bg-[#15161d]/90 px-3 py-2 text-sm ring-1 ring-white/10 max-md:hidden">
        <button className="rounded-full bg-white/10 px-3 py-1" onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
        <button className={`rounded-full px-3 py-1 ${auto ? 'bg-[#3fbf8a] text-black' : 'bg-white/10'}`} onClick={() => setAuto(!auto)}>Auto-advance</button>
        <button className="rounded-full bg-white/10 px-3 py-1" onClick={() => { setSel((s) => Math.max(0, s - 1)); setAuto(false) }}>◀ Turn</button>
        <button className="rounded-full bg-white/10 px-3 py-1" onClick={() => { setSel((s) => Math.min(ep.turns.length, s + 1)); setAuto(false) }}>Turn ▶</button>
        {[1, 2, 4].map((v) => (
          <button key={v} className={`rounded-full px-2.5 py-1 ${speed === v ? 'bg-white text-black' : 'bg-white/10'}`} onClick={() => setSpeed(v)}>{v}×</button>
        ))}
        <button className={`rounded-full px-3 py-1 ${xray ? 'bg-[#e07a5a] text-black' : 'bg-white/10'}`} onClick={() => setXray(!xray)} title="Show which boxes are actually bad (the agent cannot see this)">X-ray bad boxes</button>
        <input type="range" className="ml-2 flex-1" min={0} max={Math.max(0, (replay?.frames.length ?? 1) - 1)} step={1}
          value={Math.floor(frameRef.current)} onChange={(e) => { frameRef.current = Number(e.target.value) }} disabled={!replay} />
      </div>
    </div>
  )
}
