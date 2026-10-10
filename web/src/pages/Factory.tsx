// Bottling line demo: the simulated MuJoCo run played back on a factory floor built from the
// CC0 Kenney Factory Kit. 1 kit unit = 0.25 m; sim (x along line, y lateral, z up, metres) maps to
// kit space (x, y up, z) via K.
import { Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { Clone, ContactShadows, Environment, Lightformer, OrbitControls, useGLTF } from '@react-three/drei'
import { Bloom, EffectComposer, N8AO, SMAA, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'

const BASE = `${import.meta.env.BASE_URL}`
const S = 0.25 // metres per kit unit
const BELT_TOP = 0.4 // kit units: top of conveyor models
const kit = (name: string) => `${BASE}assets/kenney/${name}.glb`

interface Demo {
  line: { length: number; belt_z: number; weigh: [number, number]; push_x: number; bottle_r: number; bottle_h: number }
  frames: { t: number; b: (number[] | null)[]; pusher: number; checked: number; rejected: number }[]
  readings: Record<string, number>
  fate: Record<string, string>
  settings: Record<string, number>
}

function Model({ name, position = [0, 0, 0], rotation = [0, 0, 0], scale = 1 }: {
  name: string; position?: [number, number, number]; rotation?: [number, number, number]; scale?: number | [number, number, number]
}) {
  const { scene } = useGLTF(kit(name))
  useMemo(() => scene.traverse((o) => { if ((o as THREE.Mesh).isMesh) { o.castShadow = true; o.receiveShadow = true } }), [scene])
  return <Clone object={scene} position={position} rotation={rotation} scale={scale} />
}

// sim metres -> kit units (belt surface at BELT_TOP)
const toKit = (x: number, y: number, z: number, beltZ: number): [number, number, number] =>
  [x / S, BELT_TOP + (z - beltZ) / S, -y / S]

function bottleGeometry(r: number, h: number) {
  // amber glass bottle profile (sim radius r, height h), lathed, in kit units
  const R = r / S, H = h / S
  const pts = [
    [0, 0], [R * 0.92, 0], [R, 0.04 * H], [R, 0.55 * H], [R * 0.95, 0.62 * H], [R * 0.45, 0.78 * H],
    [R * 0.32, 0.86 * H], [R * 0.32, 0.97 * H], [R * 0.38, 0.98 * H], [R * 0.38, H], [0, H],
  ].map(([a, b]) => new THREE.Vector2(a, b))
  const g = new THREE.LatheGeometry(pts, 40)
  g.translate(0, -H / 2, 0)
  return g
}

function Bottles({ demo, frameRef }: { demo: Demo; frameRef: React.RefObject<number> }) {
  const n = demo.frames[0].b.length
  const geo = useMemo(() => bottleGeometry(demo.line.bottle_r, demo.line.bottle_h), [demo])
  const refs = useRef<(THREE.Mesh | null)[]>([])
  const q = useMemo(() => new THREE.Quaternion(), [])
  // change of basis sim (z up) -> kit (y up): kit = (x, z, -y), a -90 deg rotation about x
  const c = useMemo(() => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2), [])
  const cInv = useMemo(() => c.clone().invert(), [c])
  useFrame(() => {
    const f = demo.frames[Math.min(demo.frames.length - 1, Math.floor(frameRef.current))]
    for (let i = 0; i < n; i++) {
      const m = refs.current[i]
      const b = f.b[i]
      if (!m) continue
      m.visible = !!b
      if (!b) continue
      const [x, y, z] = toKit(b[0], b[1], b[2], demo.line.belt_z)
      m.position.set(x, y, z)
      // conjugate the sim rotation into kit space; the lathe axis (kit y) is the sim bottle axis (z)
      q.set(b[4], b[5], b[6], b[3])
      m.quaternion.copy(c).multiply(q).multiply(cInv)
    }
  })
  return (
    <group>
      {Array.from({ length: n }, (_, i) => (
        <mesh key={i} ref={(el) => { refs.current[i] = el }} geometry={geo} castShadow receiveShadow>
          <meshPhysicalMaterial color="#6b3a12" roughness={0.08} metalness={0} transmission={0.35} thickness={0.4}
            clearcoat={1} clearcoatRoughness={0.05} ior={1.5} />
        </mesh>
      ))}
    </group>
  )
}

function Pusher({ demo, frameRef }: { demo: Demo; frameRef: React.RefObject<number> }) {
  const rod = useRef<THREE.Group>(null!)
  const px = demo.line.push_x / S
  useFrame(() => {
    const f = demo.frames[Math.min(demo.frames.length - 1, Math.floor(frameRef.current))]
    // sim pusher y: 0.12 retracted -> -0.10 extended; kit z = -y / S
    rod.current.position.z = -f.pusher / S
  })
  return (
    <group position={[px, BELT_TOP + 0.05, 0]}>
      <Model name="piston-round" position={[0, -0.05, -1.35]} rotation={[0, 0, 0]} scale={0.9} />
      <group ref={rod}>
        <mesh castShadow position={[0, 0.12, 0]}>
          <boxGeometry args={[0.5, 0.3, 0.14]} />
          <meshStandardMaterial color="#e58a2b" roughness={0.45} metalness={0.2} />
        </mesh>
        <mesh castShadow position={[0, 0.12, -0.5]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 1, 16]} />
          <meshStandardMaterial color="#c9ccd3" metalness={1} roughness={0.2} />
        </mesh>
      </group>
    </group>
  )
}

function Line({ demo }: { demo: Demo }) {
  const L = demo.line.length / S
  const tiles = []
  for (let x = 1; x < L; x += 2) tiles.push(x)
  const w0 = demo.line.weigh[0] / S, w1 = demo.line.weigh[1] / S
  const px = demo.line.push_x / S
  return (
    <group>
      {tiles.map((x) => <Model key={x} name="conveyor-long-sides" position={[x, 0, 0]} />)}
      {/* checkweigher: low scanner housing over the weigh platform, with a display */}
      <Model name="scanner-low" position={[(w0 + w1) / 2, 0, 0]} rotation={[0, Math.PI / 2, 0]} scale={1.15} />
      <Model name="screen-wide" position={[(w0 + w1) / 2, 0, -1.6]} rotation={[0, 0, 0]} scale={0.9} />
      {/* acoustic crack tester (next pack step): tall scanner arch */}
      <Model name="scanner-high" position={[w1 + 1.6, 0, 0]} rotation={[0, Math.PI / 2, 0]} scale={1.1} />
      {/* feeder hopper at the start, case packer at the end */}
      <Model name="hopper-high-round" position={[-0.9, 0, 0]} scale={1.1} />
      <Model name="robot-arm-a" position={[L + 0.7, 0, -1.3]} rotation={[0, -Math.PI / 2, 0]} scale={1.5} />
      <Model name="box-large" position={[L + 0.9, 0, 0]} rotation={[0, Math.PI / 2, 0]} />
      <Model name="box-small" position={[L + 2.3, 0, 0.9]} rotation={[0, 0.3, 0]} />
      <Model name="box-small" position={[L + 2.3, 0.55, 0.9]} rotation={[0, -0.2, 0]} />
      <Model name="box-wide" position={[L + 2.6, 0, -0.5]} />
      {/* reject chute and bin on the far side */}
      <Model name="conveyor-bars-sides" position={[px + 0.9, -0.1, 1.25]} rotation={[0, Math.PI / 2, 0.12]} scale={[1.2, 1, 1]} />
      <Model name="box-large" position={[px + 0.9, 0, 2.4]} scale={[1.2, 1, 1]} />
      <Model name="warning-orange" position={[px + 0.9, 0.01, 3.4]} scale={1.2} />
    </group>
  )
}

function Building() {
  const tiles = []
  for (let x = -6; x <= 30; x += 2) for (let z = -8; z <= 8; z += 2) tiles.push([x, z])
  return (
    <group>
      {tiles.map(([x, z]) => <Model key={`${x},${z}`} name="floor-large" position={[x, 0, z]} />)}
      {Array.from({ length: 10 }, (_, i) => -4 + i * 4).map((x) => (
        <Model key={x} name={x % 8 === 0 ? 'structure-window-wide' : 'structure-wall'} position={[x, 0, -9.5]} scale={2} />
      ))}
      <Model name="catwalk-straight" position={[4, 2.8, -7.5]} scale={2} />
      <Model name="catwalk-straight" position={[6, 2.8, -7.5]} scale={2} />
      <Model name="catwalk-straight" position={[8, 2.8, -7.5]} scale={2} />
      <Model name="machine-window" position={[-4, 0, -4]} scale={1.4} />
      <Model name="machine" position={[-4, 0, 3.5]} rotation={[0, Math.PI, 0]} scale={1.4} />
      <Model name="pipe-large-long" position={[2, 3.6, -8.8]} rotation={[0, 0, Math.PI / 2]} scale={1.4} />
      <Model name="top-large-checkerboard" position={[-0.9, 0.005, 1.8]} />
    </group>
  )
}

function Clock({ demo, frameRef, playing, speed }: { demo: Demo; frameRef: React.RefObject<number>; playing: boolean; speed: number }) {
  useFrame((_, dt) => {
    if (!playing) return
    frameRef.current = (frameRef.current + dt * 30 * speed) % demo.frames.length
  })
  return null
}

export function Factory() {
  const [demo, setDemo] = useState<Demo | null>(null)
  const [playing, setPlaying] = useState(true)
  const [speed, setSpeed] = useState(1)
  const frameRef = useRef(0)
  const [hud, setHud] = useState({ t: 0, checked: 0, rejected: 0 })
  useEffect(() => { fetch(`${BASE}data/factory/bottling_demo.json`).then((r) => r.json()).then(setDemo) }, [])
  useEffect(() => {
    const id = setInterval(() => {
      if (!demo) return
      const f = demo.frames[Math.floor(frameRef.current)]
      if (f) setHud({ t: f.t, checked: f.checked, rejected: f.rejected })
    }, 100)
    return () => clearInterval(id)
  }, [demo])

  if (!demo) return <div className="grid h-full place-items-center bg-[#2a2b38] text-white/60">Loading line…</div>
  const L = demo.line.length / S
  const throughput = hud.t > 1 ? ((hud.checked - hud.rejected) / hud.t) * 60 : 0
  return (
    <div className="relative h-full w-full bg-[#2a2b38]">
      <Canvas shadows dpr={[1, 2]} camera={{ position: [L * 0.5, 15, 21], fov: 36 }} gl={{ antialias: false, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.0 }}>
        <color attach="background" args={['#2a2b38']} />
        <fog attach="fog" args={['#2a2b38', 22, 48]} />
        <ambientLight intensity={0.55} />
        <directionalLight position={[8, 16, 9]} intensity={2.4} castShadow shadow-mapSize={[4096, 4096]}
          shadow-camera-left={-20} shadow-camera-right={30} shadow-camera-top={15} shadow-camera-bottom={-15} shadow-bias={-0.0003} />
        <Environment resolution={256}>
          <Lightformer form="rect" intensity={2} position={[0, 10, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[30, 30, 1]} />
          <Lightformer form="rect" intensity={1.2} color="#ffd9b0" position={[-10, 4, 6]} rotation={[0, Math.PI / 2, 0]} scale={[10, 4, 1]} />
          <Lightformer form="rect" intensity={0.8} color="#b9c7ff" position={[20, 4, -6]} rotation={[0, -Math.PI / 2, 0]} scale={[10, 4, 1]} />
        </Environment>
        <Suspense fallback={null}>
          <Building />
          <Line demo={demo} />
          <Pusher demo={demo} frameRef={frameRef} />
          <Bottles demo={demo} frameRef={frameRef} />
          <ContactShadows position={[L / 2, 0.01, 0]} scale={[L + 8, 10]} opacity={0.35} blur={2.5} far={4} />
        </Suspense>
        <Clock demo={demo} frameRef={frameRef} playing={playing} speed={speed} />
        <OrbitControls makeDefault target={[L * 0.5, 0.2, 1]} maxPolarAngle={Math.PI / 2.1} minDistance={4} maxDistance={40} />
        <EffectComposer multisampling={0}>
          <N8AO aoRadius={0.6} intensity={2} distanceFalloff={1} halfRes />
          <Bloom intensity={0.3} luminanceThreshold={0.9} mipmapBlur />
          <SMAA />
          <Vignette offset={0.3} darkness={0.45} />
        </EffectComposer>
      </Canvas>

      <div className="pointer-events-none absolute left-4 top-20 w-[300px] rounded-2xl bg-[#17181f]/90 p-4 text-white shadow-2xl ring-1 ring-white/10">
        <div className="text-xs uppercase tracking-wider text-white/50">Line 1 · Bottling · commissioning</div>
        <div className="mt-2 grid grid-cols-2 gap-3 font-mono">
          <div><div className="text-[11px] text-white/50">shift time</div><div className="text-2xl">{hud.t.toFixed(1)}s</div></div>
          <div><div className="text-[11px] text-white/50">good / min</div><div className="text-2xl text-[#7ee0b0]">{throughput.toFixed(1)}</div></div>
          <div><div className="text-[11px] text-white/50">checked</div><div className="text-2xl">{hud.checked}</div></div>
          <div><div className="text-[11px] text-white/50">rejected</div><div className="text-2xl text-[#ffb27a]">{hud.rejected}</div></div>
        </div>
        <div className="mt-3 border-t border-white/10 pt-2 text-[11px] leading-5 text-white/60">
          belt {demo.settings.belt_speed_mps} m/s · reject below {demo.settings.reject_below_g} g · pusher offset {demo.settings.pusher_offset_mm} mm
        </div>
      </div>
      <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-full bg-[#17181f]/90 px-3 py-2 text-sm text-white ring-1 ring-white/10">
        <button className="rounded-full bg-white/10 px-3 py-1" onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
        {[0.5, 1, 2].map((s) => (
          <button key={s} className={`rounded-full px-2.5 py-1 ${speed === s ? 'bg-white text-black' : 'bg-white/10'}`} onClick={() => setSpeed(s)}>{s}×</button>
        ))}
      </div>
    </div>
  )
}
