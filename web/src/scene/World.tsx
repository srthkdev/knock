// The Knock table rendered with React Three Fiber.
// Everything inside <MjFrame> uses MuJoCo coordinates (z up, metres), so simulator
// trajectories can be applied directly.
import { useEffect, useMemo, useRef, type ReactNode } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Environment, Html, Lightformer, OrbitControls, RoundedBox, SoftShadows, Text, Trail } from '@react-three/drei'
import { Bloom, EffectComposer, N8AO, SMAA, Vignette } from '@react-three/postprocessing'
import * as THREE from 'three'
import type { LevelBundle } from '../data'

export interface ObjPose { x: number; z: number; q: [number, number, number, number] }
export interface Effect { kind: 'ripple' | 'arrow' | 'weight'; obj: string; t0: number; value?: number; label?: string }
export interface Frame {
  poses: Record<string, ObjPose>
  striker: [number, number, number] | null
  ghost: ObjPose | null
  ghostObj: string | null
  effects: Effect[]
  trail: string | null
  now: number
}

export const emptyFrame = (): Frame => ({ poses: {}, striker: null, ghost: null, ghostObj: null, effects: [], trail: null, now: 0 })

type LevelGeom = Pick<LevelBundle, 'task' | 'objects'>

// ------------------------------------------------------------------ camera
export function cameraFor(level: LevelGeom, framing: 'default' | 'hero' = 'default') {
  const far = Math.max(level.task.target_x + level.task.tol, 0.6)
  const hero = framing === 'hero'
  const look = new THREE.Vector3(far * (hero ? 0.5 : 0.62), hero ? 0.12 : 0.06, hero ? 0.26 : 0.02) // MuJoCo coords; hero looks above the table
  const dist = (1.3 + 0.8 * far) * (hero ? 1.35 : 1.45)
  const eye = look.clone().add(new THREE.Vector3(-0.5, -0.66, 0.56).multiplyScalar(dist))
  const toThree = (v: THREE.Vector3) => new THREE.Vector3(v.x, v.z, -v.y)
  return { eye: toThree(eye), look: toThree(look) }
}

function CameraRig({ level, auto, framing }: { level: LevelGeom; auto: boolean; framing: 'default' | 'hero' }) {
  const { camera } = useThree()
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null
  const goal = useMemo(() => cameraFor(level, framing), [level, framing])
  const settled = useRef(false)
  const placed = useRef(false)
  useEffect(() => { settled.current = false }, [goal])
  useFrame((state, dt) => {
    if (!controls) return
    if (!placed.current) {
      camera.position.copy(goal.eye)
      controls.target.copy(goal.look)
      controls.update()
      placed.current = true
    }
    if (!settled.current || auto) {
      const k = 1 - Math.exp(-dt * 2.2)
      const t = state.clock.elapsedTime
      const sway = auto ? new THREE.Vector3(Math.sin(t * 0.15) * 0.12, 0, Math.cos(t * 0.15) * 0.06) : new THREE.Vector3()
      camera.position.lerp(goal.eye.clone().add(sway), k)
      controls.target.lerp(goal.look, k)
      controls.update()
      if (camera.position.distanceTo(goal.eye) < 0.01 && controls.target.distanceTo(goal.look) < 0.01) settled.current = true
    }
  })
  return null
}

// ------------------------------------------------------------------ table
function tableTexture() {
  const c = document.createElement('canvas')
  c.width = c.height = 1024
  const g = c.getContext('2d')!
  g.fillStyle = '#d9d0bf'
  g.fillRect(0, 0, 1024, 1024)
  // subtle paper grain
  const img = g.getImageData(0, 0, 1024, 1024)
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (Math.random() - 0.5) * 7
    img.data[i] += n; img.data[i + 1] += n; img.data[i + 2] += n
  }
  g.putImageData(img, 0, 0)
  // 10 cm grid (texture spans 1 m)
  for (let i = 0; i <= 10; i++) {
    g.strokeStyle = i % 5 === 0 ? 'rgba(60,55,45,0.22)' : 'rgba(60,55,45,0.10)'
    g.lineWidth = i % 5 === 0 ? 3 : 2
    const p = (i / 10) * 1024
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, 1024); g.stroke()
    g.beginPath(); g.moveTo(0, p); g.lineTo(1024, p); g.stroke()
  }
  const t = new THREE.CanvasTexture(c)
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.repeat.set(8, 8)
  t.anisotropy = 8
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

function Table({ level }: { level: LevelGeom }) {
  const tex = useMemo(tableTexture, [])
  const span = ((level.objects.length - 1) * 0.35) / 2 + 0.2
  const far = Math.max(level.task.target_x + level.task.tol + 0.3, 1.0)
  const ticks = []
  for (let x = 0; x <= far + 1e-6; x += 0.25) ticks.push(Math.round(x * 100) / 100)
  return (
    <group>
      <mesh receiveShadow position={[0, 0, 0]}>
        <planeGeometry args={[8, 8]} />
        <meshStandardMaterial map={tex} roughness={0.92} metalness={0} />
      </mesh>
      {/* start line */}
      <mesh position={[0, 0, 0.0005]}>
        <planeGeometry args={[0.006, span * 2]} />
        <meshBasicMaterial color="#2a2a2a" />
      </mesh>
      {level.objects.map((o) => (
        <mesh key={o.id} position={[far / 2, o.y, 0.0004]}>
          <planeGeometry args={[far + 0.2, 0.004]} />
          <meshBasicMaterial color="#b9b3a6" />
        </mesh>
      ))}
      {ticks.map((x) => (
        <group key={x} position={[x, -span - 0.02, 0.001]}>
          <mesh position={[0, 0.03, 0]}>
            <planeGeometry args={[0.004, 0.04]} />
            <meshBasicMaterial color="#8d877a" />
          </mesh>
          <Text fontSize={0.034} color="#8d877a" anchorX="center" anchorY="top" position={[0, 0, 0]}>
            {x === 0 ? '0 m' : `${x.toFixed(2)}`}
          </Text>
        </group>
      ))}
      <TargetBand level={level} span={span} />
    </group>
  )
}

const bandMaterial = () => new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  uniforms: { uTime: { value: 0 }, uColor: { value: new THREE.Color('#34b37f') } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: `
    uniform float uTime; uniform vec3 uColor; varying vec2 vUv;
    void main(){
      float edge = smoothstep(0.0, 0.08, vUv.x) * smoothstep(0.0, 0.08, 1.0 - vUv.x);
      float stripes = 0.5 + 0.5 * sin((vUv.y * 60.0 - uTime * 1.5));
      float a = mix(0.85, 0.55, edge) + 0.12 * stripes * edge;
      vec3 c = mix(uColor * 1.25, uColor, edge);
      gl_FragColor = vec4(c, a);
    }`,
})

function TargetBand({ level, span }: { level: LevelGeom; span: number }) {
  const mat = useMemo(bandMaterial, [])
  useFrame((s) => { mat.uniforms.uTime.value = s.clock.elapsedTime })
  const { target_x, tol } = level.task
  return (
    <group>
      <mesh position={[target_x, 0, 0.0012]} material={mat}>
        <planeGeometry args={[2 * tol, span * 2]} />
      </mesh>
      <Text fontSize={0.045} color="#2f6f5e" position={[target_x, span + 0.05, 0.002]} anchorX="center" anchorY="bottom" fontWeight={700}>
        target
      </Text>
    </group>
  )
}

// ------------------------------------------------------------------ objects
function Cube({ obj, frame, selected, hovered, onSelect, onHover, label = true }: {
  obj: LevelBundle['objects'][number]
  frame: React.RefObject<Frame>
  selected?: boolean
  hovered?: boolean
  onSelect?: (id: string) => void
  onHover?: (id: string | null) => void
  label?: boolean
}) {
  const ref = useRef<THREE.Group>(null!)
  const mat = useRef<THREE.MeshPhysicalMaterial>(null!)
  useFrame(() => {
    const p = frame.current.poses[obj.id]
    const g = ref.current
    if (p) {
      g.position.set(p.x, obj.y, p.z)
      g.quaternion.set(p.q[1], p.q[2], p.q[3], p.q[0])
    } else {
      g.position.set(0, obj.y, obj.half)
      g.quaternion.identity()
    }
    if (mat.current) mat.current.emissiveIntensity = selected ? 0.22 : hovered ? 0.12 : 0
  })
  const edge = 2 * obj.half
  const color = useMemo(() => {
    const c = new THREE.Color(obj.color)
    const hsl = { h: 0, s: 0, l: 0 }
    c.getHSL(hsl)
    if (hsl.l > 0.66) c.setHSL(hsl.h, Math.max(hsl.s, 0.22), 0.5)
    return '#' + c.getHexString()
  }, [obj.color])
  return (
    <group ref={ref}>
      <RoundedBox args={[edge, edge, edge]} radius={0.005} smoothness={4} castShadow receiveShadow
        onClick={onSelect ? (e) => { e.stopPropagation(); onSelect(obj.id) } : undefined}
        onPointerOver={onHover ? (e) => { e.stopPropagation(); onHover(obj.id) } : undefined}
        onPointerOut={onHover ? () => onHover(null) : undefined}>
        <meshPhysicalMaterial ref={mat} color={color} roughness={0.42} clearcoat={0.35} clearcoatRoughness={0.3}
          emissive={color} emissiveIntensity={0} />
      </RoundedBox>
      {label && (
        <Html position={[0, 0, obj.half + 0.045]} center distanceFactor={2.0} zIndexRange={[10, 0]}>
          <div className={`select-none rounded-full px-2.5 py-0.5 text-[15px] font-semibold shadow-sm ${selected ? 'bg-ink text-white' : 'bg-white/90 text-ink'}`}>
            {obj.id}
          </div>
        </Html>
      )}
    </group>
  )
}

function Ghost({ level, frame }: { level: LevelGeom; frame: React.RefObject<Frame> }) {
  const ref = useRef<THREE.Group>(null!)
  const tag = useRef<HTMLDivElement>(null)
  const half = level.objects[0].half
  useFrame(() => {
    const f = frame.current
    const g = ref.current
    const show = !!(f.ghost && f.ghostObj)
    if (tag.current) tag.current.style.display = show ? 'block' : 'none'
    if (!show) { g.visible = false; return }
    const o = level.objects.find((x) => x.id === f.ghostObj)!
    const gp = f.ghost!
    g.visible = true
    g.position.set(gp.x, o.y, gp.z)
    g.quaternion.set(gp.q[1], gp.q[2], gp.q[3], gp.q[0])
  })
  return (
    <group ref={ref} visible={false}>
      <RoundedBox args={[2 * half, 2 * half, 2 * half]} radius={0.005}>
        <meshStandardMaterial color="#3fbf8a" transparent opacity={0.28} depthWrite={false} />
      </RoundedBox>
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(2 * half, 2 * half, 2 * half)]} />
        <lineBasicMaterial color="#2f6f5e" transparent opacity={0.8} />
      </lineSegments>
      <Html position={[0, 0, half + 0.05]} center distanceFactor={1.8}>
        <div ref={tag} style={{ display: 'none' }} className="select-none whitespace-nowrap rounded-full bg-mint-deep/90 px-2 py-0.5 text-[11px] font-medium text-white">ideal flick</div>
      </Html>
    </group>
  )
}

function Striker({ frame }: { frame: React.RefObject<Frame> }) {
  const ref = useRef<THREE.Mesh>(null!)
  useFrame(() => {
    const s = frame.current.striker
    ref.current.visible = !!s
    if (s) ref.current.position.set(s[0], s[1], s[2])
  })
  return (
    <mesh ref={ref} castShadow visible={false}>
      <sphereGeometry args={[0.012, 32, 32]} />
      <meshPhysicalMaterial color="#c9ccd3" metalness={1} roughness={0.18} />
    </mesh>
  )
}

function Effects({ level, frame }: { level: LevelGeom; frame: React.RefObject<Frame> }) {
  // Up to 3 ripple rings, a force arrow, and a weight readout, driven by frame.effects.
  const rings = [useRef<THREE.Mesh>(null!), useRef<THREE.Mesh>(null!), useRef<THREE.Mesh>(null!)]
  const arrow = useRef<THREE.Group>(null!)
  useFrame(() => {
    const f = frame.current
    const ripple = f.effects.find((e) => e.kind === 'ripple')
    rings.forEach((r, i) => {
      const m = r.current
      if (!ripple) { m.visible = false; return }
      const o = level.objects.find((x) => x.id === ripple.obj)!
      const p = f.poses[o.id] ?? { x: 0, z: o.half }
      const age = f.now - ripple.t0 - i * 0.12
      m.visible = age > 0 && age < 0.9
      const s = 0.02 + age * 0.35
      m.position.set(p.x - o.half - 0.002, o.y, p.z)
      m.scale.set(s, s, s)
      ;(m.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.7 * (1 - age / 0.9))
    })
    const push = f.effects.find((e) => e.kind === 'arrow')
    arrow.current.visible = !!push
    if (push) {
      const o = level.objects.find((x) => x.id === push.obj)!
      const p = f.poses[o.id] ?? { x: 0, z: o.half }
      const len = 0.05 + Math.min(0.25, Math.log10(1 + (push.value ?? 1)) * 0.18)
      arrow.current.position.set(p.x - o.half - 0.01, o.y, o.half)
      arrow.current.scale.set(len, 1, 1)
    }
  })
  return (
    <group>
      {rings.map((r, i) => (
        <mesh key={i} ref={r} rotation={[0, Math.PI / 2, 0]} visible={false}>
          <ringGeometry args={[0.9, 1, 64]} />
          <meshBasicMaterial color="#3fbf8a" transparent opacity={0.6} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      ))}
      <group ref={arrow} visible={false}>
        <mesh position={[-0.5, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.006, 0.006, 1, 16]} />
          <meshStandardMaterial color="#e07a5a" />
        </mesh>
        <mesh position={[-0.02, 0, 0]} rotation={[0, 0, -Math.PI / 2]} scale={[1, 0.04, 1]}>
          <coneGeometry args={[0.018, 1, 24]} />
          <meshStandardMaterial color="#e07a5a" />
        </mesh>
      </group>
    </group>
  )
}

function FlickTrail({ level, frame }: { level: LevelGeom; frame: React.RefObject<Frame> }) {
  const ref = useRef<THREE.Mesh>(null!)
  useFrame(() => {
    const f = frame.current
    const id = f.trail
    if (!id) { ref.current.visible = false; return }
    const o = level.objects.find((x) => x.id === id)!
    const p = f.poses[id]
    ref.current.visible = true
    if (p) ref.current.position.set(p.x, o.y, 0.006)
  })
  return (
    <Trail width={1.2} length={7} color="#e07a5a" attenuation={(w) => w * w} decay={1.5}>
      <mesh ref={ref} visible={false}>
        <sphereGeometry args={[0.001]} />
        <meshBasicMaterial color="#e07a5a" />
      </mesh>
    </Trail>
  )
}

function WeightBadge({ level, frame }: { level: LevelGeom; frame: React.RefObject<Frame> }) {
  const ref = useRef<THREE.Group>(null!)
  const text = useRef<HTMLDivElement>(null!)
  useFrame(() => {
    const f = frame.current
    const w = f.effects.find((e) => e.kind === 'weight')
    ref.current.visible = !!w
    if (text.current) text.current.style.display = w ? 'block' : 'none'
    if (w && text.current) {
      const o = level.objects.find((x) => x.id === w.obj)!
      const p = f.poses[o.id] ?? { x: 0, z: o.half }
      ref.current.position.set(p.x + o.half + 0.07, o.y, p.z + 0.03)
      text.current.textContent = w.label ?? ''
      text.current.style.opacity = String(Math.min(1, (f.now - w.t0) * 3))
    }
  })
  return (
    <group ref={ref} visible={false}>
      <Html center distanceFactor={1.8}>
        <div ref={text} style={{ display: 'none' }} className="select-none whitespace-nowrap rounded-lg bg-ink px-2.5 py-1 font-mono text-[14px] text-white shadow-lg" />
      </Html>
    </group>
  )
}

// ------------------------------------------------------------------ canvas
export function World({ level, frame, auto = false, selected, hovered, onSelect, onHover, children, post = true, framing = 'default' }: {
  level: LevelGeom
  frame: React.RefObject<Frame>
  auto?: boolean
  selected?: string | null
  hovered?: string | null
  onSelect?: (id: string) => void
  onHover?: (id: string | null) => void
  children?: ReactNode
  post?: boolean
  framing?: 'default' | 'hero'
}) {
  const cam = cameraFor(level, framing)
  return (
    <Canvas shadows dpr={[1, 2]} camera={{ position: cam.eye.toArray(), fov: 34, near: 0.02, far: 40 }}
      gl={{ antialias: false, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 0.88 }}
      onPointerMissed={() => onHover?.(null)}>
      <color attach="background" args={['#ebe6dc']} />
      <fog attach="fog" args={['#ebe6dc', 4.5, 12]} />
      <SoftShadows size={18} samples={12} focus={0.6} />
      <ambientLight intensity={0.22} />
      <directionalLight position={[-1.2, 3.2, 1.6]} intensity={2.0} castShadow shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-2.5} shadow-camera-right={2.5} shadow-camera-top={2.5} shadow-camera-bottom={-2.5} shadow-bias={-0.0004} />
      <Environment resolution={256}>
        <Lightformer form="rect" intensity={1.4} position={[0, 4, 0]} rotation={[Math.PI / 2, 0, 0]} scale={[6, 6, 1]} />
        <Lightformer form="rect" intensity={1.2} position={[-4, 1.5, 2]} rotation={[0, Math.PI / 2, 0]} scale={[4, 2, 1]} />
        <Lightformer form="rect" intensity={0.8} color="#ffd9c7" position={[4, 1, -2]} rotation={[0, -Math.PI / 2, 0]} scale={[4, 2, 1]} />
      </Environment>
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <Table level={level} />
        {level.objects.map((o) => (
          <Cube key={o.id} obj={o} frame={frame} selected={selected === o.id} hovered={hovered === o.id}
            onSelect={onSelect} onHover={onHover} />
        ))}
        <Ghost level={level} frame={frame} />
        <Striker frame={frame} />
        <Effects level={level} frame={frame} />
        <WeightBadge level={level} frame={frame} />
        <FlickTrail level={level} frame={frame} />
        {children}
      </group>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={0.6} maxDistance={6}
        maxPolarAngle={Math.PI / 2.15} />
      <CameraRig level={level} auto={auto} framing={framing} />
      {post && (
        <EffectComposer multisampling={0}>
          <N8AO aoRadius={0.12} intensity={1.6} distanceFalloff={0.6} halfRes />
          <Bloom intensity={0.25} luminanceThreshold={0.92} mipmapBlur />
          <SMAA />
          <Vignette offset={0.25} darkness={0.35} />
        </EffectComposer>
      )}
    </Canvas>
  )
}
