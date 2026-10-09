// Static data exported by `python -m knock.webexport` into public/data.
export type Pose = [t: number, x: number, z: number, qw: number, qx: number, qy: number, qz: number]

export interface SoundFeatures {
  ring_time_s: number
  distinct_hits: number
  brightness_hz: number
  dominant_pitch_hz: number
  loudness_db_over_floor: number
}

export interface LevelSummary {
  id: string
  kind: 'slide' | 'find_and_slide'
  property: string | null
  budget: number
  goal: string
  color: string
  n_objects: number
}

export interface EpisodeSummary {
  success: boolean
  correct_object: boolean
  error_m: number | null
  probes: number | null
}

export interface RunSummary {
  run: string
  label: string
  n: number
  solved: number
  episodes: Record<string, EpisodeSummary>
}

export interface Index {
  levels: LevelSummary[]
  runs: RunSummary[]
}

export interface BundleObject {
  id: string
  edge_m: number
  half: number
  color: string
  y: number
  mjcf: string
  lift_g: number
  tap: { audio: string; features: SoundFeatures }
  shake: { audio: string; features: SoundFeatures }
  reveal?: Record<string, unknown>
}

export interface LevelBundle {
  id: string
  public: { goal: string; probe_budget: number; target_band_x: [number, number] }
  task: {
    kind: 'slide' | 'find_and_slide'
    property: string | null
    property_text: string
    object_id: string | null
    target_x: number
    tol: number
    budget: number
  }
  objects: BundleObject[]
  answer: string
  oracle: { object: string; impulse_Ns: number; predicted_x: number } | null
  physics: { push_seconds: number; push_range: [number, number]; flick_range: [number, number] }
}

export interface ReplayStep {
  action: { type: 'probe' | 'flick'; probe?: string; object: string; force_N?: number; impulse_Ns?: number }
  reading: {
    force_N?: number
    moved_mm?: number
    weight_g?: number
    sound?: SoundFeatures
    error?: string
  }
  reasoning: string | null
  traj?: Pose[]
  audio?: string
}

export interface Replay {
  run: string
  label: string
  level: string
  steps: ReplayStep[]
  result: { success: boolean; correct_object: boolean; final_x: number; error_m: number; impulse_Ns: number; probes_used: number } | null
  tainted: boolean
  oracle: { object: string; impulse_Ns: number; predicted_x: number }
  oracle_traj: Pose[] | null
}

const base = `${import.meta.env.BASE_URL}data/`
export const dataUrl = (p: string) => base + p

const cache = new Map<string, Promise<unknown>>()
function get<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(path, fetch(dataUrl(path)).then((r) => {
      if (!r.ok) throw new Error(`${path}: ${r.status}`)
      return r.json()
    }))
  }
  return cache.get(path) as Promise<T>
}

export const loadIndex = () => get<Index>('index.json')
export const loadLevel = (id: string) => get<LevelBundle>(`levels/${id}.json`)
export const loadReplay = (run: string, id: string) => get<Replay>(`replays/${run}/${id}.json`)

export const PROPERTY_TEXT: Record<string, string> = {
  hollow: 'is hollow',
  solid: 'is solid',
  rattles: 'rattles inside',
  heaviest: 'is the heaviest',
  lightest: 'is the lightest',
}

export function goalText(task: LevelBundle['task']) {
  return task.kind === 'slide'
    ? `Flick ${task.object_id} so it stops inside the green band`
    : `Find the cube that ${PROPERTY_TEXT[task.property ?? ''] ?? task.property}, then flick it into the green band`
}
