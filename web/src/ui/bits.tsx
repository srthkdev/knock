import { useEffect, useRef, useState, type ReactNode } from 'react'
import { drawSpectrogram, spectrogram, type Spectrogram } from '../audio'
import type { ReplayStep, SoundFeatures } from '../data'

export function stepTitle(s: Pick<ReplayStep, 'action'>) {
  const a = s.action
  if (a.type === 'flick') return `flick ${a.object}`
  return `${a.probe} ${a.object}`
}

export function readingText(s: Pick<ReplayStep, 'action' | 'reading'>): string {
  const a = s.action, r = s.reading
  if (r.error) return r.error
  if (a.type === 'flick') return `${(a.impulse_Ns ?? 0).toFixed(2)} N·s`
  switch (a.probe) {
    case 'push': return `${r.force_N?.toFixed(2)} N → moved ${r.moved_mm?.toFixed(0)} mm`
    case 'lift': return `${r.weight_g?.toFixed(0)} g`
    case 'tap': return soundText(r.sound!, 'tap')
    case 'shake': return soundText(r.sound!, 'shake')
  }
  return ''
}

export function soundText(s: SoundFeatures, kind: string) {
  if (kind === 'shake') return s.loudness_db_over_floor < 10 ? 'silent' : `${s.distinct_hits} hits, ${s.loudness_db_over_floor.toFixed(0)} dB`
  return `${s.dominant_pitch_hz} Hz, rings ${s.ring_time_s.toFixed(2)} s`
}

const ICON: Record<string, string> = { push: '→', lift: '↑', tap: '◉', shake: '≈', flick: '⟶' }
export const probeIcon = (p: string) => ICON[p] ?? '·'

export function SpectrogramView({ path, progress = 1, className = '', silent = false }: { path: string; progress?: number; className?: string; silent?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [spec, setSpec] = useState<Spectrogram | null>(null)
  useEffect(() => {
    let alive = true
    spectrogram(path).then((s) => alive && setSpec(s)).catch(() => {})
    return () => { alive = false }
  }, [path])
  useEffect(() => {
    if (spec && ref.current) drawSpectrogram(ref.current, spec, progress)
  }, [spec, progress])
  return (
    <div className={`relative overflow-hidden rounded-xl bg-[#141418] ${className}`}>
      <canvas ref={ref} width={360} height={110} className="block h-full w-full" />
      <span className="absolute left-2 top-1.5 font-mono text-[10px] text-white/50">12 kHz</span>
      <span className="absolute bottom-1.5 left-2 font-mono text-[10px] text-white/50">50 Hz</span>
      {silent && progress >= 1 && <span className="absolute inset-0 grid place-items-center text-xs text-white/60">silence: nothing audible</span>}
    </div>
  )
}

export function Typing({ text, progress }: { text: string; progress: number }) {
  const n = Math.floor(text.length * Math.min(1, progress))
  return <span className={progress < 1 ? 'caret' : ''}>{text.slice(0, n)}</span>
}

export function Pill({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'good' | 'bad' | 'dark' }) {
  const cls = {
    neutral: 'bg-black/5 text-ink',
    good: 'bg-mint/15 text-mint-deep',
    bad: 'bg-coral/15 text-[#a8432c]',
    dark: 'bg-ink text-white',
  }[tone]
  return <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${cls}`}>{children}</span>
}

// A yantra (the square frame with four gates, interlocking triangles, centre point) inside a gear.
export function Logo({ className = 'h-7' }: { className?: string }) {
  return (
    <svg viewBox="0 0 128 128" className={className} aria-hidden>
      <rect width="128" height="128" rx="28" fill="#1c1d1f" />
      <path d="M26 26 H54 V17 H74 V26 H102 V54 H111 V74 H102 V102 H74 V111 H54 V102 H26 V74 H17 V54 H26 Z" fill="none" stroke="#f2a541" strokeWidth="4.5" strokeLinejoin="round" />
      <circle cx="64" cy="64" r="30" fill="none" stroke="#5fd3a0" strokeWidth="7" strokeDasharray="5.9 5.88" />
      <circle cx="64" cy="64" r="26" fill="#1c1d1f" stroke="#5fd3a0" strokeWidth="3" />
      <path d="M64 44 L81.3 74 L46.7 74 Z" fill="#e07a5a" />
      <path d="M64 84 L46.7 54 L81.3 54 Z" fill="none" stroke="#f2a541" strokeWidth="3.5" strokeLinejoin="round" />
      <circle cx="64" cy="64" r="4" fill="#ffffff" />
    </svg>
  )
}
