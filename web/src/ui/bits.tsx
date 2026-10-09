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

export function Logo({ className = 'h-7' }: { className?: string }) {
  return (
    <svg viewBox="0 0 128 128" className={className} aria-hidden>
      <defs>
        <linearGradient id="lt" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#f3a07f" /><stop offset="1" stopColor="#e07a5a" /></linearGradient>
      </defs>
      <rect width="128" height="128" rx="28" fill="#1c1d1f" />
      <g fill="none" stroke="#5fd3a0" strokeLinecap="round" strokeWidth="5">
        <path d="M86 44 a22 22 0 0 1 0 40" opacity="0.95" />
        <path d="M96 34 a36 36 0 0 1 0 60" opacity="0.6" />
        <path d="M106 26 a50 50 0 0 1 0 76" opacity="0.3" />
      </g>
      <g stroke="#1c1d1f" strokeWidth="2" strokeLinejoin="round">
        <path d="M48 30 L74 44 L48 58 L22 44 Z" fill="url(#lt)" />
        <path d="M22 44 L48 58 L48 92 L22 78 Z" fill="#c4553a" />
        <path d="M48 58 L74 44 L74 78 L48 92 Z" fill="#a8432c" />
      </g>
      <circle cx="78" cy="64" r="4.5" fill="#5fd3a0" />
    </svg>
  )
}
