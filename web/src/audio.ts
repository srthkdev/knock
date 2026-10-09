// WebAudio playback of probe sounds and a small FFT for live spectrograms.
import { dataUrl } from './data'

let ctx: AudioContext | null = null
const buffers = new Map<string, Promise<AudioBuffer>>()

export function audioContext() {
  ctx ??= new AudioContext()
  return ctx
}

export function loadSound(path: string) {
  if (!buffers.has(path)) {
    buffers.set(path, fetch(dataUrl(path))
      .then((r) => r.arrayBuffer())
      .then((b) => audioContext().decodeAudioData(b)))
  }
  return buffers.get(path)!
}

export let muted = false
export const setMuted = (m: boolean) => { muted = m }

export async function playSound(path: string, gain = 1.0) {
  const ac = audioContext()
  if (ac.state === 'suspended') await ac.resume()
  if (muted) return
  const buf = await loadSound(path)
  const src = ac.createBufferSource()
  const g = ac.createGain()
  g.gain.value = gain
  src.buffer = buf
  src.connect(g).connect(ac.destination)
  src.start()
}

/** A short synthetic knock used for flick strikes. */
export function playClick(strength = 0.5) {
  if (muted) return
  const ac = audioContext()
  const n = Math.floor(ac.sampleRate * 0.05)
  const buf = ac.createBuffer(1, n, ac.sampleRate)
  const d = buf.getChannelData(0)
  for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (ac.sampleRate * 0.004)) * 0.6 * strength
  const src = ac.createBufferSource()
  src.buffer = buf
  src.connect(ac.destination)
  src.start()
}

// ---------------------------------------------------------------- spectrogram
function fft(re: Float32Array, im: Float32Array) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    const wr = Math.cos(ang), wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2
        const xr = re[b] * cr - im[b] * ci
        const xi = re[b] * ci + im[b] * cr
        re[b] = re[a] - xr; im[b] = im[a] - xi
        re[a] += xr; im[a] += xi
        const t = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = t
      }
    }
  }
}

export interface Spectrogram { cols: Float32Array[]; bins: number; sampleRate: number; duration: number }

const specCache = new Map<string, Promise<Spectrogram>>()
export function spectrogram(path: string): Promise<Spectrogram> {
  if (!specCache.has(path)) {
    specCache.set(path, loadSound(path).then((buf) => {
      const x = buf.getChannelData(0)
      const N = 1024, hop = 256
      const win = new Float32Array(N).map((_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)))
      const cols: Float32Array[] = []
      for (let s = 0; s + N <= x.length; s += hop) {
        const re = new Float32Array(N), im = new Float32Array(N)
        for (let i = 0; i < N; i++) re[i] = x[s + i] * win[i]
        fft(re, im)
        const mag = new Float32Array(N / 2)
        for (let i = 0; i < N / 2; i++) mag[i] = 20 * Math.log10(Math.hypot(re[i], im[i]) + 1e-9)
        cols.push(mag)
      }
      return { cols, bins: N / 2, sampleRate: buf.sampleRate, duration: buf.duration }
    }))
  }
  return specCache.get(path)!
}

const MAGMA = [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]]
function magma(u: number) {
  const t = Math.min(1, Math.max(0, u)) * (MAGMA.length - 1)
  const i = Math.floor(t), f = t - i
  const a = MAGMA[i], b = MAGMA[Math.min(i + 1, MAGMA.length - 1)]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** Draw a log-frequency spectrogram (50 Hz to 12 kHz), revealing up to `progress` (0..1). */
export function drawSpectrogram(canvas: HTMLCanvasElement, s: Spectrogram, progress = 1) {
  const g = canvas.getContext('2d')!
  const W = canvas.width, H = canvas.height
  const img = g.createImageData(W, H)
  const fmin = Math.log(50), fmax = Math.log(12000)
  const hzPerBin = s.sampleRate / 2 / s.bins
  const lo = -10, hi = 55 // dB range after the shared gain used for export
  const shown = Math.floor(W * progress)
  for (let x = 0; x < W; x++) {
    const col = s.cols[Math.min(s.cols.length - 1, Math.floor((x / W) * s.cols.length))]
    for (let y = 0; y < H; y++) {
      const f = Math.exp(fmax - ((fmax - fmin) * y) / H)
      const v = col[Math.min(s.bins - 1, Math.round(f / hzPerBin))]
      const [r, gg, b] = x < shown ? magma((v - lo) / (hi - lo)) : [20, 20, 24]
      const k = (y * W + x) * 4
      img.data[k] = r; img.data[k + 1] = gg; img.data[k + 2] = b; img.data[k + 3] = 255
    }
  }
  g.putImageData(img, 0, 0)
}
