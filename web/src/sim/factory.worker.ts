// Runs the factory simulation off the main thread.
import { runFactory } from './factory'

self.onmessage = async (e: MessageEvent) => {
  const { id, layout, hidden, seconds, seed, fps, assets } = e.data
  try {
    const report = await runFactory(layout, hidden, seconds, seed, fps, (f) => self.postMessage({ id, progress: f }), assets)
    self.postMessage({ id, report })
  } catch (err) {
    self.postMessage({ id, error: String(err) })
  }
}
