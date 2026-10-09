import { readFileSync } from 'node:fs'
import { simulateFlick, simulatePush } from '../src/sim/physics'
const cases = JSON.parse(readFileSync(new URL('./parity_cases.json', import.meta.url), 'utf8'))
let worst = 0
for (const c of cases) {
  const f = await simulateFlick(c.xml, c.J)
  const p = await simulatePush(c.xml, c.F, 0.4)
  const ef = Math.abs(f.distance - c.flick), ep = Math.abs(p.distance - c.push)
  worst = Math.max(worst, ef, ep)
  if (ef > 1e-4 || ep > 1e-4) console.log(c.id, 'flick', c.flick.toFixed(5), f.distance.toFixed(5), 'push', c.push.toFixed(5), p.distance.toFixed(5))
}
console.log(`${cases.length} objects, worst |browser - python| = ${(worst * 1000).toFixed(4)} mm`)
