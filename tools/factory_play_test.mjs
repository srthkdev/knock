// Drive the factory Play page: build the reference quality gate, test run, hand over, screenshot.
import { chromium } from 'playwright';
const [,, url, out] = process.argv;
const browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const logs = [];
page.on('pageerror', e => logs.push('PAGEERROR ' + e.message));
page.on('console', m => { if (m.type() === 'error') logs.push(m.text().slice(0, 300)); });
await page.goto(url, { waitUntil: 'load' });
await page.waitForFunction(() => window.__play, null, { timeout: 20000 });
await page.evaluate(() => {
  const p = window.__play, v = 1.0
  p.place('belt', 1, 2, 'E', { speed: v }); p.place('scale', 2, 2, 'E', { speed: v, reject_below_g: Number(new URL(location.href).searchParams.get('thr') ?? 950) })
  p.place('belt', 3, 2, 'E', { speed: v }); p.place('belt', 4, 2, 'E', { speed: v })
  p.place('pusher', 4, 3, 'S', { offset_mm: 250, stroke_ms: 450 }); p.place('bin', 4, 1, 'E'); p.place('bin', 5, 1, 'E')
  for (const y of [2, 3, 4]) p.place('belt', 5, y, 'N', { speed: v })
  for (const x of [5, 6, 7, 8, 9, 10]) p.place('belt', x, 5, 'E', { speed: v })
});
await page.waitForTimeout(500);
await page.evaluate(() => window.__play.test());
const t0 = Date.now();
await page.waitForFunction(() => document.body.innerText.includes('Test 1'), null, { timeout: 120000 });
console.log('test run ms', Date.now() - t0);
await page.waitForTimeout(4000);
await page.screenshot({ path: out.replace('.png', '-test.png') });
await page.evaluate(() => window.__play.handover());
await page.waitForFunction(() => document.body.innerText.includes('Production run'), null, { timeout: 180000 });
await page.waitForTimeout(5000);
await page.screenshot({ path: out });
console.log(await page.evaluate(() => document.body.innerText.match(/(PASSED|FAILED)[^\n]*/)?.[0]));
console.log(JSON.stringify(logs.slice(0, 10)));
await browser.close();
