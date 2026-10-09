// Usage: node app_shot.mjs <url> <out.png> [wait_ms] [width] [height]
import { chromium } from 'playwright';
const [,, url, out, wait = '5000', w = '1440', h = '900'] = process.argv;
const browser = await chromium.launch({
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true,
  args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'],
});
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 });
const logs = [];
page.on('pageerror', e => logs.push('PAGEERROR ' + e.message));
page.on('console', m => { if (['error', 'warning'].includes(m.type())) logs.push(m.type() + ' ' + m.text().slice(0, 300)); });
await page.goto(url, { waitUntil: 'load' });
await page.waitForTimeout(+wait);
await page.screenshot({ path: out });
console.log(JSON.stringify(logs.slice(0, 15), null, 1));
await browser.close();
