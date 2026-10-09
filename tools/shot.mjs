// Usage: node shot.mjs <url-or-file> <out.png> [width] [height]
import { chromium } from 'playwright';
const [,, target, out, w = '1200', h = '900'] = process.argv;
const url = target.startsWith('http') ? target : 'file://' + target;
const browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(url, { waitUntil: 'load' });
await page.waitForTimeout(1500);
const audio = await page.$$eval('audio', as => as.map(a => ({ ok: a.src.startsWith('data:audio/wav'), len: a.src.length })));
await page.screenshot({ path: out, fullPage: true });
console.log(JSON.stringify({ errors, audioCount: audio.length, allAudioOk: audio.every(a => a.ok) }));
await browser.close();
