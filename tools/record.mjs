// Usage: node record.mjs <url> <out.webm> <seconds> [width] [height]
import { chromium } from 'playwright';
import { renameSync } from 'node:fs';
const [,, url, out, secs = '20', w = '1280', h = '720'] = process.argv;
const browser = await chromium.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--use-angle=metal', '--enable-gpu', '--ignore-gpu-blocklist'] });
const ctx = await browser.newContext({ viewport: { width: +w, height: +h }, recordVideo: { dir: '/tmp/pwvid', size: { width: +w, height: +h } } });
const page = await ctx.newPage();
await page.goto(url, { waitUntil: 'load' });
await page.waitForTimeout(+secs * 1000);
const v = page.video();
await ctx.close();
renameSync(await v.path(), out);
await browser.close();
console.log('saved', out);
