// Record docs/demo.gif from the web viewer, headlessly.
//
//   cd docs/web && python3 -m http.server 8899 &
//   node record_gif.mjs http://localhost:8899/index.html /tmp/frames [run] [every]
//   python make_gif.py /tmp/frames ../demo.gif
//
// Each GIF frame is the viewer seeked to a recorded simulation frame, so the
// GIF shows the same trajectory the page replays.
import { chromium } from "playwright";

const [url, outDir, run = "nominal", every = "4", exe] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: exe || process.env.CHROMIUM_PATH || undefined,
  args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, colorScheme: "dark", deviceScaleFactor: 1 });
await page.goto(url, { waitUntil: "networkidle" });
await page.click(`.seg [data-run="${run}"]`);
await page.waitForTimeout(1500);
await page.click("#play");  // pause autoplay
const stage = page.locator(".stage3d");
await stage.scrollIntoViewIfNeeded();
const max = Number(await page.getAttribute("#seek", "max"));
let n = 0;
for (let f = 0; f <= max; f += Number(every)) {
  await page.evaluate(v => { const s = document.getElementById("seek"); s.value = v; s.dispatchEvent(new Event("input")); }, f);
  await page.waitForTimeout(40);
  await stage.screenshot({ path: `${outDir}/f${String(n++).padStart(4, "0")}.png` });
}
console.log("frames:", n);
await browser.close();
