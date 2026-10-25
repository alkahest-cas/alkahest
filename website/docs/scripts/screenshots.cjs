// Screenshot the docs in light/dark at desktop and phone widths.
//   node scripts/screenshots.cjs <base-url> <out-dir> [page ...]
// Needs `playwright` resolvable (not a dependency of the site build).
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const base = process.argv[2] || "http://localhost:47321/docs/";
const out = process.argv[3] || "shots";
const pages = process.argv.slice(4).length ? process.argv.slice(4) : ["", "guide/calculus/", "api/diff/"];
const sizes = [
  ["desktop", 1440, 900],
  ["mobile", 390, 844],
];

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  for (const scheme of ["light", "dark"]) {
    for (const [label, width, height] of sizes) {
      const ctx = await browser.newContext({ viewport: { width, height }, colorScheme: scheme, deviceScaleFactor: label === "mobile" ? 2 : 1 });
      const page = await ctx.newPage();
      const errors = [];
      page.on("pageerror", (e) => errors.push(String(e)));
      page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
      for (const p of pages) {
        await page.goto(base + p, { waitUntil: "networkidle" });
        await page.waitForTimeout(250);
        const name = (p.replace(/\/$/, "").replace(/\//g, "_") || "home") + `-${label}-${scheme}.png`;
        await page.screenshot({ path: path.join(out, name), fullPage: false });
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
        if (overflow > 0) console.log(`  horizontal overflow ${overflow}px on ${p} (${label})`);
      }
      if (process.env.INTERACT) {
        await page.goto(base + "guide/calculus/", { waitUntil: "networkidle" });
        await page.keyboard.press("Control+k");
        await page.waitForTimeout(300);
        await page.keyboard.type(process.env.QUERY || "integ");
        await page.waitForTimeout(300);
        await page.screenshot({ path: path.join(out, `search-${label}-${scheme}.png`) });
        await page.keyboard.press("Escape");
        if (label === "mobile") {
          await page.click(".subbar [data-drawer-open]");
          await page.waitForTimeout(400);
          await page.screenshot({ path: path.join(out, `drawer-${label}-${scheme}.png`) });
          await page.click(".scrim", { position: { x: 370, y: 400 } });
          await page.click(".menu-btn");
          await page.waitForTimeout(200);
          await page.screenshot({ path: path.join(out, `menu-${label}-${scheme}.png`) });
        }
      }
      if (errors.length) console.log(`  console errors (${label} ${scheme}):`, errors.slice(0, 5));
      await ctx.close();
    }
  }
  await browser.close();
  console.log("done →", out);
})();
