import path from 'path';
import fs from 'fs';
import chalk from 'chalk';
import { chromium } from 'playwright';

export async function recordCommand(
  opts: {
    code?: string;
    output: string;
    url: string;
    serverUrl: string;
    width: string;
    height: string;
    delay: string;
    headless?: boolean;
  },
) {
  const outputPath = path.resolve(opts.output);
  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });

  // Health-check the Python server before touching Playwright
  console.log(chalk.bold('\nRecording notebook demo'));
  console.log(chalk.dim(`  URL:        ${opts.url}`));
  console.log(chalk.dim(`  Server:     ${opts.serverUrl}`));
  console.log(chalk.dim(`  Output:     ${outputPath}`));

  try {
    const res = await fetch(`${opts.serverUrl}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    console.log(chalk.green('  Server:     online ✓\n'));
  } catch (e) {
    console.error(chalk.red(`\n✗ Python server not reachable at ${opts.serverUrl}/health`));
    console.error(chalk.dim(`  Error: ${e}`));
    console.error(chalk.dim('  Start the server first: pnpm start   or   bash server/start.sh'));
    process.exit(1);
  }

  const videoDir = fs.mkdtempSync('/tmp/alkahest-rec-');
  const headless = opts.headless ?? !process.env.DISPLAY;
  console.log(chalk.dim(`  Headless: ${headless}`));

  // Encode cells into ?demo= URL param so the Notebook pre-populates them.
  // ?zen=1  — hides toolbar/nav for a clean recording
  // ?mode=server — forces server execution (alkahest + sympy available)
  // ?autorun=1 — auto-runs cells without needing the "Run all" button
  //   (needed because ?zen=1 hides the toolbar that contains "Run all")
  let targetUrl = opts.url;
  let numCells = 0;
  if (opts.code) {
    const raw = fs.readFileSync(opts.code, 'utf-8');
    const cellCodes = raw.split(/\n# ---\n/).map((c) => c.trim()).filter(Boolean);
    numCells = cellCodes.length;
    const encoded = Buffer.from(JSON.stringify(cellCodes)).toString('base64');
    targetUrl = `${opts.url}?demo=${encoded}&mode=server&zen=1&autorun=1`;
    console.log(chalk.dim(`  Cells:    ${numCells}\n`));
  } else {
    targetUrl = `${opts.url}?zen=1`;
  }

  const browser = await chromium.launch({ headless });
  const context = await browser.newContext({
    viewport: { width: Number(opts.width), height: Number(opts.height) },
    recordVideo: {
      dir: videoDir,
      size: { width: Number(opts.width), height: Number(opts.height) },
    },
  });

  const page = await context.newPage();
  await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: 30_000 });
  await page.waitForSelector('.cm-editor', { timeout: 20_000 });
  console.log(chalk.cyan('  Page loaded'));

  // Brief pause so the first frame shows the loaded cells
  await delay(1500);

  // ?autorun=1 triggers execution automatically (used with ?zen=1 since the toolbar is hidden).
  // For sessions without autorun, fall back to clicking "Run all" in the visible toolbar.
  const hasAutoRun = targetUrl.includes('autorun=1');
  if (!hasAutoRun) {
    await page.click('button:has-text("Run all")');
  }
  console.log(chalk.cyan('  Running cells…'));

  // Poll server health while waiting; abort if it goes offline
  let serverDied = false;
  const healthInterval = setInterval(async () => {
    try {
      const r = await fetch(`${opts.serverUrl}/health`, { signal: AbortSignal.timeout(3000) });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
    } catch {
      serverDied = true;
      console.error(chalk.red('\n✗ Server disconnected mid-recording — aborting'));
    }
  }, 3000);

  // Wait until every cell spinner is gone (all cells done) or server dies.
  // Use a generous timeout (120s) to accommodate slow computations like SymPy.
  await page.waitForFunction(() => {
    return document.querySelectorAll('.animate-spin').length === 0;
  }, { timeout: 120_000, polling: 500 }).catch(() => {
    if (!serverDied) console.log(chalk.yellow('  Warning: timed out waiting for cells to finish'));
  });

  clearInterval(healthInterval);

  if (serverDied) {
    await context.close();
    await browser.close();
    process.exit(1);
  }

  // Extra pause — wait for any async output rendering (KaTeX, images)
  await delay(1000);

  // Slow-scroll to show all cells and outputs
  console.log(chalk.cyan('  Scrolling to show all content…'));
  const pageHeight = await page.evaluate(() => document.body.scrollHeight);
  const viewportHeight = Number(opts.height);
  if (pageHeight > viewportHeight) {
    const scrollSteps = Math.ceil((pageHeight - viewportHeight) / 60);
    for (let i = 0; i < scrollSteps; i++) {
      await page.evaluate(() => window.scrollBy(0, 60));
      await delay(40);
    }
    // Hold at the bottom
    await delay(1200);
    // Scroll back to top
    for (let i = scrollSteps; i > 0; i--) {
      await page.evaluate(() => window.scrollBy(0, -60));
      await delay(30);
    }
    await delay(500);
  }

  console.log(chalk.green('  All cells done — holding final frame'));
  await delay(2000);

  await context.close();
  await browser.close();

  // Move video — use copy+delete to handle cross-device filesystems
  const videos = fs.readdirSync(videoDir).filter((f) => f.endsWith('.webm'));
  if (videos.length === 0) {
    console.error(chalk.red('No video captured.'));
    process.exit(1);
  }

  const src = path.join(videoDir, videos[0]);
  fs.copyFileSync(src, outputPath);
  fs.unlinkSync(src);
  try { fs.rmdirSync(videoDir); } catch {}

  console.log(chalk.green(`\n✓ Saved: ${outputPath}`));
}

function delay(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}
