import path from 'path';
import fs from 'fs';
import chalk from 'chalk';
import { chromium } from 'playwright';

function parseCellFile(filePath: string): string[] {
  const raw = fs.readFileSync(filePath, 'utf-8');
  return raw.split(/\n# ---\n/).map((c) => c.trim()).filter(Boolean);
}

function encodeCells(codes: string[]): string {
  // base64**url**, not base64. The value travels in a query string, where
  // `URLSearchParams.get` decodes `+` as a space — so a plain base64 payload
  // that happens to contain a `+` arrives corrupted, `atob` throws, and the
  // notebook quietly falls back to the default starter cells. The recording
  // then looks fine and shows the wrong notebook.
  return Buffer.from(JSON.stringify(codes), 'utf-8')
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

export async function recordCommand(
  opts: {
    code?: string;
    codeLeft?: string;
    codeRight?: string;
    output: string;
    url: string;
    serverUrl: string;
    width: string;
    height: string;
    delay: string;
    layout?: string;
    headless?: boolean;
    hideLineNumbers?: boolean;
    pace?: string;
  },
) {
  // Every wait in this function is expressed as a base duration times `pace`.
  // The defaults were tuned for a quick smoke check; a demo someone is meant to
  // read needs two or three times as long on each frame.
  const pace = Number(opts.pace) || 1;
  const hold = (ms: number) => delay(Math.round(ms * pace));
  const outputPath = path.resolve(opts.output);
  const outputDir = path.dirname(outputPath);
  fs.mkdirSync(outputDir, { recursive: true });

  const layout = opts.layout ?? (opts.codeLeft || opts.codeRight ? 'split' : 'single');
  const isSplit = layout === 'split';
  const width = Number(opts.width) || (isSplit ? 1920 : 1280);
  const height = Number(opts.height) || (isSplit ? 1080 : 720);

  console.log(chalk.bold('\nRecording notebook demo'));
  console.log(chalk.dim(`  Layout:     ${layout}`));
  console.log(chalk.dim(`  URL:        ${opts.url}`));
  console.log(chalk.dim(`  Server:     ${opts.serverUrl}`));
  console.log(chalk.dim(`  Viewport:   ${width}x${height}`));
  console.log(chalk.dim(`  Output:     ${outputPath}`));

  try {
    const res = await fetch(`${opts.serverUrl}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    console.log(chalk.green('  Server:     online\n'));
  } catch (e) {
    console.error(chalk.red(`\n✗ Python server not reachable at ${opts.serverUrl}/health`));
    console.error(chalk.dim(`  Error: ${e}`));
    console.error(chalk.dim('  Start the server first: pnpm start   or   bash server/start.sh'));
    process.exit(1);
  }

  const videoDir = fs.mkdtempSync('/tmp/alkahest-rec-');
  const headless = opts.headless ?? !process.env.DISPLAY;
  console.log(chalk.dim(`  Headless: ${headless}`));

  let targetUrl = opts.url;
  let numCells = 0;
  const recordingQs = new URLSearchParams();
  recordingQs.set('zen', '1');
  if (opts.hideLineNumbers) recordingQs.set('hideLineNumbers', '1');
  const recordingSuffix = `?${recordingQs.toString()}`;

  if (isSplit) {
    const leftFile = opts.codeLeft ?? opts.code;
    const rightFile = opts.codeRight ?? opts.code;
    if (!leftFile || !rightFile) {
      console.error(chalk.red('\n✗ Split layout requires --code-left and --code-right (or --code for both).'));
      process.exit(1);
    }
    const leftCells = parseCellFile(leftFile);
    const rightCells = parseCellFile(rightFile);
    numCells = leftCells.length + rightCells.length;
    const leftEnc = encodeCells(leftCells);
    const rightEnc = encodeCells(rightCells);
    const base = opts.url.replace(/\/$/, '');
    const compareQs = new URLSearchParams(recordingQs);
    compareQs.set('left', leftEnc);
    compareQs.set('right', rightEnc);
    compareQs.set('mode', 'server');
    compareQs.set('autorun', '1');
    targetUrl = `${base}/compare?${compareQs.toString()}`;
    console.log(chalk.dim(`  Left cells:  ${leftCells.length}`));
    console.log(chalk.dim(`  Right cells: ${rightCells.length}\n`));
  } else if (opts.code) {
    const cellCodes = parseCellFile(opts.code);
    numCells = cellCodes.length;
    const encoded = encodeCells(cellCodes);
    const demoQs = new URLSearchParams(recordingQs);
    demoQs.set('demo', encoded);
    demoQs.set('mode', 'server');
    demoQs.set('autorun', '1');
    targetUrl = `${opts.url}?${demoQs.toString()}`;
    console.log(chalk.dim(`  Cells:    ${numCells}\n`));
  } else {
    targetUrl = `${opts.url}${recordingSuffix}`;
  }

  const browser = await chromium.launch({ headless });
  const context = await browser.newContext({
    viewport: { width, height },
    recordVideo: {
      dir: videoDir,
      size: { width, height },
    },
  });

  const page = await context.newPage();
  await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: 30_000 });

  // Wait until zen layout is applied and demo cells are rendered (avoids capturing
  // the navbar or empty starter notebook in the first frames).
  await page.waitForSelector('[data-recording-ready="true"]', { timeout: 30_000 }).catch(async () => {
    await page.waitForSelector('.cm-editor', { timeout: 20_000 });
  });
  console.log(chalk.cyan('  UI ready'));

  const codeCellCount = isSplit
    ? numCells
    : await page.locator('.cm-editor').count();
  if (codeCellCount > 0) {
    await page.waitForFunction(
      (n) => document.querySelectorAll('.cm-editor').length >= n,
      codeCellCount,
      { timeout: 20_000 },
    ).catch(() => {});
  }

  await hold(800);
  console.log(chalk.cyan('  Running cells…'));

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

  // "Nothing is running" is also true *before* the first cell starts and in the
  // gap between two cells, so waiting on that alone stops the recording after
  // cell one and captures a notebook full of unexecuted cells. A run is over
  // when no cell is running *and* no code cell is still idle.
  await page.waitForFunction(() => {
    const running = document.querySelectorAll('[data-cell-status="running"]').length;
    const pending = document.querySelectorAll(
      '[data-cell-type="code"][data-cell-status="idle"]',
    ).length;
    return running === 0 && pending === 0;
  }, { timeout: 180_000, polling: 500 }).catch((err) => {
    if (!serverDied) {
      console.log(chalk.yellow(`  Warning: stopped waiting for cells — ${err}`));
    }
  });

  // A demo may show an error on purpose (alkahest refusing something is worth
  // filming), so this reports rather than fails — but a recording that quietly
  // captured a traceback is worth knowing about before you publish it.
  const errored = await page.$$eval('[data-cell-status="error"]', (els) => els.length);
  if (errored > 0) {
    console.log(chalk.yellow(`  Note: ${errored} cell(s) finished with an error`));
  }

  clearInterval(healthInterval);

  if (serverDied) {
    await context.close();
    await browser.close();
    process.exit(1);
  }

  await hold(2000);

  console.log(chalk.cyan('  Scrolling to show all content…'));
  const pageHeight = await page.evaluate(() => document.body.scrollHeight);
  const viewportHeight = height;
  if (pageHeight > viewportHeight) {
    const scrollSteps = Math.ceil((pageHeight - viewportHeight) / 50);
    for (let i = 0; i < scrollSteps; i++) {
      await page.evaluate(() => window.scrollBy(0, 50));
      await hold(50);
    }
    if (isSplit) {
      // For split layout: hold at bottom showing outputs, then end
      await hold(4000);
    } else {
      await hold(1500);
      for (let i = scrollSteps; i > 0; i--) {
        await page.evaluate(() => window.scrollBy(0, -50));
        await hold(35);
      }
      await hold(500);
    }
  }

  console.log(chalk.green('  All cells done — holding final frame'));
  await hold(isSplit ? 500 : 2500);

  await context.close();
  await browser.close();

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
