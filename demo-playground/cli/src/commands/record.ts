import path from 'path';
import fs from 'fs';
import chalk from 'chalk';
import { chromium } from 'playwright';

function parseCellFile(filePath: string): string[] {
  const raw = fs.readFileSync(filePath, 'utf-8');
  return raw.split(/\n# ---\n/).map((c) => c.trim()).filter(Boolean);
}

function encodeCells(codes: string[]): string {
  return Buffer.from(JSON.stringify(codes)).toString('base64');
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
  },
) {
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
    targetUrl = `${base}/compare?left=${leftEnc}&right=${rightEnc}&mode=server&zen=1&autorun=1`;
    console.log(chalk.dim(`  Left cells:  ${leftCells.length}`));
    console.log(chalk.dim(`  Right cells: ${rightCells.length}\n`));
  } else if (opts.code) {
    const cellCodes = parseCellFile(opts.code);
    numCells = cellCodes.length;
    const encoded = encodeCells(cellCodes);
    targetUrl = `${opts.url}?demo=${encoded}&mode=server&zen=1&autorun=1`;
    console.log(chalk.dim(`  Cells:    ${numCells}\n`));
  } else {
    targetUrl = `${opts.url}?zen=1`;
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

  await delay(800);
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

  await page.waitForFunction(() => {
    return document.querySelectorAll('[data-cell-status="running"]').length === 0;
  }, { timeout: 180_000, polling: 500 }).catch(() => {
    if (!serverDied) console.log(chalk.yellow('  Warning: timed out waiting for cells to finish'));
  });

  clearInterval(healthInterval);

  if (serverDied) {
    await context.close();
    await browser.close();
    process.exit(1);
  }

  await delay(1500);

  console.log(chalk.cyan('  Scrolling to show all content…'));
  const pageHeight = await page.evaluate(() => document.body.scrollHeight);
  const viewportHeight = height;
  if (pageHeight > viewportHeight) {
    const scrollSteps = Math.ceil((pageHeight - viewportHeight) / 50);
    for (let i = 0; i < scrollSteps; i++) {
      await page.evaluate(() => window.scrollBy(0, 50));
      await delay(50);
    }
    if (isSplit) {
      // For split layout: hold at bottom showing outputs, then end
      await delay(4000);
    } else {
      await delay(1500);
      for (let i = scrollSteps; i > 0; i--) {
        await page.evaluate(() => window.scrollBy(0, -50));
        await delay(35);
      }
      await delay(500);
    }
  }

  console.log(chalk.green('  All cells done — holding final frame'));
  await delay(isSplit ? 500 : 2500);

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
