#!/usr/bin/env node
import { Command } from 'commander';
import chalk from 'chalk';
import path from 'path';
import { fileURLToPath } from 'url';
import { startCommand } from './commands/start.js';
import { recordCommand } from './commands/record.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Resolve demo-videos/ relative to the repo's demo-playground root, not cwd
const VIDEOS_DIR = path.resolve(__dirname, '../../../demo-videos');

const program = new Command();

program
  .name('alkahest-demo')
  .description(
    chalk.bold('Alkahest Demo Playground CLI') +
    '\nLaunch, record, and orchestrate demos of the Alkahest CAS library.',
  )
  .version('0.1.0');

program
  .command('start')
  .description('Start the web app and Python execution server')
  .option('-p, --web-port <port>', 'Next.js port', '3000')
  .option('-s, --server-port <port>', 'Python server port', '8000')
  .option('--no-open', 'Do not open the browser')
  .action(startCommand);

program
  .command('record')
  .description('Record a scripted notebook demo as a video')
  .option('-c, --code <file>', 'Python file to inject into cells')
  .option('-o, --output <file>', 'Output video file', path.join(VIDEOS_DIR, `alkahest-demo-${Date.now()}.webm`))
  .option('--url <url>', 'Playground URL', 'http://localhost:3000')
  .option('--server-url <url>', 'Python execution server URL (for health checks)', 'http://localhost:8000')
  .option('--layout <mode>', 'Recording layout: single or split (side-by-side compare)', 'single')
  .option('--code-left <file>', 'Left panel cells (split layout); split on "# ---"')
  .option('--code-right <file>', 'Right panel cells (split layout); split on "# ---"')
  .option('--width <px>', 'Viewport width (default 1920 for split, 1280 for single)', '')
  .option('--height <px>', 'Viewport height (default 1080 for split, 720 for single)', '')
  .option('--delay <ms>', 'Delay between typing characters (ms)', '40')
  .option('--hide-line-numbers', 'Hide code cell line numbers in the recording (?hideLineNumbers=1)')
  .option('--pace <multiplier>', 'Scale every pause and scroll step; >1 is slower and more readable', '1')
  .action(recordCommand);

program.parse();
