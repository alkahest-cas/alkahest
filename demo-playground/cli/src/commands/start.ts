import { spawn, type ChildProcess } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '../../../');

export async function startCommand(opts: {
  webPort: string;
  serverPort: string;
  open: boolean;
}) {
  console.log(chalk.bold('\nalkahest demo playground\n'));

  const children: ChildProcess[] = [];

  function cleanup() {
    children.forEach((c) => c.kill());
    process.exit(0);
  }
  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);

  // ── Python server ────────────────────────────────────────────────────────
  console.log(chalk.cyan('→ Starting Python execution server on port ' + opts.serverPort));
  const serverProc = spawn('bash', ['server/start.sh'], {
    cwd: ROOT,
    env: { ...process.env, PORT: opts.serverPort },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  children.push(serverProc);

  serverProc.stdout?.on('data', (d: Buffer) =>
    process.stdout.write(chalk.dim('[server] ') + d.toString()),
  );
  serverProc.stderr?.on('data', (d: Buffer) =>
    process.stderr.write(chalk.dim('[server] ') + d.toString()),
  );

  // ── Next.js web app ──────────────────────────────────────────────────────
  console.log(chalk.cyan('→ Starting web app on port ' + opts.webPort));

  // Ensure .env.local exists with the right server URL
  const envContent = [
    `NEXT_PUBLIC_PYTHON_WS_URL=ws://localhost:${opts.serverPort}`,
    `NEXT_PUBLIC_PYTHON_HTTP_URL=http://localhost:${opts.serverPort}`,
    `PYTHON_SERVER_URL=http://localhost:${opts.serverPort}`,
  ].join('\n');

  const fs = await import('fs');
  fs.writeFileSync(path.join(ROOT, 'web/.env.local'), envContent);

  const webProc = spawn('pnpm', ['--filter', 'web', 'dev', '-p', opts.webPort], {
    cwd: ROOT,
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
    shell: true,
  });
  children.push(webProc);

  let webReady = false;
  webProc.stdout?.on('data', (d: Buffer) => {
    const line = d.toString();
    process.stdout.write(chalk.dim('[web] ') + line);
    if (!webReady && line.includes('localhost:')) {
      webReady = true;
      const url = `http://localhost:${opts.webPort}`;
      console.log(chalk.green(`\n✓ Playground ready at ${chalk.bold(url)}\n`));
      if (opts.open) openBrowser(url);
    }
  });
  webProc.stderr?.on('data', (d: Buffer) =>
    process.stderr.write(chalk.dim('[web] ') + d.toString()),
  );

  // Keep alive
  await new Promise(() => {});
}

function openBrowser(url: string) {
  const { platform } = process;
  const cmd =
    platform === 'darwin' ? 'open' :
    platform === 'win32' ? 'start' :
    'xdg-open';
  spawn(cmd, [url], { stdio: 'ignore', detached: true }).unref();
}
