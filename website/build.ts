#!/usr/bin/env bun
// Sanity-check the static site in src/ (served as-is by GitHub Pages):
// every local href/src must resolve to a file, and benchmarks.json must parse.
import { readdir, readFile, stat } from "node:fs/promises";
import { join, relative } from "node:path";

const root = join(import.meta.dir, "src");
const files: string[] = [];
async function walk(dir: string) {
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) await walk(p);
    else files.push(p);
  }
}
await walk(root);

// Paths served by other deployments into the same Pages site.
const external = ["/docs/", "/playground/"];
let bad = 0;
for (const f of files.filter((f) => f.endsWith(".html"))) {
  const html = await readFile(f, "utf8");
  for (const [, url] of html.matchAll(/(?:href|src)="(\/[^"#?]*)/g)) {
    if (external.some((x) => url.startsWith(x))) continue;
    const target = join(root, url.endsWith("/") ? url + "index.html" : url);
    if (!(await stat(target).catch(() => null))) {
      console.error(`${relative(root, f)}: broken link ${url}`);
      bad++;
    }
  }
}
JSON.parse(await readFile(join(root, "data/benchmarks.json"), "utf8"));
if (bad) process.exit(1);
console.warn(`ok: ${files.length} files, no broken local links`);
