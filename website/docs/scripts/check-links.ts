// Internal link checker over dist/: every href/src under the base path must
// resolve to a file, and every #fragment to an id on the target page.
//   bun scripts/check-links.ts          (exit 1 on any broken link)
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const DIST = join(import.meta.dir, "..", "dist");
const BASE = (process.env.DOCS_BASE ?? "/docs/").replace(/\/?$/, "/");

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith(".html") ? [p] : [];
  });
}

const files = walk(DIST);
const idCache = new Map<string, Set<string>>();
function ids(file: string): Set<string> {
  let s = idCache.get(file);
  if (!s) {
    s = new Set();
    const html = readFileSync(file, "utf8");
    for (const m of html.matchAll(/\sid="([^"]+)"/g)) s.add(decodeEntities(m[1]));
    idCache.set(file, s);
  }
  return s;
}
function decodeEntities(s: string) {
  return s.replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&#39;/g, "'").replace(/&amp;/g, "&");
}
function urlFor(file: string): string {
  const rel = relative(DIST, file).replace(/\\/g, "/");
  return BASE + rel.replace(/index\.html$/, "");
}
function target(path: string): string | null {
  let f = join(DIST, path.slice(BASE.length));
  if (path.endsWith("/")) f = join(f, "index.html");
  if (existsSync(f) && statSync(f).isFile()) return f;
  if (existsSync(join(f, "index.html"))) return join(f, "index.html");
  return null;
}

let broken = 0;
let checked = 0;
const report = new Map<string, string[]>();
for (const file of files) {
  const html = readFileSync(file, "utf8");
  if (html.includes('http-equiv="refresh"')) continue;
  const here = urlFor(file);
  for (const m of html.matchAll(/\s(?:href|src)="([^"]+)"/g)) {
    let href = decodeEntities(m[1]);
    if (/^(https?:|mailto:|data:|javascript:)/.test(href)) continue;
    checked++;
    let [path, frag] = href.split("#") as [string, string | undefined];
    path = path.split("?")[0];
    if (!path) path = here;
    else if (!path.startsWith("/")) path = new URL(path, "http://x" + here).pathname;
    if (!path.startsWith(BASE)) {
      // links to the landing page / playground live outside the docs tree
      if (["/", "/playground/", "/agent/"].includes(path)) continue;
      (report.get(here) ?? report.set(here, []).get(here)!).push(`outside base: ${href}`);
      broken++;
      continue;
    }
    const t = target(decodeURIComponent(path));
    if (!t) {
      (report.get(here) ?? report.set(here, []).get(here)!).push(`missing page: ${href}`);
      broken++;
      continue;
    }
    if (frag && frag !== "top" && t.endsWith(".html")) {
      const id = decodeURIComponent(frag);
      if (!ids(t).has(id)) {
        (report.get(here) ?? report.set(here, []).get(here)!).push(`missing anchor: ${href}`);
        broken++;
      }
    }
  }
}
for (const [page, errs] of report) {
  console.log(page);
  for (const e of [...new Set(errs)]) console.log("   ", e);
}
console.log(`${files.length} pages, ${checked} internal links checked, ${broken} broken`);
process.exit(broken ? 1 : 0);
