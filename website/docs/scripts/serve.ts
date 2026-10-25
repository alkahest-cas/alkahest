// Preview server: serves dist/ under /docs/ (the production base path).
//   bun scripts/serve.ts [port]
import { existsSync, statSync } from "node:fs";
import { join, normalize } from "node:path";

const DIST = join(import.meta.dir, "..", "dist");
const BASE = "/docs/";
const port = Number(process.argv[2] ?? process.env.PORT ?? 4321);

Bun.serve({
  port,
  fetch(req) {
    const url = new URL(req.url);
    let p = decodeURIComponent(url.pathname);
    if (p === "/" || p === "/docs") return Response.redirect(BASE, 302);
    if (!p.startsWith(BASE)) return new Response("Not under /docs/ (the landing page is served separately)", { status: 404 });
    p = normalize(p.slice(BASE.length));
    if (p.startsWith("..")) return new Response("Bad path", { status: 400 });
    let f = join(DIST, p);
    if (existsSync(f) && statSync(f).isDirectory()) {
      if (!url.pathname.endsWith("/")) return Response.redirect(url.pathname + "/", 301);
      f = join(f, "index.html");
    }
    if (!existsSync(f)) return new Response("Not found", { status: 404 });
    return new Response(Bun.file(f));
  },
});
console.log(`http://localhost:${port}${BASE}`);
