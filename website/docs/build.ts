#!/usr/bin/env bun
// Static documentation site for Alkahest.
//
//   bun run build            → website/docs/dist/  (served under /docs/)
//   DOCS_BASE=/ bun run build  for a site rooted elsewhere
//
// Inputs: docs/mdbook/src/ (guides, ordered by SUMMARY.md), the Python API
// extracted statically by scripts/extract_api.py, and content/ (home page
// samples, all runnable: see scripts/check_examples.py).

import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, normalize, relative, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { ApiRenderer, type ApiData, type SearchEntry } from "./lib/api";
import { escapeHtml, firstParagraphText, highlight, initHighlighter, renderMarkdown, stripTags, truncate } from "./lib/markdown";
import { GITHUB_REPO, renderPage, type NavGroup, type NavItem, type Page } from "./lib/layout";

const HERE = import.meta.dir;
const REPO = resolve(HERE, "../..");
const SRC = join(REPO, "docs/mdbook/src");
const DIST = join(HERE, "dist");
const BASE = (process.env.DOCS_BASE ?? "/docs/").replace(/\/?$/, "/");
const EDIT = `${GITHUB_REPO}/edit/main/`;
const BLOB = `${GITHUB_REPO}/blob/main/`;
const TREE = `${GITHUB_REPO}/tree/main/`;

const cargo = await readFile(join(REPO, "Cargo.toml"), "utf8");
const VERSION = cargo.match(/^version\s*=\s*"([^"]+)"/m)?.[1] ?? "dev";

// ---------------------------------------------------------------------------
// Guide structure: SUMMARY.md order, with section labels that partition it
// (the order is never changed, only labelled).
// ---------------------------------------------------------------------------

type GuideEntry = { file: string; slug: string; title: string; depth: number; children: GuideEntry[] };

const summary = await readFile(join(SRC, "SUMMARY.md"), "utf8");
const flat: GuideEntry[] = [];
const roots: GuideEntry[] = [];
for (const line of summary.split("\n")) {
  const m = line.match(/^(\s*)- \[([^\]]+)\]\(\.?\/?([^)]+\.md)\)/);
  if (!m) continue;
  const depth = Math.floor(m[1].length / 2);
  const e: GuideEntry = { file: m[3], slug: m[3].replace(/\.md$/, ""), title: m[2], depth, children: [] };
  flat.push(e);
  if (depth === 0) roots.push(e);
  else roots[roots.length - 1].children.push(e);
}

const SECTIONS: [string, string[]][] = [
  ["Get started", ["intro", "getting-started", "python-api"]],
  ["Core concepts", ["kernel", "representations", "parsing"]],
  ["Simplification", ["simplification"]],
  ["Mathematics", ["calculus", "transformations", "probability", "vector-quaternion", "units", "tensor", "controls", "orthopoly", "noncommutative", "specfun"]],
  ["Numerics and codegen", ["codegen", "ball-arithmetic", "ode-dae"]],
  ["Solving and optimisation", ["solving", "positivity", "linear-programming"]],
  ["Ecosystem", ["interop", "rl"]],
  ["Trust and verification", ["derivations", "claim-graphs", "lean-certs", "errors"]],
  ["Agent loops", ["search-plumbing"]],
  ["Project", ["stability"]],
];
{
  const listed = new Set(SECTIONS.flatMap(([, s]) => s));
  const missing = roots.filter((r) => !listed.has(r.slug));
  if (missing.length) SECTIONS.push(["More", missing.map((r) => r.slug)]);
}

// python-api.md is the doorway to the generated reference: it lives at /api/.
const guideHref = (slug: string) => (slug === "python-api" ? `${BASE}api/` : `${BASE}guide/${slug}/`);
const slugs = new Set(flat.map((e) => e.slug));

// ---------------------------------------------------------------------------
// Link rewriting
// ---------------------------------------------------------------------------

const OLD_SITE = /^https?:\/\/alkahest-cas\.github\.io\/alkahest\/?(.*)$/;

function rewriteOldSite(rest: string): string {
  const [path, hash = ""] = rest.split("#");
  const h = hash ? `#${hash}` : "";
  if (!path || path === "index.html") return `${BASE}${h}`;
  if (path.startsWith("rustdoc")) return "https://docs.rs/alkahest-cas";
  const api = path.match(/^api\/(?:api\/)?([\w-]+)?(?:\.html)?\/?$/);
  if (api) {
    const name = api[1];
    if (!name || name === "index" || name === "api") return `${BASE}api/${h}`;
    return `${BASE}api/${name}/${h}`;
  }
  const page = path.match(/^([\w-]+)\.html$/);
  if (page && slugs.has(page[1])) return guideHref(page[1]) + h;
  return `${BASE}${h}`;
}

function resolveLink(href: string, fromDir: string = SRC): string {
  if (!href) return href;
  if (href.startsWith("~/")) return BASE + href.slice(2);
  const old = href.match(OLD_SITE);
  if (old) return rewriteOldSite(old[1]);
  if (/^(https?:|mailto:|#)/.test(href)) return href;
  const [path, hash = ""] = href.split("#");
  const h = hash ? `#${hash}` : "";
  const abs = normalize(join(fromDir, path));
  const relToSrc = relative(SRC, abs);
  if (!relToSrc.startsWith("..") && relToSrc.endsWith(".md")) {
    const slug = relToSrc.replace(/\.md$/, "");
    if (slugs.has(slug)) return guideHref(slug) + h;
  }
  const relToRepo = relative(REPO, abs);
  if (relToRepo.startsWith("..")) return href;
  const isDir = existsSync(abs) && !/\.\w+$/.test(abs);
  return (isDir ? TREE : BLOB) + relToRepo + h;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

await initHighlighter();
await rm(DIST, { recursive: true, force: true });
await mkdir(DIST, { recursive: true });

const pages: Page[] = [];
const search: SearchEntry[] = [];

// --- API data ---------------------------------------------------------------
const cacheDir = join(HERE, ".cache");
await mkdir(cacheDir, { recursive: true });
const apiJson = join(cacheDir, "api.json");
const py = process.env.PYTHON ?? "python3";
const ex = spawnSync(py, [join(HERE, "scripts/extract_api.py"), REPO, apiJson], { stdio: ["ignore", "inherit", "inherit"] });
if (ex.status !== 0) throw new Error("extract_api.py failed");
const api: ApiData = JSON.parse(await readFile(apiJson, "utf8"));

const API_GROUP_TITLES: Record<string, string> = {
  Core: "Core",
  Advanced: "Advanced",
  More: "More",
  Submodules: "Submodules",
};
const apiNav: NavGroup[] = [{ title: "Overview", items: [{ title: "Python API", href: `${BASE}api/` }] }];
for (const g of ["Core", "Advanced", "Submodules", "More"]) {
  const items: NavItem[] = api.pages
    .filter((p) => p.group === g)
    .map((p) => ({ title: shortApiTitle(p.title), href: `${BASE}api/${p.slug}/` }));
  if (items.length) apiNav.push({ title: API_GROUP_TITLES[g], items });
}
apiNav.push({
  title: "Elsewhere",
  items: [
    { title: "Error codes", href: `${BASE}guide/errors/` },
    { title: "Rust API on docs.rs", href: "https://docs.rs/alkahest-cas", external: true },
  ],
});

function shortApiTitle(t: string): string {
  return t.replace(/\s+API$/, "").replace(/^Bessel, Airy, zeta and polylogarithm functions$/, "Special functions");
}

// --- guide nav ----------------------------------------------------------------
const bySlug = new Map(flat.map((e) => [e.slug, e]));
const guideNav: NavGroup[] = SECTIONS.map(([title, ss]) => ({
  title,
  items: ss
    .map((s) => bySlug.get(s))
    .filter((e): e is GuideEntry => !!e)
    .map((e) => ({
      title: e.title,
      href: guideHref(e.slug),
      children: e.children.map((c) => ({ title: c.title, href: guideHref(c.slug) })),
    })),
}));

// Reading order for prev/next: home, then SUMMARY order.
const order: { title: string; href: string }[] = [{ title: "Overview", href: BASE }, ...flat.map((e) => ({ title: e.title, href: guideHref(e.slug) }))];
function neighbours(href: string) {
  const i = order.findIndex((o) => o.href === href);
  return { prev: i > 0 ? order[i - 1] : undefined, next: i >= 0 && i < order.length - 1 ? order[i + 1] : undefined };
}

// --- guide pages ------------------------------------------------------------
const apiIntro: { html: string; title: string } = { html: "", title: "Python API reference" };
for (const e of flat) {
  let md = await readFile(join(SRC, e.file), "utf8");
  if (e.slug === "python-api") {
    // The chapter's opening points at the Sphinx site this reference replaces.
    md = md.replace(/^The Sphinx-generated[\s\S]*?(?=^Conceptual chapters)/m, "");
  }
  const r = renderMarkdown(md, { resolveLink: (h) => resolveLink(h, dirname(join(SRC, e.file))) });
  const title = r.title || e.title;
  const href = guideHref(e.slug);
  const desc = truncate(firstParagraphText(md) || `${title} — Alkahest documentation.`, 158);
  indexGuide(md, title, href);
  if (e.slug === "python-api") {
    apiIntro.html = r.html;
    continue; // rendered as the API overview below
  }
  const group = SECTIONS.find(([, ss]) => ss.includes(e.slug) || ss.includes(flat.find((f) => f.children.includes(e))?.slug ?? ""))?.[0];
  pages.push({
    path: `guide/${e.slug}/index.html`,
    href,
    title,
    description: desc,
    eyebrow: group,
    section: "guide",
    body: r.html,
    headings: r.headings,
    editUrl: `${EDIT}docs/mdbook/src/${e.file}`,
    math: r.math,
    ...neighbours(href),
  });
}

// --- API pages ----------------------------------------------------------------
const apiR = new ApiRenderer(BASE, (h) => resolveLink(h));
const apiOrder = api.pages.map((p) => ({ title: shortApiTitle(p.title), href: `${BASE}api/${p.slug}/` }));
const SPHINX_FILES = new Set(["core", "simplify", "diff", "poly", "numerics", "transform", "pari", "units", "tensor", "controls", "orthopoly", "noncommutative", "specfun", "matrix", "ode", "solve", "linprog", "logic", "codegen", "workload", "autoresearch", "errors"]);
api.pages.forEach((p, i) => {
  const r = apiR.page(p);
  const href = `${BASE}api/${p.slug}/`;
  search.push(...r.search);
  const n = p.sections.reduce((a, s) => a + s.items.reduce((b, it) => b + (it.t === "syms" ? it.syms.length : 0), 0), 0);
  const desc = truncate(
    r.description ? stripTags(r.description) : `Python API reference for ${p.title.replace(/ API$/, "").toLowerCase()} in Alkahest: ${n} documented symbols with signatures, parameters and source links.`,
    158,
  );
  pages.push({
    path: `api/${p.slug}/index.html`,
    href,
    title: p.title,
    docTitle: `${p.title}${/API|alkahest\./.test(p.title) ? "" : " API"}`,
    description: desc,
    eyebrow: p.group === "Submodules" ? "Python API · Submodule" : "Python API",
    section: "api",
    body: r.html,
    headings: r.headings,
    editUrl: SPHINX_FILES.has(p.slug) ? `${EDIT}docs/sphinx/api/${p.slug}.rst` : `${BLOB}website/docs/scripts/extract_api.py`,
    math: r.math,
    prev: i > 0 ? apiOrder[i - 1] : { title: "Python API", href: `${BASE}api/` },
    next: i < apiOrder.length - 1 ? apiOrder[i + 1] : undefined,
  });
});

// --- API overview -------------------------------------------------------------
{
  const cards = apiNav
    .filter((g) => g.title !== "Overview" && g.title !== "Elsewhere")
    .map(
      (g) =>
        `<h2 id="${g.title.toLowerCase()}">${escapeHtml(g.title)}</h2><div class="ref-grid">` +
        g.items
          .map((it) => {
            const p = api.pages.find((x) => `${BASE}api/${x.slug}/` === it.href)!;
            const n = p.sections.reduce((a, s) => a + s.items.reduce((b, x) => b + (x.t === "syms" ? x.syms.filter((y) => !("ref" in y)).length : 0), 0), 0);
            return `<a class="ref-card" href="${it.href}"><span class="ref-title">${escapeHtml(it.title)}</span><span class="ref-count">${n} symbol${n === 1 ? "" : "s"}</span></a>`;
          })
          .join("") +
        `</div>`,
    )
    .join("\n");
  const s = api.stats;
  const body = `<p class="lead">Every public name in <code>alkahest.__all__</code>, with its signature, docstring and a link to the source that defines it.</p>
${apiIntro.html}
<div class="callout callout-note"><div class="callout-title">How this reference is built</div><p>It is generated statically from the source tree, without importing the compiled extension: Python modules are read with <code>ast</code>, and the native functions' docstrings come from the PyO3 doc comments in <code>alkahest-py/src/</code>, which is exactly the text <code>help()</code> shows. ${s.public} public names, ${s.with_doc} of ${s.symbols} entries with a docstring. The Rust crate is documented on <a href="https://docs.rs/alkahest-cas" rel="noopener">docs.rs</a>.</p></div>
${cards}`;
  pages.push({
    path: "api/index.html",
    href: `${BASE}api/`,
    title: "Python API reference",
    docTitle: "Python API reference",
    description: `Reference for every public name in the alkahest Python package: ${s.public} functions, classes and constants with signatures and docstrings.`,
    eyebrow: "Python API",
    section: "api",
    body,
    headings: apiNav.filter((g) => g.title !== "Overview" && g.title !== "Elsewhere").map((g) => ({ depth: 2, id: g.title.toLowerCase(), text: g.title })),
    editUrl: `${EDIT}docs/mdbook/src/python-api.md`,
    next: apiOrder[0],
  });
}

// --- Home -------------------------------------------------------------------
{
  const sample = async (f: string) => highlight(await readFile(join(HERE, "content", f), "utf8"), "python");
  const card = (slug: string, title: string, text: string) =>
    `<a class="card" href="${guideHref(slug)}"><span class="card-title">${escapeHtml(title)}</span><span class="card-text">${escapeHtml(text)}</span></a>`;
  const body = `
<section class="hero">
  <div class="eyebrow">Documentation</div>
  <h1>Alkahest documentation</h1>
  <p class="lead">Alkahest is a computer algebra system with a Rust core and a Python API, built for both humans and agents. Every computation produces a derivation log, a growing subset exports Lean 4 proofs, and when it cannot answer it refuses with a coded error.</p>
  <div class="hero-links">
    <a class="btn btn-primary" href="${guideHref("getting-started")}">Get started</a>
    <a class="btn" href="${BASE}api/">Python API reference</a>
  </div>
</section>

<div class="callout callout-important"><div class="callout-title">Beta</div><p>The API is still moving and Alkahest will contain bugs, including answers that are wrong without saying so. Treat a result that matters as unverified until you have checked it: read the derivation log, export the Lean certificate where one exists, or cross-check against an independent source.</p></div>

<h2 id="install">Install</h2>
<div class="tabs" data-tabs>
  <div class="tab-list" role="tablist">
    <button type="button" role="tab" class="selected" aria-selected="true" data-tab-target="install-py">Python</button>
    <button type="button" role="tab" aria-selected="false" data-tab-target="install-rs">Rust</button>
  </div>
  <div class="tab-panel" id="install-py" role="tabpanel">
    ${highlight("pip install alkahest", "bash")}
    <p class="muted">Python 3.9 to 3.14. The default wheel includes the e-graph simplifier, the Gröbner solver, the Cranelift JIT and the multi-core paths. LLVM JIT builds are on <a href="https://github.com/alkahest-cas/alkahest/releases" rel="noopener">GitHub Releases</a>; see <a href="${guideHref("getting-started")}">Getting started</a>.</p>
  </div>
  <div class="tab-panel" id="install-rs" role="tabpanel" hidden>
    ${highlight("cargo add alkahest-cas", "bash")}
    <p class="muted">The crate is <code>alkahest-cas</code>. It needs FLINT 3.3 or newer, GMP and MPFR at build time (<code>apt install libflint-dev libgmp-dev libmpfr-dev</code>, <code>brew install flint</code>). API docs are on <a href="https://docs.rs/alkahest-cas" rel="noopener">docs.rs</a>.</p>
  </div>
</div>

<h2 id="quickstart">Quickstart</h2>
<p>Every computation starts with an <code>ExprPool</code>, which owns the expressions. Operations return a <code>DerivedResult</code> whose <code>.value</code> is the answer and whose <code>.steps</code> record how it was reached.</p>
${await sample("quickstart.py")}

<div class="tabs" data-tabs>
  <div class="tab-list" role="tablist">
    <button type="button" role="tab" class="selected" aria-selected="true" data-tab-target="ex-solve">Solve a system</button>
    <button type="button" role="tab" aria-selected="false" data-tab-target="ex-codegen">Compile to native code</button>
    <button type="button" role="tab" aria-selected="false" data-tab-target="ex-budget">Budgets and Lean</button>
  </div>
  <div class="tab-panel" id="ex-solve" role="tabpanel">${await sample("solve.py")}</div>
  <div class="tab-panel" id="ex-codegen" role="tabpanel" hidden>${await sample("codegen.py")}</div>
  <div class="tab-panel" id="ex-budget" role="tabpanel" hidden>${await sample("budget.py")}</div>
</div>

<h2 id="guides">Guides</h2>
<div class="cards">
  ${card("calculus", "Calculus", "Differentiation, integration with verification, limits, series and asymptotics of sums.")}
  ${card("solving", "Solving", "Polynomial and rational systems by Gröbner bases, with numeric and certified roots.")}
  ${card("simplification", "Simplification", "A rule-based rewriter and an e-graph saturation engine over the same expression pool.")}
  ${card("specfun", "Special functions", "Bessel, Airy, zeta and polylogarithm functions, with DLMF conventions.")}
  ${card("codegen", "Codegen and JIT", "Compile expressions to native code, C, StableHLO or CUDA; evaluate over NumPy arrays.")}
  ${card("lean-certs", "Lean certificates", "Emit Lean 4 source for derivations, and see which operations can be certified.")}
  ${card("budgets", "Budgets and cancellation", "Bound every call by time, steps or memory, cancel it from outside, and keep runs deterministic.")}
  ${card("errors", "Error handling", "A structured exception hierarchy where every error carries a stable code and a remediation hint.")}
  ${card("ball-arithmetic", "Ball arithmetic", "Rigorous interval evaluation with Arb, and global bounds on boxes.")}
  ${card("ode-dae", "ODEs and DAEs", "Symbolic ODE and DAE modelling, structural analysis and automatic index reduction.")}
  ${card("positivity", "Positivity certificates", "Sum-of-squares and Putinar certificates that are re-verified exactly.")}
  ${card("search-plumbing", "Agent loops", "Batch fan-out, compact result envelopes, claim graphs and cross-CAS testing for search loops.")}
</div>

<h2 id="reference">Reference</h2>
<div class="ref-grid">
  <a class="ref-card" href="${BASE}api/"><span class="ref-title">Python API</span><span class="ref-count">${api.stats.public} public names</span></a>
  <a class="ref-card" href="${guideHref("errors")}"><span class="ref-title">Error codes</span><span class="ref-count">E-SUBSYSTEM-NNN</span></a>
  <a class="ref-card" href="${guideHref("stability")}"><span class="ref-title">Stability policy</span><span class="ref-count">What may change</span></a>
  <a class="ref-card" href="https://docs.rs/alkahest-cas" rel="noopener"><span class="ref-title">Rust API ↗</span><span class="ref-count">docs.rs/alkahest-cas</span></a>
</div>
`;
  pages.push({
    path: "index.html",
    href: BASE,
    title: "Alkahest documentation",
    docTitle: "Overview",
    description:
      "Documentation for Alkahest, a computer algebra system with a Rust core and Python API: install, quickstart, guides on calculus, solving, codegen and Lean certificates, and the API reference.",
    section: "home",
    wide: true,
    body,
    headings: [
      { depth: 2, id: "install", text: "Install" },
      { depth: 2, id: "quickstart", text: "Quickstart" },
      { depth: 2, id: "guides", text: "Guides" },
      { depth: 2, id: "reference", text: "Reference" },
    ],
    editUrl: `${BLOB}website/docs/build.ts`,
    next: order[1],
  });
  search.push({ t: "Install", u: `${BASE}#install`, s: "pip install alkahest · cargo add alkahest-cas", k: "section", p: "Overview" });
  search.push({ t: "Quickstart", u: `${BASE}#quickstart`, s: "ExprPool, diff, integrate, simplify, parse, limit", k: "section", p: "Overview" });
}

// --- search index for guides --------------------------------------------------
function indexGuide(md: string, title: string, href: string) {
  // chunk by h2/h3; strip code blocks to keep the index small
  const lines = md
    .replace(/```[\s\S]*?```/g, " ")
    .split("\n")
    .filter((l) => !/^\s*(\||<|-{3,}|={3,})/.test(l)) // tables, raw HTML, rules
    .map((l) => l.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1").replace(/^\s*>\s?/, ""));
  let cur = { h: title, id: "", text: [] as string[] };
  const flush = () => {
    const text = cur.text.join(" ").replace(/[`*|#]/g, "").replace(/(^|\s)_([^_]+)_(?=\s|$)/g, "$1$2").replace(/\(([^)]*\.md[^)]*|https?:[^)]*)\)/g, "").replace(/\s+/g, " ").trim();
    search.push({ t: cur.h, u: href + (cur.id ? `#${cur.id}` : ""), s: truncate(text, 180) || undefined, k: cur.id ? "heading" : "page", p: cur.id ? title : undefined });
  };
  const used = new Map<string, number>();
  for (const l of lines) {
    const m = l.match(/^(#{2,3})\s+(.*)$/);
    if (m) {
      flush();
      const raw = m[2].replace(/`/g, "");
      let id = raw.trim().toLowerCase().replace(/\s/g, "-").replace(/[^\p{L}\p{N}_-]/gu, "");
      const n = used.get(id) ?? 0;
      used.set(id, n + 1);
      if (n) id = `${id}-${n}`;
      cur = { h: raw.replace(/\*\*/g, ""), id, text: [] };
    } else if (!/^#\s/.test(l)) cur.text.push(l);
  }
  flush();
}

// --- write ------------------------------------------------------------------
let mathUsed = false;
for (const p of pages) {
  mathUsed ||= !!p.math;
  const out = join(DIST, p.path);
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, renderPage(p, guideNav, apiNav, BASE, VERSION));
}
// legacy-style redirect stubs so /docs/guide/ and old mdBook names resolve
await mkdir(join(DIST, "guide"), { recursive: true });
await writeFile(join(DIST, "guide/index.html"), redirect(guideHref("intro")));
for (const e of flat) {
  await writeFile(join(DIST, `${e.slug}.html`), redirect(guideHref(e.slug)));
}

await mkdir(join(DIST, "assets"), { recursive: true });
await cp(join(HERE, "assets"), join(DIST, "assets"), { recursive: true });
if (mathUsed) {
  const kdir = join(HERE, "node_modules/katex/dist");
  await mkdir(join(DIST, "assets/katex"), { recursive: true });
  await cp(join(kdir, "katex.min.css"), join(DIST, "assets/katex/katex.min.css"));
  await cp(join(kdir, "fonts"), join(DIST, "assets/katex/fonts"), { recursive: true });
}
await writeFile(join(DIST, "search-index.json"), JSON.stringify(search));
await writeFile(join(DIST, ".nojekyll"), "");

function redirect(to: string) {
  return `<!doctype html><meta charset="utf-8"><title>Redirecting…</title><link rel="canonical" href="${to}"><meta http-equiv="refresh" content="0; url=${to}"><meta name="robots" content="noindex"><p><a href="${to}">${to}</a></p>`;
}

console.log(`docs: ${pages.length} pages, ${search.length} search entries → ${relative(process.cwd(), DIST) || "."} (base ${BASE})`);
