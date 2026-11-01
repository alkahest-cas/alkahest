// Page shell: top bar (shared with the landing page and playground, see
// DESIGN.md), sidebar, content column, "On this page" TOC.

import { escapeHtml, type Heading } from "./markdown";

export type NavItem = { title: string; href: string; children?: NavItem[]; external?: boolean };
export type NavGroup = { title: string; items: NavItem[] };

export type PageLink = { title: string; href: string };

export type Page = {
  path: string; // output path relative to dist, e.g. "guide/calculus/index.html"
  href: string; // URL, e.g. "/docs/guide/calculus/"
  title: string; // <h1>
  docTitle?: string; // <title> prefix; defaults to title
  description: string;
  eyebrow?: string;
  section: "home" | "guide" | "api";
  body: string;
  headings: Heading[];
  prev?: PageLink;
  next?: PageLink;
  editUrl?: string;
  math?: boolean;
  wide?: boolean; // home page: no TOC column
};

export const GITHUB_REPO = "https://github.com/alkahest-cas/alkahest";

const GITHUB_ICON =
  '<svg viewBox="0 0 16 16" width="20" height="20" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>';

const MENU_ICON =
  '<svg viewBox="0 0 20 20" width="20" height="20" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" d="M3 6h14M3 10h14M3 14h14"/></svg>';
const SEARCH_ICON =
  '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><circle cx="9" cy="9" r="5.5" fill="none" stroke="currentColor" stroke-width="1.6"/><path d="m13.2 13.2 3.8 3.8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
const SIDEBAR_ICON =
  '<svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><rect x="3" y="4" width="14" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 4v12" stroke="currentColor" stroke-width="1.5"/></svg>';
const ARROW_EXT =
  '<svg class="ext" viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><path d="M4 2h6v6M10 2 3 9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
const EDIT_ICON =
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M11.013 1.427a1.75 1.75 0 0 1 2.474 0l1.086 1.086a1.75 1.75 0 0 1 0 2.474l-8.61 8.61c-.21.21-.47.364-.756.445l-3.251.93a.75.75 0 0 1-.927-.928l.929-3.25c.081-.286.235-.547.445-.758l8.61-8.61Zm.176 4.823L9.75 4.81l-6.286 6.287a.253.253 0 0 0-.064.108l-.558 1.953 1.953-.558a.253.253 0 0 0 .108-.064Zm1.238-3.763a.25.25 0 0 0-.354 0L10.811 3.75l1.439 1.44 1.263-1.263a.25.25 0 0 0 0-.354Z"/></svg>';

export const FAVICON =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#c41e3a"/><text x="16" y="23" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-weight="700" font-size="21" fill="#fff">a</text></svg>',
  );

// The root of the GitHub Pages site (the landing page lives at "/").
function siteRoot(base: string): string {
  return base.replace(/docs\/$/, "");
}

export function topBar(base: string): string {
  const root = siteRoot(base);
  return `<header class="topbar">
  <div class="topbar-inner">
    <a class="wordmark" href="${root}">alkahest</a>
    <nav class="topnav" id="topnav" aria-label="Site">
      <a href="${base}" class="current" aria-current="page">Docs</a>
      <a href="${root}playground/">Playground</a>
      <a href="${root}agent/">Agent</a>
      <a class="gh" href="${GITHUB_REPO}" aria-label="GitHub">${GITHUB_ICON}</a>
    </nav>
    <button class="menu-btn" type="button" aria-label="Menu" aria-expanded="false" aria-controls="topnav">${MENU_ICON}</button>
  </div>
</header>`;
}

function navList(items: NavItem[], current: string, depth = 0): string {
  return (
    `<ul class="nav-list${depth ? " nested" : ""}">` +
    items
      .map((it) => {
        const active = !it.external && it.href === current;
        const inTrail = it.children?.some((c) => c.href === current);
        const a = `<a href="${it.href}"${active ? ' class="active" aria-current="page"' : ""}${it.external ? ' rel="noopener"' : ""}>${escapeHtml(it.title)}${it.external ? ARROW_EXT : ""}</a>`;
        const kids = it.children?.length ? navList(it.children, current, depth + 1) : "";
        return `<li${inTrail ? ' class="open"' : ""}>${a}${kids}</li>`;
      })
      .join("") +
    `</ul>`
  );
}

function navGroups(groups: NavGroup[], current: string): string {
  return groups
    .map((g) => `<div class="nav-group"><div class="nav-title">${escapeHtml(g.title)}</div>${navList(g.items, current)}</div>`)
    .join("");
}

export function sidebar(guide: NavGroup[], api: NavGroup[], page: Page, base: string): string {
  const onApi = page.section === "api";
  return `<aside class="sidebar" id="sidebar" aria-label="Documentation">
  <div class="sidebar-top">
    <button class="search-btn" type="button" data-search-open>${SEARCH_ICON}<span>Search docs</span><kbd class="kbd-hint">⌘K</kbd></button>
    <div class="side-tabs" role="tablist">
      <a href="${base}" role="tab" data-tab="guide" class="${onApi ? "" : "selected"}" aria-selected="${!onApi}">Guide</a>
      <a href="${base}api/" role="tab" data-tab="api" class="${onApi ? "selected" : ""}" aria-selected="${onApi}">API reference</a>
    </div>
  </div>
  <div class="sidebar-scroll">
    <nav class="side-nav" data-panel="guide"${onApi ? " hidden" : ""}>${navGroups(guide, page.href)}</nav>
    <nav class="side-nav" data-panel="api"${onApi ? "" : " hidden"}>${navGroups(api, page.href)}</nav>
  </div>
  <div class="sidebar-foot">
    <div class="theme-switch" role="radiogroup" aria-label="Theme">
      <button type="button" data-theme-set="system" aria-label="System theme" title="System"><svg viewBox="0 0 16 16" width="14" height="14"><rect x="1.5" y="2.5" width="13" height="9" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M5.5 14h5" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg></button>
      <button type="button" data-theme-set="light" aria-label="Light theme" title="Light"><svg viewBox="0 0 16 16" width="14" height="14"><circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3.05 3.05l1.13 1.13M11.82 11.82l1.13 1.13M3.05 12.95l1.13-1.13M11.82 4.18l1.13-1.13" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg></button>
      <button type="button" data-theme-set="dark" aria-label="Dark theme" title="Dark"><svg viewBox="0 0 16 16" width="14" height="14"><path d="M13.5 9.6A5.8 5.8 0 0 1 6.4 2.5a5.8 5.8 0 1 0 7.1 7.1Z" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg></button>
    </div>
    <a class="side-foot-link" href="https://pypi.org/project/alkahest/" rel="noopener">PyPI${ARROW_EXT}</a>
    <a class="side-foot-link" href="https://docs.rs/alkahest-cas" rel="noopener">docs.rs${ARROW_EXT}</a>
  </div>
</aside>`;
}

function toc(headings: Heading[], editUrl?: string): string {
  const hs = headings.filter((h) => h.depth <= 3);
  if (!hs.length && !editUrl) return `<aside class="toc"></aside>`;
  return `<aside class="toc" aria-label="On this page">
  <div class="toc-inner">
    ${hs.length ? `<div class="toc-title">On this page</div><ul>${hs.map((h) => `<li class="d${h.depth}"><a href="#${h.id}">${escapeHtml(h.text)}</a></li>`).join("")}</ul>` : ""}
    ${editUrl ? `<a class="toc-edit" href="${editUrl}" rel="noopener">${EDIT_ICON}Edit this page</a>` : ""}
    <a class="toc-top" href="#top">Back to top</a>
  </div>
</aside>`;
}

function pager(p: Page): string {
  if (!p.prev && !p.next) return "";
  const cell = (l: PageLink | undefined, dir: "prev" | "next") =>
    l
      ? `<a class="pager-${dir}" href="${l.href}" rel="${dir}"><span class="pager-dir">${dir === "prev" ? "Previous" : "Next"}</span><span class="pager-title">${escapeHtml(l.title)}</span></a>`
      : "<span></span>";
  return `<nav class="pager" aria-label="Pagination">${cell(p.prev, "prev")}${cell(p.next, "next")}</nav>`;
}

export function renderPage(p: Page, guide: NavGroup[], api: NavGroup[], base: string, version: string): string {
  const docTitle = `${p.docTitle ?? p.title} · Alkahest Docs`;
  const head = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(docTitle)}</title>
<meta name="description" content="${escapeHtml(p.description)}">
<meta property="og:title" content="${escapeHtml(docTitle)}">
<meta property="og:description" content="${escapeHtml(p.description)}">
<meta property="og:type" content="website">
<meta name="theme-color" content="#f7f6f3" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#111110" media="(prefers-color-scheme: dark)">
<link rel="icon" href="${FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="${base}assets/style.css?v=${version}">
${p.math ? `<link rel="stylesheet" href="${base}assets/katex/katex.min.css">` : ""}
<script>try{var t=localStorage.getItem("alkahest-theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}</script>
<script defer src="${base}assets/app.js?v=${version}"></script>
</head>`;
  const body = `<body data-base="${base}" data-section="${p.section}">
<a class="skip" href="#content">Skip to content</a>
${topBar(base)}
<div class="subbar">
  <button class="subbar-btn" type="button" data-drawer-open aria-controls="sidebar">${SIDEBAR_ICON}<span>${p.section === "api" ? "API reference" : "Guide"}</span></button>
  <button class="subbar-btn" type="button" data-search-open aria-label="Search">${SEARCH_ICON}<span>Search</span></button>
</div>
<div class="layout${p.wide ? " layout-wide" : ""}" id="top">
${sidebar(guide, api, p, base)}
<div class="scrim" data-drawer-close></div>
<main class="main" id="content">
  <article class="doc${p.section === "api" ? " doc-api" : ""}${p.wide ? " doc-home" : ""}">
    ${p.eyebrow ? `<div class="eyebrow">${escapeHtml(p.eyebrow)}</div>` : ""}
    ${p.wide ? "" : `<h1>${escapeHtml(p.title)}</h1>`}
    ${p.body}
  </article>
  ${pager(p)}
  <footer class="doc-foot">
    ${p.editUrl ? `<a href="${p.editUrl}" rel="noopener">${EDIT_ICON}Edit on GitHub</a>` : "<span></span>"}
    <span class="foot-meta">Alkahest ${escapeHtml(version)} · <a href="https://areggevorgyan.github.io/" rel="noopener">Built by Areg</a></span>
  </footer>
</main>
${p.wide ? "" : toc(p.headings, p.editUrl)}
</div>
<div class="search" id="search" hidden role="dialog" aria-modal="true" aria-label="Search">
  <div class="search-backdrop" data-search-close></div>
  <div class="search-panel">
    <div class="search-field">${SEARCH_ICON}<input id="search-input" type="search" placeholder="Search guides and API" autocomplete="off" spellcheck="false" aria-label="Search"><button type="button" class="search-close" data-search-close aria-label="Close search"><kbd class="hide-sm">Esc</kbd><span class="show-sm">Cancel</span></button></div>
    <ul class="search-results" id="search-results" role="listbox"></ul>
    <div class="search-foot"><span><kbd>↑</kbd><kbd>↓</kbd> to navigate</span><span><kbd>↵</kbd> to open</span></div>
  </div>
</div>
</body>
</html>
`;
  return head + "\n" + body;
}
