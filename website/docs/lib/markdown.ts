// Markdown → HTML for guide pages and docstrings.
//
// One `Marked` instance per render call collects headings (for the
// "On this page" TOC) and rewrites links through a caller-supplied resolver.

import { Marked, type Tokens } from "marked";
import { createHighlighter, type Highlighter } from "shiki";
import katex from "katex";

export type Heading = { depth: number; id: string; text: string };

let hl: Highlighter | null = null;
const LANGS = ["python", "bash", "json", "toml", "rust", "c", "lean", "lisp", "text"];
const ALIASES: Record<string, string> = {
  py: "python",
  python3: "python",
  pycon: "python",
  sh: "bash",
  shell: "bash",
  console: "bash",
  zsh: "bash",
  smt: "lisp",
  smt2: "lisp",
  smtlib: "lisp",
  lean4: "lean",
  "": "text",
  txt: "text",
  plain: "text",
  mlir: "text",
};

export async function initHighlighter() {
  hl = await createHighlighter({
    themes: ["github-light", "github-dark-dimmed"],
    langs: LANGS.filter((l) => l !== "text"),
  });
}

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function stripTags(s: string): string {
  return s
    .replace(/<[^>]+>/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}

const COPY_ICON =
  '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25ZM1.75 5H3v1.5H1.75a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25V13H10v1.25A1.75 1.75 0 0 1 8.25 16h-7.5A1.75 1.75 0 0 1 0 14.25v-7.5C0 5.784.784 5 1.75 5Z"/></svg>';

const LANG_LABEL: Record<string, string> = {
  python: "Python",
  bash: "Shell",
  json: "JSON",
  toml: "TOML",
  rust: "Rust",
  c: "C",
  lean: "Lean",
  lisp: "SMT-LIB",
  text: "",
};

export function highlight(code: string, lang0: string): string {
  const raw = (lang0 || "").trim().split(/\s+/)[0].toLowerCase();
  let lang = ALIASES[raw] ?? raw;
  if (!LANGS.includes(lang)) lang = "text";
  const label = raw === "smt" || raw === "smt2" ? "SMT-LIB" : (LANG_LABEL[lang] ?? raw);
  code = code.replace(/\n+$/, "");
  let inner: string;
  if (hl && lang !== "text") {
    inner = hl.codeToHtml(code, {
      lang,
      themes: { light: "github-light", dark: "github-dark-dimmed" },
      defaultColor: false,
    });
  } else {
    inner = `<pre class="shiki"><code>${escapeHtml(code)}</code></pre>`;
  }
  return (
    `<div class="code" data-lang="${lang}">` +
    (label ? `<span class="code-lang">${label}</span>` : "") +
    `<button class="copy" type="button" aria-label="Copy code">${COPY_ICON}<span>Copy</span></button>` +
    inner +
    `</div>`
  );
}

// mdBook's heading-id scheme (so old `#anchors` keep working).
export function slugify(text: string): string {
  return stripTags(text)
    .trim()
    .toLowerCase()
    .replace(/\s/g, "-")
    .replace(/[^\p{L}\p{N}_-]/gu, "");
}

export function renderMath(html: string): { html: string; used: boolean } {
  let used = false;
  const unesc = (s: string) => stripTags(s);
  html = html.replace(/<div class="kx-display">([\s\S]*?)<\/div>/g, (_, tex) => {
    used = true;
    try {
      return `<div class="math-display">${katex.renderToString(unesc(tex), { displayMode: true, throwOnError: false })}</div>`;
    } catch {
      return `<pre>${tex}</pre>`;
    }
  });
  html = html.replace(/<span class="kx">([\s\S]*?)<\/span>/g, (_, tex) => {
    used = true;
    try {
      return katex.renderToString(unesc(tex), { throwOnError: false });
    } catch {
      return `<code>${tex}</code>`;
    }
  });
  // $$ ... $$ display blocks in guide markdown (outside code, already escaped)
  html = html.replace(/<p>\$\$([\s\S]+?)\$\$<\/p>/g, (_, tex) => {
    used = true;
    return `<div class="math-display">${katex.renderToString(unesc(tex), { displayMode: true, throwOnError: false })}</div>`;
  });
  return { html, used };
}

export type RenderOpts = {
  resolveLink: (href: string) => string;
  collectHeadings?: boolean;
  idPrefix?: string;
  shiftHeadings?: number;
};

export function renderMarkdown(md: string, opts: RenderOpts): { html: string; headings: Heading[]; title: string; math: boolean } {
  const headings: Heading[] = [];
  const used = new Map<string, number>();
  let title = "";
  const m = new Marked({ gfm: true });
  m.use({
    renderer: {
      heading(text: string, level: number, raw: string) {
        if (level === 1 && !title) {
          title = stripTags(text);
          return ""; // the page template renders the h1
        }
        let id = (opts.idPrefix ?? "") + slugify(raw.replace(/`/g, ""));
        const n = used.get(id) ?? 0;
        used.set(id, n + 1);
        if (n) id = `${id}-${n}`;
        const lvl = Math.min(6, level + (opts.shiftHeadings ?? 0));
        if (opts.collectHeadings !== false && (level === 2 || level === 3)) headings.push({ depth: level, id, text: stripTags(text) });
        return `<h${lvl} id="${id}"><a class="hanchor" href="#${id}" aria-hidden="true" tabindex="-1">#</a>${text}</h${lvl}>\n`;
      },
      code(code: string, info: string | undefined) {
        return highlight(code, info ?? "");
      },
      link(href: string, t: string | null, text: string) {
        const h = opts.resolveLink(href);
        const ext = /^https?:\/\//.test(h);
        return `<a href="${escapeHtml(h)}"${t ? ` title="${escapeHtml(t)}"` : ""}${ext ? ' rel="noopener"' : ""}>${text}</a>`;
      },
      table(header: string, body: string) {
        return `<div class="table-wrap"><table><thead>${header}</thead><tbody>${body}</tbody></table></div>\n`;
      },
      blockquote(quote: string) {
        // GitHub-style alerts: > [!NOTE] / > [!IMPORTANT] / > [!WARNING]
        const mm = quote.match(/^<p>\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/);
        if (mm) {
          const kind = mm[1].toLowerCase();
          const body = quote.replace(mm[0], "<p>");
          const label = kind[0].toUpperCase() + kind.slice(1);
          return `<div class="callout callout-${kind}"><div class="callout-title">${label}</div>${body}</div>\n`;
        }
        return `<blockquote>${quote}</blockquote>\n`;
      },
    },
  });
  // raw HTML in markdown (e.g. README-style <h3 align>) is kept as-is.
  let html = m.parse(md, { async: false }) as string;
  const r = renderMath(html);
  return { html: r.html, headings, title, math: r.used };
}

export function firstParagraphText(md: string): string {
  const tokens = new Marked().lexer(md);
  for (const t of tokens) {
    if (t.type === "paragraph") {
      const txt = stripTags(new Marked().parseInline((t as Tokens.Paragraph).text, { async: false }) as string)
        .replace(/\s+/g, " ")
        .trim();
      if (txt.length > 30) return txt;
    }
  }
  return "";
}

export function truncate(s: string, n: number): string {
  if (s.length <= n) return s;
  const cut = s.slice(0, n - 1);
  const sp = cut.lastIndexOf(" ");
  return (sp > n * 0.6 ? cut.slice(0, sp) : cut).replace(/[,.;:—\s]+$/, "") + "…";
}
