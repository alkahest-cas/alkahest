// Python API reference pages, rendered from the JSON emitted by
// scripts/extract_api.py.

import { escapeHtml, renderMarkdown, slugify, type Heading } from "./markdown";

export type Block =
  | { t: "md"; md: string; title?: string }
  | { t: "params"; title: string; items: { name: string; type: string; desc: string }[] };

export type Member = {
  qualname: string;
  name: string;
  kind: string;
  signature: string | null;
  returns?: string | null;
  blocks: Block[];
  summary?: string;
  source: string | null;
};

export type Sym = Member & {
  value?: string | null;
  bases?: string[] | null;
  origin?: string;
  from_reference?: boolean;
  see_below?: string;
  members: Member[];
};

export type ApiItem = { t: "prose"; md: string } | { t: "syms"; syms: (Sym | { ref: string })[] };
export type ApiSection = { title: string | null; items: ApiItem[] };
export type ApiPage = { slug: string; group: string; title: string; sections: ApiSection[] };
export type ApiData = { pages: ApiPage[]; stats: Record<string, number> };

export type SearchEntry = { t: string; u: string; s?: string; k: string; p?: string };

const KIND_LABEL: Record<string, string> = {
  function: "function",
  class: "class",
  exception: "exception",
  data: "constant",
  module: "module",
  method: "method",
  property: "property",
  attribute: "attribute",
  staticmethod: "staticmethod",
  classmethod: "classmethod",
};

function splitTop(s: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = "";
  let q: string | null = null;
  for (const ch of s) {
    if (q) {
      cur += ch;
      if (ch === q) q = null;
      continue;
    }
    if (ch === '"' || ch === "'") q = ch;
    if ("([{".includes(ch)) depth++;
    if (")]}".includes(ch)) depth--;
    if (ch === "," && depth === 0) {
      out.push(cur.trim());
      cur = "";
    } else cur += ch;
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

function renderParam(p: string): string {
  if (p === "*" || p === "/") return `<span class="sig-sep">${p}</span>`;
  const eq = p.match(/^([^=:]+?)(\s*:\s*[^=]+?)?(\s*=\s*.+)?$/);
  if (!eq) return escapeHtml(p);
  const [, name, ann, def] = eq;
  return (
    `<span class="sig-p">${escapeHtml(name.trim())}</span>` +
    (ann ? `<span class="sig-ann">: ${escapeHtml(ann.replace(/^\s*:\s*/, ""))}</span>` : "") +
    (def ? `<span class="sig-def">=${escapeHtml(def.replace(/^\s*=\s*/, ""))}</span>` : "")
  );
}

export function renderSignature(prefix: string, name: string, sig: string | null, kind: string, returns?: string | null): string {
  const pre = prefix ? `<span class="sig-prefix">${escapeHtml(prefix)}</span>` : "";
  const nm = `<span class="sig-name">${escapeHtml(name)}</span>`;
  if (kind === "property" || kind === "attribute" || kind === "data" || kind === "module") {
    return `${pre}${nm}${returns ? `<span class="sig-ann">: ${escapeHtml(returns)}</span>` : ""}`;
  }
  if (!sig) return `${pre}${nm}`;
  const m = sig.match(/^\((.*)\)\s*(?:->\s*(.+))?$/s);
  if (!m) return `${pre}${nm}${escapeHtml(sig)}`;
  const params = splitTop(m[1]);
  const ret = m[2] ? `<span class="sig-ret"> → ${escapeHtml(m[2])}</span>` : "";
  const flat = params.join(", ");
  if (flat.length + name.length < 64) {
    return `${pre}${nm}(${params.map(renderParam).join(", ")})${ret}`;
  }
  return `${pre}${nm}(\n${params.map((p) => "    " + renderParam(p)).join(",\n")}\n)${ret}`;
}

export class ApiRenderer {
  constructor(
    private base: string,
    private resolveLink: (href: string) => string,
  ) {}

  md(s: string): { html: string; math: boolean } {
    const r = renderMarkdown(s, { resolveLink: this.resolveLink, collectHeadings: false, shiftHeadings: 2 });
    return { html: r.html, math: r.math };
  }

  blocks(blocks: Block[]): { html: string; math: boolean } {
    let math = false;
    const html = blocks
      .map((b) => {
        if (b.t === "md") {
          const r = this.md(b.md);
          math ||= r.math;
          return (b.title ? `<div class="doc-sub">${escapeHtml(b.title)}</div>` : "") + r.html;
        }
        const items = b.items
          .map((it) => {
            const d = it.desc ? this.md(it.desc) : { html: "", math: false };
            math ||= d.math;
            const typ = it.type ? this.md(it.type).html.replace(/^<p>|<\/p>\s*$/g, "") : "";
            return `<div class="param"><dt>${it.name ? `<code class="pname">${escapeHtml(it.name)}</code>` : ""}${typ ? `<span class="ptype">${typ}</span>` : ""}</dt>${d.html ? `<dd>${d.html}</dd>` : ""}</div>`;
          })
          .join("");
        return `<div class="doc-sub">${escapeHtml(b.title)}</div><dl class="params">${items}</dl>`;
      })
      .join("\n");
    return { html, math };
  }

  page(p: ApiPage): { html: string; headings: Heading[]; math: boolean; search: SearchEntry[]; description: string } {
    const headings: Heading[] = [];
    const search: SearchEntry[] = [];
    let math = false;
    let description = "";
    const url = `${this.base}api/${p.slug}/`;
    const out: string[] = [];
    const usedIds = new Set<string>();
    for (const sec of p.sections) {
      if (sec.title) {
        let id = slugify(sec.title);
        if (usedIds.has(id)) id += "-section";
        usedIds.add(id);
        headings.push({ depth: 2, id, text: sec.title });
        out.push(`<h2 id="${id}"><a class="hanchor" href="#${id}" aria-hidden="true" tabindex="-1">#</a>${escapeHtml(sec.title)}</h2>`);
        search.push({ t: sec.title, u: `${url}#${id}`, k: "section", p: p.title });
      }
      for (const it of sec.items) {
        if (it.t === "prose") {
          const r = this.md(it.md);
          math ||= r.math;
          if (!description) description = it.md.replace(/[`*_\[\]]|\(http[^)]*\)/g, "").split("\n\n")[0].replace(/\s+/g, " ");
          out.push(`<div class="prose-block">${r.html}</div>`);
          continue;
        }
        for (const s of it.syms) {
          if ("ref" in s) continue;
          const r = this.sym(s, url, p.title, search, headings);
          math ||= r.math;
          out.push(r.html);
        }
      }
    }
    return { html: out.join("\n"), headings, math, search, description };
  }

  sym(s: Sym, url: string, pageTitle: string, search: SearchEntry[], headings: Heading[]): { html: string; math: boolean } {
    const id = s.qualname;
    const prefix = s.qualname.includes(".") ? s.qualname.slice(0, s.qualname.lastIndexOf(".") + 1) : "";
    let math = false;
    const last = headings[headings.length - 1];
    if (!(last && last.depth === 2 && last.text === s.qualname)) headings.push({ depth: 3, id, text: s.qualname });
    search.push({ t: s.qualname, u: `${url}#${id}`, s: s.summary || undefined, k: s.kind, p: pageTitle });
    const kind = KIND_LABEL[s.kind] ?? s.kind;
    const sigPrefix = s.kind === "class" || s.kind === "exception" ? `${s.kind === "exception" ? "exception" : "class"} ` : "";
    const sig = renderSignature(prefix, s.name, s.signature, s.kind);
    const bases = s.bases?.length ? `<span class="sig-bases">(${s.bases.map(escapeHtml).join(", ")})</span>` : "";
    const head = `<div class="sym-head">
  <span class="kind kind-${s.kind}">${kind}</span>
  <pre class="sig"><code>${sigPrefix ? `<span class="sig-kw">${sigPrefix}</span>` : ""}${sig}${s.signature ? "" : bases}</code></pre>
  <div class="sym-links"><a class="hanchor-inline" href="#${escapeHtml(id)}" aria-label="Link to ${escapeHtml(s.qualname)}">#</a>${s.source ? `<a class="src" href="${escapeHtml(s.source)}" rel="noopener">source</a>` : ""}</div>
</div>`;
    let body = "";
    if (s.value != null && s.kind === "data") {
      body += `<pre class="sig sig-value"><code>${escapeHtml(s.name)} = ${escapeHtml(s.value)}</code></pre>`;
    }
    if (s.blocks.length) {
      const b = this.blocks(s.blocks);
      math ||= b.math;
      body += b.html;
    } else if (s.see_below) {
      body += `<p class="muted">Documented together with <a href="#${escapeHtml(s.see_below)}"><code>${escapeHtml(s.see_below)}</code></a>.</p>`;
    } else if (s.kind === "module") {
      body += `<p>See the <a href="${this.base}api/mod-${s.name.replace(/_/g, "-")}/"><code>alkahest.${escapeHtml(s.name)}</code> reference</a>.</p>`;
    } else {
      body += `<p class="muted nodoc">No docstring.</p>`;
    }
    if (s.members?.length) {
      const props = s.members.filter((m) => m.kind === "property" || m.kind === "attribute");
      const meths = s.members.filter((m) => !(m.kind === "property" || m.kind === "attribute"));
      const renderMembers = (ms: Member[]) =>
        ms
          .map((m) => {
            search.push({ t: m.qualname, u: `${url}#${m.qualname}`, s: m.summary || undefined, k: m.kind, p: pageTitle });
            const b = m.blocks.length ? this.blocks(m.blocks) : { html: "", math: false };
            math ||= b.math;
            const label = m.kind === "staticmethod" || m.kind === "classmethod" ? `<span class="kind kind-small">${m.kind}</span>` : "";
            return `<div class="member" id="${escapeHtml(m.qualname)}">
  <div class="member-head"><pre class="sig sig-member"><code>${renderSignature("", m.name, m.signature, m.kind, m.returns)}</code></pre>${label}<div class="sym-links"><a class="hanchor-inline" href="#${escapeHtml(m.qualname)}" aria-label="Link to ${escapeHtml(m.qualname)}">#</a>${m.source ? `<a class="src" href="${escapeHtml(m.source)}" rel="noopener">source</a>` : ""}</div></div>
  ${b.html ? `<div class="member-body">${b.html}</div>` : ""}
</div>`;
          })
          .join("\n");
      if (props.length) body += `<div class="members"><div class="members-title">Properties</div>${renderMembers(props)}</div>`;
      if (meths.length) body += `<div class="members"><div class="members-title">Methods</div>${renderMembers(meths)}</div>`;
    }
    return { html: `<section class="sym" id="${escapeHtml(id)}">${head}<div class="sym-body">${body}</div></section>`, math };
  }
}
