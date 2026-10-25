// Alkahest docs: theme, copy buttons, tabs, drawer, TOC scrollspy, search.
(() => {
  "use strict";
  const BASE = document.body.dataset.base || "/docs/";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch {} },
  };

  // ---- theme ----------------------------------------------------------------
  function applyTheme(mode) {
    const root = document.documentElement;
    if (mode === "light" || mode === "dark") root.dataset.theme = mode;
    else delete root.dataset.theme;
    $$("[data-theme-set]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.themeSet === (mode || "system"))));
    $$("[data-theme-set]").forEach((b) => b.setAttribute("role", "radio"));
  }
  applyTheme(store.get("alkahest-theme") || "system");
  $$("[data-theme-set]").forEach((b) =>
    b.addEventListener("click", () => {
      const m = b.dataset.themeSet;
      store.set("alkahest-theme", m === "system" ? null : m);
      applyTheme(m);
    }),
  );

  // ---- top-bar menu (mobile) -------------------------------------------------
  const menuBtn = $(".menu-btn");
  const topnav = $("#topnav");
  menuBtn?.addEventListener("click", () => {
    const open = topnav.classList.toggle("open");
    menuBtn.setAttribute("aria-expanded", String(open));
  });

  // ---- sidebar drawer + tabs ---------------------------------------------------
  const body = document.body;
  const openDrawer = () => { body.classList.add("drawer-open"); $(".sidebar a.active")?.scrollIntoView({ block: "center" }); };
  const closeDrawer = () => body.classList.remove("drawer-open");
  $$("[data-drawer-open]").forEach((b) => b.addEventListener("click", openDrawer));
  $$("[data-drawer-close]").forEach((b) => b.addEventListener("click", closeDrawer));
  $$(".sidebar a").forEach((a) => a.addEventListener("click", closeDrawer));
  $$(".side-tabs [data-tab]").forEach((tab) =>
    tab.addEventListener("click", (e) => {
      e.preventDefault();
      $$(".side-tabs [data-tab]").forEach((t) => {
        const on = t === tab;
        t.classList.toggle("selected", on);
        t.setAttribute("aria-selected", String(on));
      });
      $$(".side-nav").forEach((p) => (p.hidden = p.dataset.panel !== tab.dataset.tab));
    }),
  );
  // keep the active item visible in the sidebar
  const active = $(".sidebar-scroll a.active");
  if (active) {
    const sc = $(".sidebar-scroll");
    const r = active.getBoundingClientRect(), s = sc.getBoundingClientRect();
    if (r.bottom > s.bottom - 40 || r.top < s.top) sc.scrollTop += r.top - s.top - s.height / 3;
  }

  // ---- content tabs ----------------------------------------------------------------
  $$("[data-tabs]").forEach((group) => {
    const btns = $$("[data-tab-target]", group);
    btns.forEach((b) =>
      b.addEventListener("click", () => {
        btns.forEach((x) => {
          const on = x === b;
          x.classList.toggle("selected", on);
          x.setAttribute("aria-selected", String(on));
          const panel = document.getElementById(x.dataset.tabTarget);
          if (panel) panel.hidden = !on;
        });
      }),
    );
  });

  // ---- copy buttons ------------------------------------------------------------------
  $$(".code .copy").forEach((btn) =>
    btn.addEventListener("click", async () => {
      const code = btn.parentElement.querySelector("pre code")?.innerText ?? "";
      try {
        await navigator.clipboard.writeText(code.replace(/\n$/, ""));
      } catch {
        const ta = document.createElement("textarea");
        ta.value = code;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
      }
      const label = btn.querySelector("span");
      btn.classList.add("copied");
      if (label) label.textContent = "Copied";
      setTimeout(() => { btn.classList.remove("copied"); if (label) label.textContent = "Copy"; }, 1600);
    }),
  );

  // ---- TOC scrollspy -----------------------------------------------------------------
  const tocLinks = $$(".toc li a");
  if (tocLinks.length && "IntersectionObserver" in window) {
    const byId = new Map(tocLinks.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
    const targets = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
    let current = null;
    const setActive = (id) => {
      if (id === current) return;
      current = id;
      tocLinks.forEach((a) => a.classList.toggle("active", a === byId.get(id)));
      const a = byId.get(id);
      const box = $(".toc-inner");
      if (a && box) {
        const r = a.getBoundingClientRect(), b = box.getBoundingClientRect();
        if (r.top < b.top + 40 || r.bottom > b.bottom - 40) box.scrollTop += r.top - b.top - b.height / 3;
      }
    };
    const update = () => {
      const y = 110;
      let best = targets[0];
      for (const t of targets) {
        if (t.getBoundingClientRect().top - y <= 0) best = t;
        else break;
      }
      if (best) setActive(best.id);
    };
    let ticking = false;
    window.addEventListener("scroll", () => {
      if (!ticking) { ticking = true; requestAnimationFrame(() => { update(); ticking = false; }); }
    }, { passive: true });
    update();
  }

  // ---- search ----------------------------------------------------------------------------
  const dlg = $("#search");
  const input = $("#search-input");
  const list = $("#search-results");
  let index = null;
  let loading = null;
  let results = [];
  let sel = 0;
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
  $$(".kbd-hint").forEach((k) => (k.textContent = isMac ? "⌘K" : "Ctrl K"));

  function load() {
    if (index) return Promise.resolve(index);
    if (!loading)
      loading = fetch(BASE + "search-index.json")
        .then((r) => r.json())
        .then((d) => {
          index = d.map((e) => ({ ...e, tl: e.t.toLowerCase(), sl: (e.s || "").toLowerCase(), pl: (e.p || "").toLowerCase() }));
          return index;
        });
    return loading;
  }
  function open() {
    dlg.hidden = false;
    body.classList.add("search-open");
    closeDrawer();
    input.value = "";
    input.focus();
    load().then(() => render(""));
  }
  function close() {
    dlg.hidden = true;
    body.classList.remove("search-open");
  }
  $$("[data-search-open]").forEach((b) => b.addEventListener("click", open));
  $$("[data-search-close]").forEach((b) => b.addEventListener("click", close));
  document.addEventListener("keydown", (e) => {
    if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      dlg.hidden ? open() : close();
    } else if (e.key === "/" && dlg.hidden && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "")) {
      e.preventDefault();
      open();
    } else if (e.key === "Escape") {
      if (!dlg.hidden) close();
      else closeDrawer();
    }
  });

  // subsequence match: returns score or -1
  function fuzzy(hay, needle) {
    let hi = 0, score = 0, run = 0, first = -1;
    for (let i = 0; i < needle.length; i++) {
      const c = needle[i];
      const j = hay.indexOf(c, hi);
      if (j < 0) return -1;
      if (first < 0) first = j;
      run = j === hi ? run + 1 : 0;
      score += 1 + run * 2 + (j === 0 || /[\s._\-(]/.test(hay[j - 1]) ? 3 : 0);
      hi = j + 1;
    }
    return score - first * 0.05 - (hay.length - needle.length) * 0.01;
  }
  const KIND_BOOST = { page: 6, section: 3, heading: 2, function: 4, class: 5, exception: 3, module: 4, data: 1, method: 0, property: 0, attribute: 0 };
  function score(e, q, terms) {
    let s = 0;
    const t = e.tl;
    const bare = t.slice(t.lastIndexOf(".") + 1);
    if (t === q || bare === q) s += 100;
    else if (t.startsWith(q) || bare.startsWith(q)) s += 60;
    else if (t.includes(q)) s += 40;
    else {
      const f = q.length >= 2 && !q.includes(" ") ? fuzzy(t, q) : -1;
      if (f > q.length * 2.2) s += 10 + f;
    }
    let all = true;
    for (const w of terms) {
      if (t.includes(w)) s += 8;
      else if (e.sl.includes(w)) s += 3;
      else if (e.pl.includes(w)) s += 1;
      else all = false;
    }
    if (!s) return 0;
    if (!all && s < 40) return 0;
    return s + (KIND_BOOST[e.k] || 0) - t.length * 0.02;
  }
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  function mark(text, terms) {
    let h = esc(text);
    for (const w of terms) {
      if (w.length < 2) continue;
      h = h.replace(new RegExp("(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
    }
    return h;
  }
  const SUGGEST = ["integrate", "simplify", "solve", "ExprPool", "Budget", "to_lean", "compile_expr", "Getting started"];
  function render(qraw) {
    const q = qraw.trim().toLowerCase();
    if (!index) return;
    if (!q) {
      results = SUGGEST.map((s) => index.find((e) => e.tl === s.toLowerCase())).filter(Boolean);
    } else {
      const terms = q.split(/\s+/).filter(Boolean);
      results = [];
      for (const e of index) {
        const s = score(e, q, terms);
        if (s > 0) results.push([s, e]);
      }
      results.sort((a, b) => b[0] - a[0]);
      const seen = new Set();
      results = results.map((r) => r[1]).filter((e) => (seen.has(e.u) ? false : seen.add(e.u))).slice(0, 40);
    }
    sel = 0;
    const terms = q.split(/\s+/).filter(Boolean);
    if (!results.length) {
      list.innerHTML = `<li class="search-empty">No results for “${esc(qraw.trim())}”</li>`;
      return;
    }
    const mono = new Set(["function", "class", "exception", "module", "data", "method", "property", "attribute", "staticmethod", "classmethod"]);
    list.innerHTML = results
      .map((e, i) => {
        const kind = e.k === "page" ? "guide" : e.k === "heading" ? "section" : e.k;
        return `<li><a href="${esc(e.u)}" role="option" aria-selected="${i === 0}" data-i="${i}">
<div class="sr-top"><span class="sr-kind">${esc(kind)}</span><span class="sr-title${mono.has(e.k) ? " mono" : ""}">${mark(e.t, terms)}</span>${e.p ? `<span class="sr-page">${esc(e.p)}</span>` : ""}</div>
${e.s ? `<div class="sr-snippet">${mark(e.s, terms)}</div>` : ""}</a></li>`;
      })
      .join("");
  }
  function move(d) {
    const items = $$("a[role=option]", list);
    if (!items.length) return;
    items[sel]?.setAttribute("aria-selected", "false");
    sel = (sel + d + items.length) % items.length;
    items[sel].setAttribute("aria-selected", "true");
    items[sel].scrollIntoView({ block: "nearest" });
  }
  input?.addEventListener("input", () => render(input.value));
  input?.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") {
      const a = $$("a[role=option]", list)[sel];
      if (a) { e.preventDefault(); close(); location.href = a.href; }
    }
  });
  list?.addEventListener("mousemove", (e) => {
    const a = e.target.closest("a[role=option]");
    if (!a) return;
    const i = Number(a.dataset.i);
    if (i === sel) return;
    $$("a[role=option]", list)[sel]?.setAttribute("aria-selected", "false");
    sel = i;
    a.setAttribute("aria-selected", "true");
  });
  list?.addEventListener("click", (e) => { if (e.target.closest("a")) close(); });
})();
