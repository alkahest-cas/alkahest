// Alkahest landing page: nav, tabs, copy buttons and benchmark rendering.
(() => {
  "use strict";

  // ── Mobile menu ──────────────────────────────────────────────────────
  const menuBtn = document.querySelector(".menu-btn");
  const nav = document.getElementById("nav");
  if (menuBtn && nav) {
    menuBtn.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", String(open));
    });
  }

  // ── Tabs (keyboard: ←/→ move between tabs) ───────────────────────────
  function wireTabs(tablist, onSelect) {
    const tabs = [...tablist.querySelectorAll('[role="tab"]')];
    const select = (tab, focus) => {
      for (const t of tabs) {
        const on = t === tab;
        t.setAttribute("aria-selected", String(on));
        t.tabIndex = on ? 0 : -1;
        const panel = t.getAttribute("aria-controls");
        if (panel && document.getElementById(panel)) document.getElementById(panel).hidden = !on;
      }
      if (focus) tab.focus();
      if (onSelect) onSelect(tab);
    };
    tabs.forEach((tab, i) => {
      tab.addEventListener("click", () => select(tab, false));
      tab.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        select(next, true);
      });
    });
    return select;
  }
  document.querySelectorAll(".install .tabs").forEach((tl) => wireTabs(tl));

  // ── Copy buttons ─────────────────────────────────────────────────────
  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-copy]");
    if (!btn) return;
    try {
      await navigator.clipboard.writeText(btn.dataset.copy);
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Press ⌘C";
    }
    setTimeout(() => (btn.textContent = "Copy"), 1500);
  });

  // ── Benchmarks ───────────────────────────────────────────────────────
  const SYSTEM_LABEL = {
    alkahest: "Alkahest",
    sympy: "SymPy",
    symengine: "SymEngine",
    mathematica: "Mathematica",
    numpy: "NumPy",
    "sympy-lambdify": "SymPy lambdify",
    scipy: "SciPy",
  };
  const label = (r) => SYSTEM_LABEL[r.system] || r.label || r.system;
  // A result counts only if it finished with a checked-correct answer.
  const done = (r) => r.value != null && (r.status == null || r.status === "ok");
  const FAIL = { timeout: "timeout", wrong: "wrong answer", incomplete: "incomplete", error: "error" };
  const failText = (r, c) =>
    r.status && r.status !== "timeout" ? FAIL[r.status] || r.status : `timeout${c.timeout_s ? ` > ${c.timeout_s} s` : ""}`;
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

  function fmt(v, unit) {
    if (v == null) return "—";
    let ms = unit === "s" ? v * 1000 : unit === "us" || unit === "µs" ? v / 1000 : v;
    if (unit && !["ms", "s", "us", "µs"].includes(unit)) {
      return `${v >= 100 ? Math.round(v).toLocaleString() : +v.toPrecision(3)} ${unit}`;
    }
    if (ms >= 1000) return `${+(ms / 1000).toPrecision(3)} s`;
    if (ms >= 1) return `${+ms.toPrecision(3)} ms`;
    return `${+(ms * 1000).toPrecision(3)} µs`;
  }

  // Lower is better unless the case says otherwise (e.g. throughput).
  const better = (c, a, b) => (c.higher_is_better ? a > b : a < b);

  function headline(c) {
    const me = c.results.find((r) => r.system === "alkahest");
    if (!me || !done(me)) return null;
    const others = c.results.filter((r) => r.system !== "alkahest");
    if (!others.length) return null;
    // Compare against the fastest competitor that finished; a timeout counts as a win.
    const finished = others.filter(done);
    if (!finished.length) return { text: `others time out`, loss: false };
    const best = finished.reduce((a, b) => (better(c, a.value, b.value) ? a : b));
    const ratio = c.higher_is_better ? me.value / best.value : best.value / me.value;
    if (ratio >= 1.05) return { text: `${fmtRatio(ratio)} faster than ${label(best)}`, loss: false };
    if (ratio <= 0.95) return { text: `${fmtRatio(1 / ratio)} slower than ${label(best)}`, loss: true };
    return { text: `on par with ${label(best)}`, loss: false };
  }
  const fmtRatio = (r) => (r >= 100 ? `${Math.round(r).toLocaleString()}×` : r >= 10 ? `${r.toFixed(0)}×` : `${r.toFixed(1)}×`);

  function caseBars(c) {
    // Log scale: these span several orders of magnitude.
    const vals = c.results.filter((r) => done(r) && r.value > 0).map((r) => r.value);
    if (!vals.length) return "";
    const hasTimeout = c.results.some((r) => !done(r));
    const max = Math.max(...vals, hasTimeout && c.timeout_s ? toUnit(c.timeout_s, c.unit) : 0);
    const min = Math.min(...vals);
    const lo = Math.log10(min) - 0.6;
    const hi = Math.log10(max);
    const pct = (v) => (c.higher_is_better ? Math.max(4, (v / max) * 100) : Math.max(4, ((Math.log10(v) - lo) / (hi - lo || 1)) * 100));
    const h = headline(c);
    const rows = c.results
      .map((r) => {
        const me = r.system === "alkahest";
        const timeout = !done(r);
        const width = timeout ? 100 : pct(r.value);
        const val = timeout ? failText(r, c) : fmt(r.value, c.unit);
        return `<div class="sys${me ? " me" : ""}" title="${esc(label(r))}${r.version ? " " + esc(r.version) : ""}">${esc(label(r))}</div>
          <div class="bar-track"><div class="bar${me ? " me" : ""}${timeout ? " timeout" : ""}" style="width:calc(${width}% - 84px)"></div><span class="bar-val">${esc(val)}</span></div>`;
      })
      .join("");
    return `<div class="bench-case">
      <h4><span title="${esc(c.detail || "")}">${esc(c.task)}</span>${h ? `<span class="speed${h.loss ? " loss" : ""}">${esc(h.text)}</span>` : ""}</h4>
      <div class="bars" role="img" aria-label="${esc(c.task)}: ${esc(c.results.map((r) => `${label(r)} ${done(r) ? fmt(r.value, c.unit) : failText(r, c)}`).join(", "))}">${rows}</div>
    </div>`;
  }
  const toUnit = (s, unit) => (unit === "s" ? s : unit === "us" || unit === "µs" ? s * 1e6 : s * 1000);

  function renderHero(data) {
    const card = document.getElementById("hero-bench");
    if (!card) return;
    const tablist = card.querySelector(".tabs");
    const body = card.querySelector(".bench-body");
    const suites = (data.suites || []).filter((s) => s.cases && s.cases.length);
    if (!suites.length) {
      body.innerHTML = `<div class="bench-empty">Benchmarks are being regenerated.</div>`;
      return;
    }
    tablist.innerHTML = suites
      .map((s, i) => `<button class="tab" role="tab" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" data-suite="${esc(s.id)}">${esc(s.label)}</button>`)
      .join("");
    const show = (tab) => {
      const s = suites.find((x) => x.id === tab.dataset.suite);
      body.innerHTML = `${s.blurb ? `<p class="bench-blurb">${esc(s.blurb)}</p>` : ""}${s.cases.slice(0, 3).map(caseBars).join("")}`;
    };
    wireTabs(tablist, show);
    show(tablist.querySelector('[aria-selected="true"]'));
    const meta = card.querySelector(".bench-meta");
    if (meta) meta.textContent = `Median wall time, log scale · Alkahest ${data.alkahest_version || ""}`;
  }

  function renderTables(data) {
    const host = document.getElementById("perf-tables");
    if (!host) return;
    const suites = (data.suites || []).filter((s) => s.cases && s.cases.length);
    host.innerHTML = suites
      .map((s) => {
        const systems = [];
        for (const c of s.cases) for (const r of c.results) if (!systems.includes(r.system)) systems.push(r.system);
        systems.sort((a, b) => (a === "alkahest" ? -1 : b === "alkahest" ? 1 : 0));
        const versions = {};
        for (const c of s.cases) for (const r of c.results) if (r.version) versions[r.system] = r.version;
        const head = systems.map((x) => `<th>${esc(SYSTEM_LABEL[x] || x)}${versions[x] ? ` <span style="font-weight:400">${esc(versions[x])}</span>` : ""}</th>`).join("");
        const rows = s.cases
          .map((c) => {
            const ok = c.results.filter(done);
            const best = ok.length ? ok.reduce((a, b) => (better(c, a.value, b.value) ? a : b)).system : null;
            const cells = systems
              .map((x) => {
                const r = c.results.find((y) => y.system === x);
                if (!r) return `<td class="num">—</td>`;
                const txt = done(r) ? fmt(r.value, c.unit) : failText(r, c);
                const cls = ["num", x === "alkahest" ? "me" : "", x === best ? "best" : ""].join(" ");
                return `<td class="${cls}"${r.note ? ` title="${esc(r.note)}"` : ""}>${esc(txt)}</td>`;
              })
              .join("");
            const h = headline(c);
            return `<tr><td><span class="task">${esc(c.task)}</span>${c.detail ? `<span class="det">${esc(c.detail)}</span>` : ""}${c.note ? `<details class="note"><summary>Note</summary>${esc(c.note)}</details>` : ""}</td>${cells}<td class="num">${h ? esc(h.text) : ""}</td></tr>`;
          })
          .join("");
        return `<div class="perf-suite"><h3>${esc(s.label)}</h3>${s.blurb ? `<p>${esc(s.blurb)}</p>` : ""}
          <div class="table-scroll"><table class="perf"><thead><tr><th>Task</th>${head}<th>Alkahest vs best other</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
      })
      .join("");
    const m = document.getElementById("perf-method");
    if (m) {
      m.innerHTML = `Alkahest ${esc(data.alkahest_version || "")}${data.alkahest_commit ? ` (${esc(data.alkahest_commit)})` : ""}, generated ${esc(data.generated || "")} on ${esc(data.machine || "one machine")}. ` +
        `${esc(data.methodology || "")} ` +
        `Reproduce with <code>python benchmarks/website/run_all.py</code> — ` +
        `<a href="https://github.com/alkahest-cas/alkahest/blob/main/benchmarks/website/RESULTS.md">full results and caveats</a>.`;
    }
  }

  fetch("/data/benchmarks.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((data) => {
      renderHero(data);
      renderTables(data);
    })
    .catch(() => {
      const body = document.querySelector("#hero-bench .bench-body");
      if (body) body.innerHTML = `<div class="bench-empty">Benchmarks unavailable.</div>`;
    });
})();
