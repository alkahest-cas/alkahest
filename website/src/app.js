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
    // The install rows are whole-block buttons with a separate hint; plain copy buttons use their own text.
    const hint = btn.querySelector(".copy-hint") || btn;
    try {
      await navigator.clipboard.writeText(btn.dataset.copy);
      hint.textContent = "Copied";
    } catch {
      hint.textContent = "Press ⌘C";
    }
    setTimeout(() => (hint.textContent = "Copy"), 1500);
  });

  // ── Benchmarks ───────────────────────────────────────────────────────
  const SYSTEM_LABEL = {
    alkahest: "alkahest",
    sympy: "SymPy",
    symengine: "SymEngine",
    mathematica: "Mathematica",
    mpmath: "mpmath",
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

  // Median speedup against each other system, over the tasks where both finished.
  function overview(suites) {
    const cases = suites.flatMap((s) => s.cases);
    const stats = {};
    for (const c of cases) {
      const me = c.results.find((r) => r.system === "alkahest");
      if (!me || !done(me)) continue;
      for (const r of c.results) {
        if (r.system === "alkahest") continue;
        const st = (stats[r.system] ||= { n: 0, ratios: [], k: 0, wins: 0, losses: 0, dnf: 0 });
        st.n++;
        if (!done(r)) { st.dnf++; continue; }
        const ratio = c.higher_is_better ? me.value / r.value : r.value / me.value;
        st.ratios.push(ratio);
        st.k++;
        if (ratio >= 1.05) st.wins++;
        else if (ratio <= 0.95) st.losses++;
      }
    }
    const order = ["sympy", "mathematica", "symengine", "numpy"];
    const cards = order
      .filter((x) => stats[x] && stats[x].n >= 3)
      .map((x) => {
        const st = stats[x];
        const rs = st.ratios.sort((a, b) => a - b);
        const mid = rs.length >> 1;
        const g = rs.length % 2 ? rs[mid] : Math.sqrt(rs[mid - 1] * rs[mid]);
        const loss = g < 0.95;
        const big = loss ? `${fmtRatio(1 / g)} slower` : g < 1.05 ? "On par" : `${fmtRatio(g)} faster`;
        const ties = st.k - st.wins - st.losses;
        const strip =
          "<i></i>".repeat(st.wins) + '<i class="t"></i>'.repeat(st.dnf) + '<i class="l"></i>'.repeat(ties + st.losses);
        const name = SYSTEM_LABEL[x] || x;
        const rec = [
          `faster on ${st.wins + st.dnf} of ${st.n} tasks`,
          st.dnf ? `${name} did not finish ${st.dnf}` : "",
        ].filter(Boolean).join(", ");
        return `<div class="ov"><div class="num${loss ? " loss" : ""}">${esc(big)}</div>
          <div class="vs">than ${esc(name)} on the median task</div>
          <div class="wl" role="img" aria-label="${esc(rec)}">${strip}</div>
          <div class="rec">${esc(rec[0].toUpperCase() + rec.slice(1))}</div></div>`;
      })
      .join("");
    // Biggest wins as bars, so the overview still shows the actual timings.
    const top = cases
      .map((c) => ({ c, h: headline(c) }))
      .filter((x) => x.h && !x.h.loss)
      .sort((a, b) => ratioOf(b.c) - ratioOf(a.c))
      .slice(0, 4)
      .map((x) => caseBars(x.c))
      .join("");
    return `<div class="overview">${cards}</div><div class="bench-grid">${top}</div>`;
  }
  const ratioOf = (c) => {
    const me = c.results.find((r) => r.system === "alkahest");
    const ok = c.results.filter((r) => r.system !== "alkahest" && done(r));
    if (!me || !done(me)) return 0;
    if (!ok.length) return 1e9;
    return Math.min(...ok.map((r) => (c.higher_is_better ? me.value / r.value : r.value / me.value)));
  };

  function renderBench(data) {
    const card = document.getElementById("bench");
    if (!card) return;
    const tablist = card.querySelector(".tabs");
    const body = card.querySelector(".bench-body");
    const suites = (data.suites || []).filter((s) => s.cases && s.cases.length);
    if (!suites.length) {
      body.innerHTML = `<div class="bench-empty">Benchmarks are being regenerated</div>`;
      return;
    }
    const tabs = [{ id: "_overview", label: "Overview" }, ...suites.map((s) => ({ id: s.id, label: s.label.replace(" / ", " & ") }))];
    tablist.innerHTML = tabs
      .map((s, i) => `<button class="tab" role="tab" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}" data-suite="${esc(s.id)}">${esc(s.label)}</button>`)
      .join("");
    const show = (tab) => {
      const id = tab.dataset.suite;
      if (id === "_overview") {
        body.innerHTML = overview(suites);
        return;
      }
      const s = suites.find((x) => x.id === id);
      body.innerHTML = `<div class="bench-grid">${s.cases.map(caseBars).join("")}</div>`;
    };
    wireTabs(tablist, show);
    show(tablist.querySelector('[aria-selected="true"]'));

    const m = document.getElementById("perf-method");
    if (m) {
      const versions = {};
      for (const s of suites) for (const c of s.cases) for (const r of c.results) if (r.version && r.system !== "alkahest") versions[r.system] = r.version;
      const vs = Object.entries(versions).map(([k, v]) => `${SYSTEM_LABEL[k] || k} ${v}`).join(", ");
      m.innerHTML =
        `alkahest ${esc(data.alkahest_version || "")} against ${esc(vs)}, run ${esc(data.generated || "")} on ${esc(data.machine || "one machine")}. ` +
        `${esc(data.methodology || "")} ` +
        `Reproduce with <code>python benchmarks/website/run_all.py</code>, or read the <a href="https://github.com/alkahest-cas/alkahest/blob/main/benchmarks/website/RESULTS.md">full results and caveats</a>`;
    }
  }

  fetch("/data/benchmarks.json")
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then(renderBench)
    .catch(() => {
      const body = document.querySelector("#bench .bench-body");
      if (body) body.innerHTML = `<div class="bench-empty">Benchmarks unavailable</div>`;
    });
})();
