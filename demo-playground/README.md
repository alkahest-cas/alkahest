# Alkahest Demo Playground

An interactive web application for trying, demoing, testing, and recording the [Alkahest](https://alkahest-cas.github.io/) computer algebra system. It has two tabs: **Math** (the default), where you type ordinary mathematics, and **Python**, a notebook for the Python API. It shares its look and top bar with the landing page and the docs, and can be driven from the CLI for recordings.

## Features

| Feature | Description |
|---------|-------------|
| **Math tab** | One line of ordinary maths per cell (`integrate sin(x)^2 dx`, `d/dx x^3 sin(x)`, `solve …`), live "Interpreted as" preview, LaTeX results, coded refusals, verification badges |
| **Python tab** | Jupyter-style code cells with CodeMirror editor |
| **Dual execution** | WASM (Pyodide) for SymPy/pure Python; server kernel for alkahest |
| **Rich outputs** | LaTeX (KaTeX), matplotlib figures, HTML, JSON, **Lean 4 certificates** |
| **Lean verification** | Export `.lean` proofs from alkahest; typecheck via server (`lake env lean`) |
| **Local wheel** | Upload a `.whl` to test a dev build of alkahest |
| **Recording** | In-browser `MediaRecorder` or headless Playwright from the CLI |
| **CLI orchestration** | `start`, `record` commands |

## Architecture

```
demo-playground/
├── web/      Next.js 14 app (TypeScript, Tailwind, CodeMirror, KaTeX)
├── server/   FastAPI + Jupyter kernel (Python, ipykernel)
└── cli/      Node.js orchestration CLI (Commander, Playwright)
```

**Execution routing** — the browser detects `import alkahest` / `from alkahest` in cell code and sends those cells to the Python server automatically. Pure-Python cells run in Pyodide (browser WASM). This decision is configurable in Settings.

**Math tab** — the browser sends each line to `POST /math` on the Alkahest server. The server parses a small maths language and calls Alkahest functions directly; it never executes Python from the request (see [Math tab](#math-tab)). Nothing in the Math tab runs in the browser, and there is no SymPy fallback.

**No AI agent here.** The AI agent chat that used to live in this app has been removed; the site's Agent link goes to the separate agent page.

## Prerequisites

- **Node.js** ≥ 18 with **pnpm** (`npm i -g pnpm`)
- **Python** ≥ 3.9

## Quick start

### 1. Configure environment

```bash
cd demo-playground
cp .env.example web/.env.local   # optional: server URL / token overrides
```

### 2. Install dependencies

```bash
# Node packages
pnpm install

# Python server (creates a venv automatically)
# The start script below handles this, but you can also run manually:
# cd server && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### 3. Start everything

```bash
# Option A — use the CLI (recommended)
pnpm start
# or: npx tsx cli/src/index.ts start

# Option B — manually in two terminals
#   Terminal 1:  cd server && bash start.sh
#   Terminal 2:  cd web   && pnpm dev
```

Open [http://localhost:3000](http://localhost:3000).

### Hosted demo (GitHub Pages)

A static build runs in the browser at **[alkahest-cas.github.io/playground](https://alkahest-cas.github.io/playground/)**:

- **Math tab** — needs an Alkahest server. With none configured it says so and shows how to start one; it does not compute anything in the browser. An HTTPS page can only reach an `https://` server (mixed content), so for a local server use the locally served playground.
- **Python tab, WASM** — SymPy and pure Python cells work with no server.
- **Bring your own backend** — open settings (`Ctrl+/`) and set an Alkahest server URL + token (Math and Python tabs), or a remote Jupyter URL + token (Python tab only).

Pushes to `main` under `demo-playground/` deploy automatically via `.github/workflows/playground.yml`.

---

## Math tab

The default tab. Each cell is one line of maths; **Enter** runs it and moves to the next line, **↑/↓** move between lines, and the symbol palette (π, e, i, ∞, √, powers, fractions, ∫, d/dx, ∑, lim, sin/cos/log, Greek letters) inserts text at the cursor. While you type, the line is parsed on the server and shown back as *Interpreted as …* in KaTeX, so `1/2x`, `sin x^2` or `xy` show exactly how they were read.

What it understands (case-insensitive command words; unicode such as `π ∞ √ ² · × ÷ − ∫ ∑ ∂ →` and Greek letters is accepted):

| Form | Examples | Alkahest call |
|------|----------|---------------|
| integrate | `integrate sin(x)^2 dx`, `∫ x e^x dx`, `integrate 1/(1+x^2) from 0 to oo`, `integral of t^2 dt`, `integrate(f, x, 0, 1)` | `ak.integrate` |
| derivative | `d/dx x^3 sin(x)`, `d²/dx² f`, `d^3/dx^3 …`, `derivative of tan(x)`, `differentiate … with respect to t`, `diff(f, x, 2)`, `∂/∂y …` | `ak.diff` (repeated) |
| solve | `solve x^2 - 5x + 6 = 0`, `solve x^2 + y^2 = 1, x - y = 0 for x, y`, `solve(eq1, eq2, x, y)`; a bare equation also solves | `ak.solve` |
| simplify | `simplify (x^2-1)/(x-1)`; a bare expression (`sin(pi/6) + sqrt(8)`) is simplified and, if constant, approximated | `ak.simplify`, `simplify_trig`, `simplify_log_exp`, `cancel` |
| factor / expand | `factor x^4 - 1`, `factor x^2 - y^2`, `factor 360`, `expand (x+1)^5` | `UniPoly`/`MultiPoly.factor_z`, `number_theory.factorint`, `simplify_expanded` |
| limit | `limit sin(x)/x as x -> 0`, `… as x -> 0+`, `… from the left`, `lim x->0 …`, `lim_{x->0} …`, `limit(f, x, 0)` | `ak.limit` |
| series | `series exp(x) at x = 0 to order 6`, `taylor sin(x) around x = pi up to order 4`, `series(f, x, 0, 8)` | `ak.series` |
| sum / product | `sum 1/k^2 for k = 1 to oo`, `product … for k = 1 to n`, `sum(f, k, 1, oo)` | `ak.sum_definite`, `ak.product_definite` |
| numeric | `numeric pi to 50 digits`, `N[sqrt(2)]`, `N[pi, 50]`, `approx e` | `ak.evaluate(…, precision_bits=…)` (rigorous ball) |
| define | `f = x^2 sin(x)`, `g(t) = t^2 + 1`, `F = integrate x^2 dx`, then `d/dx f`, `g(3)` in later lines | — |

Expressions use the usual precedence with implicit multiplication (`2x`, `x sin(x)`, `(x+1)(x-1)`), `^` or `**`, `!`, `|x|`, `e`, `i`, `pi`, `oo`. Functions: sin, cos, tan, sec, csc, cot, asin/arcsin, acos, atan, sinh, cosh, tanh and inverses, exp, log/ln, log10, log2, `log(x, b)`, sqrt, cbrt, abs, sign, floor, ceil, gamma, erf, erfc, zeta, atan2, min, max, factorial. Decimals are read as exact rationals (`0.5` is `1/2`).

**Results** render as LaTeX. Badges never overstate: `verified` means Alkahest checked the result exactly (e.g. the derivative of the antiderivative is the integrand); `Lean certificate` means Alkahest generated a Lean 4 proof, which you can open and check with **Verify in Lean** (unchecked until you do); `certified digits` marks a rigorous ball enclosure.

**Refusals** show the code and Alkahest's own message, framed as *Alkahest declined because …* (for example `integrate e^(x^2) dx` → `E-INT-004`). Input the playground cannot read, or input over its limits, gets an `E-PLAYGROUND-NNN` code instead:

| Code | Meaning |
|------|---------|
| `E-PLAYGROUND-001` | Could not parse the line |
| `E-PLAYGROUND-002` | Input or result too large (500 characters, nesting depth, result size) |
| `E-PLAYGROUND-003` | Unknown or disallowed name (e.g. anything starting with `_`) |
| `E-PLAYGROUND-004` | A number, power, factorial, order or digit count over the cap (`2^2^2^2^2^2`) |
| `E-PLAYGROUND-005` | Ambiguous variable (say `dx`, `for x`, `as t -> 0`) |
| `E-PLAYGROUND-006` | Hard time limit hit (the worker process was killed) |
| `E-PLAYGROUND-007` | Memory limit hit |
| `E-PLAYGROUND-008` | Not applicable (e.g. `numeric` of an expression with free symbols) |
| `E-PLAYGROUND-009` | Playground bug (not a verdict about the maths) |
| `E-PLAYGROUND-010` | Server busy |

### Endpoint contract

```
POST /math            {"input": "d/dx f", "definitions": [{"name": "f", "params": [], "value": "(x^2 * sin(x))"}]}
POST /math/interpret  same body; parses only (live preview)

→ {"ok": true, "kind": "diff", "latex": "...", "text": "...", "interpreted_latex": "...",
   "verification": "certificate_available", "verified": false, "certificate": "import Mathlib...",
   "approx": "...", "notes": [...], "side_conditions": [...], "definition": {...}, "elapsed_ms": 0.4}
→ {"ok": false, "interpreted_latex": "...", "error_code": "E-INT-004", "error_message": "...",
   "error_detail": "...", "error_hint": "...", "error_source": "alkahest" | "input" | "playground"}
```

`definitions` is how earlier lines' names reach later ones: the browser sends back the `definition` objects earlier responses returned, and the server re-parses their `value` with the same restricted grammar (so editing them gains nothing).

### Safety model

The endpoint is designed so it can be hosted publicly:

- **No Python execution.** `server/mathlang.py` is a hand-written tokenizer and parser (no `eval`, no `ast`, no imports of user text); `server/math_exec.py` maps each syntax-tree node onto an explicit whitelist of Alkahest calls. Strings, quotes, `.`, `;`, `_`-prefixed names and anything outside the grammar are refused before Alkahest is imported.
- **Size caps before computing:** 500 characters, 2,000 syntax nodes, nesting depth 60, 100-digit literals, integer powers up to 100,000 bits, symbolic exponents up to 1,000, `n!` up to 1000, derivative order 20, series order 30, 1,000 digits, 32 definitions of up to 20,000 characters, 256 KiB request bodies.
- **Each computation runs in a fresh `python -I` worker process** (`server/math_worker.py`) under `RLIMIT_AS` (`ALKAHEST_MATH_MEMORY_MB`, default 2048), `RLIMIT_CPU`, `RLIMIT_FSIZE=0`, `RLIMIT_NOFILE`, no core dumps, a minimal environment, inside `ak.context(budget=ak.Budget(wall_ms=..., max_bytes=...))`, and is SIGKILLed with its process group at a hard deadline (`ALKAHEST_MATH_TIMEOUT_S`, default 15 s). The cooperative budget alone cannot interrupt one FLINT call; the process kill can.
- **Concurrency cap** (`ALKAHEST_MATH_CONCURRENCY`, default 2), with `E-PLAYGROUND-010` when the queue does not clear.
- **Token:** if `ALKAHEST_SERVER_TOKEN` is set the Math endpoints require it, unless `ALKAHEST_MATH_PUBLIC=1` opens just them (the kernel, wheel-install and Lean endpoints stay protected).

These `ALKAHEST_MATH_*` variables are read by the server; put them in `server/.env.local`, which `start.sh` loads.

Remaining risks before putting it on the open internet: there is no per-client rate limiting (put it behind a proxy that has one); each request spawns a process (about 0.15 s and up to the memory cap, times the concurrency); CORS is `*`; `/math/interpret` parses in the server process itself (bounded by the size caps, but not sandboxed); and the rest of the app (`/sessions`, `/ws`, wheel install, `/verify-lean`) runs arbitrary code by design and must not be exposed without the token.

Tests: `cd server && pytest` (parser tests need nothing; executor and HTTP tests need an Alkahest build, `fastapi` and `httpx`).

---

## Python tab (notebook)

Open it with the **Python** tab (or `?tab=python`). It starts from a starter notebook with markdown sections and code cells covering simplification, differentiation, integration, and trigonometric identities (`demos/default_notebook.py`). Use **Clear notebook** in the toolbar to reset to a single empty code cell.

- **⌘ Enter** — run focused cell
- **Run all** — execute all cells top-to-bottom
- **Install wheel** — upload a `.whl` and restart the kernel with the new build
- The execution backend badge (`wasm` / `server`) shows per-cell routing

### Displaying LaTeX

From any Python cell, print a LaTeX string wrapped in `$$`:

```python
from alkahest import latex
result = ak.diff(pool, expr, x)
print(f"$${latex(result.value)}$$")
```

The output renderer detects `$$...$$` and renders it with KaTeX.

SymPy expressions use `IPython.display.Math` which the Jupyter kernel surfaces as `text/latex` — rendered automatically.

### Lean 4 certificates

Alkahest records every rewrite in a derivation log and can export a Mathlib proof file via `to_lean()` or `DerivedResult.certificate`.

In a **server** cell (requires `pnpm start` and a built alkahest wheel):

```python
import alkahest as ak
from alkahest import latex
from playground_helpers import display_lean_cert

pool = ak.ExprPool()
x = pool.symbol("x")
result = ak.simplify(x + pool.integer(0))
print(f"$${latex(result.value)}$$")
display_lean_cert(result, operation="simplify")  # renders a certificate panel
```

The output area shows a **Lean 4 certificate** panel with Copy, Download, and **Verify in Lean** (runs `lake env lean` on the demo server).

**Server setup for verification** (one-time per machine):

```bash
# From the alkahest repo root
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
cd lean && lake update && lake exe cache get
```

Set `ALKAHEST_LEAN_PROJECT` if the repo is not at the default path relative to the server.

---

## Settings

Click the gear icon (⚙) in the navbar to configure:

- **Server URL** — Alkahest server address (default `http://localhost:8000`); used by the Math tab and by alkahest cells in the Python tab
- **Backend type** — Alkahest server or remote **Jupyter Server** (URL + token; Python tab only)
- **Access token** — Jupyter token or Alkahest `ALKAHEST_SERVER_TOKEN` (stored in browser)
- **Execution mode** — Auto / WASM only / Server only
- **Ctrl+/** — toggle settings from anywhere (except while typing in a cell)

Settings are persisted in `localStorage`.

---

## Recording

### In-browser (for quick demos)

Click **Record** in the navbar. The browser will ask for screen/tab capture permission. When you stop recording, the clip downloads as `.webm`.

### CLI — scripted notebook recording

Record a notebook session by injecting a Python file:

```bash
npx tsx cli/src/index.ts record \
  --code examples/my_demo.py \
  --output recordings/demo.webm
```

Cells are delimited by `# ---` in the file:

```python
# Cell 1
import alkahest as ak
pool = ak.ExprPool()
x = pool.symbol("x")

# ---

# Cell 2
result = ak.diff(pool, x**pool.integer(3), x)
print(f"$${ak.latex(result.value)}$$")
```

### CLI — side-by-side compare (1080p)

Record Alkahest vs SymPy in two columns (e.g. Groebner basis benchmark):

```bash
npx tsx cli/src/index.ts record \
  --layout split \
  --code-left groebner-comparison/demos/alkahest_panel.py \
  --code-right groebner-comparison/demos/sympy_panel.py \
  --output groebner-comparison/recordings/out.webm \
  --width 1920 --height 1080
```

Or use `bash groebner-comparison/record.sh`. See [`groebner-comparison/README.md`](groebner-comparison/README.md).

Use `?zen=1` automatically in CLI mode; the recorder waits until cells are rendered before capturing.

### CLI — removed: agent demo recording

The `demo <prompt>` command (which drove the old AI agent chat and recorded it) was removed together with the agent chat. `record` covers scripted notebook recordings; `?demo=` payloads open the Python tab automatically.

---

## Connecting a remote server

For demos that need a powerful GPU or larger memory, run the Python server on a remote host:

```bash
# On the remote machine:
git clone https://github.com/alkahest-cas/alkahest
cd alkahest/demo-playground/server
bash start.sh

# Locally, update Settings (or web/.env.local):
NEXT_PUBLIC_PYTHON_WS_URL=ws://your-host:8000
NEXT_PUBLIC_PYTHON_HTTP_URL=http://your-host:8000
```

SSH tunneling also works:
```bash
ssh -L 8000:localhost:8000 user@your-host
```

---

## Testing a local alkahest build

1. Build a wheel: `maturin build --release` (from the repo root)
2. Click **Install wheel** in the toolbar and select the `.whl` from `target/wheels/`
3. The kernel restarts with the new build; run your cells normally

Or use the server start script, which auto-installs the latest wheel from `dist/`:

```bash
# Build first
maturin build --release --out dist/

# Then start — the script picks it up automatically
bash server/start.sh
```

---

## Adding visualizations

The output renderer handles these MIME types from the kernel:

| MIME | Rendered as |
|------|-------------|
| `text/plain` | Monospace text |
| `text/latex` | KaTeX math block |
| `text/html` | Sanitized HTML (KaTeX auto-render for `$...$`) |
| `image/png` | Inline image |
| `image/svg+xml` | Inline SVG |
| `application/json` | Formatted JSON |

**Matplotlib:**
```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Example")
plt.show()  # captured automatically by the kernel
```

**Plotly:**
```python
pip install plotly   # once, or add to server/requirements.txt
import plotly.express as px
fig = px.line(x=[1, 2, 3], y=[1, 4, 9])
fig.show()   # renders as interactive HTML
```

**Robotics / 3D:**
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ... kinematics visualization ...
```

---

## Project structure

```
demo-playground/
├── .env.example            Environment variable template
├── package.json            pnpm workspace root
├── pnpm-workspace.yaml
│
├── web/                    Next.js 14 frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             Math | Python tabs
│   │   │   ├── compare/page.tsx     Side-by-side recording view
│   │   │   ├── api/jupyter-proxy/   CORS proxy for remote Jupyter (dev server only)
│   │   │   └── globals.css          Design tokens (shared with the site), light + dark
│   │   ├── components/
│   │   │   ├── math/                MathNotebook, MathCell, Tex
│   │   │   ├── notebook/            Cell, Notebook, Output, LeanCertificate
│   │   │   └── ui/                  SiteHeader, PlaygroundBar, Settings
│   │   └── lib/
│   │       ├── math-api.ts          /math client
│   │       ├── math-starter.ts      Starter lines, examples, palette
│   │       └── execution.ts         WASM/server routing (Python tab)
│   └── public/
│       └── pyodide-worker.js        Web Worker for Pyodide
│
├── server/                 Python execution backend
│   ├── main.py             FastAPI app
│   ├── mathlang.py         Math tab language: tokenizer, parser, LaTeX (no Alkahest, no eval)
│   ├── math_exec.py        Syntax tree → whitelisted Alkahest calls
│   ├── math_service.py     Sandboxed worker processes, limits, timeouts
│   ├── math_worker.py      One request per `python -I` process
│   ├── kernel_manager.py   Jupyter kernel wrapper (Python tab)
│   ├── tests/              pytest suite for the Math tab
│   ├── requirements.txt
│   └── start.sh            Bootstraps venv and starts uvicorn
│
└── cli/                    Orchestration CLI
    └── src/
        ├── index.ts
        └── commands/
            ├── start.ts    Start web + server
            └── record.ts   Scripted notebook recording
```
