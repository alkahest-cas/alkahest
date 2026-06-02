# Alkahest Demo Playground

An interactive web application for demoing, testing, and recording the [Alkahest](https://alkahest-cas.github.io/) CAS Python library. Designed to match the alkahest visual identity and be controllable entirely from the CLI.

## Features

| Feature | Description |
|---------|-------------|
| **Notebook mode** | Jupyter-style code cells with CodeMirror editor |
| **Dual execution** | WASM (Pyodide) for SymPy/pure Python; server kernel for alkahest |
| **Rich outputs** | LaTeX (KaTeX), matplotlib figures, HTML, JSON, **Lean 4 certificates** |
| **Lean verification** | Export `.lean` proofs from alkahest; typecheck via server (`lake env lean`) |
| **Local wheel** | Upload a `.whl` to test a dev build of alkahest |
| **Agent chat** | Provider-agnostic AI agent (Vercel AI SDK) with `run_python` tool |
| **Recording** | In-browser `MediaRecorder` or headless Playwright from the CLI |
| **CLI orchestration** | `start`, `record`, `demo <prompt>` commands |

## Architecture

```
demo-playground/
├── web/      Next.js 14 app (TypeScript, Tailwind, CodeMirror, KaTeX)
├── server/   FastAPI + Jupyter kernel (Python, ipykernel)
└── cli/      Node.js orchestration CLI (Commander, Playwright)
```

**Execution routing** — the browser detects `import alkahest` / `from alkahest` in cell code and sends those cells to the Python server automatically. Pure-Python cells run in Pyodide (browser WASM). This decision is configurable in Settings.

**Agent** — uses [Vercel AI SDK](https://ai-sdk.dev/) which is provider-agnostic. Supported providers include Anthropic, OpenAI, Google, Mistral, Groq, xAI, DeepSeek, Together.ai, Fireworks, Cerebras, and any **OpenAI-compatible** endpoint (Ollama, vLLM, LiteLLM, etc.). Configure via Settings (gear icon) or `AI_PROVIDER` / `AI_MODEL` in `web/.env.local`. The agent has a `run_python` tool wired to the same Jupyter kernel and is primed with the full alkahest skill document.

## Prerequisites

- **Node.js** ≥ 18 with **pnpm** (`npm i -g pnpm`)
- **Python** ≥ 3.9
- An API key for at least one AI provider (for agent chat)

## Quick start

### 1. Configure environment

```bash
cd demo-playground
cp .env.example web/.env.local
# Edit web/.env.local — at minimum set ANTHROPIC_API_KEY (or whichever provider you use)
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

- **WASM** — SymPy and pure Python cells work with no server.
- **Bring your own backend** — open settings (`Ctrl+/`) and set an Alkahest server URL + token, or a remote Jupyter URL + token, to run `import alkahest` and use a private kernel.
- **Agent chat** — requires self-hosting (`pnpm start`); the hosted site is notebook-only for the agent.

Pushes to `main` under `demo-playground/` deploy automatically via `.github/workflows/playground.yml`.

---

## Notebook mode

The default view is a notebook with two starter cells — one using alkahest, one using SymPy for comparison.

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

## Agent chat mode

Navigate to **Agent** in the navbar (or `/agent`).

The agent:
1. Receives your natural-language request
2. Writes Python code using alkahest (and/or SymPy for comparison)
3. Runs it via the `run_python` tool on the shared Jupyter kernel
4. Can emit Lean certificates (`display_lean_cert`) and typecheck them with `verify_lean`
5. Shows reasoning text, code, and outputs inline
6. Accepts follow-up messages (stateful kernel — variables persist)

The agent is primed with the full `alkahest-skill/alkahest.md` document from the repo.

---

## Settings

Click the gear icon (⚙) in the navbar to configure:

- **Server URL** — Python server address (default `http://localhost:8000`)
- **Backend type** — Alkahest server or remote **Jupyter Server** (URL + token)
- **Access token** — Jupyter token or Alkahest `ALKAHEST_SERVER_TOKEN` (stored in browser)
- **Execution mode** — Auto / WASM only / Server only
- **Ctrl+/** — toggle settings from anywhere (except while typing in a cell)
- **AI provider** — Anthropic, OpenAI, Google, Mistral, Groq, xAI, DeepSeek, Together.ai, Fireworks, Cerebras, or OpenAI-compatible (custom base URL)
- **AI model** — any model string supported by your provider
- **OpenAI-compatible** — optional base URL and API key (browser-local or via `OPENAI_COMPATIBLE_*` env vars)

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

### CLI — agent demo recording

Give the agent a prompt and capture the full interaction:

```bash
npx tsx cli/src/index.ts demo \
  "Demonstrate how alkahest differentiates a product of trig functions and show the steps" \
  --output recordings/agent-demo.webm
```

This command:
1. Starts servers (if not already running)
2. Opens a headless browser with Playwright
3. Navigates to the Agent page and sends your prompt
4. Waits for the agent to complete
5. Saves the video to the output file

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
│   │   │   ├── page.tsx             Notebook page
│   │   │   ├── agent/page.tsx       Agent chat page
│   │   │   ├── api/agent/route.ts   AI SDK streaming endpoint
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── notebook/            Cell, Notebook, Output
│   │   │   ├── agent/               AgentChat
│   │   │   └── ui/                  Nav, Settings
│   │   └── lib/
│   │       ├── alkahest-skill.ts    Agent system prompt
│   │       └── execution.ts         WASM/server routing
│   └── public/
│       └── pyodide-worker.js        Web Worker for Pyodide
│
├── server/                 Python execution backend
│   ├── main.py             FastAPI app
│   ├── kernel_manager.py   Jupyter kernel wrapper
│   ├── requirements.txt
│   └── start.sh            Bootstraps venv and starts uvicorn
│
└── cli/                    Orchestration CLI
    └── src/
        ├── index.ts
        └── commands/
            ├── start.ts    Start web + server
            ├── record.ts   Scripted notebook recording
            └── demo.ts     Agent demo + recording
```
