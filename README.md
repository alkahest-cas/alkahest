# Groebner basis comparison demo (v2)

Side-by-side **Alkahest** vs **SymPy** recording of cyclic-5 (lex) at **1920x1080**.

## Worktree

```bash
# This folder is a git worktree on branch demo/groebner-comparison-v2
git worktree list | grep groebner-comparison-demo-v2
```

## Record

1. Build alkahest with groebner and start the playground:

```bash
cd demo-playground
pnpm install
pnpm start   # web :3000, server :8000
```

2. In another terminal:

```bash
bash record.sh
# -> recordings/groebner-cyclic5-1080p.webm
```

## Panel sources

| Panel | File |
|-------|------|
| Alkahest (left) | `demos/alkahest_panel.py` |
| SymPy (right) | `demos/sympy_panel.py` |

Cells are split on `# ---` (markdown problem statement, then code).

## Recording fixes (v2)

- Zen mode applies on first paint (no navbar flash).
- Recorder waits for `data-recording-ready` before capturing execution.
- `/compare` route shows both panels at 1080p width.
