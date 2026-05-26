# Groebner basis comparison demo

Side-by-side **Alkahest** vs **SymPy** recording of cyclic-5 (lex) at **1920x1080**.

## Record

1. Build alkahest with groebner and start the playground:

```bash
cd demo-playground
pnpm install
pnpm start   # web :3000, server :8000
```

2. In another terminal:

```bash
bash groebner-comparison/record.sh
# -> groebner-comparison/recordings/groebner-cyclic5-1080p.webm
```

## Panel sources

| Panel | File |
|-------|------|
| Alkahest (left) | `demos/alkahest_panel.py` |
| SymPy (right) | `demos/sympy_panel.py` |

Cells are split on `# ---` (markdown problem statement, then code).

## CLI (manual)

```bash
npx tsx cli/src/index.ts record \
  --layout split \
  --code-left groebner-comparison/demos/alkahest_panel.py \
  --code-right groebner-comparison/demos/sympy_panel.py \
  --output groebner-comparison/recordings/groebner-cyclic5-1080p.webm \
  --width 1920 --height 1080
```
