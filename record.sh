#!/usr/bin/env bash
# Record side-by-side Groebner comparison at 1080p.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PLAYGROUND="$(cd "$ROOT/../../demo-playground" 2>/dev/null || cd "$ROOT/../../../demo-playground" && pwd)"
# Worktree layout: groebner-comparison-demo-v2 is under tmp/temp-alkahest/
if [[ -d "$ROOT/../demo-playground" ]]; then
  PLAYGROUND="$(cd "$ROOT/../demo-playground" && pwd)"
fi
if [[ -d "$ROOT/demo-playground" ]]; then
  PLAYGROUND="$(cd "$ROOT/demo-playground" && pwd)"
fi

OUT="${1:-$ROOT/recordings/groebner-cyclic5-1080p.webm}"
mkdir -p "$(dirname "$OUT")"

WEB_PORT="${WEB_PORT:-3001}"
SERVER_PORT="${SERVER_PORT:-8001}"

cd "$PLAYGROUND"
npx tsx cli/src/index.ts record \
  --layout split \
  --code-left "$ROOT/demos/alkahest_panel.py" \
  --code-right "$ROOT/demos/sympy_panel.py" \
  --output "$OUT" \
  --url "http://localhost:${WEB_PORT}" \
  --server-url "http://localhost:${SERVER_PORT}" \
  --width 1920 \
  --height 1080

echo "Wrote $OUT"
