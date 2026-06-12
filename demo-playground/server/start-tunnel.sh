#!/usr/bin/env bash
# Ephemeral Cloudflare quick tunnel → local demo server (port 8000).
# For GitHub Pages playground (HTTPS) — see temp-alkahest/demoing/remote-server.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
LOG_FILE="${LOG_FILE:-/tmp/alkahest-cloudflared.log}"
TUNNEL_URL_FILE="${TUNNEL_URL_FILE:-${SCRIPT_DIR}/../../temp-alkahest/demoing/tunnel-url.txt}"
PID_FILE="${PID_FILE:-/tmp/alkahest-cloudflared.pid}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared not found. Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
  exit 1
fi

if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "Demo server not reachable on http://127.0.0.1:${PORT}"
  echo "Start it first: cd ${SCRIPT_DIR} && ./start.sh --daemon"
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Stopping existing quick tunnel (PID $(cat "$PID_FILE"))..."
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  sleep 1
fi

rm -f "$LOG_FILE"
nohup cloudflared tunnel --url "http://127.0.0.1:${PORT}" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG_FILE" 2>/dev/null | head -1 || true)
  if [ -n "$URL" ]; then
    break
  fi
  sleep 1
done

if [ -z "$URL" ]; then
  echo "Timed out waiting for tunnel URL. See ${LOG_FILE}"
  exit 1
fi

# Wait until reachable (quick tunnels can take a few seconds)
for _ in $(seq 1 20); do
  if curl -sf "${URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

TOKEN=""
if [ -f "${SCRIPT_DIR}/.env.local" ]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env.local"
fi

mkdir -p "$(dirname "$TUNNEL_URL_FILE")"
cat >"$TUNNEL_URL_FILE" <<EOF
# Alkahest demo server — Cloudflare quick tunnel (ephemeral)
# Restart: demo-playground/server/start-tunnel.sh  (URL changes each run)
# Backend: cd demo-playground/server && ./start.sh --daemon

${URL}

Playground settings (https://alkahest-cas.github.io/playground/):
  Server URL:  ${URL}
  WebSocket:   wss://${URL#https://}
  Token:       ${ALKAHEST_SERVER_TOKEN:-<set in server/.env.local>}
EOF

echo "Tunnel URL: ${URL}"
echo "Saved to:   ${TUNNEL_URL_FILE}"
echo "Log:        ${LOG_FILE}  (PID $(cat "$PID_FILE"))"
echo ""
echo "GitHub Pages playground → Settings (Ctrl+/):"
echo "  Server URL: ${URL}"
if [ -n "${ALKAHEST_SERVER_TOKEN:-}" ]; then
  echo "  Token:      ${ALKAHEST_SERVER_TOKEN}"
fi
