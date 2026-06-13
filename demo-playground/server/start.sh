#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
cd "$SCRIPT_DIR"

# Optional Bearer token for remote access (see .env.example).
if [ -f ".env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.local"
  set +a
fi

# Dev PYTHONPATH (e.g. repo python/) shadows the +full wheel and disables JIT in kernels.
unset PYTHONPATH

# Lean 4 certificate verification (elan installs to ~/.elan/bin)
if [ -d "${HOME}/.elan/bin" ]; then
  export PATH="${HOME}/.elan/bin:${PATH}"
fi

# Create venv if needed
if [ ! -d ".venv" ]; then
  echo "Creating Python virtualenv..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
pip install -q --upgrade pip
pip install -q -r requirements.txt

install_alkahest_wheel() {
  local PY_TAG WHEEL_PLAT
  PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  case "$(uname -m)" in
    x86_64) WHEEL_PLAT="manylinux_2_35_x86_64" ;;
    aarch64|arm64) WHEEL_PLAT="manylinux_2_35_aarch64" ;;
    *) echo "Unsupported architecture for +full wheel: $(uname -m)"; return 1 ;;
  esac
  local ALKAHEST_VERSION="${ALKAHEST_VERSION:-3.5.0}"
  local FULL_WHEEL="https://github.com/alkahest-cas/alkahest/releases/download/v${ALKAHEST_VERSION}/alkahest-${ALKAHEST_VERSION}+full-${PY_TAG}-${PY_TAG}-${WHEEL_PLAT}.whl"
  echo "Installing alkahest +full wheel: $FULL_WHEEL"
  if pip install -q --force-reinstall "$FULL_WHEEL"; then
    return 0
  fi
  local WHEEL
  WHEEL=$(find "${REPO_ROOT}/dist" -maxdepth 1 -type f \
    -name "*+full-${PY_TAG}-${PY_TAG}-*.whl" \
    | sort -V | tail -n1)
  if [ -n "$WHEEL" ]; then
    echo "GitHub wheel failed; trying local dist wheel: $WHEEL"
    pip install -q --force-reinstall "$WHEEL"
  fi
}

install_alkahest_local() {
  local MARKER=".venv/.alkahest-local-build"
  local MANIFEST="${REPO_ROOT}/alkahest-py/Cargo.toml"
  if [ ! -f "$MANIFEST" ]; then
    return 1
  fi
  if [ "${ALKAHEST_USE_RELEASE_WHEEL:-0}" = "1" ]; then
    return 1
  fi
  if ! command -v maturin >/dev/null 2>&1; then
    pip install -q maturin
  fi
  local NEED_BUILD=0
  if [ ! -f "$MARKER" ] || [ "${ALKAHEST_FORCE_REBUILD:-0}" = "1" ]; then
    NEED_BUILD=1
  elif find "${REPO_ROOT}/alkahest-py" "${REPO_ROOT}/alkahest-core" -name '*.rs' -newer "$MARKER" -print -quit 2>/dev/null | grep -q .; then
    NEED_BUILD=1
  fi
  if [ "$NEED_BUILD" != "1" ]; then
    echo "Local alkahest build up to date ($MARKER)"
    return 0
  fi
  echo "Building local alkahest into server venv (maturin develop)..."
  if (
    cd "$REPO_ROOT"
    maturin develop --release \
      --manifest-path alkahest-py/Cargo.toml \
      --features "jit egraph parallel groebner"
  ); then
    touch "$MARKER"
    return 0
  fi
  echo "maturin develop failed; falling back to release wheel"
  rm -f "$MARKER"
  return 1
}

# Prefer local source when developing in the repo checkout; fall back to wheels.
if ! install_alkahest_local; then
  install_alkahest_wheel
fi

# Bundled wheels ship native libs under site-packages/alkahest.libs/
LIBS_DIR="$(python -c "import pathlib; import site; roots=[pathlib.Path(p) for p in site.getsitepackages()+[site.getusersitepackages()]]; found=[d for r in roots for d in [r/'alkahest.libs'] if d.is_dir()]; print(found[0] if found else '')" 2>/dev/null || true)"
if [ -n "$LIBS_DIR" ]; then
  export LD_LIBRARY_PATH="${LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Isolated Jupyter kernel (server .venv only — never touches repo .venv / agent worktrees).
if [ -n "$LIBS_DIR" ]; then
  python -m ipykernel install --user --name=alkahest-playground --display-name="Alkahest Playground (+full)"
  KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/alkahest-playground/kernel.json"
  python -c "
import json, os
p = os.path.expanduser('${KERNEL_JSON}')
d = json.load(open(p))
if 'env' not in d:
    d['env'] = {}
d['env']['LD_LIBRARY_PATH'] = '${LIBS_DIR}'
json.dump(d, open(p, 'w'), indent=2)
"
fi

PORT="${PORT:-8000}"
LOG_FILE="${LOG_FILE:-/tmp/alkahest-server.log}"

if [[ "${1:-}" == "--daemon" ]]; then
  if pgrep -f "uvicorn main:app --host 0.0.0.0 --port ${PORT}" >/dev/null 2>&1; then
    echo "Stopping existing server on port ${PORT}..."
    pkill -f "uvicorn main:app --host 0.0.0.0 --port ${PORT}" || true
    sleep 1
  fi
  echo "Starting alkahest demo server on port ${PORT} (background, log: ${LOG_FILE})..."
  if [ -n "${ALKAHEST_SERVER_TOKEN:-}" ]; then
    echo "Auth: ALKAHEST_SERVER_TOKEN is set (Bearer required for sessions / Lean / WS)."
  else
    echo "Auth: no token (open server)."
  fi
  nohup env ALKAHEST_SERVER_TOKEN="${ALKAHEST_SERVER_TOKEN:-}" uvicorn main:app --host 0.0.0.0 --port "${PORT}" >>"${LOG_FILE}" 2>&1 &
  echo "PID $!"
  sleep 1
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && echo "Health check: ok" || echo "Health check: failed (see ${LOG_FILE})"
  exit 0
fi

echo "Starting alkahest demo server on port ${PORT} (foreground, --reload)..."
exec env ALKAHEST_SERVER_TOKEN="${ALKAHEST_SERVER_TOKEN:-}" uvicorn main:app --host 0.0.0.0 --port "${PORT}" --reload
