#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Install alkahest (+full: JIT + parallel) unless a newer local wheel exists in dist/
WHEEL=$(ls ../dist/*.whl 2>/dev/null | sort -V | tail -n1)
if [ -n "$WHEEL" ]; then
  echo "Installing alkahest wheel: $WHEEL"
  pip install -q "$WHEEL"
else
  PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
  FULL_WHEEL="https://github.com/alkahest-cas/alkahest/releases/download/v3.4.0/alkahest-3.4.0+full-${PY_TAG}-${PY_TAG}-manylinux_2_35_x86_64.whl"
  echo "Installing alkahest +full wheel: $FULL_WHEEL"
  pip install -q "$FULL_WHEEL"
fi

# Bundled wheels ship native libs under site-packages/alkahest.libs/
LIBS_DIR="$(python -c "import pathlib; import site; roots=[pathlib.Path(p) for p in site.getsitepackages()+[site.getusersitepackages()]]; found=[d for r in roots for d in [r/'alkahest.libs'] if d.is_dir()]; print(found[0] if found else '')" 2>/dev/null || true)"
if [ -n "$LIBS_DIR" ]; then
  export LD_LIBRARY_PATH="${LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Jupyter kernels use ~/alkahest/.venv (alkahest-dev). Ensure +full wheel, not editable dev install.
REPO_VENV="${SCRIPT_DIR}/../../.venv"
if [ -d "$REPO_VENV" ]; then
  REPO_PY="${REPO_VENV}/bin/python"
  if [ -x "$REPO_PY" ]; then
    REPO_TAG=$("$REPO_PY" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    FULL_WHEEL_REPO="https://github.com/alkahest-cas/alkahest/releases/download/v3.4.0/alkahest-3.4.0+full-${REPO_TAG}-${REPO_TAG}-manylinux_2_35_x86_64.whl"
    echo "Ensuring alkahest +full in repo venv for alkahest-dev kernel..."
    "$REPO_PY" -m pip install -q --force-reinstall "$FULL_WHEEL_REPO"
    REPO_LIBS=$("$REPO_PY" -c "import pathlib,site; roots=[pathlib.Path(p) for p in site.getsitepackages()]; print(next((r/'alkahest.libs' for r in roots if (r/'alkahest.libs').is_dir()), ''))")
    if [ -n "$REPO_LIBS" ]; then
      "$REPO_PY" -m ipykernel install --user --name=alkahest-dev --display-name="Alkahest Dev (3.3+full)"
      KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/alkahest-dev/kernel.json"
      "$REPO_PY" -c "
import json, os
p = os.path.expanduser('${KERNEL_JSON}')
d = json.load(open(p))
d['env'] = {'LD_LIBRARY_PATH': '${REPO_LIBS}'}
json.dump(d, open(p, 'w'), indent=2)
"
    fi
  fi
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
  nohup uvicorn main:app --host 0.0.0.0 --port "${PORT}" >>"${LOG_FILE}" 2>&1 &
  echo "PID $!"
  sleep 1
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && echo "Health check: ok" || echo "Health check: failed (see ${LOG_FILE})"
  exit 0
fi

echo "Starting alkahest demo server on port ${PORT} (foreground, --reload)..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --reload
