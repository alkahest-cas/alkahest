#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
  FULL_WHEEL="https://github.com/alkahest-cas/alkahest/releases/download/v3.0.0/alkahest-3.0.0+full-${PY_TAG}-${PY_TAG}-manylinux_2_35_x86_64.whl"
  echo "Installing alkahest +full wheel: $FULL_WHEEL"
  pip install -q "$FULL_WHEEL"
fi

# Bundled wheels ship native libs under site-packages/alkahest.libs/
LIBS_DIR="$(python -c "import pathlib; import site; roots=[pathlib.Path(p) for p in site.getsitepackages()+[site.getusersitepackages()]]; found=[d for r in roots for d in [r/'alkahest.libs'] if d.is_dir()]; print(found[0] if found else '')" 2>/dev/null || true)"
if [ -n "$LIBS_DIR" ]; then
  export LD_LIBRARY_PATH="${LIBS_DIR}:${LD_LIBRARY_PATH:-}"
fi

echo "Starting alkahest demo server on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
