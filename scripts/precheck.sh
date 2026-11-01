#!/usr/bin/env bash
# Local pre-push gate: the cheap, deterministic half of CI's `Tier 1a — lint`
# job (and the Python API freeze check), in the same order, stopping at the
# first failure. Run it before pushing Rust or Python changes:
#
#     scripts/precheck.sh
#
# It does not run the test suites — `cargo test --workspace` and
# `pytest tests/` are still yours to run (see AGENTS.md / CONTRIBUTING.md).
#
# Environment:
#   LLVM_SYS_211_PREFIX   set -> also lint `--features jit` (needs LLVM 21), as CI does
#   PRECHECK_SKIP_CLIPPY  =1 -> skip the clippy passes (fmt and ruff still run)
#
# Requires FLINT where the build can find it (FLINT_LIB_DIR / FLINT_INCLUDE_DIR
# if it is not on the default search path), exactly as `cargo build` does.

set -euo pipefail

cd "$(dirname "$0")/.."

bold=$'\033[1m'; red=$'\033[31m'; yellow=$'\033[33m'; green=$'\033[32m'; reset=$'\033[0m'
if [ ! -t 1 ]; then bold='' red='' yellow='' green='' reset=''; fi

step=0
run() {
  step=$((step + 1))
  printf '\n%s==> [%d] %s%s\n' "$bold" "$step" "$*" "$reset"
  if ! "$@"; then
    printf '\n%sprecheck FAILED at step %d:%s %s\n' "$red" "$step" "$reset" "$*" >&2
    exit 1
  fi
}
warn() { printf '%swarning:%s %s\n' "$yellow" "$reset" "$*" >&2; }

# Python tools: whatever is on PATH, else the project venv (`uv sync --group dev`).
pytool() {
  local tool=$1; shift
  if command -v "$tool" >/dev/null 2>&1; then
    "$tool" "$@"
  elif [ -x ".venv/bin/$tool" ]; then
    ".venv/bin/$tool" "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv run --no-sync "$tool" "$@"
  else
    echo "error: $tool not found (install the dev group: uv sync --no-install-project --group dev)" >&2
    return 1
  fi
}

# ── Toolchain: CI lints with the *current* stable. An older local clippy
# misses lints that newer stable adds, so a clean run here can still go red in
# CI. Use stable explicitly when rustup is present, and say if it is stale.
if command -v rustup >/dev/null 2>&1; then
  export RUSTUP_TOOLCHAIN=stable
  to=()
  if command -v timeout >/dev/null 2>&1; then to=(timeout 30); fi
  # ${to[@]+...}: an empty array under `set -u` is an error in macOS's bash 3.2.
  # rustup >= 1.28 exits 100 when *any* toolchain has an update, so only a
  # status other than 0/100 means the check itself failed.
  rc=0
  check=$(${to[@]+"${to[@]}"} rustup check 2>/dev/null) || rc=$?
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 100 ]; then
    if grep -E '^stable-' <<<"$check" | grep -qi 'update available'; then
      warn "local stable is behind the current stable that CI uses:"
      grep -E '^stable-' <<<"$check" >&2
      warn "run \`rustup update stable\`, or expect CI clippy to find lints this run does not."
    fi
  else
    warn "could not run \`rustup check\` (offline?); cannot tell whether local stable is current."
  fi
else
  warn "rustup not found; using whatever cargo is on PATH (CI uses current stable)."
fi
printf '%s\n' "$(rustc --version)" "$(cargo clippy --version 2>/dev/null || echo 'clippy: not installed')"

# Same as CI's workflow-level env.
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

# ── 1. No committed _site/ (CI: "No committed _site/")
no_site() {
  local tracked
  tracked=$(git ls-files _site || true)
  if [ -n "$tracked" ]; then
    echo "_site/ must not be tracked: git rm -r --cached _site" >&2
    return 1
  fi
}
run no_site

# ── 2. rustfmt
run cargo fmt --all -- --check

# ── 3. clippy, one pass per feature set, as CI runs them. `--all-features` is
# not a substitute (it needs LLVM and the CUDA toolkit).
if [ "${PRECHECK_SKIP_CLIPPY:-0}" != 1 ]; then
  run cargo clippy --all-targets -- -D warnings
  for f in egraph parallel cranelift; do
    run cargo clippy --all-targets --features "$f" -- -D warnings
  done
  if [ -n "${LLVM_SYS_211_PREFIX:-}" ]; then
    run cargo clippy --all-targets --features jit -- -D warnings
  else
    warn "LLVM_SYS_211_PREFIX is not set: skipping \`clippy --features jit\` (CI runs it)."
  fi
  run cargo clippy --all-targets --features groebner-cuda -- -D warnings
fi

# ── 4. The extension crate with warnings denied.
run cargo check -p alkahest-py

# ── 5. ruff, same file scope as CI.
run pytool ruff check python/ tests/
run pytool ruff format --check python/ tests/

# ── 6. alkahest.__all__ is append-only vs main (CI: semver-check workflow).
if git rev-parse --verify -q origin/main >/dev/null; then
  run python3 scripts/check_api_freeze.py origin/main
else
  warn "no origin/main ref: skipping the alkahest.__all__ freeze check."
fi

printf '\n%sprecheck passed (%d steps).%s Tests are not part of it: run cargo test --workspace and pytest tests/.\n' \
  "$green" "$step" "$reset"
