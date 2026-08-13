#!/usr/bin/env bash
# verify_cuda_on_gpu.sh — run every CUDA gate on real hardware and print a verdict.
#
# The CUDA surface has no CI: `.github/workflows/cuda_nightly.yml` targets
# `[self-hosted, gpu-3090]`, no such runner is registered, and no other job
# builds the extension with `--features cuda`. Every CUDA claim this project
# makes therefore rests on someone running these commands by hand on a GPU box.
# This script is that run, so the next person does not reconstruct the
# invocation — including the environment traps — from a terminal scrollback.
#
# Usage:
#   scripts/verify_cuda_on_gpu.sh              # all gates
#   scripts/verify_cuda_on_gpu.sh --quick      # skip the Python build (~15 min faster)
#
# Exit status is the verdict: 0 iff every gate passed.
#
# Requirements: an NVIDIA GPU with driver, CUDA toolkit (`compute-sanitizer`),
# and LLVM 15 with the NVPTX target (`llc-15 --version | grep nvptx`). The
# `cuda` feature implies the LLVM JIT, so this cannot run on a box without it.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$PWD"
QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

# ---------------------------------------------------------------------------
# Environment. These are not preferences — each one is a trap that silently
# breaks the build or, worse, makes a gate pass without running.
# ---------------------------------------------------------------------------

# A user-installed gcc ahead of the system one lacks the distro's multiarch
# include path, and `llvm-sys` dies on `bits/wordsize.h: No such file`.
export PATH="/usr/bin:$PATH"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

# `cuda` needs NVPTX, which the distro's default `llc` may not be built with.
export LLVM_SYS_150_PREFIX="${LLVM_SYS_150_PREFIX:-/usr/lib/llvm-15}"

# Makes the device probes assert instead of skipping. Without it a box with no
# GPU reports every gate below as passing, having exercised nothing.
export ALKAHEST_GPU_TESTS=1

FEATURES="cuda,groebner-cuda"
GPU_TESTS=(--test nvptx_gpu --test groebner_cuda)

# Two tests deliberately request ordinals that do not exist, to prove the
# fallback path records the failure and that the count agrees with what
# launches. Under a sanitizer their expected `CUDA_ERROR_INVALID_DEVICE` is
# reported as an error — 17 of them, all on `cuDeviceGet`, none from a kernel —
# which buries a real finding in noise that is working as intended. They are
# host-side error-path tests that launch nothing, so excluding them from the
# sanitizer shard costs no kernel coverage; the ordinary `cargo test` gate above
# still runs them.
SANITIZER_SKIPS=(
    --skip requested_device_that_does_not_exist_falls_back_and_records_why
    --skip cuda_device_count_matches_the_ordinals_that_launch
)
LOG_DIR="${TMPDIR:-/tmp}/alkahest-cuda-verify-$$"
mkdir -p "$LOG_DIR"

PASSED=(); FAILED=(); SKIPPED=()

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mPASS\033[0m  %s\n' "$*"; PASSED+=("$1"); }
bad()  { printf '   \033[31mFAIL\033[0m  %s\n' "$*"; FAILED+=("$1"); }
skip() { printf '   \033[33mSKIP\033[0m  %s\n' "$*"; SKIPPED+=("$1"); }

# ---------------------------------------------------------------------------
say "Environment"
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "   no usable nvidia-smi — this script only means anything on a GPU host" >&2
    exit 2
fi
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | sed 's/^/   GPU /'
echo "   nvcc:      $(nvcc --version 2>/dev/null | tail -1 || echo 'not found')"
echo "   llc-15:    $(llc-15 --version 2>/dev/null | grep -c nvptx || echo 0) nvptx targets"
echo "   rustc:     $(rustc --version)"
echo "   sanitizer: $(compute-sanitizer --version 2>/dev/null | grep -i version | head -1)"
echo "   commit:    $(git rev-parse HEAD)"
echo "   logs:      $LOG_DIR"

# ---------------------------------------------------------------------------
say "Gate 1 — CUDA test suites (the ALKAHEST_GPU_TESTS=1 hard tier)"
# ---------------------------------------------------------------------------
# `assert_ran_on_gpu` fails here if any prime silently reduced on the CPU, which
# is the whole point of the tier: the results are correct either way, so without
# that assertion a run whose driver calls all failed reports success.
if cargo test --features "$FEATURES" "${GPU_TESTS[@]}" > "$LOG_DIR/tests.log" 2>&1; then
    n=$(grep -cE '^test .* \.\.\. ok' "$LOG_DIR/tests.log")
    ok "cargo test ($n tests, no silent CPU fallback)"
else
    bad "cargo test — see $LOG_DIR/tests.log"
fi

# ---------------------------------------------------------------------------
say "Gate 2 — compute-sanitizer"
# ---------------------------------------------------------------------------
# --target-processes all is load-bearing. The default (application-only)
# instruments `cargo`, a process that makes no CUDA calls: the step then emits a
# banner, no ERROR SUMMARY, and a green tick having inspected nothing.
#
# Scoped to the two integration targets that launch kernels; wrapping the whole
# `cargo test` drags rustdoc's doc-test runner in, where it segfaults after the
# CUDA suites pass but before the summary prints.
#
# An ERROR SUMMARY line is itself the evidence of instrumentation: with no
# kernel launched the tool says "Target application terminated before first
# instrumented API call" and prints no summary at all.
for tool in memcheck racecheck initcheck synccheck; do
    log="$LOG_DIR/$tool.log"
    compute-sanitizer --target-processes all --tool "$tool" \
        cargo test --features "$FEATURES" "${GPU_TESTS[@]}" \
        -- "${SANITIZER_SKIPS[@]}" > "$log" 2>&1
    status=$?
    summary=$(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY' "$log" | head -1)
    # The count is judged, not just the exit status. compute-sanitizer exits 0
    # while reporting errors in some modes, so trusting `$?` alone reproduces
    # the exact failure this shard exists to prevent: a green tick over a log
    # that says something is wrong.
    count=$(sed -E 's/.*SUMMARY: ([0-9]+).*/\1/' <<<"$summary")
    if [[ -z "$summary" ]]; then
        bad "$tool — no summary line: the tool inspected nothing (see $log)"
    elif [[ ! "$count" =~ ^[0-9]+$ ]]; then
        bad "$tool — unparseable summary '$summary' (see $log)"
    elif [[ "$count" -ne 0 || $status -ne 0 ]]; then
        bad "$tool — $summary (exit $status, see $log)"
    else
        ok "$tool — ${summary#*= }"
    fi
done

cat <<'COVERAGE'

   Coverage note — two of those four cannot fail on today's kernels:
     racecheck  detects __shared__ hazards; no kernel declares .shared
     synccheck  detects barrier/warp-sync misuse; no kernel emits bar.sync
   Their green ticks say "this construct is absent", not "this code is correct".
   memcheck and initcheck have both been shown to report on these kernels by
   deliberately breaking them (see the report under temp-alkahest/testing/).
COVERAGE

# ---------------------------------------------------------------------------
say "Gate 3 — Python CUDA surface"
# ---------------------------------------------------------------------------
# No CI job has ever built this. `ak.compile_cuda` once raised AttributeError on
# a build whose own capabilities() advertised `cuda: true`, and it survived three
# releases because the Rust tests all passed: the gap was in the re-export.
if [[ $QUICK -eq 1 ]]; then
    skip "maturin develop (--quick)"
elif [[ ! -x .venv/bin/python ]]; then
    skip "maturin develop (no .venv — run: uv sync --no-install-project --group dev)"
else
    if .venv/bin/python -m maturin develop --manifest-path alkahest-py/Cargo.toml \
        --release --features "cuda,groebner-cuda,egraph,groebner,parallel,cranelift" \
        > "$LOG_DIR/maturin.log" 2>&1; then
        ok "maturin develop --release --features cuda"
    else
        bad "maturin develop — see $LOG_DIR/maturin.log"
    fi
fi

if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import alkahest' 2>/dev/null; then
    if .venv/bin/python -m pytest tests/test_cuda.py tests/test_agent_contract.py -q \
        > "$LOG_DIR/pytest.log" 2>&1; then
        ok "pytest test_cuda.py + test_agent_contract.py"
    else
        bad "pytest CUDA surface — see $LOG_DIR/pytest.log"
    fi

    # A capability bit that no caller can reach is the defect contract v3 exists
    # to prevent; this checks the built extension rather than the source.
    .venv/bin/python - <<'PY' > "$LOG_DIR/caps.log" 2>&1
import alkahest as ak
caps = ak.capabilities()
feats = caps["features"]
assert caps["contract_version"] >= 3, caps["contract_version"]
assert feats["cuda"] is True, "not a cuda build"
assert feats["llvm_jit"] is True, "cuda implies alkahest-core/jit"
for gone in ("groebner_cuda", "numpy"):
    assert gone not in feats, f"{gone} came back"
for name in ("compile_cuda", "CudaCompiledFn", "cuda_device_count"):
    assert hasattr(ak, name), name
n = ak.cuda_device_count()
assert n > 0, "no device"
print(f"contract v{caps['contract_version']}, {n} device(s), entry points reachable")
PY
    if [[ $? -eq 0 ]]; then
        ok "capability contract — $(cat "$LOG_DIR/caps.log")"
    else
        bad "capability contract — see $LOG_DIR/caps.log"
    fi
else
    skip "Python gates (extension not importable)"
fi

# ---------------------------------------------------------------------------
say "Gate 4 — GpuBackendReport counts on real hardware"
# ---------------------------------------------------------------------------
# `reductions_on_gpu += 1` had never executed anywhere when it shipped. A
# counter stuck at zero makes ran_on_gpu() permanently false and the hard tier's
# assertion unpassable; one that never sees the CPU path makes a fallback
# indistinguishable from a real GPU run. Both directions have to be observed.
if cargo test --features "$FEATURES" --test groebner_cuda \
        requested_device_that_does_not_exist_falls_back_and_records_why \
        > "$LOG_DIR/fallback.log" 2>&1; then
    ok "forced fallback records first_gpu_error and still returns the right basis"
else
    bad "forced-fallback test — see $LOG_DIR/fallback.log"
fi

# ---------------------------------------------------------------------------
say "Verdict"
# ---------------------------------------------------------------------------
printf '   %d passed, %d failed, %d skipped\n' \
    "${#PASSED[@]}" "${#FAILED[@]}" "${#SKIPPED[@]}"
for f in ${FAILED+"${FAILED[@]}"}; do printf '     failed: %s\n' "$f"; done
for s in ${SKIPPED+"${SKIPPED[@]}"}; do printf '     skipped: %s\n' "$s"; done
echo "   logs kept in $LOG_DIR"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    printf '\n   \033[32mCUDA verified on this host at %s\033[0m\n' "$(git rev-parse --short HEAD)"
    exit 0
fi
printf '\n   \033[31mCUDA NOT verified: %d gate(s) failed\033[0m\n' "${#FAILED[@]}"
exit 1
