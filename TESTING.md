# Testing Architecture and Guidelines

## Overview

Testing a Computer Algebra System (CAS) with a Rust core, Python API, C math backend (GMP/FLINT/Arb), and MLIR code generation requires a multi-layered approach. Our testing philosophy focuses on fighting a two-front war:
1. **Mathematical Correctness**: Ensuring $2 + 2 = 4$, simplifications are mathematically sound, and the e-graph extracts logically equivalent expressions.
2. **Memory & Thread Safety**: Ensuring the boundaries between Python, Rust, and C do not leak memory, trigger use-after-frees, or cause data races.

This document outlines the tools we use, the invariants we test, and how our Continuous Integration (CI) pipeline is structured. **To reproduce the heavy, scheduled CI jobs on your machine** (Tier 1b + nightly matrix), skip to [§6 Running deep tests locally](#6-running-deep-tests-locally-match-scheduled-ci).

**Correctness vs performance.** The SymPy-backed oracle suite below checks that answers agree mathematically. It does **not** assert anything about runtime. Basic **wall-clock comparisons** against SymPy (and optional other CAS backends when installed) live in [`benchmarks/cas_comparison.py`](benchmarks/cas_comparison.py); see [`BENCHMARKS.md`](BENCHMARKS.md) for how to run them and interpret depth/size flags. CI’s benchmark job records those timings as artifacts for regression triage; they are informational and do not replace correctness tests.

---

## 1. Property-Based Testing (PBT)

We use PBT extensively to verify mathematical invariants across a vast state space of randomly generated Abstract Syntax Trees (ASTs).

### Rust Core (`proptest`)
* **What it tests**: Core algebraic invariants, e-graph soundness, and arithmetic correctness.
* **Key Invariants**:
  * *Idempotence*: `simplify(simplify(expr)) == simplify(expr)`
  * *E-class Equivalence*: Extracting any two nodes from the same e-class and evaluating them under identical bounds must yield the same result.
  * *Zero/One Identity*: `expr * 1 == expr`, `expr + 0 == expr`.
* **How to run (default / PR-style)**: `cargo test --workspace` — finishes quickly because Proptest uses a small default case count.
* **How to run (deep / scheduled CI parity)**: raise the case budget and use release mode (matches the `proptest` nightly shard in `.github/workflows/ci.yml`):
  ```bash
  export PROPTEST_CASES=50000
  cargo test --workspace --release
  ```

### Python API (`hypothesis`)
* **What it tests**: PyO3 bindings, edge-case floats (NaNs, infs, subnormals), and deeply nested PyTrees.
* **How to run (focused)**: `pytest tests/test_properties.py`
* **How to run (deep / scheduled CI parity)**: install the optional hypothesis stack, build the extension with Gröbner, then run the full suite with a higher example cap (matches the `hypothesis` nightly shard):
  ```bash
  pip install maturin pytest hypothesis sympy ruff mpmath symengine wolframclient
  maturin develop --manifest-path alkahest-py/Cargo.toml --features groebner
  export HYPOTHESIS_MAX_EXAMPLES=5000
  pytest tests/
  ```

---

## 2. Fuzzing (`AFL++`)

While PBT explores known bounds, coverage-guided fuzzing finds pathological edge cases, syntax horrors, and e-graph blowups. We utilize `cargo-afl` to instrument and fuzz our core Rust engine.

### Fuzzing Targets
1. **The Parser**: Feed mutated byte arrays into the parser. **Pass condition**: The parser gracefully returns a `Result::Err` rather than panicking.
2. **E-Graph Simplifier**: Feed random ASTs. **Pass condition**: The cost function must converge within a designated timeout without hitting an Out-Of-Memory (OOM) error.
3. **MLIR Lowering**: Feed valid ASTs into the custom MLIR dialect generator. **Pass condition**: No LLVM assertion failures or segfaults during compilation.

### How to Run Locally
Fuzz crates live under `fuzz/` (separate manifest, excluded from the root workspace). Targets include `fuzz_expr_builder` and `fuzz_simplifier`.

```bash
cargo install cargo-afl
cargo afl build --manifest-path fuzz/Cargo.toml --bin fuzz_expr_builder
cargo afl fuzz -i fuzz/in/expr_builder -o fuzz/out/expr_builder fuzz/target/debug/fuzz_expr_builder
# `fuzz_simplifier`: swap binary name, input `fuzz/in/simplifier`, output `fuzz/out/simplifier`.
```
CI caps each fuzz job at **2 hours** (`timeout 7200`); locally you can stop anytime or wrap the same command in `timeout`.

---

## 3. Memory Safety & Sanitizers

Because this project relies heavily on C libraries (GMP, FLINT) and exposes pointers to Python via PyO3, securing the Foreign Function Interface (FFI) is critical.

### Sanitizers (ASan, LSan, UBSan)
We compile our Rust test suite using LLVM sanitizers to catch memory violations instantly.
* **AddressSanitizer (ASan)**: Catches Out-of-Bounds accesses and Use-After-Free errors (especially critical when Python drops an object that Rust/C still expects).
* **LeakSanitizer (LSan)**: Ensures FLINT/GMP memory allocations are properly dropped.
* **UndefinedBehaviorSanitizer (UBSan)**: Catches unaligned pointers and integer overflows.

*Running with Sanitizers (requires Rust **nightly** + `rust-src`):*
```bash
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
```

**AddressSanitizer** — Tier 1 CI scopes this to `alkahest-core` only (full workspace + `build-std` is slow and easy to hit runner limits); locally you can match CI or widen:
```bash
RUSTFLAGS="-Zsanitizer=address" \
  cargo +nightly test -p alkahest-core --lib --tests \
    --target x86_64-unknown-linux-gnu \
    -Z build-std
# Optional: suppress known GMP/FLINT leak noise while debugging other issues:
#   LSAN_OPTIONS=detect_leaks=0
```

**ThreadSanitizer** — nightly `tsan` shard:
```bash
RUSTFLAGS="-Zsanitizer=thread" \
  cargo +nightly test --workspace --lib --tests \
    --target x86_64-unknown-linux-gnu \
    -Z build-std
```

**LeakSanitizer** — nightly `lsan` shard; use the repo suppression file and a symbolizer (paths may differ on your distro):
```bash
export LSAN_OPTIONS="suppressions=$PWD/lsan.supp"
export LLVM_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer-15   # or llvm-symbolizer

RUSTFLAGS="-Zsanitizer=leak" \
  cargo +nightly test --workspace --lib --tests \
    --target x86_64-unknown-linux-gnu \
    -Z build-std
```

### Valgrind
Valgrind is used exclusively on the Rust/C FFI layer (bypassing Python/PyO3 to avoid false positives from Python's `pymalloc`). It verifies that deep, long-running algebraic simplifications do not slowly leak memory.

*Local parity with the `valgrind` nightly shard:*
```bash
sudo apt-get install -y valgrind   # or your OS equivalent

export RUSTFLAGS="-C debuginfo=2 -Z dwarf-version=4"
cargo +nightly build --workspace --target x86_64-unknown-linux-gnu -Z build-std

for bin in target/x86_64-unknown-linux-gnu/debug/deps/alkahest_core-*; do
  [ -x "$bin" ] || continue
  valgrind --leak-check=full --error-exitcode=1 \
    --suppressions=valgrind.supp "$bin"
done
```

---

## 4. Oracle Cross-Validation

To guarantee our symbolic engine is structurally sound against industry standards, we run an Integration Oracle Suite. 

* **The Process**: We generate thousands of complex algebraic expressions and run them through our system, then run the exact same operations through `SymPy` (our oracle).
* **Verification**: We compute `simplify(OurAnswer - SymPyAnswer)`. The result must be exactly `0`.

SymPy is reused in a different role in **`benchmarks/cas_comparison.py`**: the same task is timed in Alkahest and SymPy so regressions in **performance** (not truth) are visible when someone inspects benchmark output or artifacts. That is complementary to this oracle, not a substitute for it.

---

## 5. CI/CD Pipeline Strategy

Given the computational expense of fuzzing and PBT, our GitHub Actions / CI pipeline is split into two tiers to maintain developer velocity while ensuring extreme rigor.

### Tier 1: Push / PR Checks (fast path)
* **Triggers**: Push or PR to `main` (not the scheduled cron).
* **Typical contents**: `cargo fmt`, `clippy`, `cargo test --workspace`, `ruff`, `pytest -m "not slow"`, ASan on `alkahest-core`, etc. (see `.github/workflows/ci.yml`).

### Tier 1b: Slow Python (sparse interpolation roadmap)
* **Triggers**: Same nightly **schedule** as Tier 2 (not on every push — keeps default CI fast).
* **Suite**: `pytest tests/test_sparse_interp.py -m slow --timeout=0 -v` after `maturin develop --features groebner`.

### Tier 2: Nightly integration (heavy matrix)
* **Triggers**: Cron on `main` (02:00 UTC); shards run in parallel.
* **Time Budget**: up to several hours per shard host cap.
* **Suites** (each is a separate matrix job): deep `proptest` (`PROPTEST_CASES=50000`, `--release`), deep `hypothesis` (`HYPOTHESIS_MAX_EXAMPLES=5000`, full `pytest tests/`), TSan, LSan, Valgrind, AFL fuzz targets, “extras” (oracle file if present, `cargo bench`, benchmark report scripts), etc.

---

## 6. Running deep tests locally (match scheduled CI)

Use this checklist when you want **more than** `cargo test --workspace` on a beefy machine. Commands mirror `.github/workflows/ci.yml` (`tier1-python-slow` + `nightly` matrix).

| CI job | What to run locally |
|--------|---------------------|
| **Tier 1b** | After `maturin develop --manifest-path alkahest-py/Cargo.toml --features groebner`: `pytest tests/test_sparse_interp.py -m slow --timeout=0 -v` |
| **proptest** | `export PROPTEST_CASES=50000` then `cargo test --workspace --release` |
| **hypothesis** | See §1 Python (`HYPOTHESIS_MAX_EXAMPLES=5000`, `pytest tests/`) |
| **tsan** | See §3 ThreadSanitizer block |
| **lsan** | See §3 LeakSanitizer block |
| **valgrind** | See §3 Valgrind block |
| **fuzz-\*** | See §2; use `--manifest-path fuzz/Cargo.toml` |
| **extras** | `maturin develop --manifest-path alkahest-py/Cargo.toml --features groebner`; `pytest tests/test_oracle.py -v` if configured; `cargo bench --workspace`; optional scripts under `benchmarks/` (may need SymPy / optional CAS; CI sets `RUN_COMMERCIAL_CAS` where applicable) |

**Lean / docs / cross-platform** workflows have their own YAML files (e.g. `.github/workflows/lean.yml`, `ci-cross.yml`); run those suites locally only when you touch those areas.

---

## Contributing

When adding a new mathematical primitive, rewrite rule, or compiler lowering pass:
1. Write a standard unit test demonstrating the basic functionality.
2. Add the primitive to the AST generator in the `proptest` suite.
3. Ensure the operation passes cleanly under AddressSanitizer (see §3).
