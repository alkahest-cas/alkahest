# Testing Architecture and Guidelines

## Overview

Testing a Computer Algebra System (CAS) with a Rust core, Python API, C math backend (GMP/FLINT/Arb), and MLIR code generation requires a multi-layered approach. Our testing philosophy focuses on fighting a two-front war:
1. **Mathematical Correctness**: Ensuring $2 + 2 = 4$, simplifications are mathematically sound, and the e-graph extracts logically equivalent expressions.
2. **Memory & Thread Safety**: Ensuring the boundaries between Python, Rust, and C do not leak memory, trigger use-after-frees, or cause data races.

This document outlines the tools we use, the invariants we test, and how our Continuous Integration (CI) pipeline is structured.

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
* **How to run**: `cargo test --release` (Proptest is integrated directly into the standard Rust test suite).

### Python API (`hypothesis`)
* **What it tests**: PyO3 bindings, edge-case floats (NaNs, infs, subnormals), and deeply nested PyTrees.
* **How to run**: `pytest tests/test_properties.py`

---

## 2. Fuzzing (`AFL++`)

While PBT explores known bounds, coverage-guided fuzzing finds pathological edge cases, syntax horrors, and e-graph blowups. We utilize `cargo-afl` to instrument and fuzz our core Rust engine.

### Fuzzing Targets
1. **The Parser**: Feed mutated byte arrays into the parser. **Pass condition**: The parser gracefully returns a `Result::Err` rather than panicking.
2. **E-Graph Simplifier**: Feed random ASTs. **Pass condition**: The cost function must converge within a designated timeout without hitting an Out-Of-Memory (OOM) error.
3. **MLIR Lowering**: Feed valid ASTs into the custom MLIR dialect generator. **Pass condition**: No LLVM assertion failures or segfaults during compilation.

### How to Run Locally
```bash
# Install AFL++ tooling
cargo install cargo-afl

# Build and run a specific fuzzer target
cargo afl build --bin fuzz_parser
cargo afl fuzz -i in/ -o out/ target/debug/fuzz_parser
```

---

## 3. Memory Safety & Sanitizers

Because this project relies heavily on C libraries (GMP, FLINT) and exposes pointers to Python via PyO3, securing the Foreign Function Interface (FFI) is critical.

### Sanitizers (ASan, LSan, UBSan)
We compile our Rust test suite using LLVM sanitizers to catch memory violations instantly.
* **AddressSanitizer (ASan)**: Catches Out-of-Bounds accesses and Use-After-Free errors (especially critical when Python drops an object that Rust/C still expects).
* **LeakSanitizer (LSan)**: Ensures FLINT/GMP memory allocations are properly dropped.
* **UndefinedBehaviorSanitizer (UBSan)**: Catches unaligned pointers and integer overflows.

*Running with Sanitizers (Requires Rust Nightly):*
```bash
RUSTFLAGS="-Zsanitizer=address" cargo +nightly test --target x86_64-unknown-linux-gnu
RUSTFLAGS="-Zsanitizer=thread" cargo +nightly test --target x86_64-unknown-linux-gnu
```

### Valgrind
Valgrind is used exclusively on the Rust/C FFI layer (bypassing Python/PyO3 to avoid false positives from Python's `pymalloc`). It verifies that deep, long-running algebraic simplifications do not slowly leak memory.

---

## 4. Oracle Cross-Validation

To guarantee our symbolic engine is structurally sound against industry standards, we run an Integration Oracle Suite. 

* **The Process**: We generate thousands of complex algebraic expressions and run them through our system, then run the exact same operations through `SymPy` (our oracle).
* **Verification**: We compute `simplify(OurAnswer - SymPyAnswer)`. The result must be exactly `0`.

SymPy is reused in a different role in **`benchmarks/cas_comparison.py`**: the same task is timed in Alkahest and SymPy so regressions in **performance** (not truth) are visible when someone inspects benchmark output or artifacts. That is complementary to this oracle, not a substitute for it.

---

## 5. CI/CD Pipeline Strategy

Given the computational expense of fuzzing and PBT, our GitHub Actions / CI pipeline is split into two tiers to maintain developer velocity while ensuring extreme rigor.

### Tier 1: Pull Request Checks (Fast)
* **Triggers**: Every commit to a PR branch.
* **Time Budget**: < 10 minutes.
* **Suites**:
  * Standard Unit Tests (`cargo test`, `pytest`).
  * Lightweight `proptest` and `hypothesis` runs (limited iterations).
  * Linter and Formatting (`clippy`, `rustfmt`, `ruff`).
  * AddressSanitizer (ASan) on a subset of FFI tests.

### Tier 2: Nightly Integration Builds (Heavy)
* **Triggers**: Nightly cron job on the `main` branch.
* **Time Budget**: 4 to 8 hours.
* **Suites**:
  * **AFL++ Fuzzing**: Runs continuously for hours attempting to crash the parser and MLIR backend.
  * **Deep PBT**: `proptest` and `hypothesis` configured to run millions of iterations.
  * **Valgrind Analysis**: Full memory profile of the Rust/C boundary.
  * **Oracle Testing**: Comparison of 10,000+ complex AST derivations against SymPy.
  * **Benchmark artifacts**: Criterion HTML + `cas_comparison.py` JSONL/Markdown (Alkahest vs SymPy and any available competitor adapters) for performance tracking; see `BENCHMARKS.md`.
  * **Lean 4 Proof Verification**: Generates output theorems and formally verifies them via the `lean` compiler.

---

## Contributing

When adding a new mathematical primitive, rewrite rule, or compiler lowering pass:
1. Write a standard unit test demonstrating the basic functionality.
2. Add the primitive to the AST generator in the `proptest` suite.
3. Ensure the operation passes cleanly under `cargo +nightly test` with ASan enabled.
