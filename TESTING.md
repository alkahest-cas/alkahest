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

  Default ``pytest tests/`` applies ``-m "not slow"`` (see ``pytest.ini``), so the long sparse_interp roadmap test is **not** part of this run; execute it via Tier 1b / ``pytest -m slow tests/test_sparse_interp.py`` as in §6.

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

### Sanitizers (ASan, LSan, TSan)
We compile our Rust test suite using LLVM sanitizers to catch memory violations instantly.
* **AddressSanitizer (ASan)**: Catches Out-of-Bounds accesses and Use-After-Free errors (especially critical when Python drops an object that Rust/C still expects).
* **LeakSanitizer (LSan)**: Ensures FLINT/GMP memory allocations are properly dropped.
* **ThreadSanitizer (TSan)**: Catches data races across the Rayon/`parallel` paths. It
  now runs `--features parallel`; until 3.8.0 it did **not**, which meant `rayon` and
  `dashmap` were not compiled into the binary it sanitized at all — `ExprPool`'s index
  was a plain `Mutex<HashMap>`, the `simplify_*_par` entry points and
  `CompiledFn::call_batch_par` did not exist, and the shard was reporting clean on code
  it had never seen. `alkahest-core/tests/parallel_stress.rs` is written for this shard:
  concurrent interning against a single-threaded node-count baseline, lock-free reads
  against a growing `boxcar::Vec`, concurrent `simplify_par` / `simplify_redex` on one
  shared pool, and nested `call_batch_par`, all from real OS threads rather than Rayon's
  own pool.

> **UndefinedBehaviorSanitizer is *not* run.** `-Zsanitizer=undefined` appears nowhere
> in this repository or in `.github/workflows/ci.yml`. If you want UB coverage you have
> to add it yourself; do not assume it is already gating anything.

> **The package is named `alkahest-cas`, not `alkahest-core`.** `alkahest-core/` is the
> *directory*; `-p alkahest-core` matches no package and the test binaries are named
> `alkahest_cas-*`. A `for bin in …/alkahest_core-*` loop expands to a literal that does
> not exist, the `[ -x "$bin" ] || continue` guard skips it, and the command **exits 0
> having checked nothing** — which is how the wrong name survived here for as long as it
> did.

*Running with Sanitizers (requires Rust **nightly** + `rust-src`):*
```bash
rustup toolchain install nightly
rustup component add rust-src --toolchain nightly
```

**AddressSanitizer** — Tier 1 CI scopes this to the `alkahest-cas` package only (full workspace + `build-std` is slow and easy to hit runner limits); locally you can match CI or widen:
```bash
RUSTFLAGS="-Zsanitizer=address" \
  cargo +nightly test -p alkahest-cas --lib --tests \
    --target x86_64-unknown-linux-gnu \
    -Z build-std
# Optional: suppress known GMP/FLINT leak noise while debugging other issues:
#   LSAN_OPTIONS=detect_leaks=0
```

**Known gap — no sanitizer sees a Python-facing path.** The PR-gating ASan job sets
`LSAN_OPTIONS: detect_leaks=0`, so it is not a leak check. The nightly LSan shard is
`--workspace`, but `alkahest-py` is a `cdylib` with zero `#[test]` functions, so no
CPython interpreter is ever started under it, and the Valgrind shard globs Rust test
binaries only. **`pytest` is never run under any sanitizer.** Until that changes, the
substitute check for the Python surface is a behavioural one: run N iterations of an
entry point on a *fresh* `ExprPool` per iteration and assert resident memory stays flat
(see [Budgets → pool lifetime](docs/mdbook/src/budgets.md#exprpool-never-reclaims)).

The same gap applies to *concurrency*, and matters more since `parallel` became a
default-wheel feature: `ExprPool` is a plain sendable `#[pyclass]`, and `py_simplify_par`
holds a `PyRef` borrow across `Python::allow_threads` while `&ExprPool` escapes into a
Rayon pool — none of which any sanitizer job constructs.
`tests/test_parallel_threadsafety.py` is the behavioural substitute: shared pool across
`threading.Thread`s, concurrent `simplify_par` / `simplify_redex` / `simplify_auto` /
`numpy_eval_par`, each asserted against the answer a lone caller gets. It proves
agreement, not absence of races; the race detection lives in the Rust `tsan` shard.

**ThreadSanitizer** — nightly `tsan` shard. Both environment variables are load-bearing;
without them the shard is red for reasons that are not races:
```bash
export TSAN_OPTIONS="suppressions=$PWD/tsan.supp"
# TSan inflates every stack frame, and `simplify/parallel.rs`'s stack governor is
# calibrated against Rayon's default 2 MiB worker stack. Without this you get a bare
# SIGSEGV — a stack overflow, not memory corruption; it does not reproduce
# uninstrumented.
export RUST_MIN_STACK=33554432

RUSTFLAGS="-Zsanitizer=thread" \
  cargo +nightly test --workspace --lib --tests \
    --features parallel \
    --target x86_64-unknown-linux-gnu \
    -Z build-std
```

`tsan.supp` suppresses `crossbeam_epoch` / `crossbeam_deque` only. Epoch-based
reclamation synchronises with an asymmetric `membarrier(2)` barrier that TSan cannot
model, so it reports a race between a reclaimer's `free` and a reader's relaxed load,
with rayon/crossbeam frames on both sides and no Alkahest frame anywhere. The
suppression is deliberately narrow: a real race inside one of our Rayon closures still
fails the shard. **Nothing with an `alkahest_cas` frame belongs in that file** — fix the
code instead.

The Python-side counterpart is `tests/test_parallel_threadsafety.py`, which drives the
same shapes through PyO3 (shared `ExprPool` across `threading.Thread`s, concurrent
`simplify_par` / `numpy_eval_par`). It is *not* run under a sanitizer — see the known
gap above — so it is an invariant test, not a race detector.

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

# The crate's test binaries are `alkahest_cas-*` (package `alkahest-cas`).
# `alkahest_core-*` matches nothing and the guard below turns that into a silent pass.
for bin in target/x86_64-unknown-linux-gnu/debug/deps/alkahest_cas-*; do
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
* **Typical contents**: `cargo fmt`, `clippy`, `cargo test --workspace`, `ruff`, `pytest` (defaults in `pytest.ini` exclude `@pytest.mark.slow`), ASan on the `alkahest-cas` package (**below** the FFI boundary — the PyO3 layer is not instrumented, and `detect_leaks` is off), the deterministic silent-error gate (`tests/silent_errors/`), CodSpeed micro-benchmarks (`.github/workflows/codspeed.yml`), etc. (see `.github/workflows/ci.yml`).

### Tier 1b: Slow Python (sparse interpolation roadmap)
* **Triggers**: Same nightly **schedule** as Tier 2 (not on every push — keeps default CI fast).
* **Suite**: `pytest tests/test_sparse_interp.py -m slow --timeout=0 -v --override-ini="addopts=-v"` after `maturin develop --features groebner`.

### Tier 2: Nightly integration (heavy matrix)
* **Triggers**: Cron on `main` (02:00 UTC); shards run in parallel.
* **Time Budget**: up to several hours per shard host cap.
* **Kani** (`.github/workflows/kani.yml`, its own workflow): the full-width model-checking harnesses on every PR, all harnesses nightly — see [§7](#7-bounded-model-checking-kani).
* **Suites** (each is a separate matrix job): deep `proptest` (`PROPTEST_CASES=50000`, `--release`), deep `hypothesis` (`HYPOTHESIS_MAX_EXAMPLES=5000`, full `pytest tests/`), TSan, LSan, Valgrind, AFL fuzz targets, “extras” (oracle file if present, `cargo bench`, benchmark report scripts), etc.

---

## 6. Running deep tests locally (match scheduled CI)

Use this checklist when you want **more than** `cargo test --workspace` on a beefy machine. Commands mirror `.github/workflows/ci.yml` (`tier1-python-slow` + `nightly` matrix).

| CI job | What to run locally |
|--------|---------------------|
| **Tier 1b** | After `maturin develop --manifest-path alkahest-py/Cargo.toml --features groebner`: `pytest tests/test_sparse_interp.py -m slow --timeout=0 -v --override-ini="addopts=-v"` |
| **proptest** | `export PROPTEST_CASES=50000` then `cargo test --workspace --release` |
| **hypothesis** | See §1 Python (`HYPOTHESIS_MAX_EXAMPLES=5000`, `pytest tests/`) |
| **tsan** | See §3 ThreadSanitizer block |
| **lsan** | See §3 LeakSanitizer block |
| **valgrind** | See §3 Valgrind block |
| **fuzz-\*** | See §2; use `--manifest-path fuzz/Cargo.toml` |
| **extras** | `maturin develop --manifest-path alkahest-py/Cargo.toml --features groebner`; `pytest tests/test_oracle.py -v` if configured; `cargo bench --workspace`; optional scripts under `benchmarks/` (may need SymPy / optional CAS; CI sets `RUN_COMMERCIAL_CAS` where applicable) |

**Lean / docs / cross-platform** workflows have their own YAML files (e.g. `.github/workflows/lean.yml`, `ci-cross.yml`); run those suites locally only when you touch those areas.

---

## 7. Bounded model checking (Kani)

[Kani](https://github.com/model-checking/kani) turns a harness into a SAT
problem over *every* input in a stated range, so a passing harness is a proof
for that range, not a sample. It only sees Rust: anything that reaches
FLINT/GMP/MPFR (`rug::Integer`, `flint::*`) is out of scope, so the harnesses
cover the pure machine-word kernels.

Harnesses live in `#[cfg(kani)] mod verification { ... }` next to the code
they check, so private helpers stay private. `build.rs` declares `cfg(kani)`
to keep `unexpected_cfgs` quiet.

```bash
# Install (once). Kani pins its own nightly toolchain and CBMC.
cargo install --locked kani-verifier
cargo kani setup

# Every harness in the crate (-j: one CBMC per core; needs terse output)
cargo kani -p alkahest-cas -Z stubbing --output-format terse -j

# The PR tier only: the full-width harnesses, a few minutes in all
cargo kani -p alkahest-cas -Z stubbing --harness _full_width --output-format terse -j

# One harness, with the counterexample trace if it fails
cargo kani -p alkahest-cas -Z stubbing --harness modular::verification::mod_inverse_small_modulus
```

```bash
# The loop-contract harnesses (`*_inductive`): a separate build, see below
ALKAHEST_KANI_LOOP_CONTRACTS=1 \
  cargo kani -p alkahest-cas -Z stubbing -Z loop-contracts --harness _inductive --output-format terse -j
```

`-Z stubbing` is required: the compositional harnesses replace a proven
function with its contract (e.g. `mul_mod` by "any value `< m`") so the caller
can be checked at full width. FLINT does not need to be installed — Kani never
links — but `build.rs`'s presence probe does run; set
`ALKAHEST_SKIP_FLINT_CHECK=1` on a machine without it. CI runs this in
`.github/workflows/kani.yml`: `*_full_width` harnesses (including the
`*_inductive_full_width` loop-contract ones) on every PR, all of them nightly.
The name decides the tier, so a harness is named `*_full_width` only once its
runtime has been measured; an unmeasured one runs nightly until it has.

**Loop contracts.** A loop whose trip count grows with the input — trial
division to `√n`, Euclid on two `u64`s — cannot be unrolled at full width. Such
a loop carries a `kani::loop_invariant`, written as
`#[cfg_attr(kani_loop_contracts, kani::loop_invariant(...))]` on the `while`, and
its harness is named `*_inductive` (`*_inductive_full_width` once measured,
which puts it in the PR tier) and gated `#[cfg(kani_loop_contracts)]`.
Kani then checks the invariant holds on entry and is preserved by one
arbitrary iteration, and continues after the loop from *any* state satisfying
it — so the harness proves panic/overflow freedom (and whatever the invariant
states) for every input, but sees nothing about the value the loop computes
beyond the invariant. The value is covered by exhaustive unit tests on small
inputs next to the harness. `build.rs` sets `cfg(kani_loop_contracts)` only
under `cargo kani` with `ALKAHEST_KANI_LOOP_CONTRACTS=1`; without it the
attributes vanish and the loops are unrolled as usual, which is what the
value-checking harnesses that run through the same loops need.

**Stubbing FLINT (FFI).** Kani cannot execute C, so a harness that reaches a
FLINT call fails as an unsupported foreign call. For the Rust code *around*
FFI calls — buffer sizing, index bounds — stub each `extern` function with a
Rust model of its documented contract:
`#[kani::stub(ffi::fmpz_get_ui_array, models::fmpz_get_ui_array)]`. The model
`assert!`s the C function's precondition and performs its writes through the
raw pointer it was given, so an undersized Rust buffer is an out-of-bounds
pointer write Kani reports. The model's `fmpz` need not be FLINT's tagged
word: in `flint::integer::verification` it simply stores the bit length
`fmpz_bits` reports. `rug`/GMP calls cannot be stubbed this way in practice
(they are generic Rust over many FFI calls); split the FLINT-facing part into
its own function first, as `fmpz_abs_words` is.

**What is proven.** "Full width" means every value the precondition allows.
Anything narrower says so in the harness's doc comment. Times are per-harness
CBMC time with the default solver (CaDiCaL), measured locally with Kani 0.68.0
running 16 harnesses at once; the whole suite took 9.5 min wall-clock.

| Harness | Property | Range | Time |
|---|---|---|---|
| `modular::…::mul_mod_in_range_full_width` | no panic, result `< m` (lossless narrowing) | all `a, b`; `m >= 1` | 0.2 s |
| `modular::…::mul_mod_matches_u64_small` | equals `a·b mod m` | `a, b, m < 2^8` | 61 s |
| `modular::…::pow_mod_in_range_full_width` | no panic, ≤ 64 rounds, result `< m` (`mul_mod` stubbed by its contract) | all `base, exp`; `m >= 1` | 4.2 s |
| `modular::…::pow_mod_exp_zero_full_width` | `x^0 = 1 mod m` (the `m = 1` bug) | all `base`; `m >= 1` | 0.1 s |
| `modular::…::mod_inverse_special_values_full_width` | `1⁻¹ = 1`; `2⁻¹ = (m+1)/2` for odd `m` | all `m >= 3` | 209 s |
| `modular::…::mod_inverse_in_range_large_modulus` | no panic / `i128` overflow, result `< m` | `m >= 2` all; `1 <= a < 2^4` | 552 s |
| `modular::…::mod_inverse_small_modulus` | result `< m`; `a·inv ≡ 1` whenever `a` is invertible (incl. `a >= m`) | `2 <= m < 2^4`; `a < 2^8` | 473 s |
| `modular::…::crt_step_u64_no_overflow_full_width` | no panic, `t < p` (inverse stubbed) | all `p >= 2`; residues `< p` | 0.6 s |
| `modular::…::crt_step_u64_solves_congruence` | `a + m·t ≡ aᵢ (mod p)`, `t < p` | `2 <= p < 2^4` | 324 s |
| `modular::…::modular_value_{add,sub,neg}_full_width` | exact canonical result | all `m >= 1`; values `< m` | 4 s / 15 s / 1.6 s |
| `modular::…::modular_value_mul_in_range_full_width` | canonical result | all `m >= 1`; values `< m` | 0.7 s |
| `modular::…::is_prime_no_panic_full_width` | no panic/overflow; each Miller–Rabin round gets `n >= 2`, `1 <= r <= 63` | every `u64` | 21 s |
| `modular::…::miller_rabin_round_no_panic_full_width` | no panic/overflow (`pow_mod`/`mul_mod` stubbed) | `n >= 2`, `1 <= r <= 63`; `d, a` all | 3.5 s |
| `modular::…::is_prime_small_agrees_with_trial_division` | agrees with trial division, unstubbed | `n < 2^6` | 531 s |
| `holonomic::modular::…::{add,sub}_mod_full_width` | exact canonical result | `m <= 2^62`; values `< m` | 0.3 s each |
| `holonomic::modular::…::index_mod_in_range_full_width` | no panic, result `< m` | all `i64`; `1 <= m <= 2^62` | 0.3 s |
| `holonomic::modular::…::pow_mod_exp_zero_full_width` | `x^0 = 1 mod m` | all `base`; `m >= 1` | 0.1 s |
| `holonomic::modular::…::inv_mod_small` | `Some(inv)` with `a·inv ≡ 1` iff `gcd(a, m) = 1` | `m, a < 2^4` | 115 s |
| `holonomic::modular::…::valuation_exact_below_cap` | `v <= cap`; exact below the cap | `x, p < 2^8`; `cap <= 4` | 5.4 s |
| `jacobian_torsion::…::{addmod,submod}_full_width` | exact canonical result | `p <= 2^63` (add), all `p` (sub); values `< p` | 0.2–0.3 s |
| `holonomic::boundary::…::ceil_div_no_overflow_full_width` | no overflow | all `a`; all `b > 0` | 0.5 s |
| `holonomic::boundary::…::ceil_div_exact_small` | `(q−1)·b < a <= q·b` | `-2^10 <= a <= 2^10`, `1 <= b <= 2^10` | 16 s |
| `calculus::puiseux::…::gcd_lcm_small` | gcd divides both; lcm a common multiple `<= a·b` | `a, b < 2^5` | 17 s |
| `simplify::rules::…::integer_sqrt_u64_is_floor_sqrt` | `⌊√n⌋`, no overflow | `n < 2^16` | 64 s |
| `simplify::rules::…::expansion_products_saturates` | exact `m^n`, else `u64::MAX` | `m < 2^8`, `n <= 9` | 34 s |
| `character::table::…::isqrt_is_floor_sqrt` | `⌊√n⌋` (f64 estimate + correction) | `n < 2^16` | 195 s |

Round 2 (overflow and indexing in the callers of those kernels). Times were
measured locally with Kani 0.68.0, two harnesses at once; **unmeasured** means
the local run was stopped before that harness finished, so it is not in the
PR tier (its name has no `_full_width`) and runs nightly only. Loop-contract
(`*_inductive…`) harnesses need `ALKAHEST_KANI_LOOP_CONTRACTS=1 -Z loop-contracts`.

| Harness | Property | Range | Tier | Time |
|---|---|---|---|---|
| `modular::…::gcd_u64_in_range_inductive_full_width` | no panic; `0` iff `a = b = 0`, else `<=` each nonzero argument (loop invariant) | all `a, b` | PR | 2.9 s |
| `modular::…::gcd_i64_sign_full_width` | no panic at `i64::MIN`; result `>= 0` except gcd `2^63` (`gcd_u64` stubbed) | all `a, b` | PR | 0.1 s |
| `modular::…::mod_inverse_in_range_inductive` | no panic / `i128` overflow, result `< m` (Bézout invariant) | all `a`; `m >= 1` | nightly | unmeasured |
| `poly::interp::…::interp_{add,sub}_mod_full_width` | exact canonical result (`p > 2^63` wrapped before) | all `p >= 1`; values `< p` | PR | 0.5 s / 0.4 s |
| `calculus::limits::…::lcm_small_no_wrap_full_width` | a `Some` is a true common multiple within the cap | `1 <= a <= 12`; all `u32 b` | PR | 0.4 s |
| `integrate::algebraic::subst::…::lcm_no_wrap_full_width` | no panic; a `Some` is a true common multiple `>= max(a, b)` | `1 <= a <= 12`; all `usize b` | PR | 133 s |
| `integrate::algebraic::subst::…::lcm_is_least_small` | least common multiple | `1 <= a, b < 2^5` | nightly | unmeasured |
| `integrate::algebraic::trager_log::…::lcm_u32_no_wrap_large_a` | no panic; a `Some` is a true common multiple | all `a`; `b < 2^4` | nightly | 33 s |
| `integrate::algebraic::trager_log::…::lcm_u32_is_least_small` | least common multiple, `None` only past `u32::MAX` | `a, b < 2^5` | nightly | 2.8 s |
| `integrate::risch::exp_case::…::perfect_square_no_overflow_full_width` | no panic / overflow | every `i64` | PR | 6.0 s |
| `integrate::risch::exp_case::…::perfect_mth_power_no_overflow` | no panic / overflow; a `true` is an exact power | all `d`, all `m` | nightly | unmeasured |
| `integrate::algebraic::elliptic_output::…::is_squarefree_no_overflow_inductive` | no panic / overflow (loop invariant) | every `i64` | nightly | unmeasured |
| `integrate::algebraic::elliptic_output::…::is_quartic_radical_no_overflow_inductive` | no panic / overflow (loop invariant) | every `i64` | nightly | unmeasured |
| `character::dixon::…::distinct_prime_factors_no_overflow_inductive` | no panic / overflow (two loop invariants) | every `u64` | nightly | unmeasured |
| `holonomic::modular::…::prime_power_in_range_full_width` | no panic / `u128` overflow; `<= 63` rounds; a `Some` is `<= 2^62` | all `p >= 2`; all `u32 e` | PR | 130 s |
| `holonomic::modular::…::prime_power_exact_small` | `p^e` when it fits, `None` exactly past `2^62` | `2 <= p < 2^8`; `e <= 8` | nightly | 52 s |
| `holonomic::telescoping2d::search::…::flatten_in_block` | no overflow; index `< box_len^num_axes` | all `box_len >= 1`; `num_axes <= 3` | nightly | unmeasured |
| `holonomic::telescoping2d::search::…::flatten_roundtrip_small` | `unflatten ∘ flatten = id` | `1 <= box_len < 2^4`; `num_axes <= 3` | nightly | unmeasured |
| `real::sos::psd::…::pack_len_exact` | `n(n+1)/2` without overflow, exact | `n < 2^32` | nightly | unmeasured |
| `eval::program::…::operand_range_no_wrap_full_width` | the range has exactly `len` entries | all `start, len` | PR | 0.04 s |
| `eval::program::…::run_in_bounds_when_well_formed_small` | a program passing `is_well_formed` runs without an out-of-bounds index | 2 ops, 5 slots, 3 operands; every `u32` index | nightly | unmeasured |
| `flint::integer::…::words_for_bits_full_width` | least word count; fits `slong` | every bit length | PR | 0.06 s |
| `flint::integer::…::abs_words_buffer_covers_fmpz_bits_small` | buffer covers `fmpz_bits`; every FLINT write in bounds (FFI stubbed by its contract) | magnitudes up to 4 words | nightly | 0.8 s |

**What is not proven.**

- That the Miller–Rabin witness sets in `is_prime` decide primality. That is a
  number-theoretic theorem (Jaeschke; Sorenson–Webster), not a bit-level fact;
  the harnesses show only that `is_prime` cannot panic or overflow.
- Full-width *value* identities through a symbolic 64/128-bit divider
  (`a·b mod m` equals its definition for all `a, b, m`, `mod_inverse_u64` for a
  64-bit modulus and a 64-bit `a`). Neither CaDiCaL nor Kissat closed the
  full-width `mul_mod` equality in 24–35 minutes; `--solver z3` was slower
  than CaDiCaL on the small-range harnesses, and `--solver bitwuzla` (0.9.1)
  reported every harness failed with zero failing checks, i.e. the back end did
  not run. Those identities are checked on small ranges instead.
- `pow_mod` for exponents above 0. Against repeated multiplication, `exp <= 3`
  with `m < 2^8` did not finish in an hour; only `x^0` and panic
  freedom/range are proven, the rest is unit-tested.
- Anything below `rug`/FLINT: the other half of `crt_combine`, `lift_crt`,
  `rational_reconstruction`, `reduce_mod`.
- FLINT itself. The `fmpz_abs_words` harness checks the Rust buffer against a
  model of FLINT's documented contract, not FLINT's C code.
- The value a loop-contract (`*_inductive`) loop computes, beyond its
  invariant. Those values are covered by exhaustive unit tests on small inputs.
- That every square is recognised by `is_perfect_square` at full width (a
  symbolic `f64` square root did not finish in 25 minutes); a unit test covers
  the bottom and top `2^16` roots and a stride through the rest.

**Writing a new harness.** What costs time is a division (or remainder) by a
*symbolic* divisor: the SAT back end bit-blasts the full 64/128-bit divider
whatever `any_where` bounds the values to, and a chain of them — Euclid,
square-and-multiply — becomes a multiplier-equivalence problem it does not
close. So prefer range facts (`r < m`) over value equalities at full width,
replace an already-proven callee by its contract with `#[kani::stub]`, and
split nested loops into separately checked functions: `is_prime`'s witness
and squaring loops went from not finishing in 35 min to 25 s together once
split. Name a harness `*_full_width` only if it covers the whole precondition
and finishes in a few minutes at most; it then runs on every PR.

---

## Contributing

When adding a new mathematical primitive, rewrite rule, or compiler lowering pass:
1. Write a standard unit test demonstrating the basic functionality.
2. Add the primitive to the AST generator in the `proptest` suite.
3. Ensure the operation passes cleanly under AddressSanitizer (see §3).
