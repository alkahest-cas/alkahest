# Contributing to Alkahest

Open work is tracked in [`ROADMAP.md`](ROADMAP.md) and GitHub issues. Changes are added to [`CHANGELOG`](CHANGELOG.md) after implementation.

## Using AI for development

AI agents must follow [`AGENTS.md`](AGENTS.md) instructions.

## Setup

End-user install commands (PyPI vs optional `+jit` / `+full` wheels, from source) are in [`README.md`](README.md) and the [**Getting started**](https://alkahest-cas.github.io/alkahest/getting-started.html) chapter of the docs.

```bash
# Prerequisites: Rust (stable + nightly), Python 3.9–3.13, uv, LLVM 15, FLINT (see README § Install)
# Install uv: https://docs.astral.sh/uv/getting-started/installation/
uv sync --no-install-project --group dev
uv run maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "cranelift jit egraph parallel groebner"
```

`--no-install-project` skips building the Rust extension during `uv sync`; `maturin develop` handles that with the right feature flags. After this, `uv run pytest`, `uv run ruff check python/`, etc. all resolve from the project venv.

## Running tests

```bash
# Rust unit + property tests
cargo test --all

# Python tests (slow sparse_interp roadmap excluded by default; see pytest.ini)
pytest

# The silent-error gate (its own Tier-1 CI step; a confident wrong answer fails here)
pytest tests/silent_errors/

# With sanitizers — same invocation CI uses. `-p alkahest-cas` (the package name;
# `alkahest-core` is only the directory) and `-Z build-std` are both required: without
# build-std the doc-test/dep binaries link without the ASan runtime and you get
# "undefined symbol: __asan_init".
RUSTFLAGS="-Zsanitizer=address" \
  cargo +nightly test -p alkahest-cas --lib --tests \
    --target x86_64-unknown-linux-gnu -Z build-std
```

See [`TESTING.md`](TESTING.md) for the full testing strategy (fuzzing, oracle cross-validation, CI tiers).

## Linting and formatting

```bash
cargo fmt --all
# One pass per feature set, which is what CI runs. `--all-features` is not the
# same thing and is not a substitute: it needs LLVM (`jit`) and the CUDA toolkit
# (`cuda`), so on a machine without them it fails to build rather than reporting
# lints, and the feature sets that *do* build stop being checked.
for f in "" egraph parallel cranelift jit groebner-cuda; do
  cargo clippy --all-targets ${f:+--features $f} -- -D warnings
done
uv run ruff check python/ tests/ --fix
uv run ruff format python/ tests/
uv run ty check python/alkahest/
```

CI enforces all of the above on every PR. Ruff and `ty` are scoped to `python/`
and `tests/` — `examples/`, `benchmarks/` and `scripts/` are not linted or run
by CI, so changes to a public API have to be swept through them by hand.

## Adding a new mathematical primitive

Every primitive must register a full bundle. Add an entry in `alkahest-core/src/primitive/`:

1. **Numerical evaluator** (`numeric_f64`) — `f(x: f64) -> f64`
2. **Ball evaluator** (`numeric_ball`) — `f(x: ArbBall) -> ArbBall`
3. **Forward-mode diff rule** (`diff_forward`) — dual-number derivative
4. **Reverse-mode diff rule** (`diff_reverse`) — adjoint propagation
5. **MLIR lowering** — entry in the `alkahest` dialect op table
6. **Lean theorem tag** — a Mathlib theorem name (or `None` if not yet mapped)

After adding the primitive:

- Add it to the proptest AST generator in `alkahest-core/src/` so property tests cover it.
- Add a unit test demonstrating the basic value and derivative.
- Run `cargo +nightly test` with ASan to confirm no FFI issues.
- Expose it to Python in `alkahest-py/src/lib.rs` and add it to `alkahest/__all__`.

## Adding a simplification rule

Rules live in `alkahest-core/src/simplify/`. Each rule is a `RewriteRule` with a stable name that appears in derivation logs and (optionally) maps to a Lean theorem.

- If the rule has side conditions (domain constraints, non-zero denominators), record them as `SideCondition` entries.
- Add the rule to the appropriate rule set (`arithmetic_rules`, `trig_rules`, `log_exp_rules_safe`, etc.).
- Add a proptest case verifying the rule is idempotent: `simplify(simplify(expr)) == simplify(expr)`.

## Accessors: property or method?

One rule, applied to every `#[pymethods]` entry in `alkahest-py/src/lib.rs`:

> **A zero-argument, O(1), non-allocating accessor that returns a scalar or a flag is a `#[getter]` (a Python property). Anything that returns a collection, allocates, or does real work is a method.**

```rust
#[getter]                            // property: reads a field, cannot fail
fn n_equations(&self) -> usize { self.inner.n_equations() }

fn polys(&self) -> Vec<PyGbPoly> { … }   // method: allocates a collection
fn rank(&self, py: Python<'_>) -> PyResult<usize> { … }  // method: real work, can fail
```

A property and a method sitting side by side on the same class is not in itself a problem — `RegularChain.n_vars` (property) next to `RegularChain.polys()` (method) is exactly what the rule asks for. What the rule rules out is the *same* kind of question being asked two different ways on two different classes, which is what made the surface unpredictable before 3.8.0.

Why the split falls there:

- A property that can raise, block, or take a noticeable amount of time is a trap — the caller reads `x.rank` as a field access. Real work stays behind parentheses.
- A method that returns a scalar is the more dangerous mistake in the other direction: `if x.n_equations:` on a bound method is always `True` and `f"{x.n_equations}"` prints `<built-in method …>`. Neither raises. Converting these was the whole point of the 3.8.0 sweep.

`tests/test_accessor_convention.py` enforces this. It pins the converted accessors at runtime and statically scans `alkahest-py/src/lib.rs` for zero-argument scalar-returning methods; if a new one is genuinely doing real work, add it to `REAL_WORK_EXEMPTIONS` there with a one-line reason. A handful of pre-3.8.0 getters return small collections (`AsymptoticReport.terms`, `CertifiedSolution.coordinates`, `PositivityCertificate.log`, …); they are grandfathered, not precedent.

Changing an existing accessor's form is a **breaking change**: record it in `CHANGELOG.md` under the release's "Behaviour changes to plan for" with a before/after line, and update every caller in `tests/`, `examples/`, `benchmarks/`, `docs/mdbook/`, `alkahest-skill/alkahest.md` and the `.pyi` stubs.

## Pull requests

- Keep PRs focused on one item from `ROADMAP.md` or one issue.
- Tier-1 CI (< 10 min) must be green before review: unit tests, lightweight proptest/hypothesis, clippy, ruff, the silent-error gate (`pytest tests/silent_errors/`), and ASan scoped to the `alkahest-cas` package. Note what that last one is *not*: it runs `cargo +nightly test -p alkahest-cas`, i.e. the crate **below** the FFI boundary, with `LSAN_OPTIONS=detect_leaks=0`. No sanitizer runs `pytest`, so a leak, UAF or data race that only appears through PyO3 is not caught by CI — see [`TESTING.md` §3](TESTING.md#3-memory-safety--sanitizers). For the concurrency half of that gap the substitute is `tests/test_parallel_threadsafety.py` (invariant checks across real Python threads on a shared `ExprPool`) plus the nightly `tsan` shard, which since 3.8.0 actually builds `--features parallel` and runs `alkahest-core/tests/parallel_stress.rs`.
- Semver is enforced automatically — `cargo semver-checks` runs on every PR and will fail if a stable API breaks.
- New stable API additions go into `alkahest_cas::stable` and `alkahest.__all__`; experimental additions go into `alkahest_cas::experimental` and `alkahest.experimental`.
- Add `[skip ci]` at the end of commit messages if changes cannot possibly effect CI.

## Rust vs Python

> **Naming.** The directory is `alkahest-core/`; the Cargo **package** it declares is
> `alkahest-cas`, and the Rust path an external crate uses is `alkahest_cas::`. Inside
> this workspace `alkahest-py` renames the dependency (`package = "alkahest-cas"`), which
> is why its sources say `alkahest_core::` — that alias is local to `alkahest-py` and is
> not what a downstream user writes. Cargo commands take the *package* name:
> `cargo test -p alkahest-cas`, never `-p alkahest-core`.

### Rust (`alkahest-cas`, in `alkahest-core/`) gets the code when...

1. It is a mathematical operation, data structure, or invariant that any front-end should see identically — e.g. polynomial normalisation, differentiation, matrix inversion, Gröbner basis.
2. It is on a hot path. Anything that iterates over `ExprId`s, touches coefficient rings, or performs codegen must be Rust.
3. It holds mutable state that must survive across Python garbage collection (pools, JIT caches, compiled kernels, MLIR modules).
4. It interacts with FFI targets (FLINT, LLVM, CUDA driver) — keep the `unsafe` surface in one crate.
5. Correctness depends on exhaustive `match` (new polynomial variant, new AST node, new error code). The compiler must enforce completeness.

### Python (`python/alkahest/`) gets the code when...

1. It's a composition of existing kernel calls — `grad = trace + diff`, pytree flattening, decorator plumbing.
2. It bridges to the Python ecosystem (NumPy, JAX, SymPy, Matplotlib). Ecosystem code changes faster than the kernel should.
3. It's sugar: default arguments, keyword-only parameters, docstring-driven overload dispatch, introspection with `inspect`.
4. It's caller-side validation whose error message is clearer in Python (wrong arity, non-iterable input, type mismatches before values cross the boundary).
5. It's experimental. `python/alkahest/experimental/` exists so that API exploration doesn't require a recompile and doesn't commit the stability surface.

### Quick reference

| Concern | Layer |
|---|---|
| New polynomial ring | `alkahest-core` |
| New simplification rule | `alkahest-core` |
| New integration heuristic | `alkahest-core` |
| New JIT backend | `alkahest-core` |
| Expose existing core fn to Python | `alkahest-py` |
| Exception-class plumbing | `alkahest-py` |
| New `@alkahest.something` decorator | `python/alkahest/` |
| NumPy/JAX/SymPy interop | `python/alkahest/` |
| Context manager / default registry | `python/alkahest/` |
| Experimental API you may throw away | `python/alkahest/experimental/` |
