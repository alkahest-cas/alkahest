# Contributing to Alkahest

Open work is tracked in [`ROADMAP.md`](ROADMAP.md) and GitHub issues. Changes are added to [`CHANGELOG`](CHANGELOG.md) after implementation.

## Using AI for development

AI agents must follow [`AGENTS.md`](AGENTS.md) instructions.

## Setup

```bash
# Prerequisites: Rust (stable + nightly), Python ≥ 3.9, maturin, LLVM 15
pip install maturin pytest hypothesis ruff
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "jit egraph parallel groebner"
```

## Running tests

```bash
# Rust unit + property tests
cargo test --all

# Python tests (slow sparse_interp roadmap excluded by default; see pytest.ini)
pytest

# With sanitizers (catches FFI memory bugs)
RUSTFLAGS="-Zsanitizer=address" cargo +nightly test --target x86_64-unknown-linux-gnu
```

See [`TESTING.md`](TESTING.md) for the full testing strategy (fuzzing, oracle cross-validation, CI tiers).

## Linting and formatting

```bash
cargo fmt --all
cargo clippy --all --all-features -- -D warnings
ruff check python/ --fix
ruff format python/
```

CI enforces all of the above on every PR.

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

## Pull requests

- Keep PRs focused on one item from `ROADMAP.md` or one issue.
- Tier-1 CI (< 10 min) must be green before review: unit tests, lightweight proptest/hypothesis, clippy, ruff, ASan on FFI tests.
- Semver is enforced automatically — `cargo semver-checks` runs on every PR and will fail if a stable API breaks.
- New stable API additions go into `alkahest_core::stable` and `alkahest.__all__`; experimental additions go into `alkahest_core::experimental` and `alkahest.experimental`.
- Add `[skip ci]` at the end of commit messages if changes cannot possibly effect CI.

## Rust vs Python

### Rust (`alkahest-core`) gets the code when...

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
