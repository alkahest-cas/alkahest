# Alkahest

[![CI](https://github.com/alkahest-cas/alkahest/actions/workflows/ci.yml/badge.svg)](https://github.com/alkahest-cas/alkahest/actions/workflows/ci.yml)
[![cross-platform CI](https://github.com/alkahest-cas/alkahest/actions/workflows/ci-cross.yml/badge.svg)](https://github.com/alkahest-cas/alkahest/actions/workflows/ci-cross.yml)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://codspeed.io/alkahest-cas/alkahest?utm_source=badge)
[![PyPI](https://img.shields.io/pypi/v/alkahest.svg)](https://pypi.org/project/alkahest/)
[![Crates.io](https://img.shields.io/crates/v/alkahest-cas.svg)](https://crates.io/crates/alkahest-cas)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://alkahest-cas.github.io/alkahest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg?url=https%3A%2F%2Fdeepwiki.com%2Falkahest-cas%2Falkahest)](https://deepwiki.com/alkahest-cas/alkahest)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A high-performance computer algebra system for Python built for both humans and agents. Symbolic operations run orders of magnitude faster than SymPy and can run on modern accelerated hardware. Every computation produces a derivation log; a meaningful subset can export Lean 4 proofs for independent verification.

**Install:** the package is published on [PyPI](https://pypi.org/project/alkahest/); use `pip install alkahest` (**Python 3.9–3.13**). Default wheels ship **Cranelift** CPU JIT (pure Rust, no LLVM). See [Install](#install) for the capability matrix and optional **`+jit`** / **`+full`** Linux wheels (GitHub Releases).

**Demo:** try the hosted **[playground](https://alkahest-cas.github.io/playground/)** (WASM in-browser, or bring your own server/Jupyter URL + token), or run [`demo-playground/`](demo-playground/) locally for the full agent and recording stack. See [`demo-playground/README.md`](demo-playground/README.md).

**Links:** [GitHub](https://github.com/alkahest-cas/alkahest) · [**RL environment**](https://app.primeintellect.ai/dashboard/environments/alkahest/alkahest-symbolic-integration) (`alkahest/alkahest-symbolic-integration` on [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments))

**Stack:** Rust kernel → FLINT/Arb (polynomials, ball arithmetic) → vendored egglog + colored e-graphs (simplification) → Cranelift/LLVM JIT + MLIR (native and GPU codegen) → PyO3 → Python

---

## Install

**Requirements:** Python **3.9–3.13** ([PyPI](https://pypi.org/project/alkahest/) `requires-python`).

```bash
pip install alkahest
```

**RL environments** (symbolic integration tasks for Prime Intellect / veRL): Python **≥ 3.10** required.

```bash
pip install "alkahest[rl]"
```

See [Reinforcement learning](#reinforcement-learning) and the [RL guide](docs/mdbook/src/rl.md).

For an isolated environment (recommended when juggling versions or building from source):

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install alkahest
```

Default PyPI wheels include the **vendored egglog** e-graph backend (`egraph`), the **Gröbner solver** (`groebner` — so `alkahest.solve`, Diophantine, homotopy, and related APIs work out of the box), and **Cranelift** Tier-1 CPU JIT (`cranelift`, pure Rust, ~2 MB larger than the interpreter-only baseline). They do **not** include LLVM JIT or `parallel`. For LLVM CPU JIT—or JIT plus parallel F4—use a **PyTorch-style** opt-in **`+jit`** / **`+full`** Linux wheel from [GitHub Releases](#opt-in-linux-wheels-jit-and-full-pytorch-style), not the default PyPI resolver path.

### Install matrix (default vs opt-in wheels)

Probe your environment after install: `alkahest.capabilities()["features"]` and `alkahest.jit_is_available()`.

| Artifact | Where | OS / arch (CI) | Python | `egraph` | `groebner` | Cranelift JIT | LLVM JIT | `parallel` |
|----------|-------|----------------|--------|----------|------------|---------------|----------|------------|
| **Default** (`pip install alkahest`) | [PyPI](https://pypi.org/project/alkahest/) | Linux manylinux x86_64; macOS arm64; Windows x86_64 | 3.9–3.13 | yes | yes | yes | no | no |
| **`+jit`** (`X.Y.Z+jit`) | GitHub Releases only | Linux x86_64 | 3.9–3.13 | yes | yes | no | yes | no |
| **`+full`** (`X.Y.Z+full`) | GitHub Releases only | Linux x86_64 | 3.9–3.13 | yes | yes | no | yes | yes |

**macOS / Windows:** default PyPI wheels include Cranelift JIT. **`+jit`** and **`+full`** are **not** built in CI (LLVM / MSYS2 constraints); use [building from source](#from-source) with `--features jit` (and `parallel` for F4 parallelism) on those platforms.

**Linux LLVM wheels** vendor LLVM and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libffi-*.so` or `libLLVM-*.so`, prepend that directory to `LD_LIBRARY_PATH`.

### Opt-in Linux wheels: `+jit` and `+full` (PyTorch-style)

**Why a separate index or direct wheel URL:** feature-heavy wheels use a PEP 440 **local version** (for example `3.7.0+jit` or `3.7.0+full`). Those builds **must not** be mixed into the main PyPI project’s simple API for the same reason PyTorch publishes CUDA wheels on `download.pytorch.org`: otherwise `pip install alkahest` could resolve a `+jit` / `+full` build as “newer” than `3.7.0` and pull LLVM (or a much larger binary) when you wanted the default wheel.

There is **no** `pip install alkahest[jit]` / `alkahest[full]` that swaps the native extension: **pip extras only add Python dependencies**, not alternate binaries for the same wheel slot.

**Until a dedicated PEP 503 simple index is published**, tagged releases attach Linux **`linux_x86_64`** wheels on [GitHub Releases](https://github.com/alkahest-cas/alkahest/releases) (CI builds them on `ubuntu-22.04`, not the manylinux image used for default wheels). Pick the `.whl` whose tags match your Python (`cp311`, etc.) and **`linux_x86_64`**.

| Local version | Cargo features | When to use |
|---------------|----------------|-------------|
| *(default PyPI)* | `egraph groebner cranelift` | Cranelift CPU JIT on all published platforms; no system LLVM. |
| `+jit` | `egraph groebner jit` | LLVM CPU JIT (Linux only in CI; larger than default; no Cranelift). |
| `+full` | `egraph groebner jit parallel` | LLVM JIT plus parallel F4 S-polynomial reduction (largest wheel; Linux only in CI). |

Direct-install examples (adjust tag and filename after checking the release assets):

```bash
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v3.7.0/alkahest-3.7.0+full-cp311-cp311-linux_x86_64.whl"
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v3.7.0/alkahest-3.7.0+jit-cp311-cp311-linux_x86_64.whl"
```

These wheels vendor LLVM (for JIT) and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libffi-*.so` or `libLLVM-*.so`, prepend that directory to `LD_LIBRARY_PATH` (or install matching system packages). Release CI uses the same `LD_LIBRARY_PATH` step when smoke-testing wheels.

If your client chokes on `+` in the URL, use percent-encoding (`3.7.0%2Bfull` in the filename segment).

After installing the **default** wheel, `alkahest.jit_is_available()` is `True` (Cranelift). After **`+jit`** or **`+full`**, it is also `True` (LLVM). Gröbner-backed APIs such as `alkahest.solve` are available in **all** wheels since `groebner` became a default feature.

*See the [install matrix](#install-matrix-default-vs-opt-in-wheels) for per-platform coverage.*

**Target layout (roadmap):** a small **extra index** URL (PEP 503) hosting only `+jit` / `+full` wheels, mirroring PyTorch’s `--extra-index-url` workflow:

```bash
pip install 'alkahest==3.7.0+full' --extra-index-url https://EXAMPLE/alkahest-extras/simple
```

### From source

Required to enable optional features (`jit`, `cuda`, `parallel`) or for development. The `groebner` and `egraph` features are already built into default wheels; a source build inherits them automatically. Prerequisites:

- **Rust** stable ≥ 1.76 and nightly:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```
- **uv** (recommended Python tool manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **LLVM 15**: `apt install llvm-15 libllvm15 llvm-15-dev` / `brew install llvm@15`
- **FLINT ≥ 2.9** (includes GMP and MPFR): `apt install libflint-dev` / `brew install flint`

```bash
# Install dev tools (maturin, pytest, ruff, ty, …) without building the Rust extension:
uv sync --no-install-project --group dev
# Build and install the extension into the project venv:
uv run maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "parallel egraph jit groebner"
```

Without `uv`, install maturin directly and run the same develop command:

```bash
pip install maturin
maturin develop --manifest-path alkahest-py/Cargo.toml --release --features "parallel egraph jit groebner"
```

Optional Cargo features: `parallel` (sharded pool + parallel F4 + `numpy_eval_par`), `egraph` (vendored egglog backend; **default** in PyPI wheels), `groebner` (Gröbner solver + Diophantine + homotopy; **default** in both the Rust crate and PyPI wheels), `cranelift` (pure-Rust Tier-1 JIT), `jit` (LLVM JIT), `cuda` (NVPTX codegen).

### Rust crate

`alkahest-cas` is also published on [crates.io](https://crates.io/crates/alkahest-cas) ([docs.rs](https://docs.rs/alkahest-cas)) for use directly from Rust without a Python runtime:

```toml
[dependencies]
alkahest-cas = "3"

# groebner is included by default; add other optional features as needed:
# alkahest-cas = { version = "3", features = ["parallel", "egraph"] }
```

**System prerequisites** (same libraries as the Python build — must be present before `cargo build`):

```bash
# Debian / Ubuntu
sudo apt-get install -y libflint-dev libgmp-dev libmpfr-dev

# macOS
brew install flint
```

The `jit` feature additionally requires LLVM 15 dev headers (`apt install llvm-15-dev` / `brew install llvm@15`). A self-contained runnable example is in [`examples/rust_quickstart/`](examples/rust_quickstart/).

---

## Quick start

```python
import alkahest as ak

caps = ak.capabilities()  # groebner, jit, egraph, parallel
pool = ak.ExprPool()
x = pool.symbol("x")

# Python int literals work in arithmetic (pool still required for symbols)
expr = x**2 + 1

# Differentiation with derivation log
result = ak.diff(ak.sin(expr), x)
print(result.value)   # 2*x*cos(x^2 + 1)
print(result.steps)   # list of rewrite steps

# Integration
r = ak.integrate(ak.exp(x), x)
print(r.value)        # exp(x)

# Simplification — use simplify_trig for sin²+cos², not the catch-all simplify
s = ak.simplify(x + 0)
print(s.value)        # x
print(ak.simplify_trig(ak.sin(x)**2 + ak.cos(x)**2).value)  # 1

# JIT-compile to native code (interpreter fallback when caps["jit"] is False)
f = ak.compile_expr(x**2 + 1, [x])
print(f([3.0]))       # 10.0

# String entry point for agents / notebooks (bindings optional)
e = ak.parse("sin(x)^2 + cos(x)^2", pool, {"x": x})
print(ak.simplify_trig(e).value)  # 1
```

Partial fractions, definite integration, and Lean certificates:

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

f = 1 / (x**2 - pool.integer(1))
print(ak.apart(f, x))  # partial fractions over ℚ

r = ak.integrate(x**2, x, pool.integer(0), pool.integer(1))  # ∫₀¹ x² dx = 1/3
print(r.value)
print(r.certificate)  # Lean 4 proof term when available
```

More runnable examples live in [`examples/`](examples/) — polynomials, Risch integration, Lean certificates, agent workflows, and more.

---

## Expression representations

| Type | Description |
|---|---|
| `Expr` | Generic hash-consed symbolic expression |
| `UniPoly` | Dense univariate polynomial (FLINT-backed) |
| `MultiPoly` | Sparse multivariate polynomial over ℤ |
| `MultiPolyFp` | Sparse multivariate polynomial over 𝔽ₚ (modular arithmetic) |
| `RationalFunction` | Quotient of polynomials with GCD normalization |
| `ArbBall` | Real interval with rigorous error bounds (Arb) |

Representation types are explicit — no silent performance cliffs. Conversion between them is always an opt-in call (`UniPoly.from_symbolic(...)`, etc.).

---

## Result objects

Most transforming operations (`diff`, `simplify`, `integrate`, `sum_*`, …) return a `DerivedResult` with:

- `.value` — the result expression
- `.steps` — derivation log (list of rewrite rules applied)
- `.certificate` — Lean 4 proof term, when available
- `.to_dict()` / `.to_json()` — versioned machine-parseable envelope; use `mode="compact"` in agent loops

Exceptions: `limit` returns a bare `Expr`, and `series` returns a `Series` (with its own `.polynomial` / `.order` fields). Use `.value` only on `DerivedResult`.

### Search plumbing (agent loops)

| Need | Entry point |
|---|---|
| Bound one candidate | `Budget` + `context(budget=…)` → `BudgetExceededError` (`E-BUDGET-*`) |
| Fan out without aborting | `batch_map` / `integrate_many` / `simplify_many` / `diff_many` |
| Compact logs | `DerivedResult.to_dict(mode="compact")` |
| Session provenance | `alkahest.research` claim graphs |
| Propose and fit a parametric family | `alkahest.ansatz` |
| Differential-test against another CAS | `alkahest.crosscheck` |
| Hand off a discrete / mixed int-real subproblem | `alkahest.smt` |

Docs: [Autoresearch / agent loops](https://alkahest-cas.github.io/alkahest/search-plumbing.html).

---

## Modules for autoresearch loops

Three submodules new in 3.8, aimed at unattended search. Each has its own chapter in the
[documentation site](https://alkahest-cas.github.io/alkahest/).

| Module | What it does |
|---|---|
| **`alkahest.ansatz`** | Parametric families with named unknown coefficients — `polynomial`, `rational`, `exponential_polynomial`, `linear_combination`, `quadratic_form` — plus `fit` (solve for the coefficients from a residual, with a verification status), `enumerate_family`, and `certify_nonneg`. This is the "guess the shape, let the CAS pin the constants" loop, done once instead of re-improvised per problem. |
| **`alkahest.crosscheck`** | Differential testing against an external CAS. `check(op, …)` runs one comparison through a ladder of increasingly semantic rungs (syntactic → normalised → numeric → invariant) and reports `agree` / `diverge` / `incomparable` / `unavailable`; `sweep()` generates a seeded corpus of them; `run_frozen_corpus()` replays the pinned cases. A missing oracle is reported as `unavailable`, never as agreement. |
| **`alkahest.smt`** | SMT-LIB 2 export (`to_smtlib`) and a bridge to z3 / cvc5 (`solve`, `supported`, `solvers`). A `sat` model is lifted to exact rationals and **substituted back and checked in-process**; an `unsat` is reported as `externally_asserted` and is deliberately not counted as machine-checked. Algebraic-number witnesses are refused (`E-SMT-003`) rather than truncated to floats. |

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

# Fit an ansatz
from alkahest.ansatz import polynomial, fit
A = polynomial(pool, [x], degree=2)
sol = fit(A, A.expr - (x**2 - pool.integer(3) * x + pool.integer(2)))
print(sol.expr, sol.status)          # (2 + x^2 + (x * -3))  exactly_verified

# Cross-check a result against SymPy
print(ak.crosscheck.check("integrate", x**2, x).outcome)     # 'agree'

# Ask whether the SMT route applies before paying for it
print(ak.smt.supported(pool.gt(x, pool.integer(0))).recommendation)   # 'prefer_in_tree'
```

---

## Known limits

Alkahest is meant to be run unattended, so the limits are documented as prominently as
the features. These are properties of the design, not open bugs — write the loop around
them.

- **`ExprPool` never reclaims.** The expression arena is append-only: no `clear`, no
  refcount, no GC. The only way to free interned nodes is to **drop the whole pool**, and
  every `Expr` / `Matrix` / `DerivedResult` holds a strong reference to its pool, so
  keeping one result keeps everything. Growth is roughly 200 bytes per node and linear
  forever (~2–3.5 KB per `integrate` call) while per-call **latency stays flat** — so a
  long-running loop on one pool dies by OOM with no slowdown to warn you first. Use **one
  pool per problem** and carry `to_dict()` envelopes, not live `Expr` handles.
  [Details](https://alkahest-cas.github.io/alkahest/budgets.html#exprpool-never-reclaims).
- **`wall_ms` is cooperative and its granularity is one primitive operation.** A call
  stops at the first checkpoint after the deadline. Past a certain degree that operation
  is a FLINT call, which no cooperative mechanism can interrupt — a 300 ms budget on a
  degree-62 integrand returns after ~2 s. Only an OS-level timeout goes below that.
- **`run_with_wall_fallback` does not bound wall time for an uncooperative callee.** It
  joins its worker before raising, so it returns when the callee returns:
  `run_with_wall_fallback(time.sleep, 3.0, budget=Budget(wall_ms=50))` raises after
  3000 ms. It exists to turn a silent truncation into a coded error, not to contain an
  unknown callee. Only `integrate` and `limit` currently honour the cooperative budget and
  release the GIL, so only they can be cancelled while already running.
- **`decide` refuses rather than answering** in cases it cannot establish. It covers
  polynomial bodies in ≤ 2 real variables with a ≤ 2-quantifier prefix, and inside that
  fragment it raises `E-CAD-001` when the only candidate solutions sit at an irrational
  boundary point that rational sampling cannot test. Same for linear algebra: an entry or
  determinant whose vanishing is undecidable gives `E-LINALG-010` / `E-MAT-004` instead of
  a guessed branch. **A refusal means undecided, not false** — a search loop that records
  it as a negative result closes a branch it never explored.
- **`Matrix.eigenvals()` can emit casus-irreducibilis cube roots.** These are correct
  under Alkahest's real cube-root convention — `eval_expr` refuses them honestly and
  `interval_eval` returns an unbounded ball — but a principal-branch evaluator (SymPy,
  NumPy) returns a confident number that is *not* an eigenvalue. Evaluate inside Alkahest
  before exporting a radical expression to another tool.
  [Details](https://alkahest-cas.github.io/alkahest/interop.html#the-interop-trap-casus-irreducibilis-cube-roots).

---

## Reinforcement learning

`alkahest.rl` exposes **verifiable RL environments** backed by the CAS. The core layer
(`alkahest.rl.core`) is trainer-agnostic; domain environments live under
`alkahest.rl.envs.*` and optionally integrate with [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/verifiers).

```bash
pip install "alkahest[rl]"   # Python ≥ 3.10; adds verifiers + datasets
```

```python
from alkahest.rl.envs.integration import IntegrationVerifier, load_environment

verifier = IntegrationVerifier()
# reward = verifier.verify(model_output, {"f_expr": f, "is_elementary": True, "pool": pool})

env = load_environment(difficulty_tier=0, n_train=1000, n_eval=100, adaptive=True)
```

| Component | Description |
|-----------|-------------|
| `IntegrationVerifier` | Layered check: symbolic diff → e-graph → interval spot checks; rewards honest refusal on NonElementary integrands |
| `load_environment()` | Returns a `verifiers.SingleTurnEnv` with Risch-tier curriculum |
| `recipes/verl_integration_reward.py` | Drop-in reward for [veRL](https://github.com/volcengine/verl) |

**Environments Hub:** [`alkahest/alkahest-symbolic-integration`](https://app.primeintellect.ai/dashboard/environments/alkahest/alkahest-symbolic-integration) — install with `prime env install alkahest/alkahest-symbolic-integration`. Publish updates from `python/alkahest/rl/envs/integration/` with `prime env push`. Full checklist in the [RL guide](docs/mdbook/src/rl.md#hub-checklist).

---

## Documentation and further reading

- [**Documentation site**](https://alkahest-cas.github.io/alkahest/) — full API reference and user guide
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — crates, directory layout, and key files
- [`ROADMAP.md`](ROADMAP.md) — planned milestones
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Rust vs Python layer guide
- [`TESTING.md`](TESTING.md) — property-based testing, fuzzing, sanitizers, CI tiers
- [`BENCHMARKS.md`](BENCHMARKS.md) — criterion and Python benchmark suites
- [`examples/`](examples/) — runnable end-to-end examples
- [`LICENSE`](LICENSE) — Apache 2.0 license

---

## Stability

Alkahest follows semantic versioning from `1.0`. The stable surface is everything re-exported from `alkahest_cas::stable` (Rust) and `alkahest.__all__` (Python). Experimental APIs live under `alkahest_cas::experimental` and `alkahest.experimental` and may change in minor releases.
