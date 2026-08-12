# Alkahest

<h3 align="center">
  <a href="https://pypi.org/project/alkahest/">PyPI</a> •
  <a href="https://crates.io/crates/alkahest-cas">crates.io</a> •
  <a href="https://alkahest-cas.github.io/alkahest/">Docs</a> •
  <a href="https://alkahest-cas.github.io/">Website</a> •
  <a href="https://app.primeintellect.ai/dashboard/environments/alkahest/alkahest-symbolic-integration">RL environment</a> •
  <a href="https://alkahest-cas.github.io/playground/">Demo</a>
</h3>

Alkahest is a high-performance computer algebra system built for both humans and agents. It is especially well suited for autoresearch agents doing work in pure and applied mathematics. Available as a Python package or a Rust crate.

The main stack is: Rust kernel → FLINT/Arb (polynomials, ball arithmetic) → egglog + colored e-graphs (simplification) → Cranelift/LLVM JIT + MLIR (native and GPU codegen) → PyO3 → Python

## Highlights

- **Verifiable by construction.** Every computation produces a derivation log; a meaningful subset can export Lean 4 proofs for independent verification.
- **520× faster than SymPy** on trig-identity simplification — and 77× faster than Mathematica on the same task ([cross-CAS report](benchmarks/results/report.md)).
- **~40× faster than SymPy** on 2-variable quadratic systems, via FLINT-backed polynomial arithmetic and a compiled F4 core ([solving guide](docs/mdbook/src/solving.md)).
- **GPU codegen is a routine operation, not a research project** — 16.2× over the CPU JIT on a 1M-point polynomial evaluation (NVPTX `sm_86`, RTX 3090). Requires a source build with `--features cuda`; the PyPI wheel has no GPU support ([GPU guide](docs/mdbook/src/gpu.md)).
- **A trained RL environment.** GRPO against the CAS verifier moved `Qwen2.5-1.5B-Instruct` from 11.7% → 15.1% on elementary integrals (+29% relative) with no stored reference answers — the CAS grades every rollout, including honest refusal on non-elementary integrands.
- **Built for agent loops.** String entry point (`ak.parse`), per-candidate `Budget`s, batched `*_many` fan-out, and compact JSON envelopes for cheap logging.
- **No silent performance cliffs.** `Expr`, `UniPoly`, `MultiPoly`, `ArbBall` and friends are explicit representations; conversion between them is always an opt-in call.

---

## Install

Python **3.9–3.13**:

```bash
pip install alkahest
pip install "alkahest[rl]"   # RL environments; Python >= 3.10
```

Rust users: `alkahest-cas = "2"` — see [Rust crate](#rust-crate).

Default wheels are batteries-included: e-graph simplification, the Gröbner solver (so `alkahest.solve`, Diophantine, and homotopy work out of the box), and the pure-Rust **Cranelift** CPU JIT, with no system LLVM required. They do **not** include LLVM JIT or `parallel` — for those, use a **PyTorch-style** opt-in **`+jit`** / **`+full`** Linux wheel from [GitHub Releases](#opt-in-linux-wheels-jit-and-full-pytorch-style), not the default PyPI resolver path.

### Install matrix (default vs opt-in wheels)

Probe your environment after install: `alkahest.capabilities()["features"]` and `alkahest.jit_is_available()`.

| Artifact | Where | OS / arch (CI) | Python | Cranelift JIT | LLVM JIT | `parallel` |
|----------|-------|----------------|--------|---------------|----------|------------|
| **Default** (`pip install alkahest`) | [PyPI](https://pypi.org/project/alkahest/) | Linux manylinux x86_64; macOS arm64; Windows x86_64 | 3.9–3.13 | yes | no | no |
| **`+jit`** (`X.Y.Z+jit`) | GitHub Releases only | Linux x86_64 | 3.9–3.13 | no | yes | no |
| **`+full`** (`X.Y.Z+full`) | GitHub Releases only | Linux x86_64 | 3.9–3.13 | no | yes | yes |

**macOS / Windows:** default PyPI wheels include Cranelift JIT. **`+jit`** and **`+full`** are **not** built in CI (LLVM / MSYS2 constraints); use [building from source](#from-source) with `--features jit` (and `parallel` for F4 parallelism) on those platforms.

**Linux LLVM wheels** vendor LLVM and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libffi-*.so` or `libLLVM-*.so`, prepend that directory to `LD_LIBRARY_PATH`.

### Opt-in Linux wheels: `+jit` and `+full` (PyTorch-style)

**Why a separate index or direct wheel URL:** feature-heavy wheels use a PEP 440 **local version** (for example `3.8.0+jit` or `3.8.0+full`). Those builds **must not** be mixed into the main PyPI project’s simple API for the same reason PyTorch publishes CUDA wheels on `download.pytorch.org`: otherwise `pip install alkahest` could resolve a `+jit` / `+full` build as “newer” than `3.8.0` and pull LLVM (or a much larger binary) when you wanted the default wheel.

There is **no** `pip install alkahest[jit]` / `alkahest[full]` that swaps the native extension: **pip extras only add Python dependencies**, not alternate binaries for the same wheel slot.

**Until a dedicated PEP 503 simple index is published**, tagged releases attach Linux **`linux_x86_64`** wheels on [GitHub Releases](https://github.com/alkahest-cas/alkahest/releases) (CI builds them on `ubuntu-22.04`, not the manylinux image used for default wheels). Pick the `.whl` whose tags match your Python (`cp311`, etc.) and **`linux_x86_64`**.

| Local version | Cargo features | When to use |
|---------------|----------------|-------------|
| *(default PyPI)* | `egraph groebner cranelift` | Cranelift CPU JIT on all published platforms; no system LLVM. |
| `+jit` | `egraph groebner jit` | LLVM CPU JIT (Linux only in CI; larger than default; no Cranelift). |
| `+full` | `egraph groebner jit parallel` | LLVM JIT plus parallel F4 S-polynomial reduction (largest wheel; Linux only in CI). |

Direct-install examples (adjust tag and filename after checking the release assets):

```bash
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v3.8.0/alkahest-3.8.0+full-cp311-cp311-linux_x86_64.whl"
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v3.8.0/alkahest-3.8.0+jit-cp311-cp311-linux_x86_64.whl"
```

These wheels vendor LLVM (for JIT) and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libffi-*.so` or `libLLVM-*.so`, prepend that directory to `LD_LIBRARY_PATH` (or install matching system packages). Release CI uses the same `LD_LIBRARY_PATH` step when smoke-testing wheels.

If your client chokes on `+` in the URL, use percent-encoding (`3.8.0%2Bfull` in the filename segment).

After installing the **default** wheel, `alkahest.jit_is_available()` is `True` (Cranelift). After **`+jit`** or **`+full`**, it is also `True` (LLVM). Gröbner-backed APIs such as `alkahest.solve` are available in **all** wheels since `groebner` became a default feature.

*See the [install matrix](#install-matrix-default-vs-opt-in-wheels) for per-platform coverage.*

**Target layout (roadmap):** a small **extra index** URL (PEP 503) hosting only `+jit` / `+full` wheels, mirroring PyTorch’s `--extra-index-url` workflow:

```bash
pip install 'alkahest==3.8.0+full' --extra-index-url https://EXAMPLE/alkahest-extras/simple
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

Optional Cargo features: `parallel` (sharded pool + parallel F4 + `numpy_eval_par`), `egraph` (vendored egglog backend; **default** in PyPI wheels), `groebner` (Gröbner solver + Diophantine + homotopy; **default** in both the Rust crate and PyPI wheels), `cranelift` (pure-Rust Tier-1 JIT), `jit` (LLVM JIT), `cuda` (NVPTX codegen — needs LLVM 15 with the NVPTX target; adds `compile_cuda`), `groebner-cuda` (CUDA Macaulay-matrix kernel — needs only `cudarc`, and is a Rust-crate entry point that no Python call reaches). Neither GPU feature is in any published wheel: see the [GPU guide](docs/mdbook/src/gpu.md).

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

## Features

| Area | What you get | Entry points |
|---|---|---|
| [Calculus](docs/mdbook/src/calculus.md) | Differentiation, Risch integration (definite and indefinite), limits, series expansion, residues | `diff` · `integrate` · `limit` · `series` · `residue` |
| [Simplification](docs/mdbook/src/simplification.md) | Rule engine plus [e-graph saturation](docs/mdbook/src/egraph.md), with domain-specific passes instead of one catch-all | `simplify` · `simplify_trig` · `simplify_log_exp` · `simplify_egraph` |
| Polynomials | FLINT-backed univariate and multivariate arithmetic, factorization over ℤ and 𝔽ₚ, sparse GCD and interpolation, resultants, partial fractions | `UniPoly.factor_z` · `factor_univariate_mod_p` · `gcd_sparse` · `resultant` · `apart` · `cancel` |
| [Solving](docs/mdbook/src/solving.md) | Gröbner bases (F4), triangular decomposition, primary decomposition, real root isolation via CAD, numerical fallback | `solve` · `real_roots` · `triangularize` · `cad_project` · `primary_decomposition` · `solve_numerical` |
| Sums and products | Gosper and [Zeilberger creative telescoping](docs/mdbook/src/telescoping.md), WZ pair verification, linear recurrence solving | `sum_indefinite` · `sum_definite` · `zeilberger` · `verify_wz_pair` · `rsolve` |
| Linear algebra | Symbolic matrices, eigenvalues and eigenvectors, Jacobians, Routh–Hurwitz stability | `Matrix.eigenvals` · `jacobian` · `routh_hurwitz` |
| [ODEs and DAEs](docs/mdbook/src/ode-dae.md) | Symbolic ODE/DAE systems, Pantelides index reduction, sensitivity and adjoint systems, acausal component modeling | `ODE` · `DAE` · `pantelides` · `dae_index_reduce` · `sensitivity_system` |
| Number theory | FLINT-backed integer theory, Diophantine equations, LLL lattice reduction, PSLQ integer-relation detection | `number_theory` · `diophantine` · `lattice` · `guess_relation` |
| [Rigorous numerics](docs/mdbook/src/ball-arithmetic.md) | Arb ball arithmetic — every float carries a proven error bound | `ArbBall` · `interval_eval` · `refine_root` |
| [Code generation](docs/mdbook/src/codegen.md) | JIT to native CPU code (Cranelift or LLVM), [NVPTX GPU kernels](docs/mdbook/src/gpu.md), C source, StableHLO, vectorized NumPy | `compile_expr` · `jit` · `numpy_eval` · `emit_c` · `to_stablehlo` · `compile_cuda` |
| Program transforms | JAX-style `trace` / `grad` / `jit` over Python functions, plus symbolic gradients and forward-mode dual-number AD | `trace_fn` · `grad` · `symbolic_grad` · `diff_forward` |
| [Verification](docs/mdbook/src/lean-certs.md) | [Derivation logs](docs/mdbook/src/derivations.md) on every result, Lean 4 certificate export, [coverage reporting](docs/mdbook/src/certificate-coverage.md) | `DerivedResult.steps` · `to_lean` · `certifiable` · `certificate_coverage` |
| [Agent loops](docs/mdbook/src/search-plumbing.md) | String parsing, [budgets and cancellation](docs/mdbook/src/budgets.md), [batched fan-out](docs/mdbook/src/batch.md), [claim graphs](docs/mdbook/src/claim-graphs.md) for session provenance | `parse` · `Budget` · `batch_map` · `research` |
| Assumptions | Domain-aware reasoning (`x > 0`, ℤ vs ℝ) with a decision procedure over quantified statements | `Assumptions` · `Domain` · `Forall` · `decide` · `satisfiable` |
| Output | LaTeX, Unicode pretty-printing, plots (2D/3D, implicit, parametric, DAG structure) | `latex` · `unicode_str` · `plot` · `plot3d` · `plot_dag` |

Full listing in the [Python API reference](https://alkahest-cas.github.io/alkahest/python-api.html).

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
- [`demo-playground/`](demo-playground/README.md) — notebook UI, agent chat, and demo recording stack (the hosted [playground](https://alkahest-cas.github.io/playground/) is the WASM build of it)
- [`LICENSE`](LICENSE) — Apache 2.0 license

---

## Stability

Alkahest follows semantic versioning from `1.0`. The stable surface is everything re-exported from `alkahest_cas::stable` (Rust) and `alkahest.__all__` (Python). Experimental APIs live under `alkahest_cas::experimental` and `alkahest.experimental` and may change in minor releases.
