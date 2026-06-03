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

**Install:** the package is published on [PyPI](https://pypi.org/project/alkahest/); use `pip install alkahest` (**Python 3.9–3.13**). See [Install](#install) below for optional **`+jit`** / **`+full`** Linux wheels (GitHub Releases or a future extras index) and building from source.

**Demo:** try the hosted **[playground](https://alkahest-cas.github.io/playground/)** (WASM in-browser, or bring your own server/Jupyter URL + token), or run [`demo-playground/`](demo-playground/) locally for the full agent and recording stack. See [`demo-playground/README.md`](demo-playground/README.md).

**Stack:** Rust kernel → FLINT/Arb (polynomials, ball arithmetic) → vendored egglog + colored e-graphs (simplification) → Cranelift/LLVM JIT + MLIR (native and GPU codegen) → PyO3 → Python

---

## Install

**Requirements:** Python **3.9–3.13** ([PyPI](https://pypi.org/project/alkahest/) `requires-python`).

```bash
pip install alkahest
```

For an isolated environment (recommended when juggling versions or building from source):

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install alkahest
```

Default PyPI wheels include the **vendored egglog** e-graph backend (`egraph` feature) and the **Gröbner solver** (`groebner` feature — so `alkahest.solve`, Diophantine, homotopy, and related APIs are available out of the box) but **not** LLVM JIT, Cranelift, or `parallel`. Numeric APIs use the tree-walking interpreter fallback. For native LLVM CPU JIT—or JIT plus parallel F4—use a **PyTorch-style** opt-in wheel (separate artifact / index), not the default PyPI resolver path. From source, add `--features cranelift` for a pure-Rust fast-compile JIT tier without system LLVM.

### Opt-in Linux wheels: `+jit` and `+full` (PyTorch-style)

**Why a separate index or direct wheel URL:** feature-heavy wheels use a PEP 440 **local version** (for example `2.0.3+jit` or `2.0.3+full`). Those builds **must not** be mixed into the main PyPI project’s simple API for the same reason PyTorch publishes CUDA wheels on `download.pytorch.org`: otherwise `pip install alkahest` could resolve a `+jit` / `+full` build as “newer” than `2.0.3` and pull LLVM (or a much larger binary) when you wanted the default wheel.

There is **no** `pip install alkahest[jit]` / `alkahest[full]` that swaps the native extension: **pip extras only add Python dependencies**, not alternate binaries for the same wheel slot.

**Until a dedicated PEP 503 simple index is published**, tagged releases attach Linux **`linux_x86_64`** wheels on [GitHub Releases](https://github.com/alkahest-cas/alkahest/releases) (CI builds them on `ubuntu-22.04`, not the manylinux image used for default wheels). Pick the `.whl` whose tags match your Python (`cp311`, etc.) and **`linux_x86_64`**.

| Local version | Cargo features | When to use |
|---------------|----------------|-------------|
| `+jit` | `egraph groebner jit` | LLVM CPU JIT (smaller than `+full`; groebner/egraph are already in default wheels). |
| `+full` | `egraph groebner jit parallel` | JIT plus parallel F4 S-polynomial reduction (largest wheel; groebner already in default). |

Direct-install examples (adjust tag and filename after checking the release assets):

```bash
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v2.3.1/alkahest-2.3.1+full-cp311-cp311-linux_x86_64.whl"
pip install "https://github.com/alkahest-cas/alkahest/releases/download/v2.3.1/alkahest-2.3.1+jit-cp311-cp311-linux_x86_64.whl"
```

These wheels vendor LLVM (for JIT) and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libffi-*.so` or `libLLVM-*.so`, prepend that directory to `LD_LIBRARY_PATH` (or install matching system packages). Release CI uses the same `LD_LIBRARY_PATH` step when smoke-testing wheels.

If your client chokes on `+` in the URL, use percent-encoding (`2.3.1%2Bfull` in the filename segment).

After installing `+jit` or `+full`, `alkahest.jit_is_available()` should be `True`. Gröbner-backed APIs such as `alkahest.solve` are available in **all** wheels (including the default PyPI wheel) since `groebner` became a default feature.

*macOS and Windows `+jit` / `+full` wheels are not produced in CI yet (LLVM / MSYS2 constraints); use [building from source](#from-source) there.*

**Target layout (roadmap):** a small **extra index** URL (PEP 503) hosting only `+jit` / `+full` wheels, mirroring PyTorch’s `--extra-index-url` workflow:

```bash
pip install 'alkahest==2.0.3+full' --extra-index-url https://EXAMPLE/alkahest-extras/simple
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
alkahest-cas = "2"

# groebner is included by default; add other optional features as needed:
# alkahest-cas = { version = "2", features = ["parallel", "egraph"] }
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

pool = ak.ExprPool()
x = pool.symbol("x")

# Differentiation with derivation log
result = ak.diff(ak.sin(x ** 2), x)
print(result.value)   # 2*x*cos(x^2)
print(result.steps)   # list of rewrite steps

# Integration
r = ak.integrate(ak.exp(x), x)
print(r.value)        # exp(x)

# Simplification
s = ak.simplify(x + pool.integer(0))
print(s.value)        # x

# JIT-compile to native code
f = ak.compile_expr(x ** 2 + pool.integer(1), [x])
print(f([3.0]))       # 10.0
```

### Explicit polynomial representations

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

# FLINT-backed univariate polynomial
p = ak.UniPoly.from_symbolic(x ** 3 + pool.integer(-2) * x + pool.integer(1), x)
print(p.degree())        # 3
print(p.coefficients())  # [1, -2, 0, 1]

# GCD
a = ak.UniPoly.from_symbolic(x ** 2 + pool.integer(-1), x)
b = ak.UniPoly.from_symbolic(x + pool.integer(-1), x)
print(a.gcd(b))          # x - 1

# Factorization over ℤ (FLINT — Zassenhaus / van Hoeij)
fac = a.factor_z()
print(int(fac.unit), fac.factor_list())  # unit and list of (UniPoly, exponent)

# Dense univariate mod p (Berlekamp / Cantor–Zassenhaus via FLINT nmod)
fp = ak.factor_univariate_mod_p([1, 0, 1], 2)  # x^2+1 over GF(2)
print(fp.factor_list())

# Rational function with automatic GCD normalization
rf = ak.RationalFunction.from_symbolic(x ** 2 + pool.integer(-1), x + pool.integer(-1), [x])
print(rf)                # x + 1

# Sparse multivariate polynomial
mp = ak.MultiPoly.from_symbolic(x ** 2 * y + x * y ** 2, [x, y])
print(mp.total_degree()) # 3
```

### Sparse multivariate interpolation (Ben-Or/Tiwari, Zippel)

Black-box recovery of sparse polynomials from evaluations, and sparse modular GCD as a substrate for faster exact GCD algorithms.

```python
import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")

# Recover a sparse univariate f ∈ 𝔽ₚ[x] from 2T black-box evaluations
p = 32749  # prime
target = lambda v: (v**5 + 3*v**3 + 7) % p  # hidden: x^5 + 3x^3 + 7
f = ak.sparse_interp_univariate(target, term_bound=3, prime=p)
print(f)   # recovered polynomial

# Recover a sparse multivariate f ∈ 𝔽ₚ[x, y] via Zippel's algorithm
target2 = lambda vals: (vals[0]**3 * vals[1]**2 + vals[0] * vals[1]**4) % p
g = ak.sparse_interp(target2, vars=[x, y], term_bound=2, degree_bound=5, prime=p)
print(g)   # recovered MultiPolyFp

# Sparse modular GCD over ℤ[x₁,...,xₙ]
f2 = ak.MultiPoly.from_symbolic((x + y) * (x - y), [x, y])
g2 = ak.MultiPoly.from_symbolic((x + y) * (x + pool.integer(1)), [x, y])
h = ak.gcd_sparse(f2, g2, term_bound=4, degree_bound=4)
print(h)   # x + y
```

### Symbolic summation (Gosper / recurrences)

Indefinite and definite sums for terms whose shift ratio `F(k+1)/F(k)` is rational in `k`—typically polynomials multiplied by `gamma` of a **linear** expression in `k`. General multivariate Zeilberger automation is partial; use `verify_wz_pair(F, G, n, k)` to check a discrete telescoping certificate after simplification.

```python
import alkahest as ak

pool = ak.ExprPool()
k = pool.symbol("k")
n = pool.symbol("n")
term = ak.simplify(k * ak.gamma(k + pool.integer(1))).value
print(ak.sum_indefinite(term, k).value)
print(ak.sum_definite(term, k, pool.integer(0), n).value)

fib = ak.solve_linear_recurrence_homogeneous(
    n, [(-1, 1), (-1, 1), (1, 1)], [pool.integer(0), pool.integer(1)]
)
```

### Difference equations / `rsolve`

Linear recurrences in one sequence with **constant coefficients** and a **polynomial** right-hand side (in the recurrence index `n`). Write shifts as `pool.func("f", [n + integer])`, pass the equation as a single expression that simplifies to zero, and optional `initials` as `{n: value}` to fix the `C0`, `C1`, … symbols.

```python
import alkahest as ak

pool = ak.ExprPool()
n = pool.symbol("n")
f = lambda *a: pool.func("f", list(a))
# f(n) - f(n-1) - 1 == 0  →  general solution n + C0
eq = ak.simplify(f(n) - f(n + pool.integer(-1)) - pool.integer(1)).value
print(ak.rsolve(eq, n, "f", None))
# Fibonacci with f(0)=0, f(1)=1
fib_eq = ak.simplify(
    f(n) - f(n + pool.integer(-1)) - f(n + pool.integer(-2))
).value
print(ak.rsolve(fib_eq, n, "f", {0: pool.integer(0), 1: pool.integer(1)}))
```

Non-homogeneous **order > 2** and sequences with **polynomial coefficients** in `n` are not implemented yet (see `RsolveError` / `E-RSOLVE-*`).

### Symbolic products (`∏`)

`product_definite(term, k, lo, hi)` closes $\prod_{i=\text{lo}}^{\text{hi}} \text{term}(i)$ (inclusive) when `term` simplifies to **ℚ(`k`)** whose numerator/denominator **factor into ℤ-linear** polynomials — the implementation expands each linear factor $\alpha k+\beta$ with $\Gamma$ shifts $\Gamma(\text{hi}+\beta/\alpha+1)/\Gamma(\text{lo}+\beta/\alpha)$ and collects $\alpha^{(\text{hi}-\text{lo}+1)\cdot e}$. `product_indefinite` returns a `Γ`/power witness `Z(k)` with `simplify`-stable ratio `Z(k+1)/Z(k)=term`. `Product(term, (k, lo, hi)).doit()` matches SymPy ergonomics (`DerivedResult`; use `.value`). Irreducible quadratics in `k`, extra symbols besides `k`, and non-integer powers are rejected (`ProductError` / `E-PROD-*`).

```python
import alkahest as ak

pool = ak.ExprPool()
k, n = pool.symbol("k"), pool.symbol("n")
P = ak.Product(k, (k, pool.integer(1), n))
print(ak.simplify(P.doit().value).value)

kp2 = k ** 2
term = ak.simplify(
    ((k + pool.integer(-1)) * (k + pool.integer(1))) / kp2
).value  # (k²-1)/k²

print(ak.simplify(
    ak.product_definite(term, k, pool.integer(2), n).value
).value)
```

### Diophantine equations

Two integer unknowns, equation as a single polynomial `= 0`: **linear** families `a·x + b·y + c = 0`, **sum of two squares** `x² + y² = n` (finitely many tuples), and **unit Pell** `x² - D·y² = 1` (fundamental solution `(x₀, y₀)` via the continued-fraction period of `√D`). The `groebner` feature is required and is **included in all PyPI wheels** since 2.3.1. API: `diophantine(equation, [x, y])` → `DiophantineSolution` with `.kind` (`parametric_linear`, `finite`, `pell_fundamental`, `no_solution`) and typed fields.

```python
import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")
sol = ak.diophantine(pool.integer(3) * x + pool.integer(5) * y - pool.integer(1), [x, y])
assert sol.kind == "parametric_linear"
pell = ak.diophantine(x**2 - pool.integer(2) * y**2 - pool.integer(1), [x, y])
assert pell.kind == "pell_fundamental" and int(str(pell.fundamental[0])) == 3
```

Quadratics with an **`x·y` cross-term**, unequal ellipse coefficients, or **generalized Pell** right-hand sides `≠ 1` are not implemented yet (`DiophantineError` / `E-DIOPH-*`).

### Integer number theory

Submodule `alkahest.number_theory`: `isprime`, `factorint`, `nextprime`, `totient`, `jacobi_symbol`, `nthroot_mod` (prime modulus; `k=2` or `gcd(k,p−1)=1`), `discrete_log` (linear scan for moderate primes), plus quadratic `DirichletChi` on odd square-free conductors. Implemented via FLINT `fmpz` in the Rust kernel; raises `NumberTheoryError` (`E-NT-*`) on invalid input.

```python
import alkahest as ak
import alkahest.number_theory as nt

assert nt.isprime(2**127 - 1)
assert nt.factorint(2**32 - 1)[65537] == 1
assert nt.discrete_log(13, 3, 17) == 4
assert pow(nt.nthroot_mod(144, 2, 401), 2, 401) == 144 % 401
```

### Noncommutative algebra

Symbols can opt out of multiplicative commutativity: `pool.symbol("A", "real", commutative=False)`. Then `A * B` and `B * A` are distinct expressions, and sorting of `Mul` factors is disabled. The egglog backend automatically falls back to the rule-based simplifier when such symbols appear.

Pauli matrices (names `sx`, `sy`, `sz`) and a minimal orthogonal Clifford pair (`cliff_e1`, `cliff_e2`) have built-in rewrite tables; combine default rules with `ak.simplify_pauli` or `ak.simplify_clifford_orthogonal`. See `examples/noncommutative.py`.

### Truncated series / Laurent tail

`series(expr, var, point, order)` builds a symbolic truncation about `(var − point)` and appends a `BigO(⋯)` remainder. Smooth functions use repeated differentiation; simple poles such as `1/x` at zero take the rational Laurent path. `Series.expr` is the pooled sum-plus-order expression; `ExprPool.big_o(inner)` constructs standalone order bounds.

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
s_cos = ak.series(ak.cos(x), x, pool.integer(0), 6)
s_inv = ak.series(x ** (-1), x, pool.integer(0), 4)
print(s_cos.expr)
```

### Rigorous interval arithmetic

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
result = ak.interval_eval(ak.sin(x), {x: ak.ArbBall(1.0, 1e-10)})
print(result)  # guaranteed enclosure of sin(1 ± 1e-10)
```

### String expressions

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

# Parse a string into a symbolic expression
e = ak.parse("x^2 + 2*x + 1", pool, {"x": x})
print(e)                    # (x^2 + (x * 2)) + 1

# Round-trip: parse then pretty-print
expr = ak.parse("sin(x)^2 + cos(x)^2", pool, {"x": x})
print(ak.latex(expr))        # \sin\!\left(x\right)^2 + \cos\!\left(x\right)^2
print(ak.unicode_str(expr))  # sin(x)² + cos(x)²
```

### Lattice reduction and approximate integer relations

Exact LLL reduction on integer bases lives under `alkahest.lattice`; for floating constants (as `float` or decimal strings) `guess_relation` searches for small integer coefficient vectors whose dot product has tiny residual relative to the working precision:

```python
import alkahest as ak

basis = ak.lattice.lll_reduce_rows([[2, 15], [1, 21]])
rel = ak.guess_relation(["1", "2", "3"], precision_bits=256)
```

The relation finder is an augmented-lattice + LLL heuristic, not Ferguson–Bailey PSLQ; treat results as exploratory unless verified independently.

### Regular chains / triangular decomposition

Lex-order Gröbner bases yield triangular sets used by the polynomial solver. The `triangularize(equations, vars)` API returns one or more `RegularChain` objects (polynomials as `GbPoly` tiles), splitting along factored bottom univariates when applicable. The built-in `solve()` routine retries backsolving from an extracted chain when the full basis is not directly triangular enough.

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")
eq1 = x**2 + y**2 - pool.integer(1)
eq2 = y - x
chains = ak.triangularize([eq1, eq2], [x, y])
assert len(chains) >= 1
```

### Primary decomposition

Lex-order Gröbner data is used to split ideals via saturations (`I : x_i^∞` with `(I + (x_i))`) and, in the zero-dimensional case, factoring a univariate polynomial in the first Lex variable. `primary_decomposition(polys, vars)` returns `PrimaryComponent` objects with `.primary()` and `.associated_prime()` Gröbner bases; `radical(polys, vars)` returns a basis for √I.

```python
import alkahest as ak

pool = ak.ExprPool()
x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
comps = ak.primary_decomposition([x * y, x * z], [x, y, z])
assert len(comps) == 2
r = ak.radical([x**2, x * y], [x, y])
assert r.contains(x)
```

### Differential algebra / Rosenfeld–Gröbner

Polynomial DAEs in implicit form `g_i(t, y, y') = 0` can be analysed by **prolongation** (formal time derivatives of each equation, with the same derivative-state extension rule as Pantelides) and **ordinary Gröbner bases** over the jet variables. Inconsistent systems yield the unit ideal. Use `rosenfeld_groebner(dae, order=..., max_prolong_rounds=...)`; when Pantelides exhausts its index cap, `dae_index_reduce(dae)` falls back to this pass.

```python
import alkahest as ak

pool = ak.ExprPool()
t = pool.symbol("t")
y = pool.symbol("y")
dy = pool.symbol("dy/dt")
dae = ak.DAE.new([dy - y, dy - y - pool.integer(1)], [y], [dy], t)
r = ak.rosenfeld_groebner(dae, max_prolong_rounds=2)
assert r.consistent is False
```

### Numerical algebraic geometry / homotopy continuation

Square polynomial systems can be solved numerically with a **total-degree** homotopy in `ℂⁿ` (`(1-t)·γ·G + t·F`), Newton polish on real projections, a conservative **Smale-style** `α` heuristic, and **`ArbBall` enclosures** attached to each coordinate. Use `solve(eqs, vars, method="homotopy")` for a list of `dict` solutions (`Expr` keys → `float`). For residuals, certification flags, and enclosures, call `solve_numerical(...)`, which returns `CertifiedSolution` objects (`.coordinates`, `.smale_certified`, `.to_dict()`, `.enclosures()`, …).

Ideals whose finite root count in `ℂⁿ` is **strictly below** the Bézout bound (often called *deficient* — e.g. the Katsura family) typically need a **polyhedral / mixed-volume** start system; only the Bézout start (`∏ deg F_i` paths) is implemented here.

```python
import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")
neg1 = pool.integer(-1)
sols = ak.solve([x**2 + neg1, y**2 + neg1], [x, y], method="homotopy")
cs = ak.solve_numerical([x**2 + neg1], [x])[0]
print(cs.coordinates, cs.smale_certified, cs.enclosures())
```

### Composable transformations

```python
import alkahest as ak

pool = ak.ExprPool()

@ak.trace(pool)
def f(x):
    return ak.sin(x ** 2)

df = ak.grad(f)          # symbolic gradient
df_fast = ak.jit(df)     # compiled gradient
```

### Plotting

Alkahest never bundles a plotting library — it detects what is installed and calls into it. The default backend is **Matplotlib** (static PNG/SVG); **Plotly** is also supported for interactive figures (and renders natively in the demo-playground notebook). A dependency-free **SVG renderer** is built into the Rust kernel and works without any plotting library installed.

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

# Curve (matplotlib by default, or backend="plotly")
ax = ak.plot(ak.sin(x), x, (-3 * 3.14159, 3 * 3.14159))

# 3-D surface
y = pool.symbol("y")
fig = ak.plot3d(ak.sin(x) * ak.cos(y), x, y, (-5, 5), (-5, 5))

# Parametric curve
t = pool.symbol("t")
ax2 = ak.plot_parametric(ak.cos(t), ak.sin(t), t, (0, 6.28318))

# Implicit curve: x² + y² = 1
ax3 = ak.plot_implicit(x**2 + y**2 - pool.integer(1), x, y, (-2, 2), (-2, 2))

# Real roots of a polynomial on the x-axis
p = ak.UniPoly.from_symbolic(x**3 - x, x)
ax4 = ak.plot_roots(p, x)

# Series truncation vs exact function
s = ak.series(ak.cos(x), x, pool.integer(0), 6)
ax5 = ak.plot_series(s, ak.cos(x), x, (-4, 4))

# Expression DAG (requires graphviz package, falls back to DOT string)
dot = ak.plot_dag(ak.sin(x**2 + pool.integer(1)))

# Dependency-free SVG (no matplotlib/plotly needed)
svg_str = ak.plot_svg(ak.sin(x), x, (-6, 6))
```

Install a backend once — alkahest uses whichever is available:

```bash
pip install matplotlib   # default; static PNG/SVG
pip install plotly       # interactive; also renders in demo-playground
```

For GPU-accelerated plots on dense grids (pairs well with the `+full` JIT wheel):

```bash
pip install fastplotlib
```

```python
from alkahest.experimental._fastplotlib import fplot, fplot3d
fplot(ak.sin(x), x, (-10, 10), n=100_000)
```

---

## Directory layout

```
alkahest/
├── alkahest-core/         # Rust kernel (published as the alkahest-cas crate)
│   ├── src/
│   │   ├── kernel/        # hash-consed expression DAG, ExprPool
│   │   ├── algebra/       # noncommutative Pauli / Clifford rules
│   │   ├── parse.rs       # Pratt expression parser (parse / ParseError)
│   │   ├── poly/          # UniPoly, MultiPoly, RationalFunction
│   │   ├── simplify/      # e-graph simplification (egglog)
│   │   ├── diff/          # symbolic differentiation
│   │   ├── integrate/     # symbolic integration
│   │   ├── calculus/      # series / limits
│   │   ├── jit/           # LLVM JIT and interpreter
│   │   ├── ball/          # Arb ball arithmetic
│   │   ├── ode/           # ODE analysis
│   │   ├── dae/           # DAE analysis and index reduction
│   │   ├── diffalg/       # Rosenfeld–Gröbner / differential elimination (groebner)
│   │   ├── solver/        # polynomial solving: Gröbner triangular, regular chains, homotopy
│   │   ├── lean/          # Lean 4 proof certificate export
│   │   ├── plot/          # SVG polyline + Graphviz DOT renderers (dependency-free)
│   │   └── primitive/     # primitive registration system
│   └── benches/           # criterion benchmarks
├── alkahest-mlir/         # MLIR dialect and lowering passes
├── alkahest-py/           # PyO3 bindings (Rust side)
├── python/alkahest/       # Python package
│   ├── _plot.py           # plotting: plot, plot3d, plot_parametric, plot_implicit, …
│   ├── _transform.py      # trace, grad, jit decorators
│   ├── _pytree.py         # JAX-style pytree flattening
│   ├── _context.py        # context manager and defaults
│   └── experimental/      # unstable API surface
│       └── _fastplotlib.py# GPU-accelerated plotting adapter
├── examples/              # runnable end-to-end examples
│   └── rust_quickstart/   # self-contained Cargo project for alkahest-cas
├── tests/                 # Python test suite (pytest + hypothesis)
├── benchmarks/            # Python benchmarks and competitor comparisons
├── fuzz/                  # AFL++ fuzz targets
├── docs/                  # mdBook and Sphinx documentation
├── website/               # landing page (alkahest-cas.github.io)
│   └── src/               # index.html + styles.css source (deployed via CI)
├── alkahest-skill/        # Skill for AI to use alkahest
├── agent-benchmark/       # benchmark for comparing AI use of alkahest vs other CAS
└── scripts/               # CI helpers (API freeze check, error codes)
```

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

Every top-level operation returns a `DerivedResult` with:

- `.value` — the result expression
- `.steps` — derivation log (list of rewrite rules applied)
- `.certificate` — Lean 4 proof term, when available

---

## Documentation and further reading

- [**Documentation site**](https://alkahest-cas.github.io/alkahest/) — full API reference and user guide
- [`ROADMAP.md`](ROADMAP.md) — planned milestones
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Rust vs Python layer guide
- [`TESTING.md`](TESTING.md) — property-based testing, fuzzing, sanitizers, CI tiers
- [`BENCHMARKS.md`](BENCHMARKS.md) — criterion and Python benchmark suites
- [`examples/`](examples/) — runnable end-to-end examples
- [`LICENSE`](LICENSE) — Apache 2.0 license

---

## Stability

Alkahest follows semantic versioning from `1.0`. The stable surface is everything re-exported from `alkahest_cas::stable` (Rust) and `alkahest.__all__` (Python). Experimental APIs live under `alkahest_cas::experimental` and `alkahest.experimental` and may change in minor releases.
