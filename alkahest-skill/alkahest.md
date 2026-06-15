# Alkahest Agent Skill

Use this skill whenever you are writing Python code that uses the `alkahest` library, or Rust code using the `alkahest-cas` crate.

## Official links

- **Repository:** [github.com/alkahest-cas/alkahest](https://github.com/alkahest-cas/alkahest)
- **Website:** [alkahest-cas.github.io/](https://alkahest-cas.github.io/)
- **Documentation:** [alkahest-cas.github.io/alkahest/](https://alkahest-cas.github.io/alkahest/)
- **API reference:** [alkahest-cas.github.io/alkahest/api/](https://alkahest-cas.github.io/alkahest/api/)
- **Playground:** [alkahest-cas.github.io/playground/](https://alkahest-cas.github.io/playground/)
- **RL environment:** [Prime Intellect Environments Hub](https://app.primeintellect.ai/dashboard/environments/alkahest/alkahest-symbolic-integration) (`alkahest/alkahest-symbolic-integration`)
- **Further reading:** [`ARCHITECTURE.md`](https://github.com/alkahest-cas/alkahest/blob/main/ARCHITECTURE.md), [`CONTRIBUTING.md`](https://github.com/alkahest-cas/alkahest/blob/main/CONTRIBUTING.md), [`TESTING.md`](https://github.com/alkahest-cas/alkahest/blob/main/TESTING.md), [`examples/`](https://github.com/alkahest-cas/alkahest/tree/main/examples/)

## Install

**Requirements:** Python **3.9–3.13** ([PyPI](https://pypi.org/project/alkahest/) `requires-python`).

```bash
pip install alkahest
```

**RL environments** (symbolic integration tasks for Prime Intellect / veRL): Python **≥ 3.10** required.

```bash
pip install "alkahest[rl]"
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

These wheels vendor LLVM (for JIT) and related `.so` files under `site-packages/alkahest.libs/`. If `import alkahest` fails with a missing `libffi-*.so` or `libLLVM-*.so`, prepend that directory to `LD_LIBRARY_PATH` (or install matching system packages).

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

The `jit` feature additionally requires LLVM 15 dev headers (`apt install llvm-15-dev` / `brew install llvm@15`). A self-contained runnable example is in [`examples/rust_quickstart/`](https://github.com/alkahest-cas/alkahest/tree/main/examples/rust_quickstart/).

---

## Core mental model

Every expression lives in an **`ExprPool`** (a hash-consed DAG). You must create a pool before making any symbolic expression. **Python `int` and `float` literals work in arithmetic** (`x + 1`, `x * 2.5`, `x**2`); use `pool.rational(p, q)` for exact rationals and `pool.integer(n)` when you need an explicit `Expr` constant (e.g. for APIs that only accept `Expr`).

```python
import alkahest as ak
from alkahest import sin, cos, exp, log, sqrt, diff, integrate, simplify, simplify_trig

caps = ak.capabilities()  # groebner, jit, egraph, parallel — probe once per session

pool = ak.ExprPool()
x = pool.symbol("x", ak.Domain.Real)   # or domain="real"
y = pool.symbol("y")

expr = x**2 + 1          # int literals in +, -, *, **, / are fine
half = pool.rational(1, 2)  # exact rationals need pool.rational
```

Arithmetic operators (`+`, `-`, `*`, `**`, `/`) are all overloaded on `Expr` — use them freely.

### Expression representations

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

## Return type: `DerivedResult`

Every top-level operation returns a `DerivedResult`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.value` | `Expr` | The result expression |
| `.steps` | `list[dict]` | Rewrite log; each step has `"rule"`, `"before"`, `"after"` keys |
| `.certificate` | `str \| None` | Lean 4 `.lean` source (Mathlib proof file), when steps are certifiable |
| `to_lean(result)` | `str` | Same as `.certificate`; also accepts `Expr` (runs `simplify` first) |

```python
result = diff(sin(x**2), x)
print(result.value)   # 2*x*cos(x^2)
print(result.steps)   # list of rewrite-rule dicts
print(result.certificate)  # or: to_lean(result)
```

---

## Simplification

```python
from alkahest import (
    simplify,            # general algebraic simplification
    simplify_trig,       # sin²+cos²=1, sin(-x)=-sin(x), …
    simplify_log_exp,    # log(exp(x))=x, exp(log(x))=x, …
    simplify_expanded,   # expand and collect
    simplify_with,       # simplify with a custom RewriteRule list
    simplify_par,        # parallel simplification (thread-pool)
    simplify_egraph,     # e-graph (egglog) simplification
    simplify_egraph_with,# e-graph with custom EgraphConfig
    collect_like_terms,  # x+x+2x → 4x
    poly_normal,         # normalize to polynomial form (raises ConversionError if not poly)
)

r = simplify(x + 0)                        # → x (algebraic rules)
r = simplify_trig(sin(x)**2 + cos(x)**2)  # → 1  (trig identities — use this, not simplify)
r = simplify_log_exp(log(exp(x)))          # → x
r = collect_like_terms(x + x + 2*x + y)   # → 4*x + y
```

**Simplifier choice:** `simplify` is a general algebraic rewriter; it does **not** apply trig identities. For `sin²+cos²` and similar, use `simplify_trig` (or `simplify_egraph`). For `log`/`exp` laws, use `simplify_log_exp`.

`simplify_egraph` / `simplify_egraph_with` run egglog e-graph saturation — use when algebraic rewriting is insufficient.

```python
from alkahest import EgraphConfig, HAS_EGRAPH

if HAS_EGRAPH:
    cfg = EgraphConfig(node_limit=50_000, iter_limit=20)
    r = simplify_egraph_with(expr, cfg)
```

---

## Differentiation

### `diff` vs `symbolic_grad` vs `grad` (keep these names)

| API | Input | Output | When to use |
|-----|--------|--------|-------------|
| **`diff`** | one `Expr`, one variable | `DerivedResult` (`.value`, `.steps`) | Single derivative; need derivation log |
| **`symbolic_grad`** | one `Expr`, `list[Expr]` vars | `list[Expr]` | All partials of one expression at once |
| **`grad`** | `TracedFn` from `@trace` | `GradTracedFn` → floats at a point | JAX-style pipeline; compose with `jit` |

Do **not** pass an `Expr` to `grad` — use `symbolic_grad` or `diff`. Do **not** pass a `TracedFn` to `symbolic_grad`.

```python
from alkahest import diff, diff_forward, symbolic_grad, jacobian

# Single variable + step log
d = diff(sin(x**2), x)          # DerivedResult; d.value = 2*x*cos(x^2)

# Forward-mode AD (cross-check)
d_fwd = diff_forward(x**2, x)

# All partials of one Expr (no TracedFn)
grads = symbolic_grad(x**2 + y**2, [x, y])  # list[Expr]: [2*x, 2*y]

# Vector-valued → matrix
J = jacobian([x**2 + y, sin(x)*y], [x, y])
entry = J.get(row, col)  # Expr
```

---

## Integration

```python
from alkahest import integrate, IntegrationError

# Basic rules
r = integrate(x**2, x)    # DerivedResult; r.value = x^3/3
r = integrate(sin(x), x)  # → -cos(x)
r = integrate(exp(x), x)  # → exp(x)
r = integrate(x**-1, x)   # → log(x)

# Rational functions — full Risch: Hermite + Rothstein–Trager + arctan + RootSum
r = integrate(pool.integer(1) / (x**2 - pool.integer(1)), x)
# → ½·(log(x−1) − log(x+1))

r = integrate(pool.integer(1) / (x**2 + pool.integer(1)), x)
# → arctan(x)

r = integrate(pool.integer(1) / (x + pool.integer(1))**2, x)
# → −1/(x+1)   (Hermite reduction for repeated factor)

# Degree-≥3 denominator → RootSum (Lazard–Rioboo–Trager)
r = integrate(pool.integer(1) / (x**3 - pool.integer(3)*x + pool.integer(1)), x)
# r.value is a RootSum node: Σ_{P(c)=0} c·log(gcd_x(numer − c·denom', denom))

# Rational coefficient × exp (rational Risch DE)
r = integrate((x - pool.integer(1)) / x**2 * exp(x), x)
# → exp(x)/x

# Non-elementary integrals raise IntegrationError with code E-INT-004
try:
    integrate(exp(x) / x, x)          # Ei function — non-elementary
except IntegrationError as e:
    print(e.code)         # E-INT-004
    print(e.remediation)  # "no elementary antiderivative (NonElementary)"

try:
    integrate(exp(x**2), x)            # Gaussian — non-elementary
except IntegrationError as e:
    print(e.code)         # E-INT-004
```

---

## Substitution and pattern matching

```python
from alkahest import subs, match_pattern, make_rule

# Substitute: values may be Expr, DerivedResult, or Python int/float (coerced to Expr)
result = subs(expr, {x: 2, y: cos(x)})

# Pattern matching
rule = make_rule("sin(?a)**2 + cos(?a)**2", pool.integer(1))
simplified = simplify_with(expr, [rule])
```

---

## Polynomial types (FLINT-backed)

All polynomial types are explicit opt-in — no silent performance cliffs.

```python
from alkahest import UniPoly, MultiPoly, RationalFunction

# Dense univariate polynomial
p = UniPoly.from_symbolic(x**3 + pool.integer(-2)*x + pool.integer(1), x)
p.degree()         # 3
p.coefficients()   # [1, -2, 0, 1]  (constant first)
p.evaluate(2.0)    # numeric eval

# GCD
a = UniPoly.from_symbolic(x**2 + pool.integer(-1), x)
b = UniPoly.from_symbolic(x + pool.integer(-1), x)
a.gcd(b)           # x - 1

# Sparse multivariate polynomial (over ℤ)
mp = MultiPoly.from_symbolic(x**2 * y + x * y**2, [x, y])
mp.total_degree()  # 3

# Rational function (GCD-normalized automatically)
rf = RationalFunction.from_symbolic(x**2 + pool.integer(-1), x + pool.integer(-1), [x])
# rf displays as x + 1
```

---

## Polynomial system solver / Gröbner basis

The `groebner` Cargo feature is **included in all PyPI wheels** since 2.3.1 (it is a default feature). No special build flag or `ImportError` guard needed — `solve`, `GroebnerBasis`, and related APIs are available after a plain `pip install alkahest`.

```python
from alkahest import solve, solve_numerical, GroebnerBasis, GbPoly

# solve(equations, vars, *, numeric=False, method="groebner")
# - method="groebner" (default): Lex/triangular path. Each finite solution is a dict
#   mapping variable Expr → Expr (symbolic) or float if numeric=True.
# - method="homotopy": numerical continuation in ℂⁿ; dict values are float.
# - If the ideal is parametric / not zero-dimensional finite, Groebner mode may
#   return a GroebnerBasis instead of a list of dicts.
solutions = solve(
    [x**2 + y**2 + pool.integer(-1), y + pool.integer(-1)*x],
    [x, y],
)
for s in solutions:
    xv, yv = s[x], s[y]

# Certified enclosures / residuals: solve_numerical(eqs, vars, ...)
```

---

## JIT compilation and numeric evaluation

```python
from alkahest import compile_expr, eval_expr, CompiledFn, CompileCache, jit_is_available

jit_is_available()   # False on default PyPI wheel; True with JIT-enabled build

# Compile (interpreter on default wheel; Cranelift/LLVM when built with those features)
f = compile_expr(x**2 + pool.integer(1), [x])   # CompiledFn
f([3.0])          # → [10.0]  (list in, list out)
f.n_inputs        # 1

# Memoize repeated compilations within a session
cache = CompileCache()
f = cache.compile(x**2, [x], pool)
print(cache.stats())   # hits, compiles, hit_rate

# Interpreter (no JIT)
val = eval_expr(x**2 + y, {x: 3.0, y: 1.0})  # float

# Vectorised evaluation (DLPack): NumPy, JAX, PyTorch CPU tensors, etc.
import numpy as np
from alkahest import numpy_eval, numpy_eval_par

xs = np.linspace(0, 1, 1_000_000)
ys = numpy_eval(f, xs)        # ndarray; much faster than a Python loop
ys = numpy_eval_par(f, xs)    # multi-core when built with --features parallel
```

---

## trace / grad / jit (JAX-style transforms)

Here **`grad`** means “gradient of a **traced** function”, not `symbolic_grad`. For partials of a bare `Expr`, use **`symbolic_grad(expr, [x, y])`** (see [Differentiation](#differentiation)).

```python
import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")

# Expr-level partials (no @trace):
partials = ak.symbolic_grad(x**2 + ak.sin(y), [x, y])  # list[Expr]

@ak.trace(pool)
def energy(x, y):
    return x**2 + ak.sin(y) * ak.exp(x)

# energy is a TracedFn
print(energy.expr)          # symbolic expression
print(energy(1.0, 0.0))     # numeric float
print(energy.symbols)       # [x, y]

# grad = gradient of TracedFn (GradTracedFn), not symbolic_grad
grad_energy = ak.grad(energy)
gs = grad_energy(1.0, 0.0)     # [∂/∂x, ∂/∂y] as floats

fast = ak.jit(energy)          # CompiledTracedFn
fast_grad = ak.jit(ak.grad(energy))  # compiled GradTracedFn
```

Non-decorator variant: `ak.trace_fn(fn, pool)`.

---

## Code emission

```python
from alkahest import horner, emit_c, to_stablehlo

poly = pool.integer(1) + pool.integer(2)*x + pool.integer(3)*x**2
print(horner(poly, x))                        # Horner-form Expr
c_code = emit_c(poly, x, "x_var", "f")       # C function string
stablehlo = to_stablehlo(sin(x)+exp(y), [x,y], fn_name="my_fn")  # StableHLO text
```

---

## Interval / ball arithmetic (Arb)

```python
from alkahest import ArbBall, interval_eval

ball = ArbBall(1.0, 1e-10)         # centre ± radius
result = interval_eval(sin(x), {x: ball})  # rigorous enclosure
```

---

## Symbolic matrices

```python
from alkahest import Matrix, jacobian

R = Matrix.from_rows([
    [cos(x), pool.integer(-1)*sin(x)],
    [sin(x), cos(x)],
])
R.rows            # 2
R.cols            # 2
R.get(0, 1)       # Expr
R.det()           # symbolic determinant
R.to_list()       # list[list[Expr]]
```

---

## ODE / DAE modeling

```python
import alkahest as ak
from alkahest import ODE, DAE, lower_to_first_order, pantelides

pool = ak.ExprPool()
t = pool.symbol("t")
y = pool.symbol("y")
k = pool.symbol("k")

ode = ODE.new([y], [pool.integer(-1)*k*y], t)
ode.order()
ode.is_autonomous()
ode.state_vars()
ode.rhs()

ode_with_ic = ode.with_ic(y, pool.integer(1))

# Second-order → first-order
ode_1st = lower_to_first_order(x, pool.integer(-1)*x, 2, t)

# DAE Pantelides index reduction
# dae = DAE.new(...)
# reduced = pantelides(dae)
```

---

## Sensitivity and adjoint systems

```python
from alkahest import sensitivity_system, adjoint_system, SensitivitySystem

ss = sensitivity_system(ode, [k])   # SensitivitySystem
ss.original_dim
ss.n_params
ss.extended_ode    # augmented ODE with sensitivity variables

adj = adjoint_system(ode, obj_grad_exprs)  # ODE run backward
```

---

## Context manager (thread-local defaults)

```python
import alkahest as ak

pool = ak.ExprPool()
with ak.context(pool=pool, domain="real", simplify=True):
    x = ak.symbol("x")   # pool and domain inferred
    y = ak.symbol("y")
    d = ak.diff(x**2 + x**2, x)  # .value is algebraically simplified

# simplify=True applies the general :func:`simplify` rewriter to results of
# diff / integrate / sum_* / product_* only — not to solve or simplify_trig.

# Inspect active context
ak.active_pool()
ak.active_domain()
ak.simplify_enabled()
ak.get_context_value("any_key")
```

---

## Error handling

All errors inherit `AlkahestError` and carry `.code`, `.remediation`, `.span`.

| Exception | Code prefix | Trigger |
|-----------|-------------|---------|
| `ConversionError` | `E-POLY-*` | Expression is not polynomial |
| `DiffError` | `E-DIFF-*` | Differentiation failed |
| `IntegrationError` | `E-INT-*` | No elementary antiderivative |
| `MatrixError` | `E-MAT-*` | Dimension mismatch, singular |
| `OdeError` | `E-ODE-*` | ODE construction failed |
| `DaeError` | `E-DAE-*` | DAE index reduction failed |
| `JitError` | `E-JIT-*` | JIT compilation failed |
| `SolverError` | `E-SOLVE-*` | Polynomial solver failed |
| `IoError` | `E-IO-*` | Pool checkpoint I/O |
| `NumberTheoryError` | `E-NT-*` | Invalid input to number-theory helpers |
| `ParseError` | `E-PARSE-*` | String parse failures |
| `RsolveError` | `E-RSOLVE-*` | Recurrence / `rsolve` failures |

```python
from alkahest import ConversionError, IntegrationError

try:
    poly_normal(sin(x), [x])
except ConversionError as e:
    print(e.code)          # "E-POLY-001"
    print(e.remediation)   # human-readable fix hint
```

---

## Available math functions

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`,
`exp`, `log`, `sqrt`, `erf`, `erfc`, `gamma`,
`abs`, `sign`, `floor`, `ceil`, `round`,
`min`, `max`, `piecewise`

All return `Expr`. They shadow Python builtins inside `alkahest` — use `alkahest.abs(expr)` to avoid ambiguity.

For `piecewise`, branch conditions must be symbolic predicates from the pool (not Python `>`):

```python
cond = pool.gt(x, pool.integer(0))
pw = alkahest.piecewise([(cond, x)], pool.integer(-1) * x)
```

---

## Pytree utilities (JAX-style)

```python
from alkahest import flatten_exprs, unflatten_exprs, map_exprs, TreeDef

leaves, treedef = flatten_exprs({"x": x_expr, "y": [y1, y2]})
reconstructed = unflatten_exprs(treedef, leaves)
mapped = map_exprs(lambda e: diff(e, x).value, {"f": f_expr})
```

---

## Parsing and pretty-printing

```python
from alkahest import parse, latex, unicode_str, ParseError

e = parse("x^2 + 2*x + 1", pool, {"x": x})
latex(e)
unicode_str(e)
```

---

## Summation, products, number theory

- Discrete summation: `sum_indefinite`, `sum_definite`, `verify_wz_pair`; linear recurrences: `solve_linear_recurrence_homogeneous`, `rsolve`.
- Symbolic products: `Product`, `product_indefinite`, `product_definite`.
- Integer number theory (FLINT-backed): `alkahest.number_theory` (`isprime`, `factorint`, `discrete_log`, …).

---

## Plotting

Alkahest never bundles a plotting library. All plot functions detect what is installed and call into it. The default backend is **Matplotlib**; **Plotly** is the interactive alternative (`backend="plotly"`).

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

# 1-D curve — uses matplotlib by default, or plotly if specified
ax  = ak.plot(ak.sin(x), x, (-6.28, 6.28))
fig = ak.plot(ak.sin(x), x, (-6.28, 6.28), backend="plotly")

# 2-D surface
ak.plot3d(ak.sin(x) * ak.cos(y), x, y, (-5, 5), (-5, 5))

# Parametric curve
t = pool.symbol("t")
ak.plot_parametric(ak.cos(t), ak.sin(t), t, (0, 6.28318))

# Implicit curve f(x, y) = 0
ak.plot_implicit(x**2 + y**2 - pool.integer(1), x, y, (-2, 2), (-2, 2))

# Root markers on x-axis
p = ak.UniPoly.from_symbolic(x**3 - x, x)
ak.plot_roots(p, x)

# Series truncation vs exact
s = ak.series(ak.cos(x), x, pool.integer(0), 6)
ak.plot_series(s, ak.cos(x), x, (-4, 4))

# Expression DAG (graphviz package → rendered figure; else DOT string)
dot_or_src = ak.plot_dag(ak.sin(x**2))

# Dependency-free SVG — no matplotlib/plotly needed
svg_str = ak.plot_svg(ak.sin(x), x, (-6, 6))
# Use in Jupyter: from IPython.display import SVG; SVG(svg_str)
```

All `plot*` functions accept `**kw` forwarded to the backend. `plot` / `plot_parametric` also accept `ax=` (matplotlib) or `fig=` (plotly) to draw onto an existing figure.

GPU-accelerated plotting for dense grids (experimental; requires `pip install fastplotlib`):

```python
from alkahest.experimental._fastplotlib import fplot, fplot3d
fplot(ak.sin(x), x, (-10, 10), n=100_000)   # 100k-point GPU line
```

---

## Reinforcement learning

`alkahest.rl` exposes **verifiable RL environments** backed by the CAS. The core layer (`alkahest.rl.core`) is trainer-agnostic; domain environments live under `alkahest.rl.envs.*` and optionally integrate with [Prime Intellect Verifiers](https://github.com/PrimeIntellect-ai/verifiers).

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

**Environments Hub:** [`alkahest/alkahest-symbolic-integration`](https://app.primeintellect.ai/dashboard/environments/alkahest/alkahest-symbolic-integration) — install with `prime env install alkahest/alkahest-symbolic-integration`. Full checklist in the [RL guide](https://alkahest-cas.github.io/alkahest/rl.html).

---

## Stable vs experimental API

Alkahest follows semantic versioning from `1.0`. The stable surface is everything re-exported from `alkahest_cas::stable` (Rust) and `alkahest.__all__` (Python). Experimental APIs live under `alkahest_cas::experimental` and `alkahest.experimental` and may change in minor releases—prefer top-level exports for agent-written code unless the user asks for experimental features.

---

## Primitive registry

```python
from alkahest import PrimitiveRegistry

reg = PrimitiveRegistry()
# reg.register(name, diff_rule, ...) — extend the kernel with custom primitives
```

---

## Key rules for agents

1. **Always create a pool first.** `ExprPool()` before any symbol or expression. Optional: `with ak.context(pool=pool): x = ak.symbol("x")` to avoid repeating `pool=`.
2. **Pool first; literals in arithmetic are OK.** Use `x + 1`, not only `x + pool.integer(1)`. Use `pool.rational(p, q)` for exact rationals; `subs` accepts Python `int`/`float` in the mapping.
3. **Read `.value` for the expression.** Top-level operations return `DerivedResult`, not `Expr`.
4. **Use specific simplifiers.** Prefer `simplify_trig`, `simplify_log_exp`, `collect_like_terms` over the catch-all `simplify` when the structure is known — it is faster.
5. **Polynomial conversions raise.** `UniPoly.from_symbolic` and `poly_normal` raise `ConversionError` for non-polynomial input — catch it.
6. **`solve` / Gröbner-side APIs are available in all PyPI wheels.** The `groebner` Cargo feature is a default since 2.3.1 — no feature flag or `ImportError` guard needed. Default PyPI wheels also include egglog (`HAS_EGRAPH` is typically `True`); use `simplify_egraph` when rule-based simplification is insufficient.
7. **`trace` requires a pool argument.** Use `@alkahest.trace(pool)` (or `trace_fn(fn, pool)`). `@alkahest.trace` alone is invalid.
8. **`grad` ≠ `symbolic_grad`.** `symbolic_grad(expr, [x, y])` → `list[Expr]`. `grad(traced_fn)` → `GradTracedFn` (needs `@trace(pool)` first). `jit` accepts `TracedFn` or `GradTracedFn`.
9. **`numpy_eval` expects a `CompiledFn`** (from `compile_expr`), not a `TracedFn`.
10. **Symbols from different pools are incompatible.** Keep one pool per computation graph.
11. **`plot*` functions detect the backend automatically.** Never import matplotlib/plotly in user code just to call `ak.plot` — let alkahest dispatch. Use `backend="plotly"` or `backend="matplotlib"` to force one. Use `plot_svg` when no plotting library is available.
12. **`plot_dag` returns a `graphviz.Source` if the `graphviz` package is installed, otherwise a raw DOT string.** Call `.render()` or `.view()` on the returned object, or pipe the string to `dot -Tpng`.
