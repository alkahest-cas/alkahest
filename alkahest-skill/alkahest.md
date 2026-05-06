# Alkahest Agent Skill

Use this skill whenever you are writing Python code that uses the `alkahest` library.

## Install

```bash
pip install alkahest
```

Source build (for optional features like JIT, Gröbner, CUDA — see README):

```bash
pip install maturin && maturin develop --release --features "parallel egraph jit groebner"
```

---

## Core mental model

Every expression lives in an **`ExprPool`** (a hash-consed DAG). You must create a pool before making any symbolic expression. Integer and rational constants must be interned through the pool — raw Python ints cannot appear directly in expressions.

```python
import alkahest as ak
from alkahest import sin, cos, exp, log, sqrt, diff, integrate, simplify

pool = ak.ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

two   = pool.integer(2)      # always intern constants
half  = pool.rational(1, 2)  # p/q form
```

Arithmetic operators (`+`, `-`, `*`, `**`, `/`) are all overloaded on `Expr` — use them freely.

---

## Return type: `DerivedResult`

Every top-level operation returns a `DerivedResult`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `.value` | `Expr` | The result expression |
| `.steps` | `list[dict]` | Rewrite log; each step has `"rule"`, `"before"`, `"after"` keys |
| `.certificate` | `str \| None` | Lean 4 proof term, when available |

```python
result = diff(sin(x**2), x)
print(result.value)   # 2*x*cos(x^2)
print(result.steps)   # list of rewrite-rule dicts
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

r = simplify(x + pool.integer(0))          # → x
r = simplify_trig(sin(x)**2 + cos(x)**2)  # → 1
r = simplify_log_exp(log(exp(x)))          # → x
r = collect_like_terms(x + x + two*x + y) # → 4*x + y
```

`simplify_egraph` / `simplify_egraph_with` run egglog e-graph saturation — use when algebraic rewriting is insufficient.

```python
from alkahest import EgraphConfig, HAS_EGRAPH

if HAS_EGRAPH:
    cfg = EgraphConfig(iterations=10, node_limit=50_000)
    r = simplify_egraph_with(expr, cfg)
```

---

## Differentiation

```python
from alkahest import diff, diff_forward, symbolic_grad, jacobian

# Symbolic (reverse-mode internally)
d = diff(sin(x**2), x)          # DerivedResult; d.value = 2*x*cos(x^2)

# Forward-mode AD
d_fwd = diff_forward(x**2, x)   # same result, forward sweep

# All partial derivatives at once (reverse-mode)
grads = symbolic_grad(x**2 + y**2, [x, y])  # returns list of Expr

# Jacobian matrix of a vector-valued function
J = jacobian([x**2 + y, sin(x)*y], [x, y])
entry = J.get(row, col)  # Expr
```

---

## Integration

```python
from alkahest import integrate, IntegrationError

r = integrate(x**2, x)    # DerivedResult; r.value = x^3/3
r = integrate(sin(x), x)  # → -cos(x)
r = integrate(exp(x), x)  # → exp(x)
r = integrate(x**-1, x)   # → log(x)

try:
    r = integrate(exp(x**2), x)   # no elementary antiderivative
except IntegrationError as e:
    print(e.code, e.remediation)
```

---

## Substitution and pattern matching

```python
from alkahest import subs, match_pattern, make_rule

# Substitute: replace symbols with expressions or numerics
result = subs(expr, {x: pool.integer(2), y: cos(x)})

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

Available when built with `--features groebner`.

```python
from alkahest import solve, GroebnerBasis, GbPoly

# solve returns list[dict[Expr, float]]
solutions = solve(
    [x**2 + y**2 + pool.integer(-1), y + pool.integer(-1)*x],  # equations = 0
    [x, y],
)
for s in solutions:
    xv = s[x]  # float
    yv = s[y]
```

---

## JIT compilation and numeric evaluation

```python
from alkahest import compile_expr, eval_expr, CompiledFn

# Compile expression to LLVM JIT
f = compile_expr(x**2 + pool.integer(1), [x])   # CompiledFn
f([3.0])          # → [10.0]  (list in, list out)
f.n_inputs        # 1

# Interpreter (no JIT)
val = eval_expr(x**2 + y, {x: 3.0, y: 1.0})  # float

# Vectorised NumPy evaluation
import numpy as np
from alkahest import numpy_eval

xs = np.linspace(0, 1, 1_000_000)
ys = numpy_eval(f, xs)   # → ndarray; uses DLPack, ≥100× faster than a loop
```

---

## trace / grad / jit (JAX-style transforms)

```python
import alkahest as ak

pool = ak.ExprPool()

@ak.trace(pool)
def energy(x, y):
    return x**2 + ak.sin(y) * ak.exp(x)

# energy is a TracedFn
print(energy.expr)          # symbolic expression
print(energy(1.0, 0.0))     # numeric float
print(energy.symbols)       # [x, y]

# Gradient
grad_energy = ak.grad(energy)   # GradTracedFn
gs = grad_energy(1.0, 0.0)     # [∂/∂x, ∂/∂y] as floats

# JIT compilation
fast = ak.jit(energy)          # CompiledTracedFn
fast(1.0, 0.0)                 # same result, LLVM-backed
fast(np.linspace(0,1,100), np.zeros(100))  # auto-vectorised
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

---

## Pytree utilities (JAX-style)

```python
from alkahest import flatten_exprs, unflatten_exprs, map_exprs, TreeDef

leaves, treedef = flatten_exprs({"x": x_expr, "y": [y1, y2]})
reconstructed = unflatten_exprs(treedef, leaves)
mapped = map_exprs(lambda e: diff(e, x).value, {"f": f_expr})
```

---

## Primitive registry

```python
from alkahest import PrimitiveRegistry

reg = PrimitiveRegistry()
# reg.register(name, diff_rule, ...) — extend the kernel with custom primitives
```

---

## Key rules for agents

1. **Always create a pool first.** `ExprPool()` before any symbol or expression.
2. **Integers must be interned.** Use `pool.integer(n)` and `pool.rational(p, q)`, never raw Python ints in expressions.
3. **Read `.value` for the expression.** Top-level operations return `DerivedResult`, not `Expr`.
4. **Use specific simplifiers.** Prefer `simplify_trig`, `simplify_log_exp`, `collect_like_terms` over the catch-all `simplify` when the structure is known — it is faster.
5. **Polynomial conversions raise.** `UniPoly.from_symbolic` and `poly_normal` raise `ConversionError` for non-polynomial input — catch it.
6. **`solve` requires the groebner feature.** Import-guard it or catch `ImportError`.
7. **`trace` requires a pool argument.** `@alkahest.trace(pool)` — the pool is mandatory.
8. **`grad` and `jit` require a `TracedFn`.** They raise `TypeError` on plain callables.
9. **`numpy_eval` expects a `CompiledFn`** (from `compile_expr`), not a `TracedFn`.
10. **Symbols from different pools are incompatible.** Keep one pool per computation graph.
