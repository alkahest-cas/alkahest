# Alkahest

A high-performance computer algebra system for Python built for both humans and agents. Symbolic operations run orders of magnitude faster than SymPy and can run on modern accelerated hardware. Every computation produces a derivation log; a meaningful subset can export Lean 4 proofs for independent verification.

**Stack:** Rust kernel → FLINT/Arb (polynomials, ball arithmetic) → egglog (e-graph simplification) → MLIR/LLVM (native and GPU codegen) → PyO3 → Python

---

## Install

```bash
pip install maturin
maturin develop --release
```

---

## Quick start

```python
import alkahest
from alkahest import ExprPool, diff, simplify, integrate, compile_expr, sin, exp

pool = ExprPool()
x = pool.symbol("x")

# Differentiation with derivation log
result = diff(sin(x ** 2), x)
print(result.value)   # 2*x*cos(x^2)
print(result.steps)   # list of rewrite steps

# Integration
r = integrate(exp(x), x)
print(r.value)        # exp(x)

# Simplification
s = simplify(x + pool.integer(0))
print(s.value)        # x

# JIT-compile to native code
f = compile_expr(x ** 2 + pool.integer(1), [x])
print(f([3.0]))       # 10.0
```

### Explicit polynomial representations

```python
from alkahest import ExprPool, UniPoly, MultiPoly, RationalFunction

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

# FLINT-backed univariate polynomial
p = UniPoly.from_symbolic(x ** 3 + pool.integer(-2) * x + pool.integer(1), x)
print(p.degree())        # 3
print(p.coefficients())  # [1, -2, 0, 1]

# GCD
a = UniPoly.from_symbolic(x ** 2 + pool.integer(-1), x)
b = UniPoly.from_symbolic(x + pool.integer(-1), x)
print(a.gcd(b))          # x - 1

# Factorization over ℤ (FLINT — Zassenhaus / van Hoeij)
fac = a.factor_z()
print(int(fac.unit), fac.factor_list())  # unit and list of (UniPoly, exponent)

# Dense univariate mod p (Berlekamp / Cantor–Zassenhaus via FLINT nmod)
fp = alkahest.factor_univariate_mod_p([1, 0, 1], 2)  # x^2+1 over GF(2)
print(fp.factor_list())

# Rational function with automatic GCD normalization
rf = RationalFunction.from_symbolic(x ** 2 + pool.integer(-1), x + pool.integer(-1), [x])
print(rf)                # x + 1

# Sparse multivariate polynomial
mp = MultiPoly.from_symbolic(x ** 2 * y + x * y ** 2, [x, y])
print(mp.total_degree()) # 3
```

### Rigorous interval arithmetic

```python
from alkahest import ExprPool, ArbBall, interval_eval, sin

pool = ExprPool()
x = pool.symbol("x")

result = interval_eval(sin(x), {x: ArbBall(1.0, 1e-10)})
print(result)  # guaranteed enclosure of sin(1 ± 1e-10)
```

### String expressions

```python
import alkahest

pool = alkahest.ExprPool()
x = pool.symbol("x")

# Parse a string into a symbolic expression
e = alkahest.parse("x^2 + 2*x + 1", pool, {"x": x})
print(e)                    # (x^2 + (x * 2)) + 1

# Round-trip: parse then pretty-print
expr = alkahest.parse("sin(x)^2 + cos(x)^2", pool, {"x": x})
print(alkahest.latex(expr))        # \sin\!\left(x\right)^2 + \cos\!\left(x\right)^2
print(alkahest.unicode_str(expr))  # sin(x)² + cos(x)²
```

### Lattice reduction and approximate integer relations

Exact LLL reduction on integer bases lives under `alkahest.lattice`; for floating constants (as `float` or decimal strings) `guess_relation` searches for small integer coefficient vectors whose dot product has tiny residual relative to the working precision:

```python
from alkahest import guess_relation
from alkahest import lattice

basis = lattice.lll_reduce_rows([[2, 15], [1, 21]])
rel = guess_relation(["1", "2", "3"], precision_bits=256)
```

The relation finder is an augmented-lattice + LLL heuristic, not Ferguson–Bailey PSLQ; treat results as exploratory unless verified independently.

### Composable transformations

```python
import alkahest

@alkahest.trace
def f(x):
    return alkahest.sin(x ** 2)

df = alkahest.grad(f)          # symbolic gradient
df_fast = alkahest.jit(df)     # compiled gradient
```

---

## Directory layout

```
alkahest/
├── alkahest-core/         # Rust kernel
│   ├── src/
│   │   ├── kernel/        # hash-consed expression DAG, ExprPool
│   │   ├── poly/          # UniPoly, MultiPoly, RationalFunction
│   │   ├── simplify/      # e-graph simplification (egglog)
│   │   ├── diff/          # symbolic differentiation
│   │   ├── integrate/     # symbolic integration
│   │   ├── jit/           # LLVM JIT and interpreter
│   │   ├── ball/          # Arb ball arithmetic
│   │   ├── ode/           # ODE analysis
│   │   ├── dae/           # DAE analysis and index reduction
│   │   ├── lean/          # Lean 4 proof certificate export
│   │   └── primitive/     # primitive registration system
│   └── benches/           # criterion benchmarks
├── alkahest-mlir/         # MLIR dialect and lowering passes
├── alkahest-py/           # PyO3 bindings (Rust side)
├── python/alkahest/       # Python package
│   ├── _transform.py      # trace, grad, jit decorators
│   ├── _pytree.py         # JAX-style pytree flattening
│   ├── _context.py        # context manager and defaults
│   └── experimental/      # unstable API surface
├── examples/              # runnable Python examples
├── tests/                 # Python test suite (pytest + hypothesis)
├── benchmarks/            # Python benchmarks and competitor comparisons
├── fuzz/                  # AFL++ fuzz targets
├── docs/                  # mdBook and Sphinx documentation
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

- [`ROADMAP.md`](ROADMAP.md) — planned milestones
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — Rust vs Python layer guide
- [`TESTING.md`](TESTING.md) — property-based testing, fuzzing, sanitizers, CI tiers
- [`BENCHMARKS.md`](BENCHMARKS.md) — criterion and Python benchmark suites
- [`docs/`](docs/) — mdBook (Rust API) and Sphinx (Python API)
- [`examples/`](examples/) — runnable end-to-end examples
- [`LICENSE`](LICENSE) - Apache 2.0 license

---

## Stability

Alkahest follows semantic versioning from `1.0`. The stable surface is everything re-exported from `alkahest_core::stable` (Rust) and `alkahest.__all__` (Python). Experimental APIs live under `alkahest_core::experimental` and `alkahest.experimental` and may change in minor releases.
