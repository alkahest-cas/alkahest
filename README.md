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

### Symbolic summation (V2-10 — Gosper / recurrences)

Indefinite and definite sums for terms whose shift ratio `F(k+1)/F(k)` is rational in `k`—typically polynomials multiplied by `gamma` of a **linear** expression in `k`. General multivariate Zeilberger automation is partial; use `verify_wz_pair(F, G, n, k)` to check a discrete telescoping certificate after simplification.

```python
import alkahest

pool = alkahest.ExprPool()
k = pool.symbol("k")
n = pool.symbol("n")
term = alkahest.simplify(k * alkahest.gamma(k + pool.integer(1))).value
print(alkahest.sum_indefinite(term, k).value)
print(alkahest.sum_definite(term, k, pool.integer(0), n).value)

fib = alkahest.solve_linear_recurrence_homogeneous(
    n, [(-1, 1), (-1, 1), (1, 1)], [pool.integer(0), pool.integer(1)]
)
```

### Difference equations / `rsolve` (V2-18)

Linear recurrences in one sequence with **constant coefficients** and a **polynomial** right-hand side (in the recurrence index `n`). Write shifts as `pool.func("f", [n + integer])`, pass the equation as a single expression that simplifies to zero, and optional `initials` as `{n: value}` to fix the `C0`, `C1`, … symbols.

```python
import alkahest

pool = alkahest.ExprPool()
n = pool.symbol("n")
f = lambda *a: pool.func("f", list(a))
# f(n) - f(n-1) - 1 == 0  →  general solution n + C0
eq = alkahest.simplify(f(n) - f(n + pool.integer(-1)) - pool.integer(1)).value
print(alkahest.rsolve(eq, n, "f", None))
# Fibonacci with f(0)=0, f(1)=1
fib_eq = alkahest.simplify(
    f(n) - f(n + pool.integer(-1)) - f(n + pool.integer(-2))
).value
print(alkahest.rsolve(fib_eq, n, "f", {0: pool.integer(0), 1: pool.integer(1)}))
```

Non-homogeneous **order > 2** and sequences with **polynomial coefficients** in `n` are not implemented yet (see `RsolveError` / `E-RSOLVE-*`).

### Symbolic products (`∏`, V2-22)

`product_definite(term, k, lo, hi)` closes \(\prod_{i=\texttt{lo}}^{\texttt{hi}} \texttt{term}(i)\) (inclusive) when `term` simplifies to **ℚ(`k`)** whose numerator/denominator **factor into ℤ-linear** polynomials — the implementation expands each linear factor \(\alpha k+\beta\) with \(\Gamma\) shifts \(\Gamma(\texttt{hi}+\beta/\alpha+1)/\Gamma(\texttt{lo}+\beta/\alpha)\) and collects \(\alpha^{(\texttt{hi}-\texttt{lo}+1)\cdot e}\). `product_indefinite` returns a `Γ`/power witness `Z(k)` with `simplify`-stable ratio `Z(k+1)/Z(k)=term`. `Product(term, (k, lo, hi)).doit()` matches SymPy ergonomics (`DerivedResult`; use `.value`). Irreducible quadratics in `k`, extra symbols besides `k`, and non-integer powers are rejected (`ProductError` / `E-PROD-*`).

```python
import alkahest

pool = alkahest.ExprPool()
k, n = pool.symbol("k"), pool.symbol("n")
P = alkahest.Product(k, (k, pool.integer(1), n))
print(alkahest.simplify(P.doit().value).value)

kp2 = k ** 2
term = alkahest.simplify(
    ((k + pool.integer(-1)) * (k + pool.integer(1))) / kp2
).value  # (k²-1)/k²

print(alkahest.simplify(
    alkahest.product_definite(term, k, pool.integer(2), n).value
).value)
```

### Diophantine equations (V2-19)

Two integer unknowns, equation as a single polynomial `= 0`: **linear** families `a·x + b·y + c = 0`, **sum of two squares** `x² + y² = n` (finitely many tuples), and **unit Pell** `x² - D·y² = 1` (fundamental solution `(x₀, y₀)` via the continued-fraction period of `√D`). Requires the `groebner` feature in the native build. API: `diophantine(equation, [x, y])` → `DiophantineSolution` with `.kind` (`parametric_linear`, `finite`, `pell_fundamental`, `no_solution`) and typed fields.

```python
import alkahest

pool = alkahest.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")
sol = alkahest.diophantine(pool.integer(3) * x + pool.integer(5) * y - pool.integer(1), [x, y])
assert sol.kind == "parametric_linear"
pell = alkahest.diophantine(x**2 - pool.integer(2) * y**2 - pool.integer(1), [x, y])
assert pell.kind == "pell_fundamental" and int(str(pell.fundamental[0])) == 3
```

Quadratics with an **`x·y` cross-term**, unequal ellipse coefficients, or **generalized Pell** right-hand sides `≠ 1` are not implemented yet (`DiophantineError` / `E-DIOPH-*`).

### Integer number theory (V3-1)

Submodule `alkahest.number_theory`: `isprime`, `factorint`, `nextprime`, `totient`, `jacobi_symbol`, `nthroot_mod` (prime modulus; `k=2` or `\gcd(k,p−1)=1`), `discrete_log` (linear scan for moderate primes), plus quadratic `DirichletChi` on odd square-free conductors. Implemented via FLINT `fmpz` in the Rust kernel; raises `NumberTheoryError` (`E-NT-*`) on invalid input.

```python
from alkahest import NumberTheoryError
from alkahest.number_theory import discrete_log, factorint, isprime, nthroot_mod

assert isprime(2**127 - 1)
assert factorint(2**32 - 1)[65537] == 1
assert discrete_log(13, 3, 17) == 4
assert pow(nthroot_mod(144, 2, 401), 2, 401) == 144 % 401
```

### Noncommutative algebra (V3-2)

Symbols can opt out of multiplicative commutativity: ``pool.symbol("A", "real", commutative=False)``. Then ``A * B`` and ``B * A`` are distinct expressions, and sorting of ``Mul`` factors is disabled. The egglog backend automatically falls back to the rule-based simplifier when such symbols appear.

Pauli matrices (names ``sx``, ``sy``, ``sz``) and a minimal orthogonal Clifford pair (``cliff_e1``, ``cliff_e2``) have built-in rewrite tables; combine default rules with ``alkahest.simplify_pauli`` or ``alkahest.simplify_clifford_orthogonal``. See ``examples/noncommutative.py``.

### Truncated series / Laurent tail (V2-15)

`series(expr, var, point, order)` builds a symbolic truncation about `(var − point)` and appends a `BigO(⋯)` remainder. Smooth functions use repeated differentiation; simple poles such as `1/x` at zero take the rational Laurent path. `Series.expr` is the pooled sum-plus-order expression; `ExprPool.big_o(inner)` constructs standalone order bounds.

```python
import alkahest

pool = alkahest.ExprPool()
x = pool.symbol("x")
s_cos = alkahest.series(alkahest.cos(x), x, pool.integer(0), 6)
s_inv = alkahest.series(x ** (-1), x, pool.integer(0), 4)
print(s_cos.expr)
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

### Regular chains / triangular decomposition (V2-11)

Lex-order Gröbner bases yield triangular sets used by the polynomial solver. The `triangularize(equations, vars)` API returns one or more `RegularChain` objects (polynomials as `GbPoly` tiles), splitting along factored bottom univariates when applicable. The built-in `solve()` routine retries backsolving from an extracted chain when the full basis is not directly triangular enough.

```python
import alkahest

pool = alkahest.ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")
eq1 = x**2 + y**2 - pool.integer(1)
eq2 = y - x
chains = alkahest.triangularize([eq1, eq2], [x, y])
assert len(chains) >= 1
```

### Primary decomposition (V2-12)

Lex-order Gröbner data is used to split ideals via saturations (`I : x_i^∞` with `(I + (x_i))`) and, in the zero-dimensional case, factoring a univariate polynomial in the first Lex variable. `primary_decomposition(polys, vars)` returns `PrimaryComponent` objects with `.primary()` and `.associated_prime()` Gröbner bases; `radical(polys, vars)` returns a basis for √I.

```python
import alkahest

pool = alkahest.ExprPool()
x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
comps = alkahest.primary_decomposition([x * y, x * z], [x, y, z])
assert len(comps) == 2
r = alkahest.radical([x**2, x * y], [x, y])
assert r.contains(x)
```

### Differential algebra / Rosenfeld–Gröbner (V2-13)

Polynomial DAEs in implicit form `g_i(t, y, y') = 0` can be analysed by **prolongation** (formal time derivatives of each equation, with the same derivative-state extension rule as Pantelides) and **ordinary Gröbner bases** over the jet variables. Inconsistent systems yield the unit ideal. Use `rosenfeld_groebner(dae, order=..., max_prolong_rounds=...)`; when Pantelides exhausts its index cap, `dae_index_reduce(dae)` falls back to this pass.

```python
import alkahest

pool = alkahest.ExprPool()
t = pool.symbol("t")
y = pool.symbol("y")
dy = pool.symbol("dy/dt")
dae = alkahest.DAE.new([dy - y, dy - y - pool.integer(1)], [y], [dy], t)
r = alkahest.rosenfeld_groebner(dae, max_prolong_rounds=2)
assert r.consistent is False
```

### Numerical algebraic geometry / homotopy continuation (V2-14)

Square polynomial systems can be solved numerically with a **total-degree** homotopy in `ℂⁿ` (`(1-t)·γ·G + t·F`), Newton polish on real projections, a conservative **Smale-style** `α` heuristic, and **`ArbBall` enclosures** attached to each coordinate. Use `solve(eqs, vars, method="homotopy")` for a list of `dict` solutions (`Expr` keys → `float`). For residuals, certification flags, and enclosures, call `solve_numerical(...)`, which returns `CertifiedSolution` objects (`.coordinates`, `.smale_certified`, `.to_dict()`, `.enclosures()`, …).

Ideals whose finite root count in `ℂⁿ` is **strictly below** the Bézout bound (often called *deficient* — e.g. the Katsura family) typically need a **polyhedral / mixed-volume** start system; only the Bézout start (`∏ deg F_i` paths) is implemented here.

```python
import alkahest

p = alkahest.ExprPool()
x, y = p.symbol("x"), p.symbol("y")
neg1 = p.integer(-1)
sols = alkahest.solve([x**2 + neg1, y**2 + neg1], [x, y], method="homotopy")
cs = alkahest.solve_numerical([x**2 + neg1], [x])[0]
print(cs.coordinates, cs.smale_certified, cs.enclosures())
```

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
│   │   ├── algebra/       # noncommutative Pauli / Clifford rules (V3-2)
│   │   ├── poly/          # UniPoly, MultiPoly, RationalFunction
│   │   ├── simplify/      # e-graph simplification (egglog)
│   │   ├── diff/          # symbolic differentiation
│   │   ├── integrate/     # symbolic integration
│   │   ├── calculus/      # series / limits (V2-15+)
│   │   ├── jit/           # LLVM JIT and interpreter
│   │   ├── ball/          # Arb ball arithmetic
│   │   ├── ode/           # ODE analysis
│   │   ├── dae/           # DAE analysis and index reduction
│   │   ├── diffalg/       # Rosenfeld–Gröbner / differential elimination (groebner)
│   │   ├── solver/        # polynomial solving: Gröbner triangular, regular chains, homotopy
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
