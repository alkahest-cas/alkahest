# Calculus

Alkahest supports symbolic differentiation and integration with full derivation logging.

## Differentiation

`diff(expr, var)` computes the symbolic derivative of `expr` with respect to `var`.

```python
from alkahest import diff, sin, cos, exp, log

pool = ExprPool()
x = pool.symbol("x")

# Polynomial
dr = diff(x**3 + pool.integer(2) * x, x)
print(dr.value)   # 3*x^2 + 2

# Chain rule
dr = diff(sin(x**2), x)
print(dr.value)   # 2*x*cos(x^2)

# Product rule
dr = diff(x * exp(x), x)
print(dr.value)   # exp(x) + x*exp(x)

# Logarithm
dr = diff(log(x**2 + pool.integer(1)), x)
print(dr.value)   # 2*x / (x^2 + 1)
```

### Registered primitives

Every primitive in the registry has a differentiation rule. The 23 currently registered primitives include:

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`, `sign`, `erf`, `erfc`, `gamma`, `floor`, `ceil`, `round`, `min`, `max`

### Derivation log

The `DerivedResult` returned by `diff` records every rule application:

```python
dr = diff(sin(x**2), x)
for step in dr.steps:
    print(f"  {step['rule']:25s}  {step['before']}  →  {step['after']}")
```

## Forward-mode automatic differentiation

`diff_forward` computes the derivative using forward-mode AD (dual numbers). It produces the same result as `diff` but through a different computational path:

```python
from alkahest import diff, diff_forward

sym = diff(x**3, x)
fwd = diff_forward(x**3, x)
# fwd.value == sym.value
```

Forward mode is useful for checking that the symbolic rules agree with dual-number evaluation.

## Symbolic gradient (`symbolic_grad`)

`symbolic_grad(expr, vars)` returns a **list of `Expr`** — one partial derivative per
variable. It does not use `@trace` and is not composable with `jit` directly.

| API | Input | Output |
|-----|--------|--------|
| `diff(expr, var)` | one variable | `DerivedResult` with `.steps` |
| `symbolic_grad(expr, vars)` | many variables | `list[Expr]` |
| `grad(traced_fn)` | `TracedFn` from `@trace` | `GradTracedFn` (numeric; see [Transformations](./transformations.md)) |

```python
from alkahest import symbolic_grad

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

expr = x**2 * y + sin(x * y)
grads = symbolic_grad(expr, [x, y])
# grads[0] = ∂/∂x = 2*x*y + y*cos(x*y)
# grads[1] = ∂/∂y = x^2 + x*cos(x*y)
```

For the JAX-style gradient of a traced Python function (compose with `jit`), use
`alkahest.grad` — **not** `symbolic_grad`. See [Transformations](./transformations.md).

## Integration

`integrate(expr, var)` computes the symbolic antiderivative of `expr` with respect to `var`.

```python
from alkahest import integrate, sin, cos, exp

# Polynomials
r = integrate(x**3, x)
print(r.value)    # x^4/4

# Known functions
r = integrate(sin(x), x)
print(r.value)    # -cos(x)

r = integrate(exp(x), x)
print(r.value)    # exp(x)

r = integrate(x**pool.integer(-1), x)
print(r.value)    # log(x)
```

### Integration rules

The integration engine applies a rule table for common forms, then escalates to the Risch decision procedure for harder cases:

**Rule table (fast path)**
- Power rule: `∫ xⁿ dx = xⁿ⁺¹/(n+1)` for integer `n ≠ -1`
- Logarithm: `∫ 1/x dx = log(x)`
- Exponential tower: `∫ exp(a*x + b) dx`, `∫ xⁿ·exp(x) dx` (poly × exp)
- Linear substitution: `∫ f(a*x + b) dx`
- Trigonometric: `∫ sin(x) dx`, `∫ cos(x) dx`, etc.
- Standard table entries for `erf`, inverse trig, etc.

**Risch algorithm (escalation)**
- **Rational functions** `A(x)/D(x)`: Hermite reduction (repeated factors → rational part), then Rothstein–Trager (rational residues → `log`), irreducible quadratics (negative discriminant → `arctan`; positive discriminant → `log` with `√Δ` coefficients), and irreducible factors of degree ≥ 3 via a `RootSum` node (Lazard–Rioboo–Trager over the number field `ℚ[t]/Q(t)`).
- **Exp tower with rational coefficient**: `∫ f(x)·exp(η) dx` where `f ∈ ℚ(x)` — solved via the rational Risch DE (Bronstein §6.1).
- **Polynomial × exp / log towers**: poly-RDE and known-table rules.

```python
# Rational functions
r = integrate(pool.integer(1) / (x**2 - pool.integer(1)), x)  # → ½·log((x-1)/(x+1))

r = integrate(pool.integer(1) / (x**2 + pool.integer(1)), x)  # → arctan(x)

# Rational coefficient × exp
r = integrate((x - pool.integer(1)) / x**2 * exp(x), x)       # → exp(x)/x

# Degree-≥3 denominator → RootSum
r = integrate(pool.integer(1) / (x**3 - pool.integer(3)*x + pool.integer(1)), x)
# r.value contains a RootSum node (sum over algebraic residues)
```

A `RootSum` answer is **verified before it is returned**, like every other
answer `integrate` gives: it is differentiated and compared numerically to the
integrand, with the `RootSum` evaluated by finding its minimal polynomial's
roots and summing the body over them in complex arithmetic. That check is an
`f64` one and the sum is ill-conditioned in the roots, so for a denominator of
degree above roughly 14 — where a *perfect* double-precision root set would
still miss the comparison tolerance — `integrate` declines with `E-INT-001`
rather than return an answer it cannot check. Such an answer is not claimed to
be non-elementary: every rational function is elementary, and `E-INT-004` is
never raised here.

**Non-elementary certification**: when the integrand is provably non-elementary (Liouville's theorem — e.g. `sin(x)/x`, `exp(x)/x`, `exp(x²)`), `integrate` raises `IntegrationError` with code `E-INT-004` (NonElementary) rather than a generic "not implemented":

```python
from alkahest import IntegrationError

try:
    integrate(exp(x) / x, x)
except IntegrationError as e:
    print(e.code)         # E-INT-004
    print(e.remediation)  # "no elementary antiderivative (NonElementary)"
```

For integrands outside the supported classes (e.g. `sqrt(P(x))`, mixed algebraic+transcendental), `integrate` raises `IntegrationError` with code `E-INT-001` (NotImplemented).

### Improper integrals over the whole real line

`integrate(f, x, -oo, oo)` for a **rational** `f` does not go through the
fundamental theorem — the antiderivative is a `RootSum` or a sum of logs and
arctangents whose limits at `±∞` the limit engine cannot establish. It takes
the residue theorem instead:

```python
neg_inf = pool.integer(-1) * pool.pos_infinity()
integrate(1 / (x**4 + pool.integer(1)), x, neg_inf, pool.pos_infinity()).value
# → π·2^(-1/2)
integrate(1 / (x**6 + pool.integer(1)), x, neg_inf, pool.pos_infinity()).value
# → 2π/3
```

Both convergence conditions are checked exactly — `deg Q ≥ deg P + 2` on the
reduced fraction, and `Q` with no real root — and a failure of either is
reported as **divergent** (`E-INT-001`), never given a finite value. That
includes cases with a Cauchy principal value: `∫ x dx/(x²+1)` has PV `0`, and
returning `0` would be wrong.

Every value is cross-checked against a rigorous enclosure of the same integral
(`validated::bounds::verified_integral` on `[-1, 1]` plus each tail mapped to
`[0, 1]` by `x = ±1/t`) before it is returned; a disagreement is treated as a
bug and declined. The route covers denominators whose Hurwitz spectral factor
is rational and, in radicals, every denominator of degree ≤ 4 after
even-normalisation. Outside that it declines explicitly — `1/(x⁸+1)` is the
smallest such case.

### Verification

A common pattern is to verify an antiderivative by differentiating it back:

```python
antideriv = integrate(expr, x).value
check = simplify(diff(antideriv, x).value)
# check.value should equal expr
```

## Higher derivatives

Chain calls to `diff`:

```python
d2 = diff(diff(sin(x), x).value, x)
print(d2.value)   # -sin(x)
```

The derivation log of the outer `diff` does not include the inner steps. If you need the full trace, concatenate `dr1.steps + dr2.steps`.

## Series and Puiseux expansion

`series(expr, var, point, order)` gives a truncated Taylor or Laurent expansion — integer
powers of `h = var - point`, with an explicit `O(h^order)` remainder:

```python
from alkahest import ExprPool, series, sin

pool = ExprPool()
x = pool.symbol("x")

print(series(sin(x), x, pool.integer(0), 6).expr)
# (x * 1) + (-1/6 * x^3) + (1/120 * x^5) + O(x^6)
```

Some expansions have **fractional** exponents and there is no `Series` that can hold one:
`√x` has valuation `1/2`. `series` refuses those with `E-SERIES-004` rather than returning
a series whose coefficients are `sqrt(0)^-1` — `Ok`, unevaluable, `NaN` on contact.

`alkahest.experimental.puiseux_series` is the sibling that expands them.

```python
from alkahest import ExprPool, sin, sqrt
from alkahest.experimental import puiseux_series

pool = ExprPool()
x = pool.symbol("x")

px = puiseux_series(sqrt(sin(x)), x, pool.integer(0), 5)

px.ramification          # 2   — the exponent lattice is (1/2)Z
px.valuation             # Fraction(1, 2)
px.remainder_order       # 5   — every omitted term has exponent >= 5
[(str(e), str(c)) for e, c in px.terms]
# [('1/2', '1'), ('5/2', '-1/12'), ('9/2', '1/1440')]
#   i.e.  sqrt(sin x) = x^(1/2) - x^(5/2)/12 + x^(9/2)/1440 + O(x^5)
```

`order` means the same thing it means for `series`: every term with exponent `< order` is
present, and the remainder is `O(h^order)`. For a *polar* expansion this is sharper than
`series`, which labels any Laurent result `O(h^1)`:

| call | result |
|---|---|
| `series(1/sin(x), x, 0, 4)` | `x^-1 + x/6 + O(x^1)` |
| `puiseux_series(1/sin(x), x, 0, 4)` | `x^-1 + x/6 + 7x^3/360 + O(x^4)` |

Where both apply the coefficients are identical — `puiseux_series` routes every analytic
sub-part through the same expansion engine `series` uses.

### The ramification index is computed, not read off the input

```python
puiseux_series(sqrt(x**2 + x**3), x, pool.integer(0), 5).ramification   # 1
```

`√(x²+x³) = x·√(1+x)` — the zero under the radical has even order, so every exponent comes
out an integer and the function is single-valued at `0`. Reporting `2` because the input
contains a square root would be a wrong claim about the branch structure.

### Every returned expansion has been checked

A `PuiseuxExpansion` is only ever constructed after verification, and `.evidence` says what
was done:

```python
px.evidence
# {'rungs': 4, 'conclusive_rungs': 4,
#  'worst_margin': -0.0023616615508199, 'power_check_passed': True}
```

* **The prefix ladder.** Each truncation of the series — starting from the empty one, whose
  residual is the function itself and whose claimed rate is the valuation — is evaluated
  against the original function at six points approaching the expansion point, and the
  residual's decay exponent is fitted and compared against the exponent of the first
  omitted term. `conclusive_rungs` counts the rungs that produced a measured rate;
  `worst_margin` is the tightest `observed − claimed` over them. At least one rung must
  measure something, or the expansion is withheld.
* **`power_check_passed`.** With ramification `e`, `S^e` has integer exponents, so it can be
  compared exactly against an independently computed expansion of `f^e` — `√(sin x)` squared
  is `sin x`. `False` means the check did not *apply*; a disagreement is a refusal.

An expansion that cannot be confirmed raises `SeriesError` with code `E-SERIES-006` instead
of being returned with a caveat: once it is in a caller's hands there is no way to tell a
checked expansion from an unchecked one.

### What it refuses

| input | code | why |
|---|---|---|
| `log(x)`, `sqrt(x)*log(x)` | `E-SERIES-005` | a logarithm is not a Puiseux series. `√x·log x` needs a Puiseux–*log* (transseries) representation, which this engine does not have — returning `x^(1/2)` for it would be a **wrong** answer, not a coarse one (opposite sign at `x=0.1`, and the relative error grows without bound as `x→0`) |
| `exp(1/x)`, `sin(1/x)` | `E-SERIES-005` | essential singularity: no exponent bounds the remainder |
| ramification past the verifiable range | `E-SERIES-005` | the decay-rate check could not distinguish the expansion from a neighbouring wrong one, and an expansion that cannot fail a check is not returned as though it passed one |
| a branch that is not real just above the point | `E-SERIES-006` | nothing could be measured |

### Exact fractional exponents from Python

`Expr.__pow__` coerces its exponent through `f64`, so `x ** Fraction(1, 3)` is **not**
`x^(1/3)` — it is the exact binary rational `6004799503160661/18014398509481984`, and
`puiseux_series` refuses it (ramification far past the verifiable range) rather than rounding
it to the fraction you probably meant. Dyadic exponents like `Fraction(3, 2)` are exact
through `f64` and work as written. For anything else, build the power node directly:

```python
cube_root = sin(x).pow_expr(pool.rational(1, 3))
px = puiseux_series(cube_root, x, pool.integer(0), 5)
px.ramification                                  # 3
[(str(e), str(c)) for e, c in px.terms]
# [('1/3', '1'), ('7/3', '-1/18'), ('13/3', '-1/3240')]
```
