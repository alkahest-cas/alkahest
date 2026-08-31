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
