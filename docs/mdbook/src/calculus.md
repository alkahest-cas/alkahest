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

### Non-smooth functions

`abs`, `sign`, `floor`, `ceil`, `round`, `min` and `max` are not differentiable
everywhere. `diff`, `diff_forward` and `symbolic_grad` return the
**almost-everywhere derivative** and state where it holds, as side conditions on
the derivation (`result.side_conditions`, also on each step in `result.steps`):

| expression | derivative | holds where |
|---|---|---|
| `abs(g)` | `sign(g)·g'` | `g ≠ 0` |
| `sign(g)` | `0` | `g ≠ 0` |
| `floor(g)`, `ceil(g)` | `0` | `g − floor(g) ≠ 0` (g not an integer) |
| `round(g)` | `0` | `g + 1/2 − floor(g + 1/2) ≠ 0` |
| `min(a, b)` | `Piecewise((a < b, a'), (a > b, b'); (a'+b')/2)` | `a − b ≠ 0` |
| `max(a, b)` | `Piecewise((a > b, a'), (a < b, b'); (a'+b')/2)` | `a − b ≠ 0` |

At a kink the value is the midpoint of the one-sided slopes, so the result is
never an unstated zero. `heaviside` keeps its distributional derivative,
`diracdelta`. `rewrite_piecewise` lowers `abs`/`sign`/`min`/`max` to `Piecewise`,
and `integrate` splits `Piecewise` integrands at the exact roots of their
conditions (rational or quadratic-surd breakpoints; a condition that is
non-polynomial, parametric or has an irreducible factor of degree ≥ 3 is refused
with `E-INT-001`) and returns an antiderivative that is continuous at every
breakpoint.

### Derivation log

The `DerivedResult` returned by `diff` records every rule application:

```python
dr = diff(sin(x**2), x)
for step in dr.steps:
    print(f"  {step['rule']:25s}  {step['before']}  →  {step['after']}")
```

## Undefined functions and Euler–Lagrange

A function name the kernel attaches no meaning to is an undefined (arbitrary
smooth) function. `diff` returns a canonical derivative node for it, with the
chain rule applied per argument slot:

```python
import alkahest as ak
pool = ak.ExprPool()
t = pool.symbol("t")
q = pool.func("q", [t])
ak.diff(q, t).value                          # Derivative(q(t), t)
ak.diff(pool.func("f", [t**2]), t).value     # 2*t*Subs(Derivative(f(_xi_1), _xi_1), _xi_1, t^2)
ak.subs(ak.diff(q, t).value, {q: ak.sin(t)}) # cos(t)
```

`euler_lagrange(L, [q1, q2], t)` returns the equations of motion
`d/dt(∂L/∂q̇) − ∂L/∂q`, and `functions_to_symbols(eoms, [q1, q2], t)` turns
`q`, `q̇`, `q̈` into plain symbols (`q1`, `q1_dot`, `q1_ddot`) for `solve`.
`replace_function(expr, f(x), body)` substitutes a definition for `f` at every
argument, derivatives included. The printed form reads back with
`parse(text, pool, {"q": q, "t": t})`; numeric evaluation refuses an undefined
function with `E-EVAL-005`.

### Calculus with undefined functions

`integrate` reverses the chain rule for undefined functions and their
derivative nodes, and refuses everything that is not an exact derivative (an
undefined function has no antiderivative of its own):

```python
f = pool.func("f", [t]); fp = ak.diff(f, t).value
ak.integrate(fp, t).value                  # f(t)
ak.integrate(f * fp, t).value              # f(t)^2 / 2
ak.integrate(fp / f, t).value              # log(f(t))
ak.integrate(t * ak.diff(fp, t).value, t).value   # t*f'(t) - f(t)
ak.integrate(f, t)                         # IntegrationError: no closed form
```

Each candidate is differentiated back and compared with the integrand before
it is returned; definite integrals are `F(b) - F(a)` (infinite bounds are
refused). `series(f, t, a, n)` expands in `f(a)`, `f'(a)`, ... . `limit` treats
`f(t)` as opaque: it is refused (`E-LIMIT-007`) unless every undefined node is
declared continuous at the point, `limit(f*cos(t), t, 2, continuous=[f])`, and
then only an expression that is jointly continuous there is evaluated by
substitution (a `(f(t) - f(a))/(t - a)` quotient needs differentiability, which
is never assumed).

`experimental.dsolve_system(equations, functions=[y(t), ...], t=t)` and
`ODE.from_equations(equations, functions, t)` accept linear systems written in
`y(t)` and `diff(y(t), t)` of any order, reduce them to first order, and
return the closed form also as `result["functions"] == {y(t): ...}`.

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
roots and summing the body over them in complex arithmetic. The sum is
ill-conditioned in the roots — a *perfect* double-precision root set misses the
comparison tolerance from degree ≈ 15 — so from degree 8 up the roots are
Newton-refined to 256 bits on the exact minimal polynomial and the body is
summed in that precision. Above degree 24 `integrate` declines with
`E-INT-001` rather than return an answer it cannot check. Such an answer is not
claimed to be non-elementary: every rational function is elementary, and
`E-INT-004` is never raised here.

The log argument of each `RootSum` is the Lazard–Rioboo–Trager one,
`gcd_x(D, A − α·D′)`, read off the **subresultant chain** of `D` and `A − t·D′`
over `ℚ[t]` (computed once) and reduced modulo the residue's minimal polynomial.
Each such argument is checked by dividing `D` and `A − α·D′` by it in `ℚ(α)[x]`
— a divisor of the right degree is the gcd — and the Euclidean gcd over the
number field is used only if that check fails. On
`∫ (x⁷+3x²−1)/(x¹⁵+x⁴+2x+1)² dx` this takes the log part from minutes to about
a second.

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

### Symbolic parameters and unnormalised integrands

When every route declines an integrand as written, `integrate` normalises it
and tries again termwise: integer powers of sums are expanded, products of
exponentials are merged into one (`e^{−aτ}·e^{−b(t−τ)}` → `e^{(b−a)τ − bt}`),
and powers of a common base are combined (`x·x^{n−1}` → `xⁿ`,
`τ·(aτ)⁻¹` → `1/a`). The terms then take closed forms that accept symbolic
parameters:

```python
k, m, a, b = (pool.symbol(s) for s in "kmab")
integrate(x * exp(-k * x), x).value        # e^{−kx}·(−x/k − 1/k²), k ≠ 0
integrate(1 / (k**2 + m**2), k).value      # atan(k/m)/m
integrate(1 / (a * x**2 + b), x).value     # atan(a·x/√(ab))/√(ab), stated: ab > 0
integrate(x**2 * exp(2*x) * sin(x), x)     # tabular (repeated by-parts) formula
```

Every division by a parameter is recorded in `result.verification["side_conditions"]`
unless its sign is known; the arctangent of a quadratic whose discriminant
`4ac − b²` has unknown sign carries `4ac − b² > 0`, a provably negative
discriminant gives logarithms instead. The answer is checked against the
original integrand before it is returned.

`integrate(f, x, 0, oo)` of a finite sum of `c·x^p·e^{−αx}` uses the Gamma
integral `Γ(p+1)/α^{p+1}` termwise, when `α > 0` and `p > −1` are established
(numerically or from symbol domains / `assumptions`):

```python
y = pool.symbol("y")
integrate(exp(-y / 2) / sqrt(y), y, pool.integer(0), pool.pos_infinity()).value
# → √2·√π   (the chi-square normaliser)
P = ak.Domain.Positive
t, ke, ka = (pool.symbol(s, P) for s in ("t", "ke", "ka"))
integrate(exp(-ke*t) - exp(-ka*t), t, pool.integer(0), pool.pos_infinity()).value
# → 1/ke − 1/ka   (Bateman AUC)
```

With the sign of `α` unknown it declines (`E-INT-001`, naming the parameter).
At other infinite bounds, the limit of an antiderivative that is a sum is taken
term by term when every term's limit exists and is finite.

### Beta, incomplete-gamma and Feynman-parameter integrals

Definite integrals whose antiderivative is not elementary — or not elementary
with a symbolic exponent — but whose value is a classical function:

| integrand, interval | value |
|---|---|
| `(x−a)^p (b−x)^q` on `[a, b]` | `(b−a)^{p+q+1} Γ(p+1)Γ(q+1)/Γ(p+q+2)` |
| `x^p/(c+dx)^r` on `[0, ∞)`, `c, d > 0` | `c^{p+1−r} d^{−p−1} B(p+1, r−p−1)` |
| `sin^p x cos^q x` on `[0, π/2]` | `½ B((p+1)/2, (q+1)/2)` |
| `x^p e^{−r x^k}` on `[lo, hi] ⊂ [0, ∞]`, `k = 1, 2`, `r > 0` | `r^{−s}/k · (Γ(s, r·loᵏ) − Γ(s, r·hiᵏ))`, `s = (p+1)/k` |
| `log Q`, `Q^{−n}` for quadratic `Q`, `Q > 0` at the ends | `atan` / `log` closed forms by the sign of `κ = (4AC−B²)/(4A²)` |

```python
x, a, b = pool.symbol("x"), pool.symbol("a"), pool.symbol("b")
r = integrate(x**a * (1 - x)**b, x, pool.integer(0), pool.integer(1))
r.value                               # gamma(1 + a)*gamma(1 + b)/gamma(2 + a + b)
r.verification["side_conditions"]     # ['(1 + a) > 0', '(1 + b) > 0']
s, c = pool.symbol("s"), pool.symbol("c", ak.Domain.Positive)
integrate(x**(s - 1) * exp(-x), x, c, pool.pos_infinity()).value   # uppergamma(s, c)
m, p = pool.symbol("m", ak.Domain.Positive), pool.symbol("p", ak.Domain.Positive)
integrate(log(m**2 - x*(1 - x)*p**2), x, pool.integer(0), pool.integer(1)).value
# Piecewise((κ > 0, 2 log m − 2 + 4√κ·atan(1/(2√κ))), (κ = 0, 2 log m − 2);
#           default = 2 log m − 2 + 2√λ·log((½+√λ)/(½−√λ)) + 2πi√λ),  κ = m²/p² − ¼ = −λ
```

Signs the formula needs (a rate `r > 0`, a linear factor's slope, `Q > 0` at the
endpoints) must be established, or the route declines; a leading coefficient of
unknown sign is allowed when it is `c·sᵏ` for one parameter `s` (the report's
`log(m² − x(1−x)·p2)` with `p2` of either sign), and the `Piecewise` then also
branches on its sign, with `Q|_{s=0}` at `s = 0`. A convergence condition on an exponent (`a + 1 > 0`) is stated
as a side condition when unknown, and refused as divergent when established
false; `∫₀¹ dx/(m² − x(1−x)p²)ⁿ` states `κ > 0` (`p² < 4m²`), since above
threshold `Q` vanishes inside the interval. Above threshold `∫ log Q` is the
principal-branch value (`log|Q| + iπ` where `Q < 0`); the Feynman `−i0`
prescription gives its complex conjugate. The quadratic forms are cross-checked
against Gauss–Legendre quadrature at sampled parameter values before they are
returned. Definite results also have their exponents re-collected for display,
so the convolution `∫₀ᵗ e^{−aτ}e^{−b(t−τ)} dτ` reads `(e^{−at} − e^{−bt})/(b − a)`.

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

### Definite elliptic integrals with rational branch points

`integrate(R(x)/sqrt(P(x)), x, a, b)` with `P` a cubic or quartic whose roots
are all **rational** (the radicand may be written factored or expanded), `R` a
rational function whose poles are rational and off `[a, b]`, and `[a, b]` —
rational bounds or `±oo` — inside one interval where `P > 0`, is evaluated
exactly:

```python
integrate(1/sqrt(x*(1 - x)*(2 - x)), x, 0, 1).value   # √2·EllipticK(1/2)
integrate(x/sqrt(x*(1 - x)*(2 - x)), x, 0, 1).value   # √2·(2K(1/2) − 2E(1/2))
integrate(1/sqrt((x-1)*(x-2)*(x-3)*(x-4)), x, 2, 3).value   # EllipticK(3/4)
```

The roots (with `∞` for a cubic) are four points of the projective line; the
cross-ratio Möbius map sending the ends of the interval's arc to `0` and `1`
gives `sin²φ = h(x)`, a rational parameter `m`, and a constant `c` with
`c² ∈ ℚ`. The `φ`-integral is then solved by an exact ansatz
`α·F + β·E + Σ γᵢ·Π(nᵢ) + sinφ·cosφ·Δ·T(sin²φ)` over `ℚ` (a linear system), and
the endpoints give `EllipticF/E(asin(√y), m)` with `y` rational, or the
complete `EllipticK/E(m)` and `EllipticPi(n, π/2, m)` at a root of `P`. Nothing
is lifted from a float, and every finite-interval value is cross-checked
against a tanh–sinh quadrature whose polynomials are Taylor-shifted to the
nearer endpoint. A divergent end, a pole on `[a, b]`, a `Π` whose path would
cross its own pole, or a radicand that does not split over `ℚ` declines to the
other routes (whose float-derived elliptic antiderivatives the definite path
refuses).

### Continuity across substitution poles

A rational function of `sin x` and `cos x` is integrated by `t = tan(x/2)`, or —
when it is invariant under `x ↦ x + π`, like `1/(1 + sin²x)` — by `t = tan x`.
Either substitution produces an antiderivative between consecutive poles of the
tangent only; written naively, its `atan` terms jump by `±π` there although the
integrand is regular. Alkahest returns antiderivatives that are valid on every
interval where the integrand is continuous:

```python
F = integrate(1 / (1 + sin(x)**2), x).value
# (x + atan(...))/√2 — no jump at π/2 + kπ
```

`atan(a·tan θ + b)` terms are rewritten into continuous `θ + atan2(N, D)` forms,
rational terms in `tan θ` are homogenised into `sin θ`/`cos θ`, and any jump that
remains on the pole lattice is computed exactly by one-sided limits and cancelled
by `J·x/T − (J/2π)·atan2(sin φ, cos φ)` (log step `int_continuous_step_correction`).

A definite integral whose antiderivative still jumps inside the interval — at a
pole of the substitution, or where the denominator of an `atan(N/D)` vanishes —
is evaluated as `F(b) − F(a) − Σ jumps`. Each branch crossing contributes exactly
`±π` times its coefficient, so only the *number and direction* of crossings is
located numerically; the corrected value is re-scanned for any remaining jump
and cross-checked against tanh–sinh quadrature before it is returned, and
refused (`E-INT-001`) otherwise. `∫ cot(x)^(p/q)` is integrated through the
reflection `cot u = tan(π/2 − u)`, so `integrate(cot(x)**(3/2), x, 1/5, 1)` is
`2.17202966873…`.

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

When every coefficient is a rational number — compositions of `+ − × ÷`, rational powers,
`exp`, `log`, `sin`, `cos`, `tan`, their hyperbolic forms, `atan`, `asin`, `atanh`, `asinh` and
`sqrt`, at a point where none of them has an irrational value — the expansion is computed with
truncated power-series arithmetic over ℚ, so `series(sin(tan(x)) - tan(sin(x)), x, 0, 15)` takes
a fraction of a millisecond. Anything else (a free parameter, `sin(1 + h)`, a branch point) is
expanded from repeated derivatives, as before.

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
present, and the remainder is `O(h^order)` — for a *polar* expansion too:

| call | result |
|---|---|
| `series(1/sin(x), x, 0, 4)` | `x^-1 + x/6 + 7x^3/360 + O(x^4)` |
| `puiseux_series(1/sin(x), x, 0, 4)` | `x^-1 + x/6 + 7x^3/360 + O(x^4)` |

(Through 3.11 `series` labelled every Laurent result `O(h^1)` and computed only `order`
coefficients past the pole, so a pole of order `≥ order` lost terms the remainder claimed.)

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

`x ** Fraction(1, 3)` is `x^(1/3)`. It used not to be: `Expr.__pow__` coerced its exponent
through `f64`, so the exponent that reached the kernel was the binary rational
`6004799503160661/18014398509481984` and `puiseux_series` refused it (ramification far past
the verifiable range) rather than rounding it to the fraction you probably meant.

An exponent with an exact value now reaches the pool exactly — a `Fraction`, a
`decimal.Decimal`, and a Python `int` of any width, including one too large for an `i64`. A
Python `float` is still a float node, because a float is what it is.

```python
cube_root = sin(x) ** Fraction(1, 3)             # or sin(x).pow_expr(pool.rational(1, 3))
px = puiseux_series(cube_root, x, pool.integer(0), 5)
px.ramification                                  # 3
[(str(e), str(c)) for e, c in px.terms]
# [('1/3', '1'), ('7/3', '-1/18'), ('13/3', '-1/3240')]
```
