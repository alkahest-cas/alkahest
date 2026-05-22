# Expression representations

Alkahest exposes multiple representation types rather than hiding everything behind a single `Expr`. This is a deliberate design decision: the representation is visible, conversions are explicit, and performance characteristics are predictable.

## Choosing a representation

| If you need... | Use |
|---|---|
| General symbolic computation | `Expr` |
| Fast univariate polynomial arithmetic | `UniPoly` |
| Sparse multivariate polynomial algebra | `MultiPoly` |
| Sparse polynomial over 𝔽ₚ (modular / interpolation) | `MultiPolyFp` |
| Rational functions with automatic cancellation | `RationalFunction` |
| Rigorous enclosures with error bounds | `ArbBall` |

Conversion to a specialized type is always an explicit opt-in:

```python
expr = x**3 + pool.integer(-2) * x + pool.integer(1)
p = UniPoly.from_symbolic(expr, x)   # explicit conversion
```

If the expression cannot be represented in the target type (e.g. `sin(x)` as a polynomial), a `ConversionError` is raised with a remediation hint.

## Expr

The generic symbolic expression. All other types convert to and from `Expr`. Built by operator overloading on the Python side:

```python
expr = x**2 + pool.integer(3) * x * y - pool.integer(1)
```

Operations like `diff`, `simplify`, and `integrate` work on `Expr` and return `DerivedResult` objects wrapping an `Expr`.

## UniPoly

Dense univariate polynomial backed by FLINT. Coefficients are exact integers or rationals stored in a FLINT polynomial object.

```python
from alkahest import UniPoly

# x^3 - 2x + 1
p = UniPoly.from_symbolic(x**3 + pool.integer(-2) * x + pool.integer(1), x)

print(p.degree())        # 3
print(p.coefficients())  # [1, -2, 0, 1]  (constant first)
print(p.leading_coeff()) # 1

# Arithmetic — all FLINT-backed, exact
q = UniPoly.from_symbolic(x + pool.integer(-1), x)
print(p * q)             # x^4 - x^3 - 2x^2 + 3x - 1
print(p.gcd(q))          # x - 1
print(p // q)            # x^2 + x - 1
print(p % q)             # 0

# Powers
r = UniPoly.from_symbolic(x + pool.integer(1), x)
print(r ** 3)            # x^3 + 3x^2 + 3x + 1
```

`UniPoly` is the right choice when you are doing heavy univariate polynomial arithmetic (GCD chains, resultants, factorization) because FLINT applies highly optimized algorithms with exact arithmetic.

## MultiPoly

Sparse multivariate polynomial over ℤ (integers). Terms are stored as a map from exponent vectors to coefficients.

```python
from alkahest import MultiPoly

# x^2*y + x*y^2 - 1
expr = x**2 * y + x * y**2 + pool.integer(-1)
mp = MultiPoly.from_symbolic(expr, [x, y])

print(mp.total_degree())     # 3
print(mp.integer_content())  # 1

# Arithmetic
mp2 = MultiPoly.from_symbolic(x * y, [x, y])
print(mp + mp2)              # x^2*y + x*y^2 + x*y - 1
print(mp * mp2)              # x^3*y^2 + x^2*y^3 - x*y
```

Variable order matters for the exponent-vector key. Pass variables in a consistent order when constructing `MultiPoly` objects that will be combined.

## MultiPolyFp

Sparse multivariate polynomial over 𝔽ₚ = ℤ/pℤ. Used as the working type for black-box sparse interpolation and sparse modular GCD.

```python
from alkahest import sparse_interp_univariate, sparse_interp, gcd_sparse, MultiPoly

p = 32749  # prime

# Recover a sparse univariate from 2T black-box evaluations (Ben-Or/Tiwari)
f = sparse_interp_univariate(lambda v: (v**5 + 3*v**3 + 7) % p, T=3, prime=p)
print(f)   # x^5 + 3*x^3 + 7 (as MultiPolyFp)

# Recover a sparse multivariate via Zippel's algorithm
f2 = sparse_interp(
    lambda vals: (vals[0]**3 * vals[1]**2 + vals[0] * vals[1]**4) % p,
    vars=[x, y], T=2, D=5, prime=p,
)
print(f2)  # x^3*y^2 + x*y^4

# Sparse modular GCD over ℤ[x₁,...,xₙ] — substrate for exact GCD algorithms
a = MultiPoly.from_symbolic((x + y) * (x - y), [x, y])
b = MultiPoly.from_symbolic((x + y) * (x + pool.integer(1)), [x, y])
h = gcd_sparse(a, b, term_bound=4, degree_bound=4)
print(h)   # x + y
```

`sparse_interp_univariate` uses Berlekamp–Massey + BSGS discrete-log + Vandermonde solve and requires exactly `2T` oracle calls. `sparse_interp` uses Zippel's variable-by-variable algorithm with batched Vandermonde lifting.

## RationalFunction

Quotient of two `MultiPoly` objects, automatically reduced by their GCD.

```python
from alkahest import RationalFunction

# (x^2 - 1) / (x - 1) → normalized to x + 1
numer = x**2 + pool.integer(-1)
denom = x + pool.integer(-1)
rf = RationalFunction.from_symbolic(numer, denom, [x])
print(rf)   # x + 1

# Arithmetic preserves the rational form
rf_x = RationalFunction.from_symbolic(x, pool.integer(1), [x])
rf_inv = RationalFunction.from_symbolic(pool.integer(1), x, [x])
print(rf_x + rf_inv)   # (x^2 + 1) / x
```

GCD normalization runs at construction, so every `RationalFunction` is in lowest terms.

## ArbBall

A real interval `[midpoint ± radius]` backed by FLINT's Arb library. Arithmetic on `ArbBall` values produces guaranteed enclosures of the true result.

```python
from alkahest import ArbBall, interval_eval, sin

# ArbBall(midpoint, radius, precision_bits=53)
a = ArbBall(2.0, 0.5)    # [1.5, 2.5]
b = ArbBall(3.0, 0.0)    # exactly 3

print(a + b)   # [4.5, 5.5]
print(a * b)   # [4.5, 7.5]

# Evaluate a symbolic expression rigorously
pool = ExprPool()
x = pool.symbol("x")
result = interval_eval(sin(x), {x: ArbBall(1.0, 1e-10)})
print(result.lo, result.hi)   # tight enclosure of sin(1)
```

The output ball is guaranteed to contain the true value for any input in the input balls. This is useful for:
- Certified numerical evaluation
- Proving bounds on symbolic expressions
- Verification workflows alongside Lean certificate export

See [Ball arithmetic](./ball-arithmetic.md) for more detail.

## Converting back to Expr

All specialized types can be converted back to a generic `Expr` for further symbolic manipulation:

```python
p = UniPoly.from_symbolic(x**2 + pool.integer(1), x)
expr_again = p.to_symbolic(pool)
dr = diff(expr_again, x)
```
