# Ball arithmetic

## Unified evaluation (experimental)

`alkahest.experimental.evaluate` provides one result contract for exact
rational, `f64`, and rigorous interval evaluation. It returns an
`EvaluationResult`; mathematically unsupported inputs return
`status == "unsupported"` and a stable `E-EVAL-*` reason code rather than raising.
Invalid API input, such as an invalid mode or zero precision, still raises
`ValueError`.

```python
from fractions import Fraction
import alkahest as ak
from alkahest.experimental import evaluate

p = ak.ExprPool()
x = p.symbol("x")
result = evaluate(x + p.rational(1, 3), {x: Fraction(1, 6)})
assert result.value == Fraction(1, 2)
assert result.backend == "exact_rational"
```

Use `mode="f64"` for ordinary floating-point evaluation and
`mode="interval"` with `ArbBall` bindings for an enclosure. In interval mode,
`result.value` and `result.enclosure` are the same `ArbBall`; its `lo` and
`hi` bound the true result. `mode="auto"` selects intervals for `ArbBall`
bindings (or an explicit precision), exact rationals when possible, and
otherwise falls back to `f64`.

Ball arithmetic provides rigorous enclosures: every operation produces an interval guaranteed to contain the true result. Alkahest uses FLINT's Arb library as the backend.

## ArbBall

An `ArbBall` represents the real interval `[midpoint ± radius]`:

```python
from alkahest import ArbBall

a = ArbBall(2.0, 0.5)    # [1.5, 2.5]
b = ArbBall(3.0, 0.0)    # exactly 3.0

print(a.mid)   # 2.0
print(a.rad)   # 0.5
print(a.lo)    # 1.5
print(a.hi)    # 2.5
```

An `ArbBall` can also carry a precision (in bits) for the midpoint:

```python
a = ArbBall(2.0, 1e-30, prec=128)  # 128-bit midpoint
```

## Ball arithmetic operations

All arithmetic on `ArbBall` values produces a guaranteed enclosure. The radius grows to account for rounding and operation error:

```python
a = ArbBall(2.0, 0.1)
b = ArbBall(3.0, 0.1)

print(a + b)    # [4.8, 5.2]  — radius grows by sum of input radii
print(a * b)    # guaranteed enclosure of [1.9, 2.1] * [2.9, 3.1]
print(a ** 2)   # [3.24, 4.41]  (squares the interval)
```

## interval_eval

`interval_eval` evaluates a symbolic expression with `ArbBall` inputs:

```python
from alkahest import ExprPool, ArbBall, interval_eval, sin, exp

pool = ExprPool()
x = pool.symbol("x")

# sin(1 ± 1e-10) — guaranteed enclosure
result = interval_eval(sin(x), {x: ArbBall(1.0, 1e-10)})
print(result.lo, result.hi)

# Multivariate
y = pool.symbol("y")
expr = sin(x) * exp(y)
result = interval_eval(expr, {
    x: ArbBall(1.0, 0.01),
    y: ArbBall(0.0, 0.01),
})
```

`interval_eval` guarantees that the output ball contains the true value for any input in the given input balls, accounting for all rounding in the intermediate computation.

## AcbBall

Complex ball arithmetic for expressions over ℂ:

```python
from alkahest import AcbBall

z = AcbBall(1.0, 0.0, 1.0, 0.0)  # 1 + i, exact
```

## Use cases

**Certified numerical evaluation.** Compute a value and prove it lies within a tight bound without symbolic proof:

```python
# Prove sin(1) ∈ [0.841, 0.842]
r = interval_eval(sin(x), {x: ArbBall(1.0, 0.0)})
assert r.lo > 0.841 and r.hi < 0.842
```

**Numerical verification of symbolic results.** After deriving a symbolic simplification, verify it numerically with rigorous bounds:

```python
# Verify sin²(x) + cos²(x) = 1 at x = 1
lhs = sin(x)**pool.integer(2) + cos(x)**pool.integer(2)
r = interval_eval(lhs, {x: ArbBall(1.0, 0.0)})
assert 1.0 in r  # ball contains 1
```

**Sensitivity analysis.** Pass an input ball representing parameter uncertainty and observe how the output uncertainty grows:

```python
# x = 1 ± 0.1 (10% uncertainty)
r = interval_eval(x**pool.integer(3), {x: ArbBall(1.0, 0.1)})
print(r)  # output uncertainty
```

## Relationship to Lean certificates

Ball arithmetic and Lean certificate export are complementary:

- Ball arithmetic gives numerical certainty within floating-point computation.
- Lean certificates give symbolic/logical certainty for the rewrite steps applied.

Combining them: `certify(interval(differentiate(f)))` gives a derivative, evaluated with rigorous interval bounds, with a machine-checkable proof of the symbolic differentiation step.
