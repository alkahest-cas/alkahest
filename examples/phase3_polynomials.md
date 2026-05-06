# Phase 3 — Polynomial Representation

This document walks through the polynomial types added in Phase 3:
`UniPoly`, `MultiPoly`, and `RationalFunction`.

All three are accessible from Python via the `alkahest` package.
Run the accompanying script to see live output:

```
.venv/bin/python3 examples/phase3_polynomials.py
```

---

## Architecture

Phase 3 sits between the symbolic kernel (Phase 1) and future algebraic
algorithms.  The data flow is:

```
ExprPool  →  from_symbolic()  →  UniPoly / MultiPoly / RationalFunction
```

Symbolic expressions are converted eagerly into dense/sparse coefficient
representations backed by FLINT 2.8.4 (`fmpz_poly_t` for `UniPoly`,
`BTreeMap<Vec<u32>, rug::Integer>` for `MultiPoly`).

---

## ExprPool — building symbolic expressions

Every polynomial starts as a symbolic expression built inside an `ExprPool`.
The pool hash-conses nodes so structurally identical sub-expressions share
the same `ExprId`.

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
two = pool.integer(2)

# x^2 + 2x + 1
expr = pool.add([pool.pow(x, two), pool.mul([two, x]), pool.integer(1)])
```

Available builders: `symbol`, `integer`, `add`, `mul`, `pow`.

---

## UniPoly — univariate polynomial over ℤ

`UniPoly` wraps FLINT's `fmpz_poly_t`.  Coefficients are stored in
ascending degree order (constant term first).

### Construction

```python
pool = ak.ExprPool()
x    = pool.symbol("x")

# x^2 + 2x + 1
p = ak.UniPoly.from_symbolic(
    pool.add([pool.pow(x, pool.integer(2)),
              pool.mul([pool.integer(2), x]),
              pool.integer(1)]),
    x,
    pool,
)

print(p)                # x^2+2*x+1
print(p.degree())       # 2
print(p.coefficients()) # [1, 2, 1]
```

### Arithmetic

All four operations are implemented via FLINT:

```python
xp1 = ak.UniPoly.from_symbolic(pool.add([x, pool.integer(1)]), x, pool)
xm1 = ak.UniPoly.from_symbolic(pool.add([x, pool.integer(-1)]), x, pool)

print(xp1 + xm1)   # 2*x
print(xp1 - xm1)   # 2
print(xp1 * xm1)   # x^2-1
print(xp1.pow(3))  # x^3+3*x^2+3*x+1
```

### GCD

```python
x2m1 = ak.UniPoly.from_symbolic(
    pool.add([pool.pow(x, pool.integer(2)), pool.integer(-1)]), x, pool
)
g = x2m1.gcd(xm1)
print(g)          # x-1
print(g.degree()) # 1
```

GCD is computed by FLINT's `fmpz_poly_gcd` and is normalised to a positive
leading coefficient.

---

## MultiPoly — multivariate polynomial over ℤ

`MultiPoly` stores terms as a sparse `BTreeMap` from exponent vector to
`rug::Integer` coefficient.  The variable ordering is fixed at construction
time by the `vars` list.

> **Display note:** variable names in the string representation use
> positional labels (`x0`, `x1`, …) matching the index in `vars`.

### Construction

```python
pool = ak.ExprPool()
x    = pool.symbol("x")
y    = pool.symbol("y")

# x^2 + xy + y^2
a = ak.MultiPoly.from_symbolic(
    pool.add([pool.pow(x, pool.integer(2)),
              pool.mul([x, y]),
              pool.pow(y, pool.integer(2))]),
    [x, y],   # variable ordering: x=x0, y=x1
    pool,
)
print(a)               # x1^2 + x0x1 + x0^2
print(a.total_degree()) # 2
```

### Integer content

The *integer content* is the GCD of all coefficients.  Dividing by it gives
the *primitive part*.

```python
# 6x + 4  →  content = 2,  primitive part = 3x + 2
b = ak.MultiPoly.from_symbolic(
    pool.add([pool.mul([pool.integer(6), x]), pool.integer(4)]),
    [x, y], pool,
)
print(b.integer_content())  # 2
```

### Arithmetic

```python
c = ak.MultiPoly.from_symbolic(pool.add([x, y]), [x, y], pool)
d = ak.MultiPoly.from_symbolic(pool.add([x, pool.integer(-1)]), [x, y], pool)

print(c + d)  # 2x + y - 1
print(c * d)  # x^2 + xy - x - y
```

---

## RationalFunction

`RationalFunction` holds a numerator and denominator `MultiPoly` and
enforces two normalisation invariants at construction:

1. **Integer content** — the combined GCD of all coefficients across both
   numerator and denominator is divided out.
2. **Positive leading coefficient** — the lexicographically-last term in the
   denominator has a positive coefficient (signs are flipped uniformly if
   needed).

```python
pool = ak.ExprPool()
x    = pool.symbol("x")
y    = pool.symbol("y")

# (6x) / 4  →  normalises to (3x) / 2
rf = ak.RationalFunction.from_symbolic(
    pool.mul([pool.integer(6), x]),   # numerator
    pool.integer(4),                   # denominator
    [x, y],
    pool,
)
print(rf)          # (3x0) / (2)
print(rf.numer())  # 3x0
print(rf.denom())  # 2

# Denominator 1 is elided
rf2 = ak.RationalFunction.from_symbolic(x, pool.integer(1), [x, y], pool)
print(rf2)   # x0

# Sign normalisation: denom leading coefficient is always positive
rf3 = ak.RationalFunction.from_symbolic(x, pool.integer(-2), [x, y], pool)
print(rf3)   # (-x0) / (2)
```

---

## Error handling

Conversion raises `ValueError` for non-polynomial expressions:

| Situation | Error message |
|---|---|
| Free symbol in polynomial | `unexpected free symbol 'y' in polynomial expression` |
| Rational or float coefficient | `non-integer coefficient (rational or float) in polynomial` |
| Negative integer exponent | `negative exponent yields a rational function, not a polynomial` |
| Non-constant exponent | `exponent is not a constant integer` |
| Zero denominator | `denominator is zero` |

```python
pool = ak.ExprPool()
x    = pool.symbol("x")
y    = pool.symbol("y")

try:
    ak.UniPoly.from_symbolic(pool.add([x, y]), x, pool)
except ValueError as e:
    print(e)   # unexpected free symbol 'y' ...
```

---

## Rust API summary

The same types are available directly in Rust:

```rust
use alkahest_core::{ExprPool, Domain, UniPoly, MultiPoly, RationalFunction};

let pool = ExprPool::new();
let x    = pool.symbol("x", Domain::Real);
let xsq  = pool.pow(x, pool.integer(2_i32));
let expr = pool.add(vec![xsq, pool.mul(vec![pool.integer(2_i32), x]), pool.integer(1_i32)]);

let p = UniPoly::from_symbolic(expr, x, &pool).unwrap();
assert_eq!(p.coefficients_i64(), vec![1, 2, 1]);  // [c0, c1, c2]

let q = UniPoly::from_symbolic(pool.add(vec![x, pool.integer(1_i32)]), x, &pool).unwrap();
let r = UniPoly::from_symbolic(pool.add(vec![x, pool.integer(-1_i32)]), x, &pool).unwrap();
assert_eq!((q * r).coefficients_i64(), vec![-1, 0, 1]);  // x^2 - 1
```
