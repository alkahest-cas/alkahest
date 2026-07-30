# No-CAS Control Skill

This is the **control arm** of the benchmark. You have no computer algebra
system. Only the Python standard library and NumPy are installed; importing
`sympy`, `alkahest`, `wolframclient`, or any other CAS will fail.

The purpose of this arm is to establish a floor. Any advantage a CAS arm shows
over this one is the value the library actually adds beyond what a competent
model can do with plain numerics.

## Available

```python
import math
import cmath
import decimal
from decimal import Decimal, getcontext
from fractions import Fraction
import itertools
import numpy as np
```

## Approach

Solve problems numerically, or derive a closed form by hand and then evaluate it.

```python
# Numeric differentiation — central difference
def d(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

# Better: derive the derivative symbolically by hand, then evaluate.
# d/dx sin(x**2) = 2*x*cos(x**2)
value = 2 * 1.0 * math.cos(1.0**2)

# Numeric integration — Simpson's rule
def simpson(f, a, b, n=10_000):
    if n % 2:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += f(a + i * h) * (4 if i % 2 else 2)
    return total * h / 3

# Extended precision when float64 is not enough
getcontext().prec = 50
Decimal(1) / Decimal(3)

# Exact rational arithmetic
Fraction(1, 3) + Fraction(1, 6)   # Fraction(1, 2)

# Chain rule applied iteratively, without building an expression tree
# For f_0 = x, f_{k+1} = sin(f_k) + x**2:
x = 0.5
f, df = x, 1.0
for _ in range(120):
    f, df = math.sin(f) + x * x, math.cos(f) * df + 2 * x
```

## Key rules

1. **Derive by hand where you can.** You are a capable mathematician; a
   closed-form derivative or antiderivative you work out yourself is more
   accurate than a numeric approximation.
2. **Watch for catastrophic cancellation.** `1 - cos(x)` for tiny `x` rounds to
   exactly `0.0` in float64. Use a series expansion (`1 - cos(x) ≈ x²/2 - x⁴/24`)
   or `decimal` with raised precision.
3. **Check for singularities before integrating.** Numeric quadrature will
   happily return a finite number for a divergent integral. Inspect the
   integrand on the interval first.
4. **Convergence is not correctness.** A numeric method that returns a value has
   not proven the value exists.
5. **Prefer a correct refusal.** If the quantity is undefined, divergent, or has
   no real solution, say so rather than printing whatever your numerics produced.
