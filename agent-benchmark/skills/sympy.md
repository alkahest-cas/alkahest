# SymPy Agent Skill

Use this skill whenever you are writing Python code that uses the `sympy` library.

## Official links

- **Repository:** [github.com/sympy/sympy](https://github.com/sympy/sympy)
- **Website:** [sympy.org](https://www.sympy.org/)
- **Documentation:** [docs.sympy.org/latest/](https://docs.sympy.org/latest/)
- **API reference:** [docs.sympy.org/latest/reference/index.html](https://docs.sympy.org/latest/reference/index.html)
- **Tutorial:** [docs.sympy.org/latest/tutorials/intro-tutorial/index.html](https://docs.sympy.org/latest/tutorials/intro-tutorial/index.html)
- **Live shell:** [live.sympy.org](https://live.sympy.org/)
- **Gotchas (read this):** [docs.sympy.org/latest/explanation/gotchas.html](https://docs.sympy.org/latest/explanation/gotchas.html)

## Install

**Requirements:** Python **3.9+**. SymPy is pure Python with a single hard
dependency (`mpmath`).

```bash
pip install sympy
```

For an isolated environment (recommended when juggling versions):

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install sympy
```

### Optional accelerators

SymPy is pure Python, so ground-domain arithmetic is the usual bottleneck. Two
optional packages change performance substantially and neither changes the API:

| Package | Effect |
|---|---|
| `gmpy2` | Replaces the pure-Python integer/rational ground types with GMP bindings. Large-integer polynomial work (GCD, factorisation, Gröbner) gets several times faster. |
| `numpy` | Required for `lambdify(..., "numpy")` vectorised evaluation. Without it `lambdify` falls back to the `math` module and evaluates one point at a time. |

```bash
pip install sympy gmpy2 numpy
```

Check which ground types are active:

```python
import sympy as sp
from sympy.polys.domains import ZZ

print(sp.__version__)
print(type(ZZ(1)))     # mpz -> gmpy2 active; int -> pure Python
```

There is no compiled extension, no JIT, and no feature-flag build. Every install
of a given SymPy version has the same capabilities — unlike libraries with
optional native backends, you never need to probe for available features.

### SymEngine backend

`symengine` is a separate C++ core with a partial SymPy-compatible API. It is
**not** a drop-in replacement: coverage is much narrower (no `integrate`, no
`dsolve`, limited `simplify`). Do not reach for it unless the user asks.

---

## Core mental model

SymPy expressions are **immutable Python objects** built from `Symbol`,
`Integer`, `Rational`, `Float`, and function applications. There is no pool, no
context, and no interning step you must perform — raw Python `int` literals work
directly in arithmetic.

```python
import sympy as sp

x = sp.Symbol("x")
y = sp.Symbol("y")
z = sp.Symbol("z")

# Several at once
a, b, c = sp.symbols("a b c")
i, j = sp.symbols("i j", integer=True)

expr = x**2 + 2 * x + 1        # Python ints are fine
half = sp.Rational(1, 2)       # exact 1/2 — NOT 0.5
```

Arithmetic operators (`+`, `-`, `*`, `**`, `/`) are overloaded on every SymPy
object. Use them freely.

### Automatic evaluation

SymPy evaluates *some* things eagerly on construction and leaves others alone.
`x - x` becomes `0` immediately; `sin(x)**2 + cos(x)**2` does not become `1`
until you ask. This is the most common source of surprise:

```python
x - x                          # 0                      (auto)
sp.sin(x)**2 + sp.cos(x)**2    # sin(x)**2 + cos(x)**2  (NOT simplified)
sp.sqrt(8)                     # 2*sqrt(2)              (auto)
sp.sqrt(x**2)                  # sqrt(x**2)             (NOT x — see assumptions)
```

To suppress automatic evaluation:

```python
sp.sympify("1+1", evaluate=False)      # 1 + 1
with sp.evaluate(False):
    e = x + x                          # x + x, unsimplified
```

### The `Rational` trap

`1/3` in Python is a float. Inside a SymPy expression that silently converts
exact arithmetic into floating point, and the error propagates through
everything downstream:

```python
x + 1/3                 # x + 0.333333333333333   ← float contamination
x + sp.Rational(1, 3)   # x + 1/3                 ← exact
x + sp.S(1)/3           # x + 1/3                 ← exact, shorthand
sp.sympify("1/3")       # 1/3                     ← exact
```

**Always use `sp.Rational(p, q)` or `sp.S(1)/q` for exact fractions.** A single
stray Python division turns an exact answer inexact with no warning.

### Inspecting structure

```python
sp.srepr(x + 1)          # "Add(Symbol('x'), Integer(1))" — exact tree
(x + 1).args             # (1, x)
(x + 1).func             # <class 'sympy.core.add.Add'>
(x**2 + y).free_symbols  # {x, y}
(x + 1).atoms(sp.Symbol)
```

### Assumptions

Assumptions are attached to symbols at creation and control which rewrites are
legal. Without them SymPy is conservative, because the identity may be false:

```python
p = sp.Symbol("p", positive=True)
r = sp.Symbol("r", real=True)
n = sp.Symbol("n", integer=True)

sp.sqrt(x**2)          # sqrt(x**2)  — x could be negative or complex
sp.sqrt(r**2)          # Abs(r)      — real, so the modulus is provable
sp.sqrt(p**2)          # p           — positive, so the modulus drops

sp.log(sp.exp(x))      # log(exp(x)) — x could be complex
sp.log(sp.exp(r))      # r
```

Note the middle case: `real=True` is **not** enough to give `r`, because a real
number can still be negative. You get `Abs(r)`, which is the correct answer. Only
`positive=True` licenses dropping the modulus.

Common keywords: `real`, `positive`, `negative`, `nonnegative`, `nonpositive`,
`nonzero`, `integer`, `rational`, `irrational`, `complex`, `finite`, `even`,
`odd`, `prime`.

Query with `.is_*`, which returns `True`, `False`, or `None` (unknown):

```python
p.is_positive     # True
x.is_positive     # None   ← unknown, NOT False
r.is_complex      # True
```

**`None` is not `False`.** `if not x.is_positive:` is a bug when the answer is
merely unknown; write `if x.is_positive is False:` when you mean it.

### The new assumptions system (`Q` / `ask` / `refine`)

A second, separate system attaches facts to a context rather than a symbol. Use
it when the fact is not known at symbol-creation time:

```python
sp.ask(sp.Q.positive(x + 1), sp.Q.positive(x))    # True
sp.refine(sp.sqrt(x**2), sp.Q.positive(x))        # x
with sp.assuming(sp.Q.positive(x)):
    sp.ask(sp.Q.positive(x**3))                   # True
```

---

## Return types: bare expressions and unevaluated objects

**SymPy returns the mathematical object itself.** There is no `.value` wrapper,
no derivation log, no verification status, and no proof certificate — a returned
expression carries no evidence about how it was obtained or whether anything
checked it.

```python
sp.diff(sp.sin(x), x)          # cos(x)   — an Expr, use it directly
sp.integrate(x**2, x)          # x**3/3
sp.limit(sp.sin(x)/x, x, 0)    # 1
```

The structural wrinkle is that operations which **fail** usually return an
*unevaluated object* rather than raising:

| Class | Produced by | Meaning |
|---|---|---|
| `Integral` | `integrate` | Could not find an antiderivative |
| `Sum` | `summation` | Could not find a closed form |
| `Product` | `product` | Could not find a closed form |
| `Derivative` | `diff` | Cannot differentiate (unknown function) |
| `Limit` | `limit` | Could not determine the limit |

```python
r = sp.integrate(sp.Function("f")(x) * sp.sin(x), x)
r                     # Integral(f(x)*sin(x), x)
r.has(sp.Integral)    # True  ← this is how you detect failure
r.doit()              # try again
```

**Check for these explicitly.** `str()` of an unevaluated `Integral` looks like a
perfectly good answer, and passing one to `float()` raises far downstream from
the actual failure.

### Failure does not always look like failure

`has(Integral)` catches only one failure mode. SymPy also returns answers in
terms of **special functions** — correct, but *not* elementary antiderivatives:

```python
sp.integrate(sp.exp(x**2), x)       # sqrt(pi)*erfi(x)/2   — has(Integral) is False
sp.integrate(sp.exp(x)/x, x)        # Ei(x)                — has(Integral) is False
sp.integrate(sp.sin(x)/x, x)        # Si(x)                — has(Integral) is False
sp.integrate(sp.sqrt(1 + x**3), x)  # gamma/hyper(...)     — has(Integral) is False
```

If the question is "does an **elementary** antiderivative exist", checking
`has(Integral)` answers *yes* for all four, and all four are wrong. Test for
special functions as well:

```python
NONELEMENTARY = (sp.erf, sp.erfi, sp.Ei, sp.Si, sp.Ci, sp.li,
                 sp.fresnels, sp.fresnelc, sp.hyper, sp.meijerg, sp.uppergamma)

result = sp.integrate(expr, x)
is_elementary = not result.has(sp.Integral) and not result.has(*NONELEMENTARY)
```

There is also a dedicated Risch implementation that signals instead of falling
back — the most reliable elementary test available:

```python
from sympy.integrals.risch import NonElementaryIntegral

r = sp.integrate(sp.exp(x**2), x, risch=True)
isinstance(r, NonElementaryIntegral)     # True
```

---

## Simplification

SymPy has one general simplifier and many targeted ones. The general one is slow
and its result is not canonical — it applies heuristics and keeps whatever comes
out smallest.

```python
sp.simplify(expr)        # general heuristic simplifier — slow, non-canonical
sp.trigsimp(expr)        # trig identities: sin²+cos²=1, double angles
sp.expand_trig(expr)     # opposite direction: expand sin(x+y), sin(2x)
sp.radsimp(expr)         # rationalise denominators containing radicals
sp.powsimp(expr)         # combine powers: x**a * x**b -> x**(a+b)
sp.expand_power_exp(expr)
sp.logcombine(expr)      # log(x) + log(y) -> log(x*y)   (needs assumptions)
sp.expand_log(expr)      # opposite direction
sp.combsimp(expr)        # factorials and binomials
sp.expand(expr)          # multiply everything out
sp.factor(expr)          # factor a polynomial over the rationals
sp.cancel(expr)          # rational function -> canonical p/q, cancels the gcd
sp.together(expr)        # combine over a common denominator
sp.apart(expr, x)        # partial fractions
sp.ratsimp(expr)         # combine into a single rational function
sp.nsimplify(expr)       # float -> plausible exact form
```

**Prefer the targeted simplifier.** `simplify` on a large expression can take
minutes and may not reach the form you want anyway. If you know the structure —
it is trig, it is rational, it is a log identity — call the specific function.

```python
sp.trigsimp(sp.sin(x)**2 + sp.cos(x)**2)     # 1
sp.cancel((x**2 - 1) / (x - 1))              # x + 1
sp.apart(1 / (x**2 - 1), x)                  # -1/(2(x+1)) + 1/(2(x-1))
sp.factor(x**4 - 1)                          # (x-1)(x+1)(x²+1)
sp.expand((x + 1)**3)                        # x³+3x²+3x+1
```

### Simplification is assumption-gated

Several identities are conditional, and SymPy refuses them rather than be wrong.
This is correct behaviour, not a missing feature — do not work around it by
string-rewriting the output:

```python
sp.simplify(sp.sqrt(x**2))                               # sqrt(x**2) — unchanged
sp.simplify(sp.sqrt(sp.Symbol("p", positive=True)**2))   # p

sp.logcombine(sp.log(x) + sp.log(y))                     # unchanged
sp.logcombine(sp.log(x) + sp.log(y), force=True)         # log(x*y)   — UNSAFE
```

`force=True` on `logcombine`, `powsimp`, and `expand_log` applies the identity
**without** checking side conditions. It produces wrong answers on negative or
complex inputs. Declare assumptions instead.

### Deciding whether two expressions are equal

`==` is **structural**, not mathematical:

```python
(x + 1)**2 == x**2 + 2*x + 1                       # False!  different trees
sp.simplify((x + 1)**2 - (x**2 + 2*x + 1)) == 0    # True
sp.expand((x + 1)**2) == x**2 + 2*x + 1            # True
((x + 1)**2).equals(x**2 + 2*x + 1)                # True (symbolic + numeric)
```

The reliable test is `sp.simplify(a - b) == 0`, or `a.equals(b)` which also does
randomised numeric sampling. Never compare two expressions with `==` and
conclude they differ.

---

## Differentiation

```python
sp.diff(sp.sin(x**2), x)          # 2*x*cos(x**2)
sp.diff(x**5, x, 3)               # 60*x**2   — third derivative
sp.diff(x**2 * y, x)              # 2*x*y     — partial
sp.diff(expr, x, 2, y, 1)         # mixed: d³/dx²dy

expr.diff(x)                      # method form, identical
```

### Unevaluated derivatives

```python
d = sp.Derivative(sp.sin(x), x)   # stays unevaluated
d.doit()                          # cos(x)
```

An unknown function differentiates to an unevaluated `Derivative`, which is
correct and expected:

```python
f = sp.Function("f")
sp.diff(f(x)**2, x)               # 2*f(x)*Derivative(f(x), x)
```

### Vector calculus

```python
# Gradient — there is no gradient() helper; build it yourself
grad = [sp.diff(expr, v) for v in (x, y, z)]
grad = sp.Matrix([expr]).jacobian([x, y, z])     # as a row Matrix

# Jacobian of a vector function
F = sp.Matrix([x**2 + y, x * y**2])
J = F.jacobian([x, y])            # Matrix([[2*x, 1], [y**2, 2*x*y]])
J[1, 0]                           # y**2   — (row, col), 0-indexed

# Hessian
sp.hessian(x**2 * y, (x, y))      # Matrix([[2*y, 2*x], [2*x, 0]])

# Implicit differentiation: dy/dx from F(x, y) = 0
sp.idiff(x**2 + y**2 - 1, y, x)   # -x/y
```

There is no forward/reverse-mode automatic differentiation and no traced-function
transform. `diff` is purely symbolic; for numeric gradients, differentiate
symbolically and then `lambdify` the result.

### Recursion depth

`diff` is recursive in Python. Deeply nested expressions raise `RecursionError`
well before they exhaust memory:

```python
e = x
for _ in range(100):
    e = sp.sin(e) + x**2
sp.diff(e, x)                     # RecursionError

import sys
sys.setrecursionlimit(20_000)     # workaround; may still crash at depth
```

If you must handle deep compositions, apply the chain rule iteratively in Python
rather than building one giant expression:

```python
import math
xv, f, df = 0.5, 0.5, 1.0
for _ in range(120):
    f, df = math.sin(f) + xv*xv, math.cos(f)*df + 2*xv
```

---

## Integration

```python
sp.integrate(x**2, x)                     # x**3/3
sp.integrate(sp.sin(x), x)                # -cos(x)
sp.integrate(1 / x, x)                    # log(x)
sp.integrate(1 / (x**2 + 1), x)           # atan(x)

# Definite
sp.integrate(x**2, (x, 0, 1))             # 1/3
sp.integrate(sp.sin(x), (x, 0, sp.pi))    # 2
sp.integrate(sp.exp(-x), (x, 0, sp.oo))   # 1

# Multiple
sp.integrate(x * y, (x, 0, 1), (y, 0, 1)) # 1/4
```

Bounds must be SymPy objects for exactness: `sp.pi`, `sp.oo`, `sp.Rational` —
not `math.pi` or `float("inf")`.

### Algorithm selection

```python
sp.integrate(expr, x, risch=True)     # Risch only; signals NonElementaryIntegral
sp.integrate(expr, x, meijerg=True)   # Meijer G-function method
sp.integrate(expr, x, manual=True)    # human-style rules; may leave it unevaluated
sp.integrate(expr, x, heurisch=True)  # heuristic Risch

from sympy.integrals.manualintegrate import manualintegrate, integral_steps
manualintegrate(x**2, x)              # x**3/3
integral_steps(x**2, x)               # a rule tree explaining the steps
```

`integral_steps` is the closest thing SymPy has to a derivation log, and it
exists only for the `manual` method.

### Divergent and improper integrals

SymPy generally detects a singularity inside the interval and reports it rather
than blindly applying the fundamental theorem:

```python
sp.integrate(1 / x, (x, -1, 1))          # nan  — pole at 0 inside the interval
sp.integrate(1 / x**2, (x, -1, 1))       # oo   — divergent
sp.integrate(1 / (x**2 - 1), (x, 0, 2))  # nan  — pole at x = 1
```

`nan` and `oo` are the signal. **Do not coerce them to a float and report the
result** — check first:

```python
r = sp.integrate(expr, (x, a, b))
if r.has(sp.nan, sp.zoo) or r in (sp.oo, -sp.oo):
    ...   # divergent or undefined
```

Principal values must be requested explicitly, and it is a method on `Integral`,
**not** a keyword argument to `integrate`:

```python
sp.Integral(1 / x, (x, -1, 1)).principal_value()   # 0
# sp.integrate(1/x, (x, -1, 1), principal_value=True)  -> TypeError
```

### Verification

Nothing verifies the answer for you. Differentiate it back:

```python
antideriv = sp.integrate(expr, x)
assert sp.simplify(sp.diff(antideriv, x) - expr) == 0
```

For definite integrals, cross-check numerically:

```python
symbolic = float(sp.integrate(expr, (x, 0, 1)))
numeric = float(sp.Integral(expr, (x, 0, 1)).evalf())   # mpmath quadrature
assert abs(symbolic - numeric) < 1e-9
```

---

## Limits and series

```python
sp.limit(sp.sin(x) / x, x, 0)          # 1
sp.limit(1 / x, x, 0, "+")             # oo    — right-hand limit
sp.limit(1 / x, x, 0, "-")             # -oo
sp.limit((1 + 1/x)**x, x, sp.oo)       # E

sp.Limit(sp.sin(x)/x, x, 0)            # unevaluated
```

The direction argument matters: the default is `"+"`, so a two-sided limit that
does not exist may still return a one-sided value. Compute both sides when it
matters.

```python
sp.series(sp.exp(x), x, 0, 4)          # 1 + x + x²/2 + x³/6 + O(x⁴)
sp.series(sp.log(x), x, 1, 3)          # expansion about x = 1
expr.series(x, 0, 5)                   # method form
expr.nseries(x, 0, 5)                  # faster, weaker O() bookkeeping

s = sp.series(sp.exp(x), x, 0, 4)
s.removeO()                            # drop the O(x⁴) term
```

**The `O()` term is part of the expression.** Feeding a series straight into
`lambdify` or `float()` fails; call `.removeO()` first.

```python
sp.residue(1 / x, x, 0)                # 1
```

---

## Logic and inequalities

SymPy's boolean layer is propositional. Real quantifier elimination is **not**
available: there is no CAD, no decision procedure over the reals, and no
`Forall`/`Exists` solver.

```python
A, B = sp.symbols("A B")
sp.And(A, B), sp.Or(A, B), sp.Not(A), sp.Implies(A, B)
A & B, A | B, ~A                        # operator forms

sp.satisfiable(A & ~A)                  # False
sp.satisfiable(A | B)                   # {A: True, B: False}  — a model
sp.simplify_logic(A & (A | B))          # A
```

For real-domain reasoning use inequality solving and set algebra:

```python
sp.reduce_inequalities(x**2 < 4, x)               # (-2 < x) & (x < 2)
sp.solveset(x**2 - 4 > 0, x, domain=sp.S.Reals)   # Union(...)

sp.Interval(0, 1)                       # [0, 1]
sp.Interval.open(0, 1)                  # (0, 1)
sp.Union(...), sp.Intersection(...), sp.Complement(...)
sp.S.Reals, sp.S.Integers, sp.S.Naturals, sp.EmptySet
```

To decide a universally quantified polynomial statement, solve for
counterexamples and check the solution set is empty:

```python
# Is x**2 >= 0 for all real x?  Look for a real counterexample.
counterexamples = sp.solveset(x**2 < 0, x, domain=sp.S.Reals)
counterexamples == sp.EmptySet          # True -> the statement holds
```

---

## Substitution and pattern matching

```python
expr.subs(x, 2)                          # single substitution
expr.subs({x: 2, y: sp.cos(x)})          # simultaneous (dict)
expr.subs([(x, y), (y, x)])              # sequential (list) — order matters!
expr.xreplace({x: y})                    # exact structural replacement, faster
```

`subs` with a list applies substitutions **in order**, so `[(x, y), (y, x)]` maps
everything to `x`. Use a dict for simultaneous substitution.

### Wildcards and matching

```python
a_ = sp.Wild("a")
b_ = sp.Wild("b")

(sp.sin(x)**2).match(sp.sin(a_)**b_)     # {a_: x, b_: 2}
(x + 3).match(a_ + b_)                   # {a_: 3, b_: x}

c_ = sp.Wild("c", exclude=[0])           # exclude trivial matches

expr.replace(sp.sin, sp.cos)                       # every sin -> cos
expr.replace(lambda e: e.is_Pow, lambda e: e.base) # predicate + transform
expr.rewrite(sp.exp)                               # trig -> exponential form
expr.rewrite(sp.sin)                               # -> sine form
```

`rewrite` is the idiomatic way to change representation:
`sp.tan(x).rewrite(sp.sin)`, `sp.exp(x).rewrite(sp.cos)`,
`sp.gamma(x).rewrite(sp.factorial)`.

---

## Polynomials

`Poly` is an explicit polynomial representation over an explicit domain. It is
much faster than the generic expression tree for polynomial algorithms.

```python
P = sp.Poly(x**3 - 2*x + 1, x)
P.degree()                # 3
P.all_coeffs()            # [1, 0, -2, 1]   — highest power first
P.coeffs()                # non-zero coefficients only
P.eval(2)                 # 5
P.domain                  # ZZ
P.as_expr()               # back to a plain Expr

sp.Poly(x**2 + y, x, y)                     # multivariate
sp.Poly(x**2 + sp.Rational(1,2), x, domain="QQ")
```

### Polynomial algorithms

```python
sp.gcd(x**6 - 1, x**4 - 1)         # x**2 - 1
sp.lcm(x**2 - 1, x - 1)
sp.div(x**3 - 1, x - 1)            # (quotient, remainder)
sp.quo(f, g, x), sp.rem(f, g, x)

sp.factor(x**4 - 1)                # (x-1)*(x+1)*(x**2+1)
sp.factor_list(x**4 - 1)           # (1, [(x-1,1), (x+1,1), (x**2+1,1)])
sp.factor(x**2 + 1, extension=sp.I)          # factor over Q(i)
sp.factor(x**2 - 2, extension=sp.sqrt(2))

sp.resultant(x**2 - 1, x - 1, x)   # 0
sp.discriminant(x**2 + 2*x + 1, x) # 0
sp.subresultants(f, g, x)

sp.roots(sp.Poly(x**2 - 1, x))       # {-1: 1, 1: 1}  — root: multiplicity
sp.real_roots(sp.Poly(x**2 - 2, x))  # [-sqrt(2), sqrt(2)]  — exact
sp.nroots(sp.Poly(x**3 - 2, x))      # numeric roots, all of them
sp.RootOf(x**5 - x - 1, 0)           # indexed exact root of an unsolvable quintic
sp.minimal_polynomial(sp.sqrt(2) + sp.sqrt(3), x)
```

`real_roots` returns exact algebraic numbers (or `RootOf` objects), not isolating
intervals. Call `.evalf()` for numbers.

### Rational functions

```python
sp.cancel((x**2 - 1) / (x - 1))       # x + 1
sp.together(1/x + 1/y)                # (x + y)/(x*y)
sp.apart(1 / (x**2 - 1), x)           # partial fractions
sp.fraction(sp.together(1/x + 1/y))   # (x + y, x*y)  — (numer, denom)
sp.numer(e), sp.denom(e)
```

**`cancel` removes removable singularities.** `cancel((x²-1)/(x-1))` is `x+1`,
which is defined at `x = 1` while the original is not. Substituting `x = 1`
afterwards silently answers a different question. Check the original denominator
before evaluating a cancelled expression at a point.

---

## Solving equations

SymPy has two solver families and they behave differently. Knowing which to use
is the single most important solving skill.

### `solve` — the legacy general solver

```python
sp.solve(x**2 - 5*x + 6, x)                 # [2, 3]            — a list
sp.solve(x**2 - 5*x + 6)                    # infers the symbol
sp.solve([x + y - 2, x - y], [x, y])        # {x: 1, y: 1}      — a dict!
sp.solve([x**2 + y**2 - 2, y - x], [x, y])  # [(-1,-1), (1,1)]  — list of tuples
sp.solve(x**2 - 1, x, dict=True)            # [{x: -1}, {x: 1}] — force dicts
sp.solve(sp.Eq(x**2, 1), x)                 # equations via Eq
```

**`solve`'s return type is not stable.** Depending on input shape it returns a
list of values, a list of tuples, a dict, or a list of dicts. Pass `dict=True`
for a predictable shape.

A bare expression is assumed equal to zero: `solve(x**2 - 1, x)` solves
`x² - 1 = 0`.

### `solveset` — the modern solver

```python
sp.solveset(x**2 - 1, x)                          # {-1, 1}   — a Set
sp.solveset(x**2 + 1, x, domain=sp.S.Reals)       # EmptySet
sp.solveset(x**2 + 1, x)                          # {-I, I}   — complex by default
sp.solveset(sp.sin(x), x, domain=sp.S.Reals)      # ImageSet — infinitely many
```

`solveset` always returns a `Set`, always respects `domain`, and represents
infinite solution families exactly. `solve` cannot do the last one — it returns
only principal solutions of `sin(x) = 0` with no indication that others exist.

**`solve` ignores symbol assumptions for domain purposes.** `solve(x**2 + 1, x)`
returns `[-I, I]` even when `x` was declared real. If the question is about real
solutions, use `solveset(..., domain=sp.S.Reals)`.

```python
sp.linsolve([x + y - 2, x - y], [x, y])            # {(1, 1)}
sp.nonlinsolve([x**2 + y**2 - 2, y - x], [x, y])   # {(-1,-1), (1,1)}
sp.nsolve(sp.cos(x) - x, x, 1.0)                   # 0.739085 — numeric, needs a guess
```

### Extraneous solutions

Squaring or clearing denominators can introduce roots that do not satisfy the
original equation. `solve` mostly filters these, but verify when it matters:

```python
sp.solve(sp.sqrt(x) + 1, x)      # []  — correctly empty
# sqrt(x) = -1 has no solution; squaring by hand would suggest x = 1.

roots = sp.solve(eq, x)
valid = [r for r in roots if sp.simplify(eq.subs(x, r)) == 0]
```

### Gröbner bases

```python
G = sp.groebner([x**2 - 1, y - x], x, y, order="lex")
G.exprs                                       # the basis polynomials
sp.groebner(polys, *gens, order="grevlex")    # usually much faster than lex
```

`order="lex"` gives a triangular basis you can back-substitute but is
exponentially slower; `grevlex` is the default choice for ideal membership.

---

## Numeric evaluation

```python
sp.N(sp.sqrt(2), 30)             # 1.41421356237309504880168872421
sp.sqrt(2).evalf(50)             # 50 significant digits
sp.pi.evalf(1000)                # arbitrary precision via mpmath
(x**2).evalf(subs={x: 3})        # 9.00000000000000
float(sp.Rational(1, 3))         # 0.3333333333333333
```

`evalf` is backed by mpmath and is *adaptive*: it raises working precision
internally to try to deliver the requested number of correct digits. It is the
right tool when float64 would lose precision:

```python
expr = (1 - sp.cos(x)) / x**2
expr.subs(x, sp.Rational(1, 10**8)).evalf(30)   # 0.5 — exact rational input
# Naive float64: (1 - math.cos(1e-8)) / 1e-16 == 0.0  ← catastrophic cancellation
```

There is **no interval or ball arithmetic** in SymPy. `evalf` tracks precision
internally and prefers returning fewer digits to returning wrong ones, but it
does not hand you a certified enclosure. For rigorous bounds, drop to mpmath's
interval context:

```python
from mpmath import iv

iv.dps = 30
enclosure = iv.sin(iv.cos(iv.mpf(1)))
enclosure.a, enclosure.b            # certified lower and upper bounds
```

**Do not `float()` the endpoints of a tight enclosure.** At 30 digits both ends
of the interval above round to the *same* float64, which may sit on the wrong
side of the true value — the enclosure guarantee is destroyed by the conversion.
If you must report float bounds, round outward explicitly:

```python
import math

lo = math.nextafter(float(enclosure.a), -math.inf)
hi = math.nextafter(float(enclosure.b), math.inf)
```

### `lambdify` — compile to a fast numeric function

```python
import numpy as np

f = sp.lambdify(x, x**2 + sp.sin(x), "numpy")
f(3.0)                                  # scalar
f(np.linspace(0, 1, 1_000_000))         # vectorised — do this, never a Python loop

g = sp.lambdify((x, y), x*y, "numpy")   # several arguments
h = sp.lambdify(x, expr, "mpmath")      # arbitrary precision
h = sp.lambdify(x, expr, "math")        # stdlib only
```

`lambdify` generates Python source and `exec`s it: fast to call, comparatively
slow to build. Build once, outside your loop.

**`lambdify` does not check the expression is numeric.** An unevaluated
`Integral`, a leftover `O()` term, or a free symbol you forgot to pass produces a
`NameError` or nonsense at call time, not at build time.

```python
sp.cse([x**2 + sp.sin(x**2)])   # ([(x0, x**2)], [x0 + sin(x0)])
sp.lambdify(x, big_expr, "numpy", cse=True)
```

---

## Matrices

```python
M = sp.Matrix([[x, 1], [0, x]])
sp.eye(3), sp.zeros(2, 3), sp.ones(2, 2), sp.diag(1, 2, 3)

M.shape          # (2, 2)  — a property, not a method
M[0, 1]          # element access, (row, col)
M.row(0), M.col(1)
M.tolist()
```

### Arithmetic

```python
A * B                       # matrix product
A @ B                       # same
A * 3                       # scalar multiply
A**2                        # matrix power
A**-1                       # inverse
A.multiply_elementwise(B)   # Hadamard product
```

### Decompositions and spectral data

```python
M.det()                  # determinant
M.trace()
M.rank()
M.T                      # transpose (property)
M.inv()                  # inverse; raises for singular
M.rref()                 # (rref_matrix, pivot_columns) — a TUPLE
M.nullspace()            # list of basis vectors
M.columnspace(), M.rowspace()

M.eigenvals()            # {eigenvalue: multiplicity}
M.eigenvects()           # [(eigenvalue, multiplicity, [eigenvectors])]
M.diagonalize()          # (P, D); raises MatrixError if defective
M.jordan_form()          # (P, J)
M.charpoly()             # characteristic polynomial as PurePoly
M.berkowitz_charpoly()   # division-free variant
# NB: there is no Matrix.minimal_polynomial(). For the minimal polynomial of an
# algebraic *number* use the free function sp.minimal_polynomial(expr, x).
M.LUdecomposition(), M.QRdecomposition(), M.cholesky()
M.exp()                  # matrix exponential
M.applyfunc(sp.simplify) # simplify every entry
```

`rref()` returns a **tuple**, not a matrix — `M.rref()[0]` is the matrix. This
trips up almost everyone once.

Symbolic matrices work throughout, but entries are not auto-simplified after
products; call `.applyfunc(sp.simplify)` before comparing or printing.

`diagonalize()` raises `MatrixError` on a defective matrix (fewer independent
eigenvectors than the algebraic multiplicity) — catch it rather than assuming
diagonalisability.

### Performance

`Matrix` is dense and pure Python. Determinants and inverses of symbolic matrices
above roughly 6×6 become very slow. Choose the method explicitly for larger work:

```python
M.det(method="berkowitz")     # division-free; better for symbolic entries
M.det(method="lu")            # faster for numeric entries
sp.SparseMatrix(...)          # mostly-zero matrices
sp.ImmutableMatrix(...)       # hashable, usable inside expressions
```

---

## Differential equations

```python
f = sp.Function("f")
ode = sp.Eq(f(x).diff(x) + f(x), 0)

sp.dsolve(ode, f(x))                     # Eq(f(x), C1*exp(-x))
sp.classify_ode(ode, f(x))               # ('separable', '1st_exact', ...)
sp.dsolve(ode, f(x), hint="separable")   # force a method

# Initial conditions
sp.dsolve(ode, f(x), ics={f(0): 1})      # Eq(f(x), exp(-x))

# Second order
ode2 = sp.Eq(f(x).diff(x, 2) + f(x), 0)
sp.dsolve(ode2, f(x))                    # C1*sin(x) + C2*cos(x)

# Systems
g = sp.Function("g")
sp.dsolve([sp.Eq(f(x).diff(x), g(x)), sp.Eq(g(x).diff(x), -f(x))])

# Verify
sol = sp.dsolve(ode, f(x))
sp.checkodesol(ode, sol)                 # (True, 0)
```

`dsolve` returns an `Eq`, not a bare expression — use `sol.rhs` for the
right-hand side. Solutions carry integration constants `C1`, `C2`, … as free
symbols.

There is no DAE index reduction, no Pantelides algorithm, no acausal component
modelling, and no numeric ODE integrator. For numeric solutions, `lambdify` the
right-hand side and hand it to `scipy.integrate.solve_ivp`. PDE support is
limited to first-order linear equations via `sp.pdsolve`.

---

## Integral transforms

```python
t, s = sp.symbols("t s")

sp.laplace_transform(sp.sin(t), t, s)
# (1/(s**2 + 1), 0, True)   ← a 3-TUPLE: (F(s), convergence_plane, conditions)

sp.laplace_transform(sp.sin(t), t, s, noconds=True)
# 1/(s**2 + 1)              ← just the transform

sp.inverse_laplace_transform(1/(s**2 + 1), s, t)   # sin(t)*Heaviside(t)
sp.fourier_transform(sp.exp(-t**2), t, s)
sp.inverse_fourier_transform(...)
sp.mellin_transform(...), sp.inverse_mellin_transform(...)
sp.sine_transform(...), sp.cosine_transform(...)
```

**`laplace_transform` returns a tuple unless you pass `noconds=True`.** Feeding
that tuple into further arithmetic is a common and confusing bug.

The inverse Laplace transform generally carries a `Heaviside(t)` factor, since it
is the unilateral transform. There is no Z-transform in SymPy.

---

## Summation and products

```python
sp.summation(k, (k, 1, n))              # n²/2 + n/2
sp.summation(1/n**2, (n, 1, sp.oo))     # pi**2/6
sp.Sum(k**2, (k, 1, n)).doit()          # method form
sp.Sum(1/k, (k, 1, n))                  # unevaluated — no closed form

sp.product(k, (k, 1, 5))                # 120
sp.Product(k, (k, 1, n)).doit()         # factorial(n)

sp.Sum(1/k**2, (k, 1, sp.oo)).is_convergent()   # True
sp.Sum(1/k, (k, 1, sp.oo)).is_convergent()      # False
```

SymPy handles Gosper-summable terms, hypergeometric sums, and many classical
series including zeta-valued ones. An unevaluated `Sum` is the failure signal.

### Number theory

```python
sp.isprime(97)                # True
sp.factorint(360)             # {2: 3, 3: 2, 5: 1}
sp.primerange(10, 30)
sp.nextprime(100), sp.prevprime(100)
sp.igcd(12, 18), sp.ilcm(4, 6)
sp.totient(12), sp.divisors(28), sp.divisor_count(28)
sp.mod_inverse(3, 7)
sp.discrete_log(41, 15, 7)
sp.binomial(10, 3), sp.factorial(10)
sp.continued_fraction(sp.Rational(415, 93))
sp.diophantine(x**2 + y**2 - 25)
```

---

## Parsing and printing

```python
sp.sympify("x**2 + 1")                         # str -> Expr
sp.parse_expr("x^2", transformations="all")    # ^ as power, implicit multiplication
sp.srepr(x + 1)                                # exact reproducible repr

sp.pprint(x**2 / 3)                     # unicode 2-D output
sp.latex(sp.Integral(x**2, x))          # LaTeX source
sp.mathematica_code(expr)
sp.ccode(x**2)                          # "pow(x, 2)"
sp.fcode(expr), sp.jscode(expr), sp.rust_code(expr), sp.julia_code(expr)
sp.python(expr)                         # Python source rebuilding the expression

sp.init_printing()                      # pretty output in a REPL/notebook
```

**Never build expressions with `eval()` on user text.** `sympify` calls `eval`
internally on strings; use `parse_expr` with explicit transformations, or build
expressions programmatically, when the input is untrusted.

### Code generation

```python
from sympy.utilities.codegen import codegen
[(name, code), (hname, header)] = codegen(("f", x**2 + y), "C99", "f")

from sympy.utilities.autowrap import autowrap
fast = autowrap(x**2, backend="cython")   # needs a compiler
```

There is no Lean, StableHLO, or proof-certificate emission.

---

## Plotting

```python
sp.plot(sp.sin(x), (x, -sp.pi, sp.pi))              # 2-D curve
sp.plot(sp.sin(x), sp.cos(x), (x, 0, 2*sp.pi))      # several curves
sp.plot_parametric(sp.cos(x), sp.sin(x), (x, 0, 2*sp.pi))
sp.plot_implicit(sp.Eq(x**2 + y**2, 1), (x, -2, 2), (y, -2, 2))

from sympy.plotting import plot3d
plot3d(x**2 + y**2, (x, -2, 2), (y, -2, 2))

p = sp.plot(sp.sin(x), show=False)
p.save("out.png")
```

Plotting requires `matplotlib`. There is no dependency-free SVG backend and no
expression-graph visualiser.

---

## Error handling

SymPy raises ordinary Python exceptions. There are **no stable error codes, no
`.remediation` hints, and no source spans** — you match on exception type and,
where necessary, message text.

| Exception | Raised by |
|---|---|
| `sympy.SympifyError` | `sympify` on unparseable input |
| `sympy.PolynomialError` | `Poly` on a non-polynomial expression |
| `sympy.matrices.exceptions.MatrixError` | singular inverse, non-diagonalisable |
| `sympy.matrices.exceptions.ShapeError` | dimension mismatch (subclass of `MatrixError`) |
| `sympy.matrices.exceptions.NonSquareMatrixError` | `det`/`inv` on a non-square matrix |
| `NotImplementedError` | algorithm not implemented for this input |
| `ValueError` / `TypeError` | generic bad arguments |
| `RecursionError` | expression too deeply nested |

Matrix exceptions are not re-exported at the top level — `sp.MatrixError` does
not exist. Import them explicitly:

```python
from sympy.matrices.exceptions import MatrixError, ShapeError

try:
    sp.Poly(sp.sin(x), x)
except sp.PolynomialError as e:
    print(e)     # "sin(x) contains an element of the set of generators."

try:
    sp.Matrix([[x, 1], [0, x]]).diagonalize()
except MatrixError as e:
    print(e)     # "Matrix is not diagonalizable"
```

The larger practical issue is that many failures are **not** exceptions — they
are unevaluated `Integral`/`Sum`/`Limit` objects, `nan`, `zoo` (complex
infinity), or `oo`. A defensive result check therefore looks like:

```python
def check(result):
    if result.has(sp.Integral, sp.Sum, sp.Limit, sp.Derivative):
        raise RuntimeError("SymPy could not evaluate this")
    if result.has(sp.nan, sp.zoo):
        raise RuntimeError("undefined")
    if result in (sp.oo, -sp.oo):
        raise RuntimeError("divergent")
    return result
```

---

## Available math functions

`sin`, `cos`, `tan`, `cot`, `sec`, `csc`,
`asin`, `acos`, `atan`, `atan2`, `acot`, `asec`, `acsc`,
`sinh`, `cosh`, `tanh`, `coth`, `asinh`, `acosh`, `atanh`, `acoth`,
`exp`, `log`, `sqrt`, `cbrt`, `root`, `Abs`, `sign`, `arg`, `conjugate`, `re`, `im`,
`floor`, `ceiling`, `frac`, `Min`, `Max`, `Piecewise`,
`gamma`, `loggamma`, `digamma`, `polygamma`, `beta`, `factorial`, `binomial`,
`erf`, `erfc`, `erfi`, `Ei`, `Si`, `Ci`, `li`, `zeta`, `polylog`,
`besselj`, `bessely`, `besseli`, `besselk`, `airyai`, `airybi`,
`elliptic_k`, `elliptic_e`, `elliptic_f`, `elliptic_pi`,
`LambertW`, `hyper`, `meijerg`, `fresnels`, `fresnelc`,
`Heaviside`, `DiracDelta`

Constants: `pi`, `E`, `I`, `oo` (infinity), `zoo` (complex infinity), `nan`,
`EulerGamma`, `GoldenRatio`, `S.Half`, `S.One`, `S.Zero`.

```python
# Piecewise: conditions are SymPy relationals; True is the fallback branch
pw = sp.Piecewise((x, x > 0), (-x, True))
pw.subs(x, -3)                                # 3
```

---

## Performance notes

SymPy is pure Python and its performance profile differs sharply from a compiled
CAS:

1. **`simplify` is the usual bottleneck.** It tries many strategies. Use a
   targeted simplifier whenever you know the structure.
2. **Use `Poly` for polynomial algorithms.** Generic `Expr` manipulation carries
   large constant factors.
3. **Install `gmpy2`.** Large-coefficient work gets several times faster with no
   code change.
4. **Never evaluate in a Python loop.** `lambdify` once, then call on a NumPy
   array.
5. **Watch recursion depth.** Deeply nested expressions raise `RecursionError`
   in `diff`, `subs`, and printing. There is no iterative fallback.
6. **`cse` before `lambdify`** for large expressions with shared subterms.
7. **Clear the cache** if memory grows without bound in a long-running process:
   `from sympy.core.cache import clear_cache; clear_cache()`. There is no
   `sp.cache` attribute on the top-level module.

---

## Versioning and stability

SymPy's public surface is large and loosely specified. The behaviour of
`simplify`, and the exact form returned by `solve` and `integrate`, changes
between minor releases. Do not assert on the structural form of a simplified
result; assert mathematical equivalence:

```python
# Fragile — breaks when simplify changes strategy
assert str(sp.simplify(e)) == "x + 1"

# Robust
assert sp.simplify(e - (x + 1)) == 0
```

Deprecations are announced with `SymPyDeprecationWarning` and typically removed
after two minor releases.

---

## Key rules for agents

1. **Raw Python ints work.** `x**2 + 1` is valid; no pool or interning step.
2. **Use `sp.Rational(p, q)`, never `1/3`.** A Python float silently destroys
   exactness for the rest of the computation.
3. **`==` is structural, not mathematical.** Use `sp.simplify(a - b) == 0` or
   `a.equals(b)` to compare expressions.
4. **Integration failure is not an exception.** Check `result.has(sp.Integral)`,
   and separately check for `erf`/`Ei`/`Si`/`hyper` if the question is whether an
   *elementary* antiderivative exists.
5. **Check for `nan`, `zoo`, and `oo`** before converting a definite integral or
   limit to a float. A divergent integral returns `oo`, not an error.
6. **`solve` ignores domain; `solveset` respects it.** For real solutions use
   `sp.solveset(eq, x, domain=sp.S.Reals)`. `solve(x**2+1, x)` returns `[-I, I]`
   even for a real symbol.
7. **`solve`'s return shape varies.** Pass `dict=True` for a predictable
   `list[dict]`.
8. **Declare assumptions rather than passing `force=True`.** `force` skips the
   side-condition check and produces wrong results for negative or complex input.
9. **`cancel` changes the domain of a function.** `(x²-1)/(x-1)` becomes `x+1`,
   which is defined at `x=1` while the original is not.
10. **`rref()` and `laplace_transform()` return tuples**, not a matrix and an
    expression. Use `M.rref()[0]` and `noconds=True`.
11. **Strip `O()` from a series** with `.removeO()` before numeric use.
12. **`lambdify` once, outside the loop**, then call it on a NumPy array.
13. **`.is_positive` can be `None`.** Unknown is not False; test with `is False`
    when you mean it.
14. **Deep expressions raise `RecursionError`.** Apply the chain rule iteratively
    rather than building a deeply nested tree.
15. **Verify your own results.** Nothing in SymPy checks an answer for you:
    differentiate an antiderivative back, or cross-check a definite integral
    against `.evalf()` quadrature.
