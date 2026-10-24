# Quickstart

One problem, worked end to end, with every output on this page produced by
running the code. If you want the full install matrix — optional wheels, JIT,
GPU, building from source — see [Getting started](./getting-started.md).

```bash
pip install alkahest
```

## A worked problem

Every expression belongs to an `ExprPool`. The pool owns the nodes; you make
symbols and integers from it, and Python's operators build trees.

```python
import alkahest as ak
from alkahest import ExprPool, diff, integrate, compile_expr, exp, sin

pool = ExprPool()
x = pool.symbol("x")

print(x * exp(x))
# => (x * exp(x))
```

Integrate it:

```python
result = integrate(x * exp(x), x)
print(result.value)
# => (exp(x) * (x + -1))
```

That is `(x − 1)eˣ`, which is right. But you should not take that on faith, and
the library does not ask you to.

## Check it yourself

An antiderivative is checkable in one step — differentiate it back:

```python
print(diff(result.value, x).value)
# => (exp(x) + (exp(x) * (x + -1)))
```

`eˣ + (x−1)eˣ = x·eˣ`, the integrand. Note that `simplify` will **not** collapse
that sum for you — it is a rule engine, not a normal form, and this combination
is not one of its rules. Reducing an expression to a recognisable form is often
your job, not the library's.

## Why it believes the answer

Two things come back with every result. The derivation log says what was done:

```python
print(result.steps[0]["rule"])
# => int_x_exp
```

And `verification` says how far the answer was checked:

```python
print(result.verification["status"])
# => numerically_checked
print(result.verification["evidence"])
# => antiderivative_numeric_samples
```

`numerically_checked` means alkahest differentiated its own answer and sampled
it against the integrand. Other statuses — `exactly_verified`,
`certificate_available`, `externally_verified` — mean stronger things; see
[Derivation logs](./derivations.md) and [Lean certificates](./lean-certs.md).
A status is a claim about evidence, so read it before trusting a result that
matters.

## The part that actually matters

Most computer algebra systems will answer this:

```text
∫₋₁¹ x⁻² dx
```

with `−2`. The power rule gives `[−x⁻¹]₋₁¹ = −1 − 1 = −2`, the arithmetic is
clean, and the answer is wrong — the integrand has a pole at `0` and the
integral diverges. A wrong answer that looks right is worse than no answer,
because nothing downstream can tell.

Alkahest refuses:

```python
try:
    integrate(x ** pool.integer(-2), x, pool.integer(-1), pool.integer(1))
except ak.IntegrationError as e:
    print(e.code)
    # => E-INT-001
    print("pole at x = 0" in str(e))
    # => True
    print("does not converge" in str(e))
    # => True
```

Every error carries a stable `.code` you can branch on, so an agent loop can
tell "this does not converge" from "this is not implemented yet" without
matching English.

A refusal is only worth anything if the library still answers the neighbouring
question it *can* answer. Move the interval off the pole and it does:

```python
print(integrate(x ** pool.integer(-2), x, pool.integer(1), pool.integer(2)).value)
# => 1/2
```

Exactly `1/2`, not `0.5` — the result stays rational.

This is the property the library is built around, and it is measured rather
than asserted: `tests/silent_errors/` is a corpus of traps like the one above,
each paired with a convergent control, run on every pull request. See
[Error handling](./errors.md) for the full code list.

## Compiling for speed

Symbolic work is for deriving; once you have an expression you want numbers out
of it fast.

```python
f = compile_expr(sin(x) * x**2, [x])
print(f([2.0]))
# => 3.637189707302727
```

On a default wheel this warns and falls back to a tree-walking interpreter,
which is correct but slow. `ak.capabilities()["jit"]` tells you which you have;
[Code generation](./codegen.md) covers the `+jit` wheels and the Cranelift
backend.

## Where to go next

| You want | Page |
|---|---|
| The install matrix, optional features, building from source | [Getting started](./getting-started.md) |
| Every function, with signatures | [Python API reference](./python-api.md) |
| Calculus in depth — limits, series, residues | [Calculus](./calculus.md) |
| Why `simplify` is several functions and not one | [Simplification](./simplification.md) |
| Distributions, expectations, entropy | [Probability and information theory](./probability.md) |
| Machine-readable results for an agent loop | [Derivation logs](./derivations.md) |

> The code on this page is executed by `tests/test_docs_quickstart.py`, which
> checks every `# =>` against real output. An earlier README quickstart shipped
> a comment claiming the derivative of `sin(x²)` when the code differentiated
> `sin(x²+1)`, because nothing ran it. Doc examples rot silently unless
> something executes them.
