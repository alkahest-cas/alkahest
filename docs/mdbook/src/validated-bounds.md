# Rigorous global bounds (Taylor models)

[Ball arithmetic](./ball-arithmetic.md) gives rigorous **pointwise** enclosures:
evaluate `f` at a ball and the true value is inside the ball you get back. A
research loop routinely needs rigorous **global** statements instead:

- the maximum of `f` on `[a,b]×[c,d]` is at most `M`;
- `∫_a^b f dx` lies in `[I₁, I₂]`;
- `f` has no root anywhere in this box.

These turn a numeric observation into a theorem, which is exactly the step that
takes a candidate from "survived the sweep" to "established".

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

r = ak.bound_on_box(x * (pool.integer(1) - x), [(x, 0.0, 1.0)])
r.lower, r.upper        # encloses the true range [0, 1/4]
r.budget_exhausted      # False — converged within the work budget

ak.verified_integral(ak.sin(x), x, 0.0, 3.14159)   # Enclosure containing ~2
ak.verified_no_roots(x*x + pool.integer(1), [(x, -5.0, 5.0)])   # "true"
ak.verified_sign(ak.exp(x), [(x, -5.0, 5.0)], "positive")       # "true"
```

## The soundness contract

**Every returned enclosure is a rigorous outer bound.** Rounding is outward at
every step, and any operation whose remainder cannot be bounded rigorously
refuses rather than returning something plausible.

The consequence worth internalising: an enclosure may be **wide**, but it is
never **wrong**. A wide-but-true bound is a fine answer for a loop — it just
means "not settled yet". A tight-but-false one is a poisoned lemma that
everything downstream inherits.

Running out of work budget is therefore *not* an error. The enclosure is
returned anyway, with `budget_exhausted = True`, still sound.

## Why not just use interval arithmetic

Naive interval evaluation is rigorous but suffers the **dependency problem**:
it forgets that the two `x`s in `x - x` are the same number.

| Expression, box | Naive intervals | Taylor model |
|---|---|---|
| `x - x`, `x ∈ [-1,1]` | `[-2, 2]` | `{0}` |
| `x(1-x)`, `x ∈ [0,1]` | `[0, 1]` | `[0, 1/4]` |

A Taylor model carries a polynomial in the box's normalised coordinates plus a
rigorously enclosing remainder interval, so cancellation happens *symbolically*
in the polynomial part and only the genuinely uncertain residue stays in the
interval.

## Branch-and-bound

`bound_on_box` runs Moore–Skelboe branch-and-bound: one pass for the minimum,
one for the maximum. Each keeps a rigorous bound on the extremum and **prunes**
sub-boxes whose enclosure proves they cannot contain it, so work concentrates
where the extremum actually is.

This pruning is not an optimisation detail, it is what makes the module usable.
A stopping rule that instead required every sub-box's enclosure to be narrower
than `tol` would be demanding a pointwise-tight model *everywhere*, which no
finite budget achieves for a function with a wide range — `exp` on `[-5,5]`
would exhaust the budget and still return an enclosure loose enough to straddle
zero, so even `exp > 0` would come back undecided.

## Three-valued predicates

`verified_no_roots` and `verified_sign` return one of `"true"`, `"false"`,
`"undecided"`. The third is never collapsed into the other two.

| Verdict | Meaning |
|---|---|
| `"true"` | Certified: the property holds everywhere on the box |
| `"false"` | Certified: it fails — proved by the enclosure, or by a rigorously evaluated witness point |
| `"undecided"` | Neither could be established within the budget and precision |

A sign predicate is a universally quantified claim, so a single point where it
provably fails disproves it. `verified_sign` uses that: it evaluates the
expression rigorously at the box centre, the per-axis endpoints and (in low
dimension) the corners, and reports `"false"` when one of those point
enclosures lies strictly on the wrong side. Without it, `x > 0` on `[-1,1]` —
plainly false — could only ever be `"undecided"`, since the range enclosure
straddles zero by construction.

## Refusals

| Code | Meaning |
|---|---|
| `E-VALIDATED-001` | No rigorous Taylor model rule for some primitive in the expression |
| `E-VALIDATED-002` | A free symbol has no interval in the box |
| `E-VALIDATED-003` | A singularity or branch cut inside the box (e.g. `1/x` over a box containing 0) |
| `E-VALIDATED-004` | An enclosure overflowed to infinity |
| `E-VALIDATED-005` | Malformed request (empty box, inverted interval, bad order) |

For `E-VALIDATED-003` the search first tries bisecting *away* from the trouble,
since a domain violation on a wide box is often a boundary effect rather than a
genuine interior pole. Only after the box has been bisected far enough for that
explanation to be exhausted does it refuse — which is the right answer for a
real interior singularity, where the range is not a bounded interval at all.

## Relation to the rest of the stack

This is the slow, certifying half of *falsify fast, certify slow*:

| | JIT / `numpy_eval` | Ball arithmetic | Validated bounds |
|---|---|---|---|
| Answers | Approximate values, fast | Rigorous at a point | Rigorous over a region |
| Cost | ~µs | ~ms | seconds, adaptive |
| Use | Kill 99% of candidates | Distinguish near-miss from hit | Promote a survivor to a theorem |

`verified_integral` complements the symbolic `integrate_definite`: use the
symbolic path when you want a closed form, and this one when you want a
guaranteed numeric interval — including for integrands with no elementary
antiderivative.

## Scope of this release

Shipped: Taylor model arithmetic over a box (arithmetic, powers, division, and
the elementary functions with rigorous remainders), range enclosure by
branch-and-bound, verified 1-D definite integrals, root absence and sign
predicates.

Not shipped: multivariate verified quadrature (`verified_integral` is 1-D),
improper or singular integrals, and Taylor-model-based ODE enclosures.
