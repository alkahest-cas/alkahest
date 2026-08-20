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

### Proving a root *exists*

`verified_no_roots` returns `"false"` only with a proof in hand, and the proof
is the intermediate value theorem. The full-box enclosure succeeding is already
a continuity certificate — a Taylor model is only ever built where every
elementary step stayed strictly inside its domain — and a box is convex. So if
two points of the box can be found where `f` is *rigorously proven* to have
opposite signs, the segment between them stays in the box and `f` vanishes
somewhere on it.

The two points do not have to be the box's own endpoints, and that is what makes
the test usable: the search subdivides the box, records the sign of any sub-box
whose enclosure has a determined one, and samples the centres of the rest.

| Box | Roots inside | Endpoint signs | Verdict |
|---|---|---|---|
| `x²−2` on `[-2,0]` | 1 | + → − | `"false"` |
| `x²−2` on `[-2,2]` | 2 | + → + | `"false"` |
| `x²−2` on `[-10,10]` | 2 | + → + | `"false"` |
| `(x²−2)(x²+1)` on `[-2,2]` | 2 | + → + | `"false"` |
| `x−y` on `[-1,1]²` | a whole line | — | `"false"` |
| `(x−1)²` on `[0,2]` | 1 (double) | + → + | `"undecided"` |

The last row is the honest limit. A double root never changes sign, so no
witness pair exists; `"undecided"` is the answer, and it is not upgraded to
`"false"` on the strength of an enclosure that merely touches zero.

### Inequalities that are tight at an endpoint

The interesting inequalities are usually the sharp ones, and sharp means the
margin goes to zero somewhere. Subdivision alone cannot certify those: where the
margin vanishes, every enclosure of the range straddles zero however fine the
boxes get.

Two separate things are done about it. `tol` is an **absolute** width, so it is
the wrong stopping rule for a sign question — an expression whose minimum is
`10⁻¹³` meets a `1e-9` tolerance while its enclosure still straddles zero.
`verified_sign` therefore re-runs the search with the sign itself as the goal,
refining while the bound straddles zero rather than to a fixed width. And where
the margin genuinely reaches zero, the box is **split**: a collar `[a, a+δ]` at
the endpoint is handled by a truncated Taylor expansion with a proven Lagrange
remainder, the rest by ordinary branch-and-bound. The pieces are closed and
share the join point, so their union is the original box.

```python
x = pool.symbol("x")
# Cusa–Huygens, denominator cleared: x(2 + cos x) − 3 sin x ≥ 0, tight at x = 0
f = x * (pool.integer(2) + ak.cos(x)) - pool.integer(3) * ak.sin(x)
ak.verified_sign(f, [(x, 0.0, 1.5)], "nonnegative")   # "true"
```

Mitrinović–Adamović, Wilker, Huygens and Jordan's inequality behave the same
way. The remainder is *proven*, not assumed: a Taylor coefficient counts as zero
only when substitution and `simplify` land on a literal integer `0` — no numeric
enclosure can prove a value is zero — and the tail is bounded by
`sup|g⁽ᵐ⁾|/m!` enclosed over the whole collar, with analyticity certified by
requiring every derivative up to `g⁽ᵐ⁾` to enclose successfully there.

The limits are worth knowing:

| Case | Verdict | Why |
|---|---|---|
| tight at an endpoint of the box | `"true"` | the expansion applies there |
| leading coefficient proven negative | `"false"` | `g < 0` just inside the endpoint |
| `"positive"` where `g` provably vanishes | `"false"` | a strict claim fails at that point |
| tight in the **interior** | `"undecided"` | the expansion does not apply |

`(x − 7/10)²(x + 1)` on `[0, 3/2]` is non-negative and touches zero in the
middle; it stays `"undecided"` rather than being upgraded on the strength of an
enclosure that merely touches zero.

### `"undecided"` is not monotone in the box

`"true"` on a box *logically* implies `"true"` on every sub-box, but this is a
search and the search does not have that property. The collar is planted at the
box's own endpoint, and its strength comes from `simplify` proving the low-order
Taylor coefficients there exactly zero. Back the box off the tight point by a
whisker and nothing vanishes exactly any more: the leading coefficient becomes
`g(δ)`, a number that has to beat both the cancellation noise of evaluating `g`
near a zero of order `j` and the linear term of the tail bound. For Cusa–Huygens
(`g ~ x⁵/60`) that failed for every left endpoint between about `10⁻³⁰⁰` and
`10⁻⁹`, while `[0, π/2]` and `[10⁻⁶, π/2]` both came back `"true"` — a dead band
in the middle of two `"true"` regions.

An `"undecided"` box that stops short of `x = 0` is therefore retried with the
collar planted at `0` — `"true"` on the larger box implies `"true"` on the
box asked about, so the retry is sound, and only `"true"` is taken from it. That
closes the band for the classical inequalities, all of which are tight at `0`.
A statement tight at some *other* point still shows the effect: if `[a, b]`
comes back `"undecided"`, try the larger box whose endpoint is the point the
inequality is tight at before concluding it is out of reach.

## Which functions are covered — ask before you build the workload

Taylor models reach the **elementary fragment**: `exp`, `log`, `sqrt`, `sin`,
`cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `asinh`,
`acosh`, `atanh`, `abs`, plus arithmetic and integer/rational powers — and,
since 3.9.0, `erf`, `erfc`, `bessel_j0`, `bessel_j1`, `digamma`, `gamma` and
`lambert_w`. Outside it are the elliptic integrals, `floor`, `ceil` and `sign`,
and so is any two-argument function such as `atan2`.

The three inverse hyperbolics carry the domain restriction their branch has:
`acosh` needs the whole box strictly above `1` and `atanh` needs it strictly
inside `(-1, 1)`. A box that only *touches* the boundary is refused with
`E-VALIDATED-003`, because the derivative is unbounded there and no Taylor
remainder exists — that is a statement about the box, not about coverage, so
`bounds_supported` still answers `True`.

That boundary is queryable, so a search loop can choose a certifiable route
instead of discovering it by hitting `E-VALIDATED-001`:

```python
ak.bounds_supported(ak.sin(x) * ak.exp(x))     # truthy
ak.bounds_supported(ak.bessel_j0(x))           # truthy since 3.9.0
answer = ak.bounds_supported(ak.floor(x))
bool(answer), answer.functions                  # (False, ['floor'])
answer.blocker                                  # "function `floor`"

# Per primitive, in the agent contract:
{row["name"] for row in ak.capabilities()["primitives"] if row["taylor_model"]}
```

**`numeric_ball` is not this flag.** It reports pointwise ball arithmetic,
which `floor` and `ceil` have; a Taylor model additionally needs a rule with a
rigorous Lagrange remainder, which a step function cannot have. Both bits are
honest — they answer different questions. `taylor_model` and
`bounds_supported` are derived by *running* the Taylor evaluator, not from a
maintained list, so neither can drift from what `bound_on_box` accepts.

A `True` answer means "will not be refused with `E-VALIDATED-001`". It is not
a promise of success: a covered function can still hit a domain violation or
an infinite enclosure on a *particular* box, which is a property of the box
and not of the expression.

## Refusals

| Code | Meaning |
|---|---|
| `E-VALIDATED-001` | No rigorous Taylor model rule for some primitive in the expression (ask `bounds_supported` first — see above) |
| `E-VALIDATED-002` | A free symbol has no interval in the box |
| `E-VALIDATED-003` | A singularity or branch cut inside the box (e.g. `1/x` over a box containing 0) |
| `E-VALIDATED-004` | An enclosure overflowed to infinity |
| `E-VALIDATED-005` | Malformed request (empty box, inverted interval, bad order) |

For `E-VALIDATED-003` the search first tries bisecting *away* from the trouble,
since a domain violation on a wide box is often a boundary effect rather than a
genuine interior pole. Only after the box has been bisected far enough for that
explanation to be exhausted does it refuse — which is the right answer for a
real interior singularity, where the range is not a bounded interval at all.

## Removable singularities in `verified_integral`

`∫₀¹ ln(1+x)/x dx = π²/12` has nothing singular about it — only the *expression*
is singular at `x = 0`, and the integrand extends continuously to 1 there. A
Taylor model still refuses, because the reciprocal's enclosure contains zero.

`verified_integral` recognises this shape. If the integrand splits as `N(x)/D(x)`
and there is a point `p` of the offending sub-interval at which `N` and `D` both
vanish, it enclosures that piece with **Cauchy's mean value theorem** instead:

```text
N(p) = D(p) = 0,  D' ≠ 0 on J   ⟹   ∀ x ∈ J\{p} :  N(x)/D(x) = N'(ξ)/D'(ξ)  for some ξ ∈ J
                                ⟹   ∫_J N/D dx ∈ |J| · range(N'/D' on J)
```

so the piece is bounded by an enclosure of `N'/D'`, which is perfectly regular.
The number returned is the integral of the continuous extension.

When `D'` vanishes at `p` as well, the step is **repeated** — up to a small
fixed depth — provided `N'` vanishes there too. That is what covers a
higher-order removable singularity such as `(1 − cos x)/x²`, whose value at
`x = 0` is `1/2` but whose denominator has a double zero: only `D'' = 2` is
bounded away from zero, and `N''/D'' = cos(x)/2` is the quotient that gets
enclosed.

Three guards keep this from swallowing a genuine pole:

- `N⁽ᵏ⁾(p) = 0` and `D⁽ᵏ⁾(p) = 0` are established **symbolically** at every
  level the descent passes (substitute the exact rational `p`, simplify,
  require a literal zero). No numeric enclosure can prove a value is exactly
  zero, so none is asked to.
- Some `D⁽ᵈ⁾` must be *certified non-vanishing* on the sub-interval. That is
  what fails for `sin(x)/x²`: `D' = 2x` vanishes at `0` but `N' = cos x` does
  not, so the descent stops with nothing proven — as it must, since the
  integral does not converge.
- `N⁽ᵏ⁾` and `D⁽ᵏ⁾` must each have a successful enclosure over the whole
  sub-interval, which certifies they are analytic — and hence that the symbolic
  derivatives really are their derivatives.

```python
ak.verified_integral(ak.log(pool.integer(1) + x) / x, x, 0.0, 1.0)  # ≈ π²/12
ak.verified_integral(ak.sin(x) / x, x, -1.0, 1.0)                   # ≈ 1.8921661
ak.verified_integral(pool.integer(1) / x, x, -1.0, 1.0)             # refuses: N(0) ≠ 0
```

## Bounded integrands whose Taylor model runs out of domain

`asin` on `[0, 1]` is bounded and continuous on the closed interval —
`asin(1) = π/2` — but the Taylor rule for `asin` needs a bound on
`1/√(1−x²)`, and there is none at `x = 1`. Bisecting does not help: every
sub-interval that still touches `1` refuses, however narrow. The same happens
for `√x` and `xˣ` at `x = 0`.

Those panels are closed with a **`width × range`** bound instead. A range needs
no derivative, so it survives the domain boundary, and it is computed by
directed-rounding interval arithmetic over the extended reals — which is what
lets `log([0, h]) = [−∞, log h]` be composed with the factor that tames it, so
`xˣ = exp(x·log x) ∈ [0, 1]` comes out bounded. By the time a panel needs this
it has been bisected down to a width of order `2⁻⁶⁰` of the interval, so the
crude bound costs essentially nothing in accuracy.

The soundness argument is that an integrand which is genuinely unbounded on the
panel has no bounded range either, so the fallback still refuses on it. Nothing
is clamped into a domain it is not proven to be in: `√` and `log` of an
interval whose lower endpoint is *strictly* negative are refused, and only a
closed endpoint sitting exactly on the domain boundary is accepted.

| Integral | Value | Status |
|---|---|---|
| `∫₀¹ asin(x) dx` | `π/2 − 1` | enclosed (`width × range` on the last panel) |
| `∫₀¹ √(1−x²) dx` | `π/4` | enclosed, same |
| `∫₀¹ √x dx` | `2/3` | enclosed, same |
| `∫₀¹ xˣ dx` | 0.78343… | enclosed, same |
| `∫₀¹ (x−1)/ln x dx` | `ln 2` | enclosed — removable at `1`, bounded at `0` |
| `∫₀¹ (1−cos x)/x² dx` | 0.48638… | enclosed (order-2 removable) |

### What is still refused

An **unbounded** integrand is refused, whether or not the integral converges,
and the message names the rule that stopped rather than implying the integral
does not exist:

| Integral | Value | Status |
|---|---|---|
| `∫₀¹ ln(1+x)/x dx` | `π²/12` | enclosed (removable) |
| `∫_{-1}^{1} sin(x)/x dx` | `2·Si(1)` | enclosed (removable) |
| `∫₀¹ −ln x dx` | 1 | refused — unbounded at `0`, and not a `0/0` quotient |
| `∫₀¹ (ln x)² dx` | 2 | refused, same reason |
| `∫₀¹ dx/√(1−x²) dx` | `π/2` | refused — unbounded at `1`, numerator does not vanish |
| `∫₀¹ ln(x)·ln(1−x) dx` | `2 − π²/6` | refused — unbounded at both ends |

These need an integrable-tail bound or a singularity-removing substitution,
neither of which can be derived rigorously from the expression alone today. The
refusal is the honest answer; widening an enclosure to make them pass would
break the contract that makes the module worth using.

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
branch-and-bound, verified 1-D definite integrals including removable
singularities, root absence, root existence and sign predicates.

Not shipped: multivariate verified quadrature (`verified_integral` is 1-D),
improper integrals, integrable-but-not-removable singularities, and
Taylor-model-based ODE enclosures.
