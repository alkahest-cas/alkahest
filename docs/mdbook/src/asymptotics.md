# Asymptotics at scale

`series` and Gruntz `limit` expand a *function*; `asymptotic_expand` handles the
power/log/exp scales of a function at infinity. What a loop working on
combinatorics and analysis needs on top of that is the asymptotics of a **sum**
— how `Σ_{k=a}^{n} f(k)` behaves as `n → ∞`. That is how a conjecture about a
growth rate actually gets settled.

```python
from alkahest import ExprPool
from alkahest.experimental import euler_maclaurin

pool = ExprPool()
k, n = pool.symbol("k"), pool.symbol("n")

r = euler_maclaurin(pool.integer(1) / k, k, 1, n, corrections=2)
r.leading           # log(n)
r.terms            # [log(n), γ, 1/(2n), −1/(12n²)] — most significant first
r.rigor            # "numerically_consistent"
```

## The formula

For a smooth summand,

```text
Σ_{k=a}^{n} f(k) = ∫_a^n f(t) dt + (f(a) + f(n))/2
                   + Σ_{j=1}^{m} B_{2j}/(2j)! · (f^{(2j-1)}(n) − f^{(2j-1)}(a))
                   + R_m
```

Read as `n → ∞`, the `a`-endpoint pieces collapse into one additive constant and
what remains is an asymptotic expansion in `n`. For `f(k) = 1/k` that is the
classical `H_n ~ log n + γ + 1/(2n) − 1/(12n²) + …`.

## The constant is fitted, and the report says so

Euler–Maclaurin does **not** determine the additive constant from the `n`-side
terms. For the harmonic numbers that constant is Euler's γ, and no amount of
boundary algebra at `k = a` produces it.

This module therefore obtains it numerically, from the exactly computed sum at
large `n`, and is explicit about that: `rigor` comes back as
`"numerically_consistent"` rather than `"proved"`, and the fitted constant
appears in `hypotheses` with status `"assumed"`. The *shape* of the expansion is
derived symbolically — only that one scalar is empirical.

```python
r.all_hypotheses_checked      # False
[s for s, _ in r.hypotheses]  # ["checked", "assumed", "assumed"]
```

This distinction is the whole reason the result is an `AsymptoticReport` rather
than a bare list of terms. A loop that records "γ ≈ 0.5772 was fitted, not
proved" can revisit it; one handed an undifferentiated expansion cannot.

## Ordering and the verification gate

Terms are returned most-significant first, and the ordering is by *magnitude at
the check points*, not by where they came from in the formula. This matters:
the constant sits below every growing term and above every decaying one, so for
`Σ k` it lands after `n²/2` and `n/2`, while for `H_n` it lands right after
`log n`.

Every term then goes through the same `o()`-gate as `asymptotic_expand`: the
truncated expansion is compared against the exactly computed sum at increasing
`n`, and a term is kept only if it genuinely refines its predecessor *against
the true value*. Terms that fail are dropped; if nothing survives, the call
refuses rather than emitting an unverified expansion. `r.verification` carries
the evidence and `r.max_relative_error` summarises it.

## Refusals

| Situation | Behaviour |
|---|---|
| Summand has no symbolic antiderivative (e.g. `exp(−k²)`) | Refuse — the integral term cannot be formed |
| Summand not numerically evaluable at the check points | Refuse — the gate has no oracle |
| No term survives the gate | Refuse — emit nothing rather than something unverified |
| `corrections` above the supported maximum | Refuse |

## Singularity analysis of generating functions

For a rational `f(z)`, the growth of `[zⁿ] f(z)` is governed entirely by the
singularity of smallest modulus:

```python
from alkahest.experimental import coefficient_asymptotics

gf = pool.integer(1) / (pool.integer(1) - z - z*z)   # Fibonacci
r = coefficient_asymptotics(gf, z, n)
r.terms[0]      # ~ C·φⁿ
```

A pole `ρ` of multiplicity `m` contributes `C · n^{m-1} · ρ^{-n}`. The *shape*
— which power of `n`, which exponential base — is exact; the single leading
constant is obtained from the exact power series by **Richardson
extrapolation**, and the report says so.

That extrapolation is not a nicety. Reading the constant off a single finite
index absorbs the subleading term: for `1/(1-z)²`, where `[zⁿ] = n+1` exactly,
taking `C = a₄₈/48 = 49/48` leaves a permanent 2% bias, so the relative error
stops shrinking as `n` grows — which is precisely what an asymptotic statement
must not do. One Richardson step cancels the `1/k` term and recovers `C = 1`.

### What it refuses

The transfer theorem needs a **unique** dominant singularity. `1/(1-z²)` has
poles at both `+1` and `−1`; its coefficients oscillate (`1,0,1,0,…`) and no
single power-law term describes them. Rather than reporting one pole as if it
won, the routine declines — as it does for a complex dominant pole (necessarily
one of a conjugate pair) and for non-rational input.

## Scope of this release

Shipped: the Euler–Maclaurin route for `Σ_{k=a}^{n} f(k)`, with Bernoulli
corrections, magnitude ordering, the numeric gate, and the checked-versus-
assumed hypothesis ledger in `AsymptoticReport`.

Also shipped: singularity analysis for **rational** generating functions.

Not shipped, and tracked as follow-up (the shared scaffolding —
`AsymptoticReport`, the gate, exact Bernoulli numbers, rational-function
extraction and a complex root finder — is already in place for them):

- **Algebraic and log-type generating functions** — the transfer theorem beyond
  poles (`√(1-4z)` for the Catalan numbers, `log` singularities). Only the
  rational case ships here.
- **Sequence asymptotics** from a closed form or a P-recursive recurrence
  (Stirling-based expansions, Poincaré–Perron growth).
- **Laplace / saddle-point / stationary-phase** asymptotics of parameter
  integrals.

Note that `Σ log k` gives Stirling's formula for `log n!` through this route
already, so the factorial case is reachable today.
