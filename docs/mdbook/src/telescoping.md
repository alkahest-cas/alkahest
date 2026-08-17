# Creative telescoping (Zeilberger's algorithm)

`zeilberger` decides combinatorial identities over a well-defined class and
hands back a **certificate** you can re-check independently. That combination —
a decision procedure whose output is a short, verifiable algebraic object — is
why it matters for an autoresearch loop: it turns "this identity survived a
numeric sweep" into "this identity is proved", without a human in the loop.

```python
import alkahest as ak

pool = ak.ExprPool()
n, k = pool.symbol("n"), pool.symbol("k")
one = pool.integer(1)

# F(n, k) = C(n, k), written as the ratio of gammas the engine recognises.
F = ak.gamma(n + one) / (ak.gamma(k + one) * ak.gamma(n - k + one))

cert = ak.zeilberger(F, n, k)
cert.order          # 1
cert.coeffs         # [a_0(n), a_1(n)] — here proportional to [-2, 1]
cert.certificate    # R(n, k)
cert.boundary       # "vanishes" — so the recurrence holds for the *sum*
```

The result says: with `S(n) = Σ_{k=k_lo}^{k_hi} F(n,k)`,

```text
Σ_i a_i(n)·S(n+i) = b(n)
```

where `b(n)` is a boundary term that the summand-level identity does not by
itself make zero. Here it *is* zero — `cert.boundary` says so, and says it after
computing it — so the recurrence reads `S(n+1) − 2·S(n) = 0`, which together
with `S(0) = 1` gives `Σ_k C(n,k) = 2^n`. When it is not zero the inhomogeneous
recurrence is returned instead, and when neither can be proved nothing is
claimed about the sum at all. That three-way verdict is the subject of
[its own section](#the-boundary-hypothesis-and-the-verdict-on-it) below, and it
is the difference between a certificate and a theorem.

## What the certificate asserts

The returned `certificate` is a rational function `R(n, k)` such that, with
`G(n,k) = R(n,k)·F(n,k)`,

```text
Σ_i a_i(n)·F(n+i, k) = G(n, k+1) − G(n, k)
```

holds **identically**. That identity in `k` is the whole of what is verified.
Because it is a rational-function identity, a reader — or a referee, or another
CAS — can check it by clearing denominators and expanding, with no reference to
how it was found.

## The boundary hypothesis, and the verdict on it

Summing that identity over `k = k_lo .. k_hi` telescopes the right-hand side to
`G(n, k_hi+1) − G(n, k_lo)` — a *boundary difference*, not zero. The familiar
homogeneous recurrence for `S(n)` therefore holds only when that difference
vanishes: the **natural boundary** hypothesis, which Zeilberger's algorithm does
not establish.

It holds in the usual case, where `F` vanishes outside `0 ≤ k ≤ n`, and that
covers every classical identity in this chapter. It fails, for instance, for
`F(n,k) = C(n,k)/(k+1)`: there `G(n,0) = −1`, and the true relation is
`(n+2)·S(n+1) − (2n+2)·S(n) = 1`, not `0`. Reading the homogeneous recurrence
off the certificate there gives a false lemma.

**`zeilberger` decides it** rather than leaving it stated. `cert.boundary` is
one of three values:

| `cert.boundary` | what may be claimed about `S(n)` |
|---|---|
| `"vanishes"` | proved: `Σ_i a_i(n)·S(n+i) = 0` |
| `"nonzero"` | proved: `Σ_i a_i(n)·S(n+i) = b(n)`, with `b(n)` in `cert.boundary_rhs` |
| `"unknown"` | **nothing.** The certificate is still a true statement about the summand |

```python
cert = ak.zeilberger(F, n, k)          # limits default to k = 0..n
cert.boundary            # "vanishes"
cert.boundary_rhs        # None — the right-hand side is 0
cert.limits              # (0, n), echoed back so the assumption is on the record
cert.boundary_reason     # why the verdict came out this way
cert.implies_sum_recurrence   # True for "vanishes" and "nonzero"
```

`"nonzero"` is a **result, not a refusal**. OEIS A279013,
`a(n) = Σ_{k=0}^{n} C(2k,k)/(k+1)·C(2n−1,n−k)`, gets a verified order-2
certificate in a tenth of a second; the homogeneous recurrence read off it fails
against the sequence at the very first term. The engine now returns the
inhomogeneous recurrence, which does hold:

```python
cert.boundary       # "nonzero"
cert.boundary_rhs   # b(n), an explicit hypergeometric term in n
# Σ_i a_i(n)·S(n+i) = b(n) — checked exactly against 2, 8, 35, 161, 768, 3773
```

### The summation range is part of the claim

The verdict is about the range in `cert.limits`, and it *changes with it* —
truncating a sum by one term generally turns `"vanishes"` into `"nonzero"`. Pass
`limits=(k_lo, k_hi)` (each an `Expr` or an `int`) to say what you are summing:

```python
cert = ak.zeilberger(F, n, k, limits=(0, n - pool.integer(1)))
cert.boundary                         # "nonzero" — the k = n term was dropped

# Or ask the same certificate about another range, without re-running the search:
cert.boundary_at(0, n)["boundary"]    # "vanishes"
```

Two things this design does deliberately. The default `(0, n)` is *stated and
echoed back*, not inferred from the summand, so a caller summing over something
else can see the mismatch. And a range the analysis cannot place — endpoints
that are not integer-affine in `n` — is `"unknown"`, never `"vanishes"`.

### What `"vanishes"` is worth

It is a proof, not a numeric check. Each endpoint of `G` is evaluated by exact
**order counting** in `Q(n)`: the multiplicity of the endpoint as a root of the
certificate's numerator and denominator, plus `−e` for every `Γ(a·n+b·k+c)^e`
factor whose argument lands on a non-positive integer there (a pole of `Γ`, or a
zero of `1/Γ`). A strictly positive total order *is* an exact zero. A negative
one means `G` is unbounded at the endpoint, and that is reported as `"unknown"`.
Nothing that merely looks like zero can produce `"vanishes"`.

The verdict also accounts for a subtlety that is easy to miss: when the limits
move with `n`, `Σ_{k=0}^{n} F(n+i,k)` is **not** `S(n+i)`. For `Σ_{k=0}^{n}
C(n,k)` the telescoped difference alone is `−1`, and it is the missing term
`C(n+1,n+1) = 1` that cancels it. The full statement is

```text
b(n) = G(n, k_hi+1) − G(n, k_lo) + Σ_i a_i(n)·D_i(n)
```

with `D_i` the finitely many values of `F` between the range at `n` and the range
at `n+i`, and it is `b(n)` that the verdict is about.

Symmetrically, `"nonzero"` needs a *witness*: an integer `n₀` at which `b(n₀)`
is nonzero in exact rational arithmetic. Sampling that finds only zeros proves
nothing and yields `"unknown"`.

`cert.boundary_term` still returns `G(n,k) = R(n,k)·F(n,k)` if you want to
discharge the hypothesis yourself.

`side_conditions` is a `list[str]`, the same shape as
`DerivedResult.verification["side_conditions"]`: things the result depends on
that were assumed rather than proved. It **tracks the verdict** — a discharged
hypothesis, a refuted one and an open one read differently, so a loop that only
reads this list still cannot mistake the three. It is never empty: even a proved
boundary is a statement about the `n` at which everything involved is defined,
and a permanent record of which range was assumed.

## Verification is not optional

Every certificate is re-checked as an exact identity in `Q(n)(k)` before it is
returned. A candidate that fails verification is discarded and the search
continues; it is never returned with a caveat. This is the same
withhold-rather-than-lie discipline as the Lean certificate exporter: in a loop,
one unverified certificate becomes a false lemma that every downstream
derivation inherits.

Verification runs in exact rational arithmetic — no floating point is involved
at any stage of the algorithm.

## The class it decides, and where it refuses

The supported class is **proper hypergeometric terms**:

```text
F(n, k) = R(n, k) · z^k · w^n · ∏_j Γ(a_j·n + b_j·k + c_j)^(e_j)
```

with `R` a rational function, `z, w` nonzero rationals, `a_j, b_j` integers and
`c_j` rational. Factorials, binomials and Pochhammer symbols are recognised and
normalised into this form.

Everything else is refused, with a structured error rather than a guess:

| Code | Meaning | What a loop should do |
|---|---|---|
| `E-HOLO-001` | Not a proper hypergeometric term | Close this branch — Zeilberger does not apply |
| `E-HOLO-002` | Search bounds exhausted | Retry with larger `max_order` / `max_degree`, or deprioritise |
| `E-HOLO-003` | A candidate failed exact verification | Report as a bug with the term |
| `E-HOLO-004` | Malformed call (`n` and `k` not distinct, non-positive bounds) | Fix the call |

`E-HOLO-002` is worth dwelling on. It does **not** mean "no recurrence exists" —
it means none was found within the bounds you set. The distinction matters: an
agent can raise the bounds and retry, whereas `E-HOLO-001` is a permanent answer
about the input and the branch can be closed for good.

## Cost and bounds

`max_order` (default 4) and `max_degree` (default 16) bound the search. Solving
the linear system over `Q(n)` at one `(J, d)` pair gets rapidly more expensive as
either grows — measured on `Σ (−1)^k C(n,k)³` at order 1, a single probe goes
from 0.7 ms at `d = 0` to 0.6 s at `d = 7` to 84 s at `d = 12`, and one extra
order costs about what three extra degrees cost.

**Both are upper bounds, not starting points.** The `(J, d)` pairs are visited by
iterative deepening, cheapest estimated candidate first, and the first relation
that passes exact verification is returned. Raising a bound therefore widens the
reach without moving where the search starts — `Σ (−1)^k C(n,k)³` is decided at
order 2 in 0.8 s at the defaults and 0.6 s at `max_order=6, max_degree=64`
(a hand-tuned `max_order=2, max_degree=4` costs 0.2 s, because a tight bound
also truncates the cheap probes the deepening would interleave first):

```python
# Same answer, same order of magnitude — the bounds are not a starting point.
cert = ak.zeilberger(F, n, k)                            # defaults
cert = ak.zeilberger(F, n, k, max_order=6, max_degree=64)
```

What the bounds *do* control is the price of a refusal: a term with no recurrence
inside them pays the full grid before raising `E-HOLO-002`. Set them to the
largest search you are willing to wait through when the answer is "no".

```python
cert = ak.zeilberger(F, n, k, max_order=2, max_degree=6)
```

## Is the order minimal?

Usually the interesting part of a certified recurrence is not that it exists but
that it is *short*: an order-4 relation where the literature records a guessed
order-5 is a result, and an order-4 relation that might have been order 3 is a
coincidence. So the question has to be asked explicitly, and the answer is on
the certificate:

```python
cert.order_is_minimal   # True only when the search established it
```

**Cheapest-first is not order-ascending.** The deepening above orders probes by
estimated cost, `3·(J−1) + d`, so it can reach a cheap order-2 probe long before
an expensive order-1 one — which is exactly what makes Dixon, Franel and Apéry
decidable at the default bounds. A returned order 2 therefore does *not*
establish that no order-1 relation exists, and `order_is_minimal` is `False` to
say so. `False` means **not established**, never "a lower order exists": a
lower-order relation that had been found would have been the one returned.

It is `True` for free at order 1, and `True` whenever the cost-ordered plan
happened to spend every lower order before the probe that succeeded — which it
does at narrow `max_degree`, since the plan interleaves less there. To get the
claim in general, ask for it:

```python
cert = ak.zeilberger(F, n, k, max_degree=6, minimal=True)
cert.order_is_minimal   # True — every degree ≤ 6 at every lower order was refused
```

`minimal=True` walks the grid order-major: every degree `0..max_degree` at order
`J` is probed and refused before order `J+1` is tried at all. Same bounds, same
exact verification, same certificate — only what was ruled out along the way
differs. The flag is computed from the probes that actually happened rather than
from the mode, so it cannot drift away from what the search did.

The price is the whole hopeless low-order sweep the default plan exists to
avoid, and it is charged against `max_degree` because that is the bound
minimality is claimed relative to. Measured on this machine at `max_order=4`:

| Summand | `max_degree` | default | `minimal=True` |
|---|---|---|---|
| `Σ C(n,k)³` (Franel) | 4 | 0.15 s | 0.14 s — default already minimal |
| `Σ C(n,k)³` (Franel) | 6 | 0.23 s | 0.23 s — default already minimal |
| `Σ C(n,k)³` (Franel) | 8 | 0.23 s | 0.56 s |
| `Σ C(n,k)³` (Franel) | 16 | 0.23 s | **9.7 s** |
| `Σ C(n,k)²C(n+k,k)²` (Apéry) | 4 | 0.07 s | 0.07 s — default already minimal |
| `Σ C(n,k)²C(n+k,k)²` (Apéry) | 6 | 0.08 s | 0.11 s |
| `Σ C(n,k)²C(n+k,k)²` (Apéry) | 8 | 0.08 s | 0.29 s |
| `Σ C(n,k)²C(n+k,k)²` (Apéry) | 16 | 0.08 s | **13.1 s** |

The default column is flat in `max_degree` — that is the deepening doing its job
— and the `minimal=True` column is not, because it is the column that has to
sweep.

The default is unchanged, deliberately — the cost-ordered plan is what makes
these terms decidable at all, and `minimal=True` is an opt-in for when
minimality is the result you intend to publish. Note the shape of the table: the
sweep grows like `3^d`, so the honest move is usually to claim minimality
against the smallest `max_degree` you are willing to state rather than against
the default 16.

## Guessing the recurrence first

The other half of the loop is `guess_holonomic`, which fits a P-recursive
recurrence to the first terms of a sequence in exact rational arithmetic — the
*guess* in guess-then-prove, with `zeilberger` supplying the proof:

```python
motzkin = [1, 1, 2, 4, 9, 21, 51, 127, 323, 835, 2188, 5798, 15511,
           41835, 113634, 310572, 853467, 2356779, 6536382, 18199284, 50852019]

guess = ak.guess_holonomic(motzkin)
guess.order, guess.degree   # (2, 1)
guess.surplus_terms         # 14 equations confirmed it without being needed
guess.confirmed             # True
```

A fitted recurrence is a conjecture, and the number that says how much of one is
`surplus_terms`. See [Guessing recurrences](guessing.md) for the guard and what
it refuses.

## The `q`-analogue (`alkahest.experimental.q_zeilberger`)

`q`-hypergeometric sums — Gaussian binomials `[n;k]_q`, `q`-Pochhammer symbols
`(a;q)_n` — are not proper hypergeometric terms in `(n,k)`, so `zeilberger`
refuses them (correctly) with `E-HOLO-001`. `q_zeilberger` is the `q`-shifted
twin of the same algorithm, and the same discipline: the certificate is
re-checked as an exact identity in `Q(q)(qⁿ)(q^k)` before it is returned.

```python
import alkahest as ak
from alkahest.experimental import q_zeilberger, qbinomial

pool = ak.ExprPool()
q, n, k = pool.symbol("q"), pool.symbol("n"), pool.symbol("k")

# Σ_k [n;k]_q² · q^{k²} = [2n;n]_q  — the q-analogue of Σ_k C(n,k)² = C(2n,n).
b = qbinomial(pool, n, k)
cert = q_zeilberger(b * b * q ** (k * k), q, n, k)

cert.order            # 1
cert.boundary         # "vanishes"
cert.support          # ("0", "n") — where the summand is proved to live
cert.sum_term(3)      # the exact q-series value S(3), a polynomial in q
```

`sum_term(n0)` is the part worth reaching for. It evaluates the sum from the
*definition* of the `q`-Pochhammer symbol, not through the shift quotients the
search used, so checking `Σ_i a_i(qⁿ)·S(n+i) = 0` against it is an independent
check of the returned recurrence rather than a restatement of the certificate.

### What it accepts

```text
F(n,k) = R(qⁿ, q^k) · z^k · w^n · q^{A·k² + B·n·k + C·n² + D·k + E·n}
         · Π_j (q^{u_j}; q^{d_j})_{v_j}^{e_j}
```

with `u_j`, `v_j` integer-affine in `n, k`. Written as an expression: the heads
`qbinomial(N, K)` and `qpochhammer(u, d, v)` (meaning `(q^u; q^d)_v`), powers of
`q` whose exponent is a degree-≤2 polynomial in `n` and `k`, powers with a base
free of `n` and `k`, and any rational function of `q`, `qⁿ`, `q^k`. Half-integer
quadratic coefficients are fine — `q^{k(k−1)/2}` is not rational in `q^k` but
all of its shift quotients are, which is the property the algorithm needs.

| Code | Meaning | What a loop should do |
|---|---|---|
| `E-HOLO-020` | Not a `q`-hypergeometric term | Close this branch |
| `E-HOLO-021` | Search bounds exhausted | Raise `max_order` / `max_degree` and retry |
| `E-HOLO-022` | A candidate failed exact verification | Report as a bug with the term |
| `E-HOLO-023` | Malformed call (`q`, `n`, `k` not distinct; non-positive bounds) | Fix the call |
| `E-HOLO-024` | In the shape of the class, outside it in substance | Close this branch |

`E-HOLO-024` is the interesting one. `(q^k; q²)_n` shifted in `k` moves its
first argument by `1`, which the base `q²` does not divide, so the shift
quotient is an *infinite* product and no algorithm in this family applies. That
is a permanent answer about the input, like `E-HOLO-020`, not a budget problem.

### The boundary verdict is two-valued here

`cert.boundary` is `"vanishes"` or `"unknown"` — there is no `"nonzero"` arm.
The sum it is about is `S(n) = Σ_{k ∈ Z} F(n,k)`, which the analysis also proves
is a *finite* sum, over the window in `cert.support`. Fixing the range at all of
`Z` is what makes the proof short: the range does not move with `n`, so there are
no `D_i` correction terms, and `"vanishes"` follows from two structural facts
about the summand alone — that it vanishes outside an affine window in `k`, and
that it is finite at every integer `k`.

The certificate is **not** evaluated at an endpoint, and that is deliberate
rather than lucky: `R` genuinely has poles at integer `k` — on the summand above
it has a double pole exactly where the summand has a double zero, and
`G(n, n+1)` is a finite *non-zero* limit of `0·∞`. What the proof does instead is
find one `k` far to the right that is past both the window and the (finitely
many) poles, where `G = R·0 = 0` with no indeterminacy, and then induct
downwards on `G(n,k) = G(n,k+1) − Σ_i a_i(qⁿ)·F(n+i,k)`, whose right-hand side
the support analysis has already shown is finite everywhere. That gives every
`G` a finite value, poles included, without evaluating the product at one; `G`
is then constant and zero beyond the window at both ends, and the sum over `Z`
telescopes to zero. (Read analytically at generic `q` with `0 < |q| < 1`; the
conclusion is an identity between rational functions of `q` that holds on an
open set, so it holds in `Q(q)`.)

What is *not* implemented is the inhomogeneous arm: computing `b(n)` for a
`q`-sum needs endpoint values of `G` that are not rational in `qⁿ`, so a
summand whose support the analysis cannot bound gets `"unknown"` and **no**
claim about its sum, not a guessed inhomogeneity.

One more caveat, and it is on every verdict's `side_conditions`: `q` is treated
as **transcendental**. Everything here is an identity in `Q(q)`. Specialising
`q` to a root of unity — which is what the `q`-supercongruence literature does —
is a separate step with its own hypotheses.

### Specialising at a root of unity (`specialize_at_root_of_unity`)

A proved `Q(q)` recurrence does not, by itself, license setting `q = ζ_d` for a
primitive `d`-th root of unity: a coefficient or a sum value can have a pole
there, and specialising anyway is exactly the `q`-analogue of the A279013
failure mode — a certificate that re-checks perfectly while the specialised
claim is false. `QZeilbergerCertificate.specialize_at_root_of_unity(d, n)`
takes that step as a **decision**, not an assumption:

```python
from alkahest.experimental import cyclotomic_polynomial, q_zeilberger, qbinomial

pool = ak.ExprPool()
q, n, k = pool.symbol("q"), pool.symbol("n"), pool.symbol("k")
b = qbinomial(pool, n, k)
cert = q_zeilberger(b * b * q ** (k * k), q, n, k)

spec = cert.specialize_at_root_of_unity(3, 2)  # q = zeta_3, at n = 2
spec.status                       # "specializes" / "obstructed" / "unknown"
spec.sum_value(0)                 # S_zeta(2), the canonical rep in Q[q]/(Phi_3)
spec.sum_valuation(0)             # the exact Phi_3-adic valuation of S(2)
spec.modulus()                    # Phi_3(q) = q^2 + q + 1, exposed for a by-hand check
```

The hypotheses — no pole in any coefficient `a_i(qⁿ)` or sum value `S(n+i)` at
`ζ_d` — are decided **exactly**, by polynomial divisibility by `Φ_d(q)` over
`Q` in the cyclotomic field `Q(ζ_d) = Q[q]/(Φ_d(q))`; nothing is evaluated
numerically at any stage. `Φ_d` is irreducible over `Q`, so "does `p` vanish at
`ζ_d`" is exactly "does `Φ_d` divide `p`", which is what makes the valuation —
and therefore the decision — exact rather than approximate. `cyclotomic_polynomial(pool, d)`
returns `Φ_d(q)` directly, so a caller can redo the whole check by hand.

`status` is three-valued, and the three are not interchangeable:

* **`"specializes"`** — proved: every coefficient and every sum value has
  non-negative `Φ_d`-adic valuation, so the specialisation map is defined on
  all of them, and the specialised identity was re-checked as an exact
  statement in `Q(ζ_d)` before being returned. Three further things are
  reported on this verdict rather than folded into it, because each of them
  makes a true verdict mean less than it looks:
  * `is_vacuous` — every coefficient died at `ζ_d` (the `q → 1` limit at
    `d = 1` is always like this), so the recurrence is `0 = 0`. Still a
    theorem; it constrains nothing.
  * `leading_coefficient_survives` — `False` means the specialised recurrence
    no longer determines the last value from the earlier ones, even though it
    is not vacuous.
  * `support_shrinks` / `effective_support` — the `q`-Lucas phenomenon:
    `[2;1]_q = 1 + q` is non-zero in `Q(q)` and zero at `ζ_2`, so the
    surviving window at a root of unity can be a strict subset of the generic
    one. It can never grow.
* **`"obstructed"`** — a pole at `ζ_d` was **exhibited**: some coefficient or
  sum value has negative `Φ_d`-adic valuation (available via `sum_valuation`
  even on this verdict, since a negative valuation *is* the obstruction).
  Nothing is offered — `sum_value` and `coefficient` raise — and this is not a
  claim that the specialised identity is false, only that this route to it is
  blocked.
* **`"unknown"`** — the generic boundary verdict was already `"unknown"`, so
  there is no proved `Q(q)` statement to specialise in the first place.

`sum_valuation(i)` is the `q`-supercongruence content in its exact form: it is
the integer `v` with `Φ_d(q)^v` dividing `S(n+i)` and `Φ_d(q)^{v+1}` not — so
`v ≥ r` is precisely the divisibility statement `Φ_d(q)^r | S(n)` that a
`q`-supercongruence asserts, decided exactly rather than checked at finitely
many numeric points.

## Method

The implementation is the standard Gosper-style reduction (Petkovšek–Wilf–
Zeilberger, *A=B*, ch. 6; Koepf, *Hypergeometric Summation*, ch. 7), carried out
over the field `Q(n)` rather than `Q`:

1. Compute the exact shift quotients `p(k) = F(n,k+1)/F(n,k)` and
   `c_i(k) = F(n+i,k)/F(n,k)` — both rational functions, which is precisely the
   property that defines the proper hypergeometric class.
2. Take `D(k)`, a common denominator of the `c_i`, and work with
   `W(n,k) = F(n,k)/D(k)`, so that `Σ_i a_i·F(n+i,k) = N(k)·W(n,k)` with
   `N(k) = Σ_i a_i·D(k)·c_i(k)` polynomial and linear in the unknowns.
3. Decompose the shift ratio of `W`, `ρ(k) = p(k)·D(k)/D(k+1)`, into Gosper
   normal form `ρ = A(k)·C(k+1)/(B(k)·C(k))`.
4. Gosper's key equation is then the polynomial identity
   `A(k)·X(k+1) − B(k−1)·X(k) = C(k)·N(k)`. Comparing coefficients of each power
   of `k` gives a linear system over `Q(n)`; solving it yields both the `a_i` and
   the certificate `R = B(k−1)·X(k) / (C(k)·D(k))`.
5. The solved pair is substituted back and checked exactly. Only then is it
   returned.

## Scope of this release

Shipped: Zeilberger's algorithm with exact certificate verification, the
`Q(n)` / `Q(n)(k)` arithmetic tower it rests on, proper-hypergeometric
recognition, the three-valued boundary verdict over a stated summation range,
explicit minimal-order certification, `guess_holonomic` — recurrence guessing
from finite data — the `q`-analogue `q_zeilberger` over `Q(q)(qⁿ)(q^k)` with
its own two-valued boundary verdict, and `specialize_at_root_of_unity` — the
step from a `Q(q)` identity to `q = ζ_d`, decided exactly in the cyclotomic
field `Q(ζ_d)` with its own three-valued verdict.

Not shipped on the `q` side: multivariate (`q`-)telescoping and an
inhomogeneous boundary arm. A `q`-sum whose support cannot be bounded is
answered `"unknown"`, never guessed. Root-of-unity specialisation covers a
single certificate at a single `(d, n)` pair — it is the mechanical step the
`q`-supercongruence literature needs, not a search over `d` or a prover for
the wider congruence statements (e.g. uniform-in-`n` supercongruences, or
`p`-adic statements not phrased as `Φ_d`-adic valuations) that literature
contains.

Not yet shipped, and tracked as follow-up work: Ore-operator closure properties
for D-finite functions (sums and products of holonomic objects) and the
differential half of the guessing front-end (fitting a linear ODE to a power
series).
`sum_indefinite` (Gosper) and `verify_wz_pair` remain the neighbouring tools for
the indefinite and WZ-pair cases respectively.
