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
```

The result says: with `S(n) = Σ_{k=k_lo}^{k_hi} F(n,k)`,

```text
Σ_i a_i(n)·S(n+i) = G(n, k_hi+1) − G(n, k_lo)
```

and that boundary difference vanishes here, so the recurrence reads
`S(n+1) − 2·S(n) = 0`, which together with `S(0) = 1` gives `Σ_k C(n,k) = 2^n`.

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

## The boundary hypothesis is yours to discharge

Summing that identity over `k = k_lo .. k_hi` telescopes the right-hand side to
`G(n, k_hi+1) − G(n, k_lo)` — a *boundary difference*, not zero. The familiar
homogeneous recurrence for `S(n)` therefore holds only when that difference
vanishes: the **natural boundary** hypothesis, which Zeilberger's algorithm does
not establish and which `zeilberger` does not check.

It holds in the usual case, where `F` vanishes outside `0 ≤ k ≤ n`, and that
covers every classical identity in this chapter. It fails, for instance, for
`F(n,k) = C(n,k)/(k+1)`: there `G(n,0) = −1`, and the true relation is
`(n+2)·S(n+1) − (2n+2)·S(n) = 1`, not `0`. Reading the homogeneous recurrence
off the certificate there gives a false lemma.

The certificate carries what you need to settle it:

```python
cert.side_conditions   # the hypothesis, stated
cert.boundary_term     # G(n, k) = R(n, k)·F(n, k)

# Substitute your own summation endpoints and check the difference is 0.
g_at_lo = ak.simplify(ak.subs(cert.boundary_term, {k: pool.integer(0)})).value
```

`side_conditions` is a `list[str]`, the same shape as
`DerivedResult.verification["side_conditions"]`: things the result depends on
that were assumed rather than proved. An empty list would be a claim; this one
is never empty.

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
`Q(n)` / `Q(n)(k)` arithmetic tower it rests on, and proper-hypergeometric
recognition.

Not yet shipped, and tracked as follow-up work: Ore-operator closure properties
for D-finite functions (sums and products of holonomic objects) and the guessing
front-end (fitting a P-recursive recurrence or a linear ODE to finite data).
`sum_indefinite` (Gosper) and `verify_wz_pair` remain the neighbouring tools for
the indefinite and WZ-pair cases respectively.
