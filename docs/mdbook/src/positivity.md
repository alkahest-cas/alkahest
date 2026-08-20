# Positivity certificates (SOS / Positivstellensatz)

`decide` answers real-algebraic questions by CAD, and pays doubly-exponential
cost for it. Most positivity questions that actually arise — is this bound
valid, is this Lyapunov candidate non-negative, is this inequality true on a
box — do not need a decision procedure at all. They need a **certificate**: a
short algebraic identity that makes the answer checkable by anyone, including a
proof assistant.

> `decide` is **not** complete in this implementation: on some sentences it
> refuses with `E-CAD-001` rather than answering. See
> [`decide` refuses rather than guessing](#decide-refuses-rather-than-guessing)
> below — this changed in 3.8 and it changed because the alternative was
> answering wrongly.

```python
import alkahest as ak

pool = ak.ExprPool()
x, y = pool.symbol("x"), pool.symbol("y")

cert = ak.sos_decompose(x*x - pool.integer(2)*x*y + y*y, [x, y])
cert.kind        # "sos"
cert.identity    # p = 1*(x - y)^2
cert.verify()    # True — re-expands to the target exactly
cert.to_lean()   # Lean 4 rendering, or None
```

Constrained, on a basic semialgebraic set:

```python
g1, g2 = x, pool.integer(1) - x          # the box 0 ≤ x ≤ 1
cert = ak.prove_nonneg(x - x*x, [x], constraints=[g1, g2])
cert.kind        # "handelman"  —  x − x² = x·(1 − x)
```

## Three outcomes, deliberately kept apart

This is the part that matters for a search loop, and the part most CAS get
wrong by collapsing:

| Outcome | Meaning | What a loop should do |
|---|---|---|
| `PositivityCertificate` | **Proved** non-negative, with a checkable witness | Record it; cite it |
| `SosError` `E-SOS-003` | **Proved not** non-negative — a witness point is in the message | The conjecture is false; kill the branch |
| `SosError` `E-SOS-002` | **Undecided**: no certificate of this shape at this degree | Record `unknown`; raise the degree, or fall back to `decide` |

> **`E-SOS-002` must be recorded as `unknown`, never as "not SOS".**
> It reports that *this* search — the LP-representable subcone described below,
> at *this* `basis_degree` (or `level`) — found nothing. Three different worlds
> produce it and the error cannot tell them apart: `p` is SOS but its Gram
> matrix lies outside the subcone; `p` is SOS only at a higher basis degree;
> `p` is non-negative but genuinely not SOS. A loop that maps `E-SOS-002` to
> "the conjecture is false" or "`p` is not a sum of squares" closes a branch on
> evidence that does not support it, and a wrongly closed branch is invisible —
> nothing downstream will ever contradict it. Only `E-SOS-003` is a refutation,
> and it carries a witness point.

`E-SOS-002` is *not* a claim that the polynomial is not a sum of squares, and
certainly not that it is negative. The canonical illustration is the
**Motzkin polynomial** `x⁴y² + x²y⁴ − 3x²y² + 1`, which is non-negative
everywhere but provably not a sum of squares *itself* — asked to decompose it
*directly* (no multiplier), this module refuses with `E-SOS-002`, correctly:
it does not report it as negative, and it does not invent a decomposition.
`sos_decompose`'s full pipeline does not stop there, though (see "What the
search actually covers" below) — it also tries multiplying by a power of
`x²+y²` before giving up, and that succeeds for Motzkin, so the *end-to-end*
call returns a certificate, not a refusal. The *homogeneous* 3-variable form
of Motzkin now certifies too, at multiplier power `N = 1` — which is the
classical fact:

```text
(x²+y²+z²)(x⁴y²+x²y⁴−3x²y²z²+z⁶)
  = (½x³y+xy³−3⁄2xyz²)² + ¾(x³y−xyz²)² + (xy²z−xz³)² + (x²yz−yz³)² + (x²y²−z⁴)²
```

> **Correction (2026-08-20).** Earlier releases of this page said the
> homogeneous ternary form "is still out of reach" and that the classical
> fact "needs `N = 2`, not `N = 1`". Both statements were wrong: the identity
> above is exactly why Motzkin is the standard example of a PSD non-SOS form
> that becomes SOS after one multiplication by `Σxᵢ²`. What was missing was a
> half-Newton-polytope reduction in the search — see "What the search
> actually covers" below.

The general point stands regardless of which examples are currently
reachable: **an `E-SOS-002` refusal is a property of the search, not of the
polynomial**, and which polynomials it applies to shifts as the search grows
more complete.

The three-way branch a loop should write:

```python
try:
    cert = ak.sos_decompose(p, [x, y])
    verdict, evidence = "nonneg", cert    # proved, with a checkable identity
except ak.SosError as e:
    if e.code == "E-SOS-003":
        verdict, evidence = "negative", str(e)   # refuted; witness point in the message
    else:
        verdict, evidence = "unknown", str(e)    # E-SOS-002 lands here — leave the branch open
```

## What the search actually covers

A sum-of-squares decomposition `p = zᵀQz` over the monomial basis `z` exists
iff there is a **positive semidefinite** Gram matrix `Q` matching `p`'s
coefficients. Deciding general PSD feasibility is a semidefinite programme,
and no floating-point SDP solver is ever trusted with a certificate here — a
rounded `Q` is not a proof.

The search tries three things, in order, before refusing:

1. **The diagonally-dominant (DSOS) subcone** — non-negative combinations of
   squares of a fixed generator set, `(e_i)²` and `(a·e_i ± b·e_j)²` for small
   coprime `(a, b)`, solved with the exact rational simplex in `real::sos::lp`
   (Bland's rule, so termination is unconditional and there are no epsilon
   tolerances). Every generator is *literally a square*, so a feasible point
   is sound by construction — but the cone is a strict subset of the SOS
   cone: diagonal dominance is not invariant under scaling the basis, so a
   perfect square as ordinary as `(x/2 + 1/3)²` has a Gram matrix — its only
   one — that is PSD but not DD.
2. **The full PSD Gram cone**, when DSOS fails (`real::sos::psd::psd_search`).
   The monomial basis is cut down first: to the monomials of degree exactly
   `d/2` when `p` is homogeneous of degree `d`, and then to the lattice
   points of `½·Newton(p)` — Reznick's theorem says the support of every
   square in every SOS decomposition of `p` already lies there, so this loses
   no certificate while removing free parameters quadratically. It then leans
   on a floating-point search
   (Jacobi eigendecomposition, PSD-cone projection, an annealed schedule of
   shrinking eigenvalue floors with several random restarts —
   `real::sos::sdp`) to *propose* a Gram matrix, which is then rounded to
   nearby rationals and re-expanded to check it equals `p` exactly before
   anything is returned. A `Some` here is always sound regardless of what the
   numeric search converged to; a `None` means only "the search did not turn
   up a certificate", never "not SOS".
3. **A Reznick multiplier search**, when even step 2 fails on `p` itself:
   tries `(x_1²+…+x_n²)^N·p` for `N = 1..4` and reruns step 2 on the product.
   Some positive-definite forms are not SOS at all (Hilbert 1888 — this is
   what Motzkin's polynomial witnesses), but Reznick's theorem guarantees
   `(Σxᵢ²)^N·p` is SOS for *some* `N`; the search does not know `N` in advance
   and reports budget exhaustion honestly rather than a disproof.

`E-SOS-002` at the end of all three is phrased as a statement about the
search, not the polynomial — the search's incompleteness, at any step.

**Step 3's search has to work harder than plain alternating projection**,
because the multiplier certificates it exists for are frequently *tight* —
Motzkin's polynomial and Robinson's form (the textbook PSD-not-SOS examples)
both have witnessing Gram matrices that are *singular*, sitting exactly on
the boundary of the PSD cone rather than its interior. A first version of
this search (annealed alternating projection with several random restarts)
converged toward that boundary monotonically (confirmed by a diagnostic
trajectory) but never reliably closed the last, asymptotically slow stretch —
the textbook behaviour of alternating projection at a tangential
(non-transversal) set intersection. The search now also tries
Douglas–Rachford splitting with over-relaxation and a facial-reduction step —
both standard escapes for exactly this stall — and with them, both
`(x²+y²)·Motzkin(x,y)` and `(x²+y²+z²)·Robinson(x,y,z)` are found and
exactly re-verified.

**The half-Newton-polytope reduction is what closed the homogeneous cases.**
The *homogeneous* ternary Motzkin form at `N = 1` used to refuse, and the
refusal was misdiagnosed as numerical hardness: a great deal of extra search
machinery (symmetry reduction, an exact zero-vector restriction, 6,000,000
Douglas–Rachford iterations) was spent on the wrong multiplier power. The
actual cause was dimension. `(x²+y²+z²)·Motzkin_hom` has a 15-monomial
degree-4 basis of which only 9 lie in `½·Newton`; the 75-parameter family
over the unreduced basis leaves the numeric search about `0.96` away from
the true certificate in parameter space — far too far to round onto it —
while the 18-parameter family over the reduced basis lands on it exactly.
Note what this is *not*: on both bases the certificate is the unique PSD
point of the affine family, rank 5, minimum eigenvalue exactly 0, so `λ_min`
does not distinguish them and more iterations do not help (4× the
Douglas–Rachford budget on the unreduced family still fails to round). With
the reduction, `(Σxᵢ²)·Motzkin_hom` and `(Σxᵢ²)·Choi–Lam` both certify at
`N = 1` in seconds.

**What's still open:** larger copositivity forms — the Horn/C₅ form (5
variables) and the C₇ form (7 variables) both admit `N = 1` certificates that
this search does not reach. Their Newton polytopes are already full, so the
reduction does not help, and their `N = 1` affine families have 420 and 2646
free parameters respectively — above `psd_search`'s numeric-search ceiling of
200, so no search is attempted at those powers at all. That is now *reported*
in the `E-SOS-002` message (lines marked `NOT SEARCHED`) rather than being
indistinguishable from an exhausted search. Closing these needs a real
interior-point SDP solve on the reduced family, not more alternating
projection.

## Constrained certificates

With constraints `g_i ≥ 0`, `prove_nonneg` searches for a **Handelman**
certificate

```text
p = Σ_α c_α · Π_i g_i^{α_i},    c_α ≥ 0 rational,   Σ_i α_i ≤ level
```

which is again an exact LP in the weights `c_α`. `level` is a user-visible
parameter; exceeding it is a refusal (`E-SOS-002`), not a failure. Handelman
is complete for polytopes given a high enough level, but the level needed is
not known in advance — so raising it on refusal is a meaningful retry.

## Verification

Every certificate is re-expanded in exact rational arithmetic and compared
against the target **identically** before it is returned. A candidate that
fails is refused (`E-SOS-005`), never returned with a caveat. `verify()` runs
the same check on demand so a downstream consumer never has to trust the
search that produced the certificate.

## When to use this versus `decide`

| | `sos_decompose` / `prove_nonneg` | `decide` (CAD) |
|---|---|---|
| Answers | Non-negativity, with a certificate | Real-algebraic sentences in ≤ 2 variables with a ≤ 2-quantifier prefix |
| Completeness | No — refuses honestly (`E-SOS-002`) | No — refuses honestly (`E-CAD-001`) |
| Cost | LP in exact rationals | Doubly exponential |
| Output | Checkable identity, Lean-exportable | Truth value (+ witness) |

The intended pattern is: try the certificate route first because it is cheap
and its output is citable; fall back to `decide` on `E-SOS-002` when you need a
verdict rather than a certificate and can afford the cost. Note that neither
route is complete, so "both refused" is a real and expected outcome — it means
*undecided by these methods*, not *false*.

## `decide` refuses rather than guessing

`decide` implements CAD over a **bounded fragment**: purely polynomial bodies over
ℚ in one or two real variables, with a quantifier prefix of at most two. Outside
that fragment it raises `CadError` (`E-CAD-001`). Inside it, there is one further
refusal, and it is the important one.

The CAD sample set is built from rational points — bracket endpoints, refined
brackets, midpoints. For a **strict** atom (`<`, `>`) that is complete: strict
solution sets are open, so if a solution exists, a whole interval of rational
points solves it too. For a **non-strict** atom (`=`, `≠`, `≤`, `≥`) the solution
set can be a single boundary point, and if that point is irrational it is never in
the sample set. Concluding "no sample satisfied it, therefore unsatisfiable" would
then be a claim about a point that was never tested — and via `∀x. φ ≡ ¬∃x. ¬φ`,
that fabricated `false` becomes a machine-checked-looking proof of a false
universal theorem.

So when a boundary root has not been shown rational and the body has a non-strict
atom, `decide` refuses:

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")

# Rational double root: found exactly, so the verdict is real.
body = pool.gt((pool.integer(3) * x + pool.integer(2)) ** pool.integer(2), pool.integer(0))
ak.decide(ak.Forall(x, body))        # (False, None) — false at x = -2/3

# Irrational double root at ±sqrt(2): refuses instead of answering.
irr = pool.gt((x ** pool.integer(2) - pool.integer(2)) ** pool.integer(2), pool.integer(0))
try:
    ak.decide(ak.Forall(x, irr))
except ak.CadError as e:
    print(e.code)                     # E-CAD-001
```

Three consequences worth planning for:

- **`E-CAD-001` is "I did not establish this", not "false".** A search loop must not
  record it as a closed branch. It is the same class of answer as `E-SOS-002`.
- **Witnesses are verified.** When `decide` reports `(True, {...})` for an
  existential, the point is substituted back and checked; if it does not satisfy the
  sentence the witness is reported as `None` rather than as a certificate that fails.
  `∃x. 3x − 2 = 0` gives `(True, {'x': '2/3'})`; `∃x. x² = 2` gives `(True, None)`,
  because no rational witness exists and a midpoint of the isolating interval is
  not one.
- **Mixed-alternation sentences refuse more often** than same-flavour ones. `∀x∃y. p > 0`
  is decided through `¬∃x∀y. p ≤ 0`, and De Morgan turns a strict body into a
  non-strict one, so it can land in the refusal case even though the original body
  was strict.

If you need an answer where `decide` refuses, the routes are: a positivity
certificate (above), `alkahest.smt` with a nonlinear-real solver, or rigorous
numerics ([validated bounds](./validated-bounds.md)) if a *quantified-over-a-box*
statement is good enough.

## Scope of this release

Shipped: exact rational SOS over the DSOS generator cone, a general PSD Gram
search (floating-point proposal, exact rational verification) for cases DSOS
alone refuses — with Douglas–Rachford splitting and a facial-reduction step
alongside the original annealed alternating projection, specifically so
boundary-only (singular Gram matrix) certificates are reachable — a Reznick
multiplier search (`(Σxᵢ²)^N·p` for `N ≤ 4`) on top of that (finds both
Motzkin's polynomial and Robinson's form), Handelman certificates on basic
semialgebraic sets, exact verification, and Lean export.

Not yet shipped: reliable certification of *every* boundary-case example —
the Horn/C₅ and C₇ copositivity forms are the ones currently out of reach,
and for a reason the error message now states outright (their affine families
are over the numeric-search ceiling, so no search runs) — a proper
interior-point solver that would close them more systematically, and
Putinar-style certificates with
genuine SOS — rather than non-negative constant — multipliers on the
*constraints*. `CertificateKind::Putinar` exists in the certificate type so
those can be added without a shape change.
