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
certainly not that it is negative. The canonical example is the **Motzkin
polynomial** `x⁴y² + x²y⁴ − 3x²y² + 1`, which is non-negative everywhere but
provably not a sum of squares. Asked to decompose it, this module refuses with
`E-SOS-002` — it does not report it as negative, and it does not invent a
decomposition. Choi–Lam and Robinson refuse the same way, and for the same
reason: **the refusal is a property of the search, not of the polynomial.**

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
and no floating-point SDP solver is allowed anywhere near a certificate here —
a rounded `Q` is not a proof.

So the search covers a **linear-programming-representable subcone**: the
non-negative combinations of squares of a fixed generator set,

```text
(e_i)²,    (a·e_i ± b·e_j)²   for small coprime (a, b)
```

solved with the exact rational simplex in `real::sos::lp` (Bland's rule, so
termination is unconditional and there are no epsilon tolerances). Every
generator is *literally a square*, so any feasible point is a sound certificate
by construction.

The `(1,1)` case alone is the classical diagonally-dominant (DSOS) cone. It is
not enough on its own: diagonal dominance is not invariant under scaling the
basis, so a perfect square as ordinary as `(x/2 + 1/3)²` has a Gram matrix —
its only one — that is PSD but not DD. The extra ratios widen the cone enough
to catch these while keeping the problem an LP.

The cone is still a strict subset of the SOS cone, which is exactly why
`E-SOS-002` is phrased as a statement about the search rather than about the
polynomial.

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

Shipped: exact rational SOS over the generator cone above, Handelman
certificates on basic semialgebraic sets, exact verification, and Lean export.

Not yet shipped: full SDP-based SOS (which needs an exact or rationally-rounded
semidefinite solver), and Putinar-style certificates with genuine SOS — rather
than non-negative constant — multipliers. `CertificateKind::Putinar` exists in
the certificate type so those can be added without a shape change.
