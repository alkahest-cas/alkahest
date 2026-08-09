# Positivity certificates (SOS / Positivstellensatz)

`decide` answers real-algebraic questions **completely**, by CAD, and pays
doubly-exponential cost for that completeness. Most positivity questions that
actually arise — is this bound valid, is this Lyapunov candidate non-negative,
is this inequality true on a box — do not need completeness. They need a
**certificate**: a short algebraic identity that makes the answer checkable by
anyone, including a proof assistant.

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
| `SosError` `E-SOS-002` | **Undecided**: no certificate of this shape at this degree | Raise the degree, or fall back to `decide` |

`E-SOS-002` is *not* a claim that the polynomial is not a sum of squares, and
certainly not that it is negative. The canonical example is the **Motzkin
polynomial** `x⁴y² + x²y⁴ − 3x²y² + 1`, which is non-negative everywhere but
provably not a sum of squares. Asked to decompose it, this module refuses with
`E-SOS-002` — it does not report it as negative, and it does not invent a
decomposition.

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
| Answers | Non-negativity, with a certificate | Any real-algebraic sentence |
| Completeness | No — refuses honestly | Yes |
| Cost | LP in exact rationals | Doubly exponential |
| Output | Checkable identity, Lean-exportable | Truth value (+ witness) |

The intended pattern is: try the certificate route first because it is cheap
and its output is citable; fall back to `decide` on `E-SOS-002` when you need
the complete answer and can afford it.

## Scope of this release

Shipped: exact rational SOS over the generator cone above, Handelman
certificates on basic semialgebraic sets, exact verification, and Lean export.

Not yet shipped: full SDP-based SOS (which needs an exact or rationally-rounded
semidefinite solver), and Putinar-style certificates with genuine SOS — rather
than non-negative constant — multipliers. `CertificateKind::Putinar` exists in
the certificate type so those can be added without a shape change.
