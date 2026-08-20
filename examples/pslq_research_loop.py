"""A complete experimental-mathematics loop recorded as a claim graph.

This is the PSLQ -> conjecture -> prove -> certify loop, end to end, with every
stage recorded into an :class:`alkahest.research.ClaimGraph` and rendered as a
research document a human referee can read:

1. **Numeric oracle.** Compute ``I = int_0^1 x/(x^2+1) dx`` to ~50 decimal
   digits with tanh-sinh quadrature in :mod:`decimal`.  Nothing symbolic is
   used here, so the value is genuinely independent of what Alkahest will
   later derive.
2. **Discovery.** Ask :func:`alkahest.guess_relation` for an integer relation
   between ``I`` and ``log 2``.  It returns ``-2 I + log 2 = 0``, which is a
   *conjecture* -- recorded with status ``unverified``, because numerics are
   evidence and not a proof.
3. **Proof.** Have Alkahest integrate symbolically, evaluate the definite
   integral by the fundamental theorem of calculus, and simplify the residual
   ``2 I - log 2`` to zero.
4. **Certification.** The residual simplification carries a Lean 4 certificate.
   The document says so *and* says that nobody has run Lean on it.
5. **Audit.** The graph is written to JSON, read back, and re-verified from its
   serialised form.

Run it::

    python examples/pslq_research_loop.py                # print the document
    python examples/pslq_research_loop.py --out ./report # also write files
"""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
from pathlib import Path

import alkahest as ak
from alkahest.research import ClaimGraph, session

#: Working precision, in decimal digits, for the numeric oracle.
DIGITS = 60

#: pi to more digits than the oracle needs (tanh-sinh needs it for its weights).
PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510"
    "58209749445923078164062862089986280348253421170679"
)


# ---------------------------------------------------------------------------
# Stage 1 -- an independent numeric oracle
# ---------------------------------------------------------------------------


def tanh_sinh(f, lo: Decimal, hi: Decimal, level: int = 8) -> Decimal:
    """Double-exponential quadrature of *f* over ``[lo, hi]`` in :mod:`decimal`.

    Uses the substitution ``x = tanh((pi/2) sinh t)``, which converges
    doubly exponentially for analytic integrands, so ~50 digits costs a few
    hundred evaluations and no symbolic machinery whatsoever.
    """
    step = Decimal(1) / (2**level)
    half_pi = PI / 2
    centre = (hi + lo) / 2
    radius = (hi - lo) / 2
    cutoff = Decimal(10) ** (-getcontext().prec)
    total = Decimal(0)
    index = 0
    while index <= 4000:
        t = step * index
        exp_t = t.exp()
        sinh_t = (exp_t - 1 / exp_t) / 2
        cosh_t = (exp_t + 1 / exp_t) / 2
        u = half_pi * sinh_t
        exp_u = u.exp()
        inv_u = 1 / exp_u
        node = (exp_u - inv_u) / (exp_u + inv_u)
        cosh_u = (exp_u + inv_u) / 2
        if 1 - abs(node) < cutoff:
            break
        weight = half_pi * cosh_t / (cosh_u * cosh_u)
        offsets = (node,) if index == 0 else (node, -node)
        for offset in offsets:
            total += weight * f(centre + radius * offset)
        index += 1
    return total * step * radius


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_loop() -> ClaimGraph:
    """Run one iteration of the loop and return its claim graph."""
    getcontext().prec = DIGITS

    # -- stage 1: numerics ------------------------------------------------
    integral = tanh_sinh(lambda x: x / (x * x + 1), Decimal(0), Decimal(1))
    log_two = Decimal(2).ln()

    # -- stage 2: integer-relation detection ------------------------------
    constants = [str(integral), str(log_two)]
    # These are decimal strings, so nothing in them tells an exact rational from
    # a truncation, and both `guess_relation` and `relation_confidence` answer
    # "unknown" on their own.  Only this loop knows the strings are quadrature
    # output good to DIGITS places, so only this loop can declare it -- `digits=`
    # is that declaration, and `180` is the width of the search, not a claim
    # about the data.  Without it the search cannot refuse, and `None` is not a
    # pass.
    relation = ak.guess_relation(constants, 180, 10_000, digits=DIGITS)
    if relation is None:
        raise SystemExit("no integer relation found; nothing to conjecture")

    confidence = ak.relation_confidence(constants, relation, digits=DIGITS)
    if not confidence["credible"]:
        raise SystemExit(
            f"relation {relation} costs ~{confidence['consumed_digits']:.0f} digits to pin "
            f"down and only {DIGITS} were computed: purchasable from the available "
            "precision, so it is evidence of nothing"
        )

    pool = ak.ExprPool()
    x = pool.symbol("x")
    integrand = x / (x ** pool.integer(2) + pool.integer(1))
    log2 = ak.log(pool.integer(2))

    coefficients = ", ".join(str(c) for c in relation)
    with session(
        title="Closed form for the integral of x/(x^2+1) over [0, 1]",
        pool=pool,
        capture=True,
        metadata={"loop": "pslq -> conjecture -> prove -> certify", "digits": DIGITS},
    ) as s:
        # -- stage 3: the conjecture (evidence only, never "proved") ------
        conjecture = s.conjecture(
            f"{relation[0]} * integral(x/(x^2+1), dx, 0, 1) + {relation[1]} * log(2) = 0",
            evidence=(
                f"integer relation [{coefficients}] found by guess_relation at "
                f"{DIGITS} decimal digits of tanh-sinh quadrature"
            ),
            label="Conjectured closed form (numeric evidence only)",
            tags=("conjecture", "pslq"),
            notes=(
                f"I ~ {str(integral)[:32]}..., log 2 ~ {str(log_two)[:32]}...; "
                "an integer relation at finite precision is not a proof."
            ),
            check={
                "kind": "numeric_relation",
                "constants": [float(integral), float(log_two)],
                "coefficients": list(relation),
                "tolerance": 1e-12,
            },
        )

        # -- stage 4: prove it symbolically -------------------------------
        # Captured automatically: an antiderivative the kernel verifies exactly.
        ak.integrate(integrand, x)

        # Captured automatically: the definite value via the FTC.  Alkahest
        # reports this one as *unverified*, and the document says so.
        definite = ak.integrate(integrand, x, pool.integer(0), pool.integer(1))

        # Captured automatically: the residual collapses to zero, which is the
        # proof of the conjecture.  Cite the conjecture so the graph records
        # that this claim settles it.
        s.cite(conjecture)
        residual = pool.integer(-relation[0]) * definite.value + pool.integer(-relation[1]) * log2
        ak.simplify(residual)

    return s.graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="directory for JSON/MD/TeX output")
    args = parser.parse_args()

    graph = run_loop()

    print(graph.to_markdown())

    print("=" * 78)
    print("Dependency structure")
    print("=" * 78)
    for claim in graph:
        cited = graph.impact(claim.id)
        print(f"  {claim.id}  {claim.mark:<24} {claim.method}")
        print(f"      depends on : {list(claim.depends_on) or 'nothing'}")
        print(f"      falsifying it would invalidate: {list(cited) or 'nothing'}")

    print()
    print("=" * 78)
    print("Audit: reload from disk and re-verify")
    print("=" * 78)
    reloaded = ClaimGraph.from_json(graph.to_json())
    assert reloaded.digest() == graph.digest(), "round trip changed the graph"
    report = reloaded.verify()
    print(report.to_markdown())
    print(f"graph digest      : {graph.digest()}")
    print(f"re-verification ok : {report.ok}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        graph.save(args.out / "claims.json")
        (args.out / "claims.md").write_text(graph.to_markdown(), encoding="utf-8")
        (args.out / "claims.tex").write_text(graph.to_latex(), encoding="utf-8")
        print(f"\nwrote claims.json, claims.md, claims.tex to {args.out}")


if __name__ == "__main__":
    main()
