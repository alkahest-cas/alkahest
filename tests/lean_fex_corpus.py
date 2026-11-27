#!/usr/bin/env python3
"""Lean corpus of FEX PDE-residual certificates (reflective ``Alkahest.Kernel.Pde``).

Writes the kernel certificate of the closed-form solutions of the FEX
benchmark problems (the ones :func:`alkahest.fex.finalize` recovers) to
``OUTPUT/fex_<problem>_<d>.lean``: Poisson ``−Δ(½|x|²) = −d``, the cubic
Schrödinger equation with the paper's potential and ``u = ⅓·exp(Σcos xᵢ/d)``,
the harmonic eigenproblem ``exp(−½|x|²)`` with ``λ = d``, and the linear
conservation law ``sin(t + π/4·Σxᵢ)``. Each theorem states the PDE residual
with Mathlib's ``deriv`` along the coordinate lines, at every point of
``ℝ^d``; the Lean CI job compiles every file.

Usage::

    python tests/lean_fex_corpus.py --output /tmp/lean_fex/

Fails when any certificate is withheld (a broken emitter).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest as ak
from alkahest._fex_lean import kernel_certificate


def _sum(terms):
    terms = list(terms)
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def cases():
    out = []
    for d in (2, 3, 10):
        pool = ak.ExprPool()
        xs = [pool.symbol(f"x{i}") for i in range(d)]
        u = pool.rational(1, 2) * _sum(v**2 for v in xs)
        out.append(("poisson", d, u, xs, None, {"rhs": pool.integer(-d)}))
    for d in (2, 6, 12):
        pool = ak.ExprPool()
        xs = [pool.symbol(f"x{i}") for i in range(d)]
        S = _sum(ak.cos(v) for v in xs)
        inv = pool.rational(1, d)
        V = (
            -pool.rational(1, 9) * ak.exp(2 * inv * S)
            + pool.rational(1, d * d) * _sum(ak.sin(v) ** 2 for v in xs)
            - inv * S
        )
        u = pool.rational(1, 3) * ak.exp(inv * S)
        out.append(("cubic_schrodinger", d, u, xs, None, {"potential": V}))
    for d in (3, 10):
        pool = ak.ExprPool()
        xs = [pool.symbol(f"x{i}") for i in range(d)]
        r2 = _sum(v**2 for v in xs)
        u = ak.exp(-pool.rational(1, 2) * r2)
        out.append(("eigen", d, u, xs, None, {"potential": r2, "eigenvalue": pool.integer(d)}))
    for d in (3, 5):
        pool = ak.ExprPool()
        xs = [pool.symbol(f"x{i}") for i in range(d)]
        c = pool.pi() / 4
        u = ak.sin(xs[0] + c * _sum(xs[1:]))
        coeffs = [(d - 1) * c] + [pool.integer(-1)] * (d - 1)
        out.append(("transport", d, u, xs, 0, {"coeffs": coeffs}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    bad = []
    n = 0
    for kind, d, u, xs, time, src in cases():
        c = kernel_certificate(kind, u, xs, time, src)
        if c["status"] != "emitted":
            bad.append((kind, d, c.get("reason")))
            continue
        with open(os.path.join(args.output, f"fex_{kind}_{d}.lean"), "w") as f:
            f.write(c["lean"])
        n += 1
    print(f"wrote {n} FEX PDE certificates to {args.output}")
    for kind, d, why in bad:
        print(f"WITHHELD {kind} d={d}: {why}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
