#!/usr/bin/env python3
"""Lean corpus of ODE solution certificates (reflective ``Alkahest.Kernel``).

Solves a fixed list of ODEs with ``alkahest.experimental.dsolve`` and writes
the certificate of every explicit solution
(:func:`alkahest.experimental.dsolve_certificate`) to ``OUTPUT/ode_<n>.lean``.
Each states, on an explicit domain, that ``S, S', …`` is a derivative chain
and that the equation with them substituted vanishes; the Lean CI job compiles
every file.

Usage::

    python tests/lean_ode_corpus.py --output /tmp/lean_ode/

Fails only when too few certificates are produced (a broken emitter), and
lists every withheld solution with its reason.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest
from alkahest import experimental as ex

# (equation in `x, y, yp, ypp, yppp`, order)
ODES = [
    ("yp - y", 1),
    ("yp - 2*x*y", 1),
    ("yp + y - x", 1),
    ("yp - y*cos(x)", 1),
    ("x*yp - y", 1),
    ("ypp + y", 2),
    ("ypp - y", 2),
    ("ypp + 4*y", 2),
    ("ypp - 3*yp + 2*y", 2),
    ("ypp + 2*yp + y", 2),
    ("ypp + yp", 2),
    ("ypp + y - x", 2),
    ("ypp - y - exp(2*x)", 2),
    ("yppp - y", 3),
    ("yppp - yp", 3),
]

MIN_CERTIFIED = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    x, y = alkahest.symbol("x"), alkahest.symbol("y")
    ds = [alkahest.symbol(n) for n in ("yp", "ypp", "yppp")]
    n = 0
    withheld = []
    for i, (src, order) in enumerate(ODES):
        eq = alkahest.parse(src)
        derivs = ds[:order]
        try:
            sols = ex.dsolve(eq, x, y, derivs)
        except Exception as e:  # the solver may refuse; record it
            withheld.append((src, f"dsolve: {type(e).__name__}"))
            continue
        explicit = [
            s for s in sols if s.get("form", "explicit") == "explicit" and s["y_of_x"] is not None
        ]
        if not explicit:
            withheld.append((src, "no explicit solution"))
            continue
        out = ex.dsolve_certificate(eq, x, y, derivs, explicit[0]["y_of_x"])
        cert = out["certificate"]
        if cert is None:
            withheld.append((src, out["detail"]))
            continue
        for token in ("sorry", "admit", "axiom", "native_decide"):
            if token in cert:
                raise SystemExit(f"{src}: certificate contains {token!r}")
        with open(os.path.join(args.output, f"ode_{i:03d}.lean"), "w", encoding="utf-8") as fh:
            fh.write(cert)
        n += 1
    print(f"wrote {n} ODE solution certificates to {args.output}")
    for src, why in withheld:
        print(f"  withheld: {src}  ({why})")
    if n < MIN_CERTIFIED:
        raise SystemExit(f"only {n} ODE certificates (expected >= {MIN_CERTIFIED})")


if __name__ == "__main__":
    main()
