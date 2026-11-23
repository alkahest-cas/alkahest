#!/usr/bin/env python3
"""Lean corpus of indefinite-integration ``HasDerivAt`` certificates.

Integrates a fixed list of integrands (polynomial × exp/sin/cos, rational
functions with log/atan parts, log/exp towers, inverse trig, algebraic
answers with ``sqrt`` / rational powers, multiple-angle trig) and writes every
certificate ``alkahest.integrate`` emits to ``OUTPUT/int_<n>.lean``. The Lean
CI job compiles each file with ``-DwarningAsError=true -DautoImplicit=false``.

Usage::

    python tests/lean_integrate_corpus.py --output /tmp/lean_integrate/

The integrator is allowed to change its answers; the corpus then certifies
the new ones. It fails only when *too few* integrands are certified (a broken
emitter), and lists every withheld one with its reason.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest

# Integrands in `alkahest.parse` syntax, variable `x`.
INTEGRANDS = [
    # polynomial × exp / sin / cos
    "x^2*sin(x)",
    "x*cos(x)",
    "x^3*exp(2*x)",
    "x*exp(x)",
    "exp(x)*sin(x)",
    "x*exp(x^2)",
    "(2*x^2+1)*exp(x^2)",
    "cos(2*x+3)",
    "x^3*sin(x^2)",
    # rational functions: log / atan parts
    "1/(x^3+1)",
    "x/(x^2+2*x+5)^2",
    "(x^4+1)/(x^4-1)",
    "1/(x^2+1)",
    "1/(x^4*(x^2+1))",
    "1/(x^2+2*x+1)",
    "1/x",
    # log / exp towers
    "log(x)^3",
    "x*log(x)^2",
    "log(x^2+1)",
    "exp(x)/(1+exp(x))",
    "exp(2*x)/(1+exp(x))^2",
    "1/(x*log(x))",
    "log(log(x))/x",
    "x*exp(x)/(1+x)^2",
    "1/(x*(log(x)^2+1))",
    "log(x)/(log(x)+1)^2",
    # inverse trig
    "atan(x)",
    "x*atan(x)",
    "asin(x)",
    "x^2*asin(x)",
    # algebraic
    "sqrt(1-x^2)",
    "x*sqrt(x^2+1)",
    "x/sqrt(1-x^4)",
    "x*sqrt(x+1)",
    "x^(3/2)",
    "(x^2+1)/sqrt(x)",
    "(1+x)^(1/3)/x",
    "(3*x-5)^2/(2*x-1)^(7/2)",
    "-x^2/(1-x^2)^(3/2)",
    # trigonometric
    "sin(x)^2",
    "sin(x)*cos(x)",
    "sin(3*x)*cos(5*x)",
    "tan(x)*sec(x)^2",
    "sinh(x)^2*cosh(x)",
    "exp(x)*sin(exp(x))*cos(exp(x))^2",
]

# Below this many certificates the emitter is considered broken.
MIN_CERTIFIED = 30


def certificates():
    """Yield ``(integrand, result)`` for every integrand that integrates."""
    x = alkahest.symbol("x")
    for src in INTEGRANDS:
        if "sec(" in src:
            f = alkahest.parse(src.replace("sec(x)^2", "cos(x)^(-2)"))
        else:
            f = alkahest.parse(src)
        yield src, alkahest.integrate(f, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)
    n = 0
    withheld = []
    for i, (src, r) in enumerate(certificates()):
        cert = r.certificate
        if cert is None:
            st = r.certificate_status
            withheld.append((src, st.get("detail") or st["reason"]))
            continue
        for token in ("sorry", "admit", "axiom"):
            if token in cert:
                raise SystemExit(f"{src}: certificate contains {token!r}")
        with open(os.path.join(args.output, f"int_{i:03d}.lean"), "w", encoding="utf-8") as fh:
            fh.write(cert)
        n += 1
    print(f"wrote {n} integration certificates to {args.output}")
    for src, why in withheld:
        print(f"  withheld: {src}  ({why})")
    if n < MIN_CERTIFIED:
        raise SystemExit(f"only {n} integration certificates (expected >= {MIN_CERTIFIED})")


if __name__ == "__main__":
    main()
