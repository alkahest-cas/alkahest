#!/usr/bin/env python3
"""
Lean SOS corpus generator.

Generates a strict, no-admission Lean proof corpus from real
``sos_decompose`` / ``prove_nonneg`` certificates. Used by the Lean CI job.

Empty, ``None``, or admission-bearing ``to_lean()`` output is never written
(withhold rather than sorry).

Usage::

    python tests/lean_sos_corpus.py --output /tmp/lean_sos/
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alkahest

FORBIDDEN_TOKENS = ("sorry", "admit", "axiom")
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_TOKENS) + r")\b")


def _square(e):
    return e * e


def _square_diff(pool):
    """(x − y)² — unconstrained SOS identity."""
    x, y = pool.symbol("x"), pool.symbol("y")
    return alkahest.sos_decompose(_square(x - y), [x, y])


def _sum_of_squares(pool):
    """x² + y² — unconstrained SOS identity."""
    x, y = pool.symbol("x"), pool.symbol("y")
    return alkahest.sos_decompose(_square(x) + _square(y), [x, y])


def _univariate_square(pool):
    """x² — unconstrained univariate SOS, via prove_nonneg with no constraints."""
    x = pool.symbol("x")
    return alkahest.prove_nonneg(_square(x), [x])


def _square_with_spectator_var(pool):
    """x² over the variable list ``[x, y]`` — ``y`` occurs in nothing.

    Binding a variable the statement never mentions is a hard
    ``unusedVariables`` error under ``-DwarningAsError=true``, and leaves ``rw
    [alkahest_sos_identity]`` with an uninstantiated ``⊢ ℝ`` side goal. Every
    case above happens to use all its variables, which is why this shipped.
    """
    x, y = pool.symbol("x"), pool.symbol("y")
    return alkahest.sos_decompose(_square(x), [x, y])


def _shifted_square_with_spectator_var(pool):
    """(x − 1)² over ``[x, y]`` — same, through the constrained-free branch."""
    x, y = pool.symbol("x"), pool.symbol("y")
    return alkahest.prove_nonneg(_square(x) - 2 * x + 1, [x, y])


def _motzkin(pool):
    """Motzkin polynomial via a Reznick multiplier (not itself SOS)."""
    x, y = pool.symbol("x"), pool.symbol("y")
    p = (
        _square(x) * _square(x) * _square(y)
        + _square(x) * _square(y) * _square(y)
        - pool.integer(3) * _square(x) * _square(y)
        + pool.integer(1)
    )
    return alkahest.sos_decompose(p, [x, y])


# (name, certificate builder). Keep this list to identities whose Lean
# actually typechecks; drop a case rather than emit sorry.
CASES = [
    ("square_diff", _square_diff),
    ("sum_of_squares", _sum_of_squares),
    ("univariate_square", _univariate_square),
    ("square_with_spectator_var", _square_with_spectator_var),
    ("shifted_square_with_spectator_var", _shifted_square_with_spectator_var),
    ("motzkin_multiplier", _motzkin),
]


def generate_proof(name: str, builder, pool) -> str:
    """Generate one sorry-free Lean proof from a positivity certificate."""
    cert = builder(pool)
    lean_src = cert.to_lean()
    if not lean_src or not str(lean_src).strip():
        raise ValueError(f"{name}: to_lean() withheld (empty/None)")
    match = _FORBIDDEN_RE.search(lean_src)
    if match:
        raise ValueError(f"{name}: generated Lean source contains {match.group(0)!r}")
    return lean_src


def main():
    parser = argparse.ArgumentParser(
        description="Generate Lean proofs for Alkahest SOS positivity certificates"
    )
    parser.add_argument("--output", default=".", help="Output directory for .lean files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    pool = alkahest.ExprPool()
    success = 0
    for name, builder in CASES:
        try:
            lean_src = generate_proof(name, builder, pool)
            out_path = os.path.join(args.output, f"{name}.lean")
            with open(out_path, "w") as f:
                f.write(lean_src)
            print(f"Generated: {out_path}")
            success += 1
        except Exception as e:
            print(f"ERROR generating {name}: {e}", file=sys.stderr)

    print(f"\n{success}/{len(CASES)} SOS proofs generated in {args.output}")
    return 0 if success == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
