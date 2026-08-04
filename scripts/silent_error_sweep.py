#!/usr/bin/env python
"""Differential / self-consistency sweep hunting for *silent errors*.

A silent error is a confident, plausible, mathematically wrong answer returned
with no exception and no verification flag.  This harness attacks that claim by
running alkahest over a generated corpus and cross-checking every non-raising
answer against independent evidence:

* **SymPy** — an independent implementation (test-only; deliberately *not* a
  dependency of alkahest).
* **mpmath** — high-precision quadrature / partial sums / directional
  evaluation, used as an oracle-free self-consistency check.  Where the two
  disagree the case is reported as ``DISPUTED`` rather than as a bug, because
  two disagreeing implementations do not say which one is wrong.

It is *not* part of the default ``pytest`` run: it needs SymPy and it is slow.
Run it directly::

    python scripts/silent_error_sweep.py                 # all surfaces
    python scripts/silent_error_sweep.py --surface limit
    python scripts/silent_error_sweep.py --verbose

Exit status is 1 when at least one ``WRONG`` verdict is produced.

Verdicts
--------
``OK``        alkahest's answer agrees with the evidence.
``REFUSED``   alkahest raised — always acceptable, never a silent error.
``WRONG``     alkahest returned a value the evidence contradicts.  This is the
              thing the project claims does not happen.
``DISPUTED``  evidence is inconclusive or the oracles disagree with each other.
``SKIP``      the case could not be evaluated on one side at all.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

import mpmath as mp
import sympy as sp

import alkahest as ak

mp.mp.dps = 40

#: Wall-clock cap on a single oracle call.  SymPy can spend minutes on a single
#: ``integrate``/``limit``; a sweep must not be held hostage by one input, and a
#: timed-out oracle is simply "no evidence", never a verdict.
ORACLE_TIMEOUT_SECONDS = 20


class _Timeout(Exception):
    pass


@contextmanager
def time_limit(seconds=ORACLE_TIMEOUT_SECONDS):
    def handler(signum, frame):
        raise _Timeout()

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def guarded(fn, *args, **kwargs):
    """Run *fn*, returning ``None`` on any failure or timeout."""
    try:
        with time_limit():
            return fn(*args, **kwargs)
    except (_Timeout, Exception):  # noqa: B014 - oracles raise anything
        return None


OK, REFUSED, WRONG, DISPUTED, SKIP = "OK", "REFUSED", "WRONG", "DISPUTED", "SKIP"


@dataclass
class Report:
    counts: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)

    def record(self, verdict, surface, label, detail=""):
        self.counts[verdict] = self.counts.get(verdict, 0) + 1
        if verdict in (WRONG, DISPUTED):
            self.findings.append((verdict, surface, label, detail))

    @property
    def total(self):
        return sum(self.counts.values())


# ---------------------------------------------------------------------------
# SymPy -> alkahest translation
# ---------------------------------------------------------------------------

_UNARY = {
    sp.sin: "sin",
    sp.cos: "cos",
    sp.tan: "tan",
    sp.exp: "exp",
    sp.log: "log",
    sp.sinh: "sinh",
    sp.cosh: "cosh",
    sp.tanh: "tanh",
    sp.asin: "asin",
    sp.acos: "acos",
    sp.atan: "atan",
}


def to_alkahest(expr, pool, var_map):
    """Translate a SymPy expression into the alkahest pool.

    Raises ``NotImplementedError`` for anything outside the shared fragment, so
    a case that cannot be expressed identically on both sides is skipped rather
    than silently compared against a different problem.
    """
    if expr.is_Integer:
        return pool.integer(int(expr))
    if expr.is_Rational:
        return pool.rational(int(expr.p), int(expr.q))
    if expr.is_Symbol:
        if expr not in var_map:
            raise NotImplementedError(f"free symbol {expr}")
        return var_map[expr]
    if expr.is_Add:
        out = to_alkahest(expr.args[0], pool, var_map)
        for a in expr.args[1:]:
            out = out + to_alkahest(a, pool, var_map)
        return out
    if expr.is_Mul:
        out = to_alkahest(expr.args[0], pool, var_map)
        for a in expr.args[1:]:
            out = out * to_alkahest(a, pool, var_map)
        return out
    if expr.is_Pow:
        base = to_alkahest(expr.base, pool, var_map)
        exponent = expr.exp
        if exponent.is_Integer:
            return base ** pool.integer(int(exponent))
        if exponent.is_Rational:
            return base ** pool.rational(int(exponent.p), int(exponent.q))
        return base ** to_alkahest(exponent, pool, var_map)
    for cls, name in _UNARY.items():
        if isinstance(expr, cls):
            return getattr(ak, name)(to_alkahest(expr.args[0], pool, var_map))
    raise NotImplementedError(f"cannot translate {expr!r}")


def numeric(expr):
    """Best-effort float value of an alkahest expression with no free symbols."""
    try:
        return float(ak.eval_expr(expr, {}))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Surface 1 — definite integration
# ---------------------------------------------------------------------------

X = sp.Symbol("x")


def _definite_corpus():
    """(integrand, lower, upper) triples in SymPy, mixing proper and improper."""
    atoms = [
        X,
        X**2,
        X**3,
        1 / X,
        1 / X**2,
        1 / (X - 1),
        1 / (X + 2),
        1 / (X**2 + 1),
        1 / (X**2 - 1),
        1 / (X**2 - 4),
        1 / (2 * X - 1),
        X / (X**2 - 1),
        (X + 1) / (X - 1),
        1 / (X - 1) ** 2,
        1 / (X**3 - X),
        sp.sin(X),
        sp.cos(X),
        sp.tan(X),
        1 / sp.cos(X),
        1 / sp.cos(X) ** 2,
        1 / sp.sin(X),
        1 / sp.sin(X) ** 2,
        sp.exp(X),
        sp.exp(-X),
        X * sp.exp(X),
        sp.log(X),
        X * sp.log(X),
        sp.sqrt(X),
        1 / sp.sqrt(X),
        X ** sp.Rational(-3, 2),
        X ** sp.Rational(-1, 3),
        1 / (1 - X),
        1 / (1 + sp.exp(X)),
        sp.sin(X) * sp.cos(X),
        sp.tan(X) ** 2,
        X / (X - 2),
        1 / (X * (X - 1)),
        sp.atan(X),
        1 / (1 + X**2) ** 2,
        sp.exp(X) / (1 + sp.exp(X)),
    ]
    bounds = [
        (0, 1),
        (-1, 1),
        (0, 2),
        (1, 2),
        (-2, 2),
        (sp.Rational(1, 2), 2),
        (0, 3),
        (-1, 2),
    ]
    return [(f, a, b) for f in atoms for (a, b) in bounds]


def _true_definite(f, a, b):
    """Ground truth for the definite integral.

    Returns ``("value", v)``, ``("divergent", None)`` or ``("unknown", None)``.
    SymPy is the primary oracle; a finite SymPy value is confirmed by mpmath
    quadrature before it is trusted, and mpmath alone can establish divergence
    when SymPy gives up.
    """
    sym = guarded(sp.integrate, f, (X, a, b))

    if sym is not None and not sym.has(sp.Integral):
        if sym in (sp.oo, -sp.oo, sp.zoo) or sym is sp.nan:
            return ("divergent", None)
        val = guarded(lambda: complex(sp.N(sym, 30)))
        if val is not None and math.isfinite(val.real) and math.isfinite(val.imag):
            if abs(val.imag) > 1e-12 * max(1.0, abs(val.real)):
                # Real integrand with a complex "value" => the real integral
                # does not exist as an ordinary Lebesgue/Riemann integral.
                return ("divergent", None)
            num = _quad(f, a, b)
            if num is None:
                return ("value", val.real)
            if abs(num - val.real) <= 1e-6 * max(1.0, abs(val.real)):
                return ("value", val.real)
            return ("unknown", None)

    num = _quad(f, a, b)
    if num is None:
        return ("divergent", None) if _blows_up(f, a, b) else ("unknown", None)
    return ("value", num)


def _quad(f, a, b):
    """mpmath quadrature, with interior singular points handed to the splitter."""
    fn = sp.lambdify(X, f, "mpmath")
    lo, hi = float(a), float(b)
    path = [lo] + _interior_singularities(f, lo, hi) + [hi]

    def run():
        with mp.workdps(40):
            val = mp.quad(fn, path)
            if not mp.isfinite(val):
                return None
            if abs(mp.im(val)) > 1e-12 * max(1.0, abs(mp.re(val))):
                return None
            return float(mp.re(val))

    return guarded(run)


def _interior_singularities(f, lo, hi):
    """Real singularities of *f* strictly inside ``(lo, hi)``, sorted."""
    sings = guarded(sp.singularities, f, X)
    if sings is None:
        return []
    out = []
    for s in sings:
        try:
            if s.is_real and lo < float(s) < hi:
                out.append(float(s))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _blows_up(f, a, b):
    """True when the integrand is unbounded strictly inside ``(a, b)``."""
    return bool(_interior_singularities(f, float(a), float(b)))


def sweep_definite(report, verbose=False):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    for f, a, b in _definite_corpus():
        label = f"∫_{a}^{b} {f} dx"
        try:
            fa = to_alkahest(f, pool, {X: x})
            lo = to_alkahest(sp.sympify(a), pool, {X: x})
            hi = to_alkahest(sp.sympify(b), pool, {X: x})
        except NotImplementedError:
            report.record(SKIP, "definite_integral", label)
            continue
        try:
            got = ak.integrate(fa, x, lo, hi).value
        except Exception as e:
            report.record(REFUSED, "definite_integral", label, str(e)[:120])
            continue
        kind, truth = _true_definite(f, a, b)
        if kind == "divergent":
            report.record(
                WRONG,
                "definite_integral",
                label,
                f"integral does not converge; alkahest returned {got}",
            )
            continue
        if kind == "unknown":
            report.record(DISPUTED, "definite_integral", label, f"got {got}")
            continue
        val = numeric(got)
        if val is None:
            report.record(
                WRONG,
                "definite_integral",
                label,
                f"returned {got}, which does not denote a real number (true value {truth})",
            )
            continue
        if abs(val - truth) <= 1e-6 * max(1.0, abs(truth)):
            report.record(OK, "definite_integral", label)
        else:
            report.record(WRONG, "definite_integral", label, f"got {val}, true value {truth}")
        if verbose:
            print(f"  {label}: {got}")


# ---------------------------------------------------------------------------
# Surface 2 — limits
# ---------------------------------------------------------------------------


def _limit_corpus():
    exprs = [
        sp.sin(X) / X,
        (1 - sp.cos(X)) / X**2,
        sp.tan(X) / X,
        X * sp.sin(1 / X),
        sp.sin(1 / X),
        sp.cos(1 / X),
        sp.exp(1 / X),
        sp.exp(-1 / X**2),
        1 / X,
        1 / X**2,
        sp.log(X),
        sp.sqrt(X),
        X**X if False else sp.exp(X * sp.log(X)),
        (1 + 1 / X) ** X,
        sp.sin(X),
        sp.cos(X),
        sp.tan(X),
        X * sp.sin(X),
        sp.sin(X) / sp.sqrt(X),
        sp.exp(-X) * sp.sin(X),
        sp.exp(X) / X**3,
        sp.log(X) / X,
        X / sp.log(X),
        (sp.exp(X) - 1) / X,
        sp.atan(X),
        sp.sin(X) ** 2,
        (X**2 - 1) / (X - 1),
        (X**2 + 1) / (X**2 - 1),
        sp.sqrt(X**2 + 1) - X,
        sp.sin(X) * sp.cos(1 / X),
    ]
    points = [0, 1, sp.oo]
    return [(e, p) for e in exprs for p in points]


def _true_limit(expr, point):
    """SymPy's two-sided limit, or ``("dne", None)`` when it does not exist."""
    try:
        left = sp.limit(expr, X, point, "-") if point is not sp.oo else None
        right = sp.limit(expr, X, point, "+")
    except Exception:
        return ("unknown", None)
    if right.has(sp.AccumBounds) or (left is not None and left.has(sp.AccumBounds)):
        return ("dne", None)
    if left is not None and sp.simplify(left - right) != 0:
        return ("dne", None)
    if right in (sp.oo, -sp.oo, sp.zoo):
        return ("infinite", right)
    if right is sp.nan:
        return ("dne", None)
    try:
        val = complex(sp.N(right, 30))
    except Exception:
        return ("unknown", None)
    if not (math.isfinite(val.real) and math.isfinite(val.imag)):
        return ("unknown", None)
    if abs(val.imag) > 1e-12:
        return ("unknown", None)
    return ("value", val.real)


def sweep_limits(report, verbose=False):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    inf = pool.pos_infinity()
    for expr, point in _limit_corpus():
        label = f"lim_{{x→{point}}} {expr}"
        try:
            fa = to_alkahest(expr, pool, {X: x})
            pt = inf if point is sp.oo else to_alkahest(sp.sympify(point), pool, {X: x})
        except NotImplementedError:
            report.record(SKIP, "limit", label)
            continue
        try:
            got = ak.limit(fa, x, pt)
        except Exception as e:
            report.record(REFUSED, "limit", label, str(e)[:100])
            continue
        kind, truth = _true_limit(expr, point)
        got_s = str(got)
        if kind == "dne":
            report.record(WRONG, "limit", label, f"limit does not exist; alkahest returned {got_s}")
            continue
        if kind == "unknown":
            report.record(DISPUTED, "limit", label, f"got {got_s}")
            continue
        if kind == "infinite":
            if "∞" in got_s:
                report.record(OK, "limit", label)
            else:
                report.record(WRONG, "limit", label, f"limit is {truth}; alkahest returned {got_s}")
            continue
        val = numeric(got)
        if val is None:
            report.record(DISPUTED, "limit", label, f"got {got_s}, not numerically evaluable")
            continue
        if abs(val - truth) <= 1e-8 * max(1.0, abs(truth)):
            report.record(OK, "limit", label)
        else:
            report.record(WRONG, "limit", label, f"got {val}, true limit {truth}")
        if verbose:
            print(f"  {label}: {got_s}")


# ---------------------------------------------------------------------------
# Surface 3 — sums and products
# ---------------------------------------------------------------------------

K = sp.Symbol("k")


def _sum_corpus():
    terms = [
        K,
        K**2,
        K**3,
        2 * K + 1,
        1 / (K * (K + 1)),
        1 / ((K + 1) * (K + 2)),
        1 / K,
        1 / K**2,
        1 / K**3,
        1 / K**4,
        2**K,
        2 ** (-K),
        (-1) ** K,
        sp.Rational(1, 2) ** K,
        3 ** (-K),
        K * 2**K,
        1 / (K**2 - 1),
    ]
    ranges = [(1, 10), (1, 5), (0, 6), (2, 8), (1, sp.oo), (5, 1), (3, 2)]
    return [(t, a, b) for t in terms for (a, b) in ranges]


def _true_sum(term, lo, hi):
    if hi is not sp.oo and int(hi) < int(lo):
        # Empty range under the standard convention.  SymPy applies the
        # reversal convention instead, so this is reported separately and never
        # scored as WRONG — see the report.
        return ("empty", 0.0)
    try:
        with mp.workdps(40):
            fn = sp.lambdify(K, term, "mpmath")
            if hi is sp.oo:
                val = mp.nsum(fn, [int(lo), mp.inf])
            else:
                val = mp.fsum(fn(j) for j in range(int(lo), int(hi) + 1))
        if not mp.isfinite(val):
            return ("divergent", None)
        return ("value", float(val))
    except Exception:
        pass
    try:
        s = sp.Sum(term, (K, lo, hi)).doit()
        if s in (sp.oo, -sp.oo, sp.zoo) or s is sp.nan or s.has(sp.Sum):
            return ("divergent", None)
        return ("value", float(sp.N(s, 30)))
    except Exception:
        return ("unknown", None)


def sweep_sums(report, verbose=False):
    pool = ak.ExprPool()
    k = pool.symbol("k")
    inf = pool.pos_infinity()
    for term, lo, hi in _sum_corpus():
        label = f"Σ_{{k={lo}}}^{{{hi}}} {term}"
        try:
            ta = to_alkahest(term, pool, {K: k})
            lo_a = pool.integer(int(lo))
            hi_a = inf if hi is sp.oo else pool.integer(int(hi))
        except (NotImplementedError, TypeError):
            report.record(SKIP, "sum", label)
            continue
        try:
            got = ak.sum_definite(ta, k, lo_a, hi_a).value
        except Exception as e:
            report.record(REFUSED, "sum", label, str(e)[:100])
            continue
        kind, truth = _true_sum(term, lo, hi)
        if kind == "unknown":
            report.record(DISPUTED, "sum", label, f"got {got}")
            continue
        val = numeric(got)
        if kind == "divergent":
            report.record(WRONG, "sum", label, f"series diverges; alkahest returned {got}")
            continue
        if kind == "empty":
            if val is not None and abs(val) > 1e-9:
                report.record(
                    DISPUTED,
                    "sum",
                    label,
                    f"empty range (hi < lo): alkahest returned {got} "
                    "(reversal convention, undocumented)",
                )
            else:
                report.record(OK, "sum", label)
            continue
        if val is None:
            report.record(DISPUTED, "sum", label, f"got {got}, not evaluable")
            continue
        if abs(val - truth) <= 1e-8 * max(1.0, abs(truth)):
            report.record(OK, "sum", label)
        else:
            report.record(WRONG, "sum", label, f"got {val}, true sum {truth}")
        if verbose:
            print(f"  {label}: {got}")


SURFACES = {
    "definite_integral": sweep_definite,
    "limit": sweep_limits,
    "sum": sweep_sums,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", choices=sorted(SURFACES), action="append")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    report = Report()
    for name in args.surface or sorted(SURFACES):
        print(f"== {name} ==")
        SURFACES[name](report, verbose=args.verbose)

    print()
    print(f"swept {report.total} inputs")
    for verdict in (OK, REFUSED, WRONG, DISPUTED, SKIP):
        if verdict in report.counts:
            print(f"  {verdict:9s} {report.counts[verdict]}")
    if report.findings:
        print()
        for verdict, surface, label, detail in report.findings:
            print(f"[{verdict}] {surface}: {label}\n    {detail}")
    return 1 if report.counts.get(WRONG) else 0


if __name__ == "__main__":
    sys.exit(main())
