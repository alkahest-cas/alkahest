"""Shared helpers for the textbook gate (``tests/textbook_gate/``).

The textbook gate is a curated suite of first-course calculus/algebra identities
(derivatives, integrals, limits, series, sums, solving, polynomial algebra,
simplification) that should keep working across changes. See
``tests/textbook_gate/README.md`` for the rationale and how to extend it.

Verification is **behavioral** (numeric point-checks or residual checks), never
string comparison against alkahest's printed normal form — that form is an
implementation detail and changes as the simplifier evolves. Two exceptions:
``assert_definite_value`` and ``assert_sum_closed_form``, which check a
concrete numeric value against a known constant, and the exact-solution
residual checks for ``solve``.

Known-broken cases must be marked ``@pytest.mark.xfail(strict=True, reason=...)``
citing the bug (e.g. "B4, report7-20.md: sum_definite rejects Σk"). ``strict=True``
means an unexpected pass turns into a hard failure — that's the signal a fix
landed and the xfail should be deleted, promoting the case to a real assertion.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import alkahest as ak

if TYPE_CHECKING:
    from collections.abc import Sequence

# Sample points deliberately avoid 0, 1, and other special values (branch
# points, removable singularities) unless a case is specifically testing one.
DEFAULT_POINTS: tuple[float, ...] = (0.3, 0.7, 1.3, 1.9, 2.4)
POSITIVE_POINTS: tuple[float, ...] = (0.2, 0.6, 1.1, 1.8, 3.2)  # log/sqrt/gamma domains
UNIT_INTERVAL_POINTS: tuple[float, ...] = (-0.8, -0.4, 0.1, 0.5, 0.9)  # asin/acos/atanh domains
SMALL_POINTS: tuple[float, ...] = (0.05, 0.1, -0.15, 0.2, -0.3)  # near-0 series/limit checks


def eval_at(expr: ak.Expr, var: ak.Expr, value: float) -> float:
    """Numerically evaluate ``expr`` with ``var`` bound to ``value``."""
    return ak.eval_expr(expr, {var: value})


def assert_matches_reference(
    expr: ak.Expr,
    var: ak.Expr,
    reference: Callable[[float], float],
    points: Sequence[float] = DEFAULT_POINTS,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """Assert ``expr`` numerically equals ``reference(x)`` at each sample point."""
    for p in points:
        got = eval_at(expr, var, p)
        want = reference(p)
        assert math.isclose(got, want, rel_tol=rtol, abs_tol=atol), (
            f"at {var}={p}: alkahest={got!r} reference={want!r} (diff={got - want!r})"
        )


def assert_derivative_matches(
    f_expr: ak.Expr,
    var: ak.Expr,
    reference_derivative: Callable[[float], float],
    points: Sequence[float] = DEFAULT_POINTS,
    **kw: object,
) -> ak.Expr:
    """Differentiate ``f_expr`` w.r.t. ``var`` and check against a known closed form.

    Returns the computed derivative Expr for callers that want extra checks.
    """
    d = ak.diff(f_expr, var).value
    assert_matches_reference(d, var, reference_derivative, points, **kw)  # type: ignore[arg-type]
    return d


def assert_integral_self_consistent(
    integrand: ak.Expr,
    var: ak.Expr,
    points: Sequence[float] = DEFAULT_POINTS,
    **kw: object,
) -> ak.Expr:
    """Integrate ``integrand``, then assert d/dvar of the result equals the integrand.

    This is the most robust check for indefinite integrals: it does not depend
    on alkahest's particular normal form for the antiderivative (which
    constant of integration, log vs -log, etc.), only on the fundamental
    theorem of calculus holding numerically. Returns the antiderivative Expr
    in case callers want extra checks (e.g. inspecting its shape).
    """
    F = ak.integrate(integrand, var).value
    dF = ak.diff(F, var).value

    def ref(x: float) -> float:
        return eval_at(integrand, var, x)

    assert_matches_reference(dF, var, ref, points, **kw)  # type: ignore[arg-type]
    return F


def assert_definite_value(
    result_expr: ak.Expr,
    expected: float,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """Assert a constant (no free-symbol) Expr numerically equals ``expected``."""
    got = ak.eval_expr(result_expr, {})
    assert math.isclose(got, expected, rel_tol=rtol, abs_tol=atol), (
        f"alkahest={got!r} expected={expected!r}"
    )


def eval_series_truncated(series_result: ak.Series, var: ak.Expr, x0: float) -> float:
    """Numerically evaluate a truncated ``alkahest.series()`` result at ``x0``.

    ``Series.expr`` keeps its trailing ``O(...)`` remainder term baked into the
    top-level sum, which ``eval_expr`` cannot evaluate directly. This walks the
    flat n-ary ``add`` node, drops the ``big_o`` child, and numerically sums
    the rest — i.e. evaluates the truncated Taylor/Laurent polynomial.
    """
    node = series_result.expr.node()
    children = node[1] if node[0] == "add" else [series_result.expr]
    total = 0.0
    for c in children:
        if c.node()[0] == "big_o":
            continue
        total += ak.eval_expr(c, {var: x0})
    return total


def assert_series_matches_reference(
    series_result: ak.Series,
    var: ak.Expr,
    reference: Callable[[float], float],
    points: Sequence[float] = SMALL_POINTS,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> None:
    """Assert a truncated series numerically approximates ``reference`` near the
    expansion point. Tolerances are loose (truncation error, not exactness) —
    callers with high-order expansions or points very close to the center
    should tighten them.
    """
    for p in points:
        got = eval_series_truncated(series_result, var, p)
        want = reference(p)
        assert math.isclose(got, want, rel_tol=rtol, abs_tol=atol), (
            f"at {var}={p}: series={got!r} reference={want!r} (diff={got - want!r})"
        )


def assert_sum_closed_form(
    term_expr: ak.Expr,
    k: ak.Expr,
    n: ak.Expr,
    lo: ak.Expr,
    reference: Callable[[int], float],
    n_values: Sequence[int] = (1, 2, 5, 10, 20),
    *,
    rtol: float = 1e-9,
    atol: float = 1e-6,
) -> ak.Expr:
    """Compute ``sum_definite(term_expr, k, lo, n)`` symbolically in ``n``, then
    check it against ``reference(ni)`` for concrete integer values of ``n`` by
    numeric substitution (mirrors the pattern in ``tests/test_sum_v210.py``).
    """
    s = ak.sum_definite(term_expr, k, lo, n).value
    for ni in n_values:
        got = ak.eval_expr(s, {n: float(ni)})
        want = reference(ni)
        assert math.isclose(got, want, rel_tol=rtol, abs_tol=atol), (
            f"at n={ni}: sum={got!r} reference={want!r} (diff={got - want!r})"
        )
    return s


def assert_solutions_satisfy(
    equations: Sequence[ak.Expr],
    variables: Sequence[ak.Expr],
    solutions: Sequence[dict],
    *,
    expected_count: int | None = None,
    atol: float = 1e-6,
) -> None:
    """Assert each returned solution dict makes every equation evaluate ~0.

    Order- and representation-independent: doesn't care which root came back
    first or what symbolic form it's printed in, only that it's an actual root.
    """
    if expected_count is not None:
        assert len(solutions) == expected_count, (
            f"expected {expected_count} solutions, got {len(solutions)}: {solutions}"
        )
    for sol in solutions:
        bindings = {}
        for var in variables:
            val = sol[var]
            bindings[var] = ak.eval_expr(val, {}) if isinstance(val, ak.Expr) else float(val)
        for eq in equations:
            residual = ak.eval_expr(eq, bindings)
            assert abs(residual) < atol, f"solution {bindings} leaves residual {residual!r} in {eq}"
