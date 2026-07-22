"""Textbook gate — ordinary differential equations.

First-course ODEs (calc 2 / intro diff-eq level) via ``alkahest.experimental.dsolve``:
separable equations, first-order linear equations, and second-order
constant-coefficient equations (distinct real roots, complex/oscillatory
roots, and repeated roots). See ``tests/textbook_gate/README.md`` for the
verification philosophy (numeric checks, never string-matching alkahest's
printed normal form).

``dsolve(equation, x, y, derivs)`` treats ``equation`` as ``equation == 0``,
written in terms of the independent variable ``x``, a plain symbol ``y``
standing for the unknown function (not ``y(x)`` notation), and a list of
plain symbols ``derivs`` standing for y', y'', ... in order. It returns a
list of solution dicts with keys ``'y_of_x'`` (the general solution, an
``Expr`` in ``x`` and named constants), ``'constants'`` (the free constant
symbols appearing in ``y_of_x``), and ``'method'``.

Verification does not check the solution's shape or the reported
``'method'`` string (e.g. separable equations are sometimes reported back as
``'linear'``) — it substitutes the candidate solution (and its derivatives)
into the original equation and checks the residual is numerically ~0 for
concrete values of the free constants, the ODE analogue of "differentiate
what you integrated" used for indefinite integrals elsewhere in this suite.
"""

from __future__ import annotations

import alkahest as ak
import alkahest.experimental as ax
import pytest

DEFAULT_POINTS: tuple[float, ...] = (0.3, 0.7, 1.3, 1.9, 2.4)
POSITIVE_POINTS: tuple[float, ...] = (0.2, 0.6, 1.1, 1.8, 3.2)  # avoid x=0 (log/1/x domain)


@pytest.fixture
def pool() -> ak.ExprPool:
    return ak.ExprPool()


@pytest.fixture
def x(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("x")


@pytest.fixture
def y(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("y")


@pytest.fixture
def yp(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("yp")


@pytest.fixture
def ypp(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("ypp")


def assert_ode_solution_satisfies(
    equation: ak.Expr,
    x: ak.Expr,
    y: ak.Expr,
    derivs: list[ak.Expr],
    sol: dict,
    const_values: list[float],
    x_points: tuple[float, ...] = DEFAULT_POINTS,
) -> None:
    """Substitute a ``dsolve()`` solution (and its derivatives) back into the
    original ``equation`` and assert the residual evaluates to ~0.

    ``sol`` is one solution dict from ``dsolve()``'s return list. This
    differentiates ``sol['y_of_x']`` the right number of times to build up
    each entry of ``derivs``, substitutes into ``equation``, binds the free
    constants to concrete numbers, and checks the residual numerically —
    order- and normal-form-independent, unlike checking the solution's shape.
    """
    y_of_x = sol["y_of_x"]
    subs_map = {y: y_of_x}
    d = y_of_x
    for dv in derivs:
        d = ak.diff(d, x).value
        subs_map[dv] = d
    residual = ak.subs(equation, subs_map)
    const_bindings = dict(zip(sol["constants"], const_values))
    for xv in x_points:
        b = dict(const_bindings)
        b[x] = xv
        val = ak.eval_expr(residual, b)
        assert abs(val) < 1e-6, f"residual={val!r} at x={xv}, constants={const_bindings}"


# --- separable ---------------------------------------------------------------


def test_separable_y_prime_eq_x_times_y(pool, x, y, yp):
    sol = ax.dsolve(yp - x * y, x, y, [yp])[0]
    assert_ode_solution_satisfies(yp - x * y, x, y, [yp], sol, [2.0])


def test_separable_y_prime_eq_2x(pool, x, y, yp):
    sol = ax.dsolve(yp - 2 * x, x, y, [yp])[0]
    assert_ode_solution_satisfies(yp - 2 * x, x, y, [yp], sol, [1.5])


def test_separable_y_prime_eq_y_over_x(pool, x, y, yp):
    # y = C*x, but the residual check needs x != 0 (equation itself divides by x).
    sol = ax.dsolve(yp - y / x, x, y, [yp])[0]
    assert_ode_solution_satisfies(yp - y / x, x, y, [yp], sol, [2.0], x_points=POSITIVE_POINTS)


# --- first-order linear -------------------------------------------------------


def test_linear_y_prime_eq_y(pool, x, y, yp):
    sol = ax.dsolve(yp - y, x, y, [yp])[0]
    assert_ode_solution_satisfies(yp - y, x, y, [yp], sol, [2.0])


def test_linear_y_prime_plus_y_eq_zero(pool, x, y, yp):
    """Same equation as ``y' = y`` up to sign (y' + y = 0 <=> y' = -y), phrased
    differently — checks the two land on equivalent (residual-satisfying)
    solutions rather than assuming the parser/solver treats them identically.
    """
    sol = ax.dsolve(yp + y, x, y, [yp])[0]
    assert_ode_solution_satisfies(yp + y, x, y, [yp], sol, [2.0])


def test_linear_y_prime_minus_3y_eq_zero(pool, x, y, yp):
    sol = ax.dsolve(yp - 3 * y, x, y, [yp])[0]
    assert_ode_solution_satisfies(yp - 3 * y, x, y, [yp], sol, [1.5])


# --- second-order constant-coefficient, distinct real roots ------------------


def test_second_order_distinct_real_roots_pm1(pool, x, y, yp, ypp):
    # y'' - y = 0: characteristic roots +-1.
    eq = ypp - y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, -1.5])


def test_second_order_distinct_real_roots_2_3(pool, x, y, yp, ypp):
    # y'' - 5y' + 6y = 0: characteristic roots 2, 3.
    eq = ypp - 5 * yp + 6 * y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, -1.5])


def test_second_order_distinct_real_roots_pm2(pool, x, y, yp, ypp):
    # y'' - 4y = 0: characteristic roots +-2.
    eq = ypp - 4 * y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, -1.5])


def test_second_order_distinct_real_roots_neg1_neg2(pool, x, y, yp, ypp):
    # y'' + 3y' + 2y = 0: characteristic roots -1, -2.
    eq = ypp + 3 * yp + 2 * y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, -1.5])


def test_second_order_distinct_real_roots_0_1(pool, x, y, yp, ypp):
    # y'' - y' = 0: characteristic roots 0, 1 (solution is C1 + C2*exp(x)).
    eq = ypp - yp
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, -1.5])


# --- second-order constant-coefficient, complex roots (oscillatory) ----------


def test_second_order_complex_roots_frequency_1(pool, x, y, yp, ypp):
    # y'' + y = 0: characteristic roots +-i, oscillates at frequency 1.
    eq = ypp + y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.5, 1.5])


def test_second_order_complex_roots_frequency_2(pool, x, y, yp, ypp):
    # y'' + 4y = 0: characteristic roots +-2i, oscillates at frequency 2.
    eq = ypp + 4 * y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, -2.0])


# --- second-order constant-coefficient, repeated root -------------------------


def test_second_order_repeated_root_at_1(pool, x, y, yp, ypp):
    # y'' - 2y' + y = 0: double root at 1, solution ~ C1*exp(x) + C2*x*exp(x).
    eq = ypp - 2 * yp + y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, 2.0])


def test_second_order_repeated_root_at_neg1(pool, x, y, yp, ypp):
    # y'' + 2y' + y = 0: double root at -1.
    eq = ypp + 2 * yp + y
    sol = ax.dsolve(eq, x, y, [yp, ypp])[0]
    assert_ode_solution_satisfies(eq, x, y, [yp, ypp], sol, [1.0, -2.0])
