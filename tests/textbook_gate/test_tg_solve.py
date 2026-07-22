"""Textbook gate — equation solving.

First-course solving: linear systems, quadratics (symbolic back-substitution),
two-variable nonlinear systems solvable via Gröbner elimination, and
cubic/quartic systems that need numeric (homotopy continuation) solving. See
``tests/textbook_gate/README.md`` for the verification philosophy — every
solution is checked by substituting it back into the original equations and
confirming the residual is ~0, never by comparing to a printed normal form.

B5 (report7-20.md): ``ak.solve(..., numeric=True)`` falls back past the
symbolic HighDegree limit (homotopy). Direct ``method="homotopy"`` and
``ak.solve_numerical`` paths are also exercised below.
"""

from __future__ import annotations

import alkahest as ak
import pytest
from _tg_helpers import assert_solutions_satisfy


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
def z(pool: ak.ExprPool) -> ak.Expr:
    return pool.symbol("z")


# --- linear systems -----------------------------------------------------


def test_solve_linear_2x2_unique(pool, x, y):
    """x + y = 3, x - y = 1 -> x=2, y=1."""
    eqs = [x + y - pool.integer(3), x - y - pool.integer(1)]
    sols = ak.solve(eqs, [x, y])
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=1)


def test_solve_linear_2x2_fractional_solution(pool, x, y):
    """2x + 3y = 7, 4x - y = 1 -> a non-integer (rational) unique solution."""
    eqs = [
        pool.integer(2) * x + pool.integer(3) * y - pool.integer(7),
        pool.integer(4) * x - y - pool.integer(1),
    ]
    sols = ak.solve(eqs, [x, y])
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=1)


def test_solve_linear_3x3_unique(pool, x, y, z):
    """x+y+z=6, x-y+z=2, x+y-z=0 -> x=1, y=2, z=3."""
    eqs = [
        x + y + z - pool.integer(6),
        x - y + z - pool.integer(2),
        x + y - z,
    ]
    sols = ak.solve(eqs, [x, y, z])
    assert_solutions_satisfy(eqs, [x, y, z], sols, expected_count=1)


def test_solve_linear_3x3_negative_coefficients(pool, x, y, z):
    """-x + 2y - z = -3, 3x - y + z = 8, x + y + z = 4 -> a second 3x3 case."""
    eqs = [
        -x + pool.integer(2) * y - z - pool.integer(-3),
        pool.integer(3) * x - y + z - pool.integer(8),
        x + y + z - pool.integer(4),
    ]
    sols = ak.solve(eqs, [x, y, z])
    assert_solutions_satisfy(eqs, [x, y, z], sols, expected_count=1)


# --- quadratics (symbolic back-substitution, degree <= 2) ---------------


def test_solve_quadratic_two_real_roots(pool, x):
    """x^2 - 4 = 0 -> x = +-2."""
    eqs = [x**2 - pool.integer(4)]
    sols = ak.solve(eqs, [x])
    assert_solutions_satisfy(eqs, [x], sols, expected_count=2)


def test_solve_quadratic_via_factoring(pool, x):
    """x^2 - 5x + 6 = 0 -> x = 2 or x = 3."""
    eqs = [x**2 - pool.integer(5) * x + pool.integer(6)]
    sols = ak.solve(eqs, [x])
    assert_solutions_satisfy(eqs, [x], sols, expected_count=2)


def test_solve_quadratic_repeated_root(pool, x):
    """x^2 - 4x + 4 = 0 -> x = 2 (double root); the solver returns two
    syntactically distinct-but-equal-valued entries (both simplify to 2),
    so we check the count it actually produces rather than assume dedup."""
    eqs = [x**2 - pool.integer(4) * x + pool.integer(4)]
    sols = ak.solve(eqs, [x])
    assert_solutions_satisfy(eqs, [x], sols, expected_count=2)
    for sol in sols:
        val = ak.eval_expr(sol[x], {})
        assert abs(val - 2.0) < 1e-9


def test_solve_quadratic_no_linear_term(pool, x):
    """x^2 - 9 = 0 -> x = +-3, a pure difference-of-squares case."""
    eqs = [x**2 - pool.integer(9)]
    sols = ak.solve(eqs, [x])
    assert_solutions_satisfy(eqs, [x], sols, expected_count=2)


# --- two-variable nonlinear systems (Gröbner elimination) ---------------


def test_solve_circle_and_line(pool, x, y):
    """x^2+y^2=25, x-y=1 -> (4,3) and (-3,-4)."""
    eqs = [x**2 + y**2 - pool.integer(25), x - y - pool.integer(1)]
    sols = ak.solve(eqs, [x, y])
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=2)


def test_solve_hyperbola_and_line(pool, x, y):
    """xy=12, x+y=7 -> (3,4) and (4,3)."""
    eqs = [x * y - pool.integer(12), x + y - pool.integer(7)]
    sols = ak.solve(eqs, [x, y])
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=2)


def test_solve_parabola_and_line(pool, x, y):
    """y=x^2, y=x+2 -> (2,4) and (-1,1)."""
    eqs = [y - x**2, y - x - pool.integer(2)]
    sols = ak.solve(eqs, [x, y])
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=2)


def test_solve_two_conics(pool, x, y):
    """x^2+y^2=13, x^2-y=7 -> two conics meeting at (+-3,2) and (+-2,-3)."""
    eqs = [x**2 + y**2 - pool.integer(13), x**2 - y - pool.integer(7)]
    sols = ak.solve(eqs, [x, y])
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=4)


def test_solve_two_circles(pool, x, y):
    """x^2+y^2=25, (x-3)^2+y^2=16 -> two intersecting circles, meeting at
    (3,4) and (3,-4) (a 3-4-5 right triangle)."""
    eqs = [x**2 + y**2 - pool.integer(25), (x - pool.integer(3)) ** 2 + y**2 - pool.integer(16)]
    sols = ak.solve(eqs, [x, y])
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=2)


# --- cubic/quartic systems (numeric, method="homotopy") -----------------


def test_solve_cubic_homotopy_plain(pool, x):
    """x^3 - 6x^2 + 11x - 6 = 0 -> roots 1, 2, 3 (symbolic back-substitution
    can't reach this — degree 3 — so this exercises the numeric path)."""
    eqs = [x**3 - pool.integer(6) * x**2 + pool.integer(11) * x - pool.integer(6)]
    sols = ak.solve(eqs, [x], method="homotopy")
    assert_solutions_satisfy(eqs, [x], sols, expected_count=3)


def test_solve_quartic_homotopy_four_real_roots(pool, x):
    """x^4 - 5x^2 + 4 = 0 -> roots -2, -1, 1, 2 (biquadratic, all real)."""
    eqs = [x**4 - pool.integer(5) * x**2 + pool.integer(4)]
    sols = ak.solve(eqs, [x], method="homotopy")
    assert_solutions_satisfy(eqs, [x], sols, expected_count=4)


def test_solve_quartic_homotopy_irrational_real_roots(pool, x):
    """x^4 - 2 = 0 has two real roots (+-2^(1/4)) and two complex ones;
    homotopy solving over the reals returns just the two real roots."""
    eqs = [x**4 - pool.integer(2)]
    sols = ak.solve(eqs, [x], method="homotopy")
    assert_solutions_satisfy(eqs, [x], sols, expected_count=2)


def test_solve_symmetric_cubic_system_homotopy(pool, x, y, z):
    """x+y+z=6, xy+yz+zx=11, xyz=6 -- the elementary symmetric polynomials
    of {1,2,3} -- solved numerically finds all 6 permutation assignments."""
    eqs = [
        x + y + z - pool.integer(6),
        x * y + y * z + z * x - pool.integer(11),
        x * y * z - pool.integer(6),
    ]
    sols = ak.solve(eqs, [x, y, z], method="homotopy")
    assert_solutions_satisfy(eqs, [x, y, z], sols, expected_count=6)


def test_solve_cubic_system_homotopy_distinct_values(pool, x, y, z):
    """A second symmetric-style cubic system: x+y+z=0, xy+yz+zx=-7, xyz=-6
    -- roots of t^3-7t-6=0, i.e. {-1,-2,3} -- again all 6 permutations."""
    eqs = [
        x + y + z,
        x * y + y * z + z * x - pool.integer(-7),
        x * y * z - pool.integer(-6),
    ]
    sols = ak.solve(eqs, [x, y, z], method="homotopy")
    assert_solutions_satisfy(eqs, [x, y, z], sols, expected_count=6)


def test_solve_sphere_and_two_planes_homotopy(pool, x, y, z):
    """A sphere/plane-type system: x^2+y^2+z^2=14, x+y+z=6, y-x=1 -- the
    sphere and two planes meet at the two permutations (1,2,3) and (2,3,1)
    of the point satisfying both linear constraints."""
    eqs = [
        x**2 + y**2 + z**2 - pool.integer(14),
        x + y + z - pool.integer(6),
        y - x - pool.integer(1),
    ]
    sols = ak.solve(eqs, [x, y, z], method="homotopy")
    assert_solutions_satisfy(eqs, [x, y, z], sols, expected_count=2)


def test_solve_cyclic_quadratic_system_homotopy(pool, x, y, z):
    """A cyclic (non-symmetric-sum) polynomial system: x^2-y=1, y^2-z=1,
    z^2-x=1. Its two real solutions are the fixed points x=y=z of
    t^2-t-1=0, i.e. the golden ratio and its conjugate."""
    eqs = [
        x**2 - y - pool.integer(1),
        y**2 - z - pool.integer(1),
        z**2 - x - pool.integer(1),
    ]
    sols = ak.solve(eqs, [x, y, z], method="homotopy")
    assert_solutions_satisfy(eqs, [x, y, z], sols, expected_count=2)


# --- solve_numerical (alternative numeric path with certification) ------


def test_solve_numerical_cubic_certified_low_residual(pool, x, y, z):
    """solve_numerical on the same symmetric cubic system: each returned
    CertifiedSolution should report a tiny residual and be Smale-certified."""
    eqs = [
        x + y + z - pool.integer(6),
        x * y + y * z + z * x - pool.integer(11),
        x * y * z - pool.integer(6),
    ]
    certified = ak.solve_numerical(eqs, [x, y, z])
    assert len(certified) == 6
    for sol in certified:
        assert sol.max_residual < 1e-6
        assert sol.smale_certified is True
    sols = [sol.to_dict() for sol in certified]
    assert_solutions_satisfy(eqs, [x, y, z], sols, expected_count=6)


def test_solve_numerical_cubic_plain(pool, x):
    """solve_numerical also handles the plain univariate cubic that the
    default (Gröbner, degree<=2) path cannot."""
    eqs = [x**3 - pool.integer(6) * x**2 + pool.integer(11) * x - pool.integer(6)]
    certified = ak.solve_numerical(eqs, [x])
    assert len(certified) == 3
    for sol in certified:
        assert sol.max_residual < 1e-6
        assert sol.smale_certified is True
    sols = [sol.to_dict() for sol in certified]
    assert_solutions_satisfy(eqs, [x], sols, expected_count=3)


def test_solve_numerical_linear_system(pool, x, y):
    """solve_numerical on a simple linear system it has no trouble with."""
    eqs = [x + y - pool.integer(3), x - y - pool.integer(1)]
    certified = ak.solve_numerical(eqs, [x, y])
    assert len(certified) == 1
    assert certified[0].max_residual < 1e-6
    assert certified[0].smale_certified is True
    sols = [sol.to_dict() for sol in certified]
    assert_solutions_satisfy(eqs, [x, y], sols, expected_count=1)


# --- degree-3+ default (Gröbner) path: documented unsupported case ------


def test_solve_default_method_rejects_cubic(pool, x):
    """Sanity check on the documented limit: the default method="groebner"
    path's symbolic back-substitution only handles degree <= 2 univariate
    polynomials, so a plain cubic raises SolverError (this is why
    method="homotopy" / solve_numerical exist as escape hatches, exercised
    above)."""
    eqs = [x**3 - pool.integer(6) * x**2 + pool.integer(11) * x - pool.integer(6)]
    with pytest.raises(ak.SolverError):
        ak.solve(eqs, [x])


# --- B5 regression: numeric=True falls back past HighDegree ---------------


def test_solve_numeric_true_falls_back_on_cubic(pool, x, y, z):
    eqs = [
        x + y + z - pool.integer(6),
        x * y + y * z + z * x - pool.integer(11),
        x * y * z - pool.integer(6),
    ]
    sols = ak.solve(eqs, [x, y, z], numeric=True)
    assert len(sols) == 6


def test_solve_numeric_true_falls_back_on_plain_cubic(pool, x):
    eqs = [x**3 - pool.integer(6) * x**2 + pool.integer(11) * x - pool.integer(6)]
    sols = ak.solve(eqs, [x], numeric=True)
    assert len(sols) >= 1
