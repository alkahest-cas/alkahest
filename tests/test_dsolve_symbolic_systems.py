"""Symbolic constant coefficients, and linear systems, through the bindings.

Two things are checked here that the Rust tests cannot: that the new surface is
actually reachable from Python (``experimental.dsolve_system``, the extra keys
on ``experimental.dsolve``'s dicts), and that the branch bookkeeping survives
the crossing — a caller who never reads ``side_conditions`` must still be able
to tell from ``method`` that an assumption was made.

Every solution these return has already passed a substitution gate in the
kernel; the assertions are about *what* comes back, not about whether it is a
solution.
"""

from __future__ import annotations

import alkahest as A
import pytest
from alkahest import experimental as ex


def _scalar(order: int = 2):
    p = A.ExprPool()
    x = p.symbol("x")
    y = p.symbol("y")
    derivs = [p.symbol("y" + "'" * k) for k in range(1, order + 1)]
    return p, x, y, derivs


# ---------------------------------------------------------------------------
# Scalar, symbolic constant coefficients
# ---------------------------------------------------------------------------


def test_damped_oscillator_solves_with_symbolic_coefficients():
    """`y'' + 2*z*w*y' + w**2*y = 0` — refused outright before 3.10."""
    p, x, y, (yp, ypp) = _scalar()
    z = p.symbol("z")
    w = p.symbol("w")
    eq = ypp + 2 * z * w * yp + w**2 * y
    branches = ex.dsolve(eq, x, y, [yp, ypp])
    assert len(branches) == 1
    sol = branches[0]
    assert len(sol["constants"]) == 2
    assert str(sol["y_of_x"]).count("exp(") == 2
    # The branch that was not taken is reported, not hidden.
    assert sol["method"] == "constant_coefficient_symbolic"
    assert len(sol["side_conditions"]) == 1
    assert "≠ 0" in sol["side_conditions"][0]
    assert any("discriminant" in n for n in sol["notes"])


def test_a_provable_double_root_carries_no_condition():
    p, x, y, (yp, ypp) = _scalar()
    a = p.symbol("a")
    eq = ypp + 2 * a * yp + a**2 * y
    sol = ex.dsolve(eq, x, y, [yp, ypp])[0]
    assert sol["method"] == "constant_coefficient_symbolic_repeated_root"
    assert sol["side_conditions"] == []
    assert sol["notes"] == []


def test_assumptions_settle_the_branch():
    p, x, y, (yp, ypp) = _scalar()
    z = p.symbol("z")
    w = p.symbol("w")
    eq = ypp + 2 * z * w * yp + w**2 * y
    assumptions = A.Assumptions(p)
    # −D/4 = w² − z²w² > 0 is the underdamped region.
    assumptions.refine(p.gt(w**2 - z**2 * w**2, p.integer(0)))
    sol = ex.dsolve(eq, x, y, [yp, ypp], assumptions)[0]
    assert sol["method"] == "constant_coefficient_symbolic_oscillatory"
    assert sol["side_conditions"] == []
    s = str(sol["y_of_x"])
    assert "cos(" in s
    assert "sin(" in s


def test_numeric_coefficients_are_unchanged():
    _p, x, y, (yp, ypp) = _scalar()
    sol = ex.dsolve(ypp + y - A.sin(x), x, y, [yp, ypp])[0]
    assert sol["method"] == "constant_coefficient"
    assert sol["side_conditions"] == []


def test_a_symbolic_cubic_is_refused():
    p, x, y, (yp, ypp, yppp) = _scalar(3)
    a, b, c = p.symbol("a"), p.symbol("b"), p.symbol("c")
    with pytest.raises(ValueError, match="degree"):
        ex.dsolve(yppp + a * ypp + b * yp + c * y, x, y, [yp, ypp, yppp])


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------


def test_two_compartment_pk_model():
    """`x' = -ka*x`, `y' = ka*x - ke*y` with symbolic rates."""
    p = A.ExprPool()
    t = p.symbol("t")
    x = p.symbol("x")
    y = p.symbol("y")
    ka = p.symbol("ka")
    ke = p.symbol("ke")
    ode = A.ODE([x, y], [-ka * x, ka * x - ke * y], t)
    sol = ex.dsolve_system(ode)
    assert sol["method"] == "linear_system_putzer"
    assert len(sol["y_of_t"]) == 2
    assert len(sol["constants"]) == 2
    # The donor compartment cannot depend on the eliminating rate.
    assert "ke" not in str(sol["y_of_t"][0])
    # ka = ke is a real degeneracy and is reported, not assumed away.
    assert len(sol["side_conditions"]) == 1
    cond = sol["side_conditions"][0]
    assert "ka" in cond
    assert "ke" in cond
    assert "≠ 0" in cond
    assert any("confluent" in n for n in sol["notes"])


def test_defective_matrix_gets_the_secular_term():
    """A single Jordan block — the case that needs more than eigenvectors."""
    p = A.ExprPool()
    t = p.symbol("t")
    x = p.symbol("x")
    y = p.symbol("y")
    ode = A.ODE([x, y], [2 * x + y, 2 * y], t)
    sol = ex.dsolve_system(ode)
    phi = sol["fundamental_matrix"]
    assert len(phi) == 2
    assert len(phi[0]) == 2
    assert "t" in str(phi[0][1]), "e^{At} must carry the t·e^{2t} off-diagonal"
    assert str(phi[1][0]) == "0"
    assert sol["side_conditions"] == []


def test_forced_system_uses_variation_of_parameters():
    p = A.ExprPool()
    t = p.symbol("t")
    x = p.symbol("x")
    y = p.symbol("y")
    ode = A.ODE([x, y], [-x + 1, x - 2 * y], t)
    sol = ex.dsolve_system(ode)
    assert sol["method"] == "linear_system_putzer_variation_of_parameters"
    assert len(sol["constants"]) == 2


def test_nonlinear_system_is_refused():
    p = A.ExprPool()
    t = p.symbol("t")
    x = p.symbol("x")
    y = p.symbol("y")
    ode = A.ODE([x, y], [x * y, y], t)
    with pytest.raises(ValueError, match="not linear"):
        ex.dsolve_system(ode)


def test_time_varying_coefficient_is_refused():
    p = A.ExprPool()
    t = p.symbol("t")
    x = p.symbol("x")
    y = p.symbol("y")
    ode = A.ODE([x, y], [t * x, y], t)
    with pytest.raises(ValueError, match="independent variable"):
        ex.dsolve_system(ode)


def test_symbolic_non_triangular_three_by_three_is_refused():
    p = A.ExprPool()
    t = p.symbol("t")
    xs = [p.symbol(n) for n in ("x", "y", "z")]
    a = [[p.symbol(f"a{i}{j}") for j in range(3)] for i in range(3)]
    rhs = [sum((a[i][j] * xs[j] for j in range(1, 3)), a[i][0] * xs[0]) for i in range(3)]
    ode = A.ODE(xs, rhs, t)
    with pytest.raises(ValueError, match="Cardano"):
        ex.dsolve_system(ode)


def test_dsolve_system_is_exported():
    assert "dsolve_system" in ex.__all__
