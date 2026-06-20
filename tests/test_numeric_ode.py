"""Tests for the built-in numeric ODE integrators (Phase 16b).

Covers ``experimental.ode_integrate_rk4`` and ``experimental.ode_integrate_rk45``
exposed through the ``alkahest.experimental`` surface.

Test cases:
- y' = y, y(0) = 1  → y(1) ≈ e    (scalar exponential growth)
- 2-state linear system vs closed form
- dsolve sanity check (already has dedicated tests; included here for cross-check)
- harmonic oscillator via lower_to_first_order
- error conditions (invalid interval, IC length mismatch)
"""

from __future__ import annotations

import math

import alkahest as A
import pytest
from alkahest import experimental as ex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_exp_ode():
    """dy/dt = y."""
    p = A.ExprPool()
    t = p.symbol("t")
    y = p.symbol("y")
    ode = A.ODE([y], [y], t)
    return ode


def make_harmonic_ode():
    """y0' = y1, y1' = -y0  (harmonic oscillator y'' + y = 0)."""
    p = A.ExprPool()
    t = p.symbol("t")
    y0 = p.symbol("y0")
    y1 = p.symbol("y1")
    neg_y0 = p.integer(-1) * y0
    ode = A.ODE([y0, y1], [y1, neg_y0], t)
    return ode


def make_two_state_ode():
    """dy1/dt = -2*y1 + y2,  dy2/dt = y1 - 2*y2."""
    p = A.ExprPool()
    t = p.symbol("t")
    y1 = p.symbol("y1")
    y2 = p.symbol("y2")
    rhs1 = p.integer(-2) * y1 + y2
    rhs2 = y1 + p.integer(-2) * y2
    ode = A.ODE([y1, y2], [rhs1, rhs2], t)
    return ode


# ---------------------------------------------------------------------------
# OdeTrajectory type
# ---------------------------------------------------------------------------


class TestOdeTrajectory:
    def test_trajectory_attributes(self):
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 0.5, h=0.1)
        assert hasattr(traj, "t")
        assert hasattr(traj, "y")
        assert callable(traj.t_final)
        assert callable(traj.y_final)

    def test_trajectory_len(self):
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 1.0, h=0.1)
        # At least 11 rows (initial + 10 steps)
        assert len(traj) >= 11
        assert len(traj.t) == len(traj.y)

    def test_t_starts_at_t_start(self):
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 1.0, h=0.1)
        assert traj.t[0] == pytest.approx(0.0)

    def test_t_final_approx_t_end(self):
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 1.0, h=0.1)
        assert traj.t_final() == pytest.approx(1.0, abs=1e-10)

    def test_y_initial_matches_ic(self):
        ode = make_harmonic_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0, 0.0], 0.0, 0.5, h=0.1)
        assert traj.y[0][0] == pytest.approx(1.0)
        assert traj.y[0][1] == pytest.approx(0.0)

    def test_repr(self):
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 0.5, h=0.1)
        r = repr(traj)
        assert "OdeTrajectory" in r

    def test_trajectory_is_OdeTrajectory(self):
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 0.5, h=0.1)
        assert isinstance(traj, ex.OdeTrajectory)


# ---------------------------------------------------------------------------
# RK4 tests
# ---------------------------------------------------------------------------


class TestRK4:
    def test_exp_growth_y_at_1_approx_e(self):
        """y' = y, y(0) = 1  →  y(1) ≈ e (within 1e-9 for h=0.001)."""
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 1.0, h=0.001)
        y1 = traj.y_final()[0]
        assert abs(y1 - math.e) < 1e-9, f"y(1) = {y1}, e = {math.e}"

    def test_exp_growth_multiple_points(self):
        """Check intermediate values against exact solution y(t) = e^t."""
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0], 0.0, 1.0, h=0.01)
        for t_val, y_row in zip(traj.t, traj.y):
            expected = math.exp(t_val)
            assert abs(y_row[0] - expected) < 1e-7, (
                f"at t={t_val:.2f}: y={y_row[0]:.8f}, exact={expected:.8f}"
            )

    def test_harmonic_oscillator(self):
        """y'' + y = 0, y(0)=1, y'(0)=0  →  y(t)=cos(t);  y(π) = -1."""
        ode = make_harmonic_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0, 0.0], 0.0, math.pi, h=0.001)
        y_end = traj.y_final()[0]
        assert abs(y_end - (-1.0)) < 1e-7, f"y(π) = {y_end}"

    def test_two_state_linear_system(self):
        """dy1/dt = -2y1 + y2, dy2/dt = y1 - 2y2, y0=[1,0].

        Exact: y1(t) = (e^{-t} + e^{-3t})/2, y2(t) = (e^{-t} - e^{-3t})/2.
        """
        ode = make_two_state_ode()
        traj = ex.ode_integrate_rk4(ode, [1.0, 0.0], 0.0, 1.0, h=0.001)
        y_fin = traj.y_final()
        exact_y1 = (math.exp(-1) + math.exp(-3)) / 2
        exact_y2 = (math.exp(-1) - math.exp(-3)) / 2
        assert abs(y_fin[0] - exact_y1) < 1e-8, f"y1(1) = {y_fin[0]}, exact = {exact_y1}"
        assert abs(y_fin[1] - exact_y2) < 1e-8, f"y2(1) = {y_fin[1]}, exact = {exact_y2}"

    def test_ic_length_mismatch_raises(self):
        ode = make_exp_ode()
        with pytest.raises(A.OdeError):
            ex.ode_integrate_rk4(ode, [1.0, 2.0], 0.0, 1.0, h=0.01)

    def test_invalid_interval_raises(self):
        ode = make_exp_ode()
        with pytest.raises(A.OdeError):
            ex.ode_integrate_rk4(ode, [1.0], 1.0, 0.0, h=0.01)

    def test_max_steps_exceeded_raises(self):
        ode = make_exp_ode()
        with pytest.raises(A.OdeError):
            ex.ode_integrate_rk4(ode, [1.0], 0.0, 100.0, h=0.01, max_steps=5)


# ---------------------------------------------------------------------------
# RK45 tests
# ---------------------------------------------------------------------------


class TestRK45:
    def test_exp_growth_y_at_1_approx_e(self):
        """y' = y, y(0) = 1  →  y(1) ≈ e (within 1e-8 for tight tolerances)."""
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk45(ode, [1.0], 0.0, 1.0, rtol=1e-9, atol=1e-12)
        y1 = traj.y_final()[0]
        assert abs(y1 - math.e) < 1e-8, f"y(1) = {y1}, e = {math.e}"

    def test_harmonic_oscillator(self):
        """y'' + y = 0, y(0)=1, y'(0)=0  →  y(π) = -1."""
        ode = make_harmonic_ode()
        traj = ex.ode_integrate_rk45(ode, [1.0, 0.0], 0.0, math.pi, rtol=1e-9, atol=1e-12)
        y_end = traj.y_final()[0]
        assert abs(y_end - (-1.0)) < 1e-8, f"y(π) = {y_end}"

    def test_two_state_linear_system(self):
        """2-state linear system vs closed form."""
        ode = make_two_state_ode()
        traj = ex.ode_integrate_rk45(ode, [1.0, 0.0], 0.0, 1.0, rtol=1e-9, atol=1e-12)
        y_fin = traj.y_final()
        exact_y1 = (math.exp(-1) + math.exp(-3)) / 2
        exact_y2 = (math.exp(-1) - math.exp(-3)) / 2
        assert abs(y_fin[0] - exact_y1) < 1e-8, f"y1(1) = {y_fin[0]}"
        assert abs(y_fin[1] - exact_y2) < 1e-8, f"y2(1) = {y_fin[1]}"

    def test_adaptive_fewer_steps_than_rk4(self):
        """RK45 uses fewer steps than equivalent fixed-step RK4."""
        ode = make_exp_ode()
        traj_rk4 = ex.ode_integrate_rk4(ode, [1.0], 0.0, 1.0, h=1e-4)
        traj_rk45 = ex.ode_integrate_rk45(ode, [1.0], 0.0, 1.0, rtol=1e-6, atol=1e-9)
        # RK4 with h=1e-4 needs 10,000 steps; RK45 should use far fewer
        assert len(traj_rk45) < len(traj_rk4) // 10

    def test_ic_length_mismatch_raises(self):
        ode = make_exp_ode()
        with pytest.raises(A.OdeError):
            ex.ode_integrate_rk45(ode, [1.0, 2.0], 0.0, 1.0)

    def test_invalid_interval_raises(self):
        ode = make_exp_ode()
        with pytest.raises(A.OdeError):
            ex.ode_integrate_rk45(ode, [1.0], 1.0, 0.0)

    def test_max_steps_exceeded_raises(self):
        ode = make_exp_ode()
        with pytest.raises(A.OdeError):
            ex.ode_integrate_rk45(ode, [1.0], 0.0, 100.0, max_steps=3)

    def test_default_tolerances_produce_reasonable_accuracy(self):
        """Default rtol=1e-6, atol=1e-9 should give at least 1e-5 accuracy."""
        ode = make_exp_ode()
        traj = ex.ode_integrate_rk45(ode, [1.0], 0.0, 1.0)
        y1 = traj.y_final()[0]
        assert abs(y1 - math.e) < 1e-5


# ---------------------------------------------------------------------------
# Cross-check: numeric integrator vs dsolve closed form
# ---------------------------------------------------------------------------


class TestNumericVsDsolve:
    def test_logistic_numeric_vs_dsolve(self):
        """Compare RK45 output to dsolve closed form for y' = y(1 - y).

        Exact: y(t) = 1 / (1 + (1/y0 - 1) * e^{-t}).
        With y(0) = 0.1: y(1) = 1/(1 + 9*e^{-1}) ≈ 0.2310...
        """
        p = A.ExprPool()
        t = p.symbol("t")
        y = p.symbol("y")
        # y' = y*(1 - y)
        rhs = y * (p.integer(1) - y)
        ode = A.ODE([y], [rhs], t)
        y0 = 0.1
        t_end = 1.0
        traj = ex.ode_integrate_rk45(ode, [y0], 0.0, t_end, rtol=1e-10, atol=1e-13)
        y_numeric = traj.y_final()[0]
        # Exact closed form: y(t) = 1/(1 + (1/y0 - 1)*e^{-t})
        exact = 1.0 / (1.0 + (1.0 / y0 - 1.0) * math.exp(-t_end))
        assert abs(y_numeric - exact) < 1e-7, f"numeric={y_numeric:.10f}, exact={exact:.10f}"

    def test_dsolve_still_works(self):
        """Quick sanity check that dsolve is still accessible and working."""
        p = A.ExprPool()
        x = p.symbol("x")
        y = p.symbol("y")
        yp = p.symbol("y'")
        # y' - y = 0 (separable: solution y = C*e^x)
        eq = yp - y
        branches = ex.dsolve(eq, x, y, [yp])
        assert len(branches) >= 1
        sol = branches[0]
        assert "exp" in str(sol["y_of_x"]).lower() or "e" in str(sol["y_of_x"]).lower()
