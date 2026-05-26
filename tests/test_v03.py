"""v0.3 feature tests: phases 14–20.

Covers:
  Phase 14 — reverse-mode AD (grad)
  Phase 15 — symbolic matrices and jacobian
  Phase 16 — ODE representation
  Phase 17 — DAE structural analysis
  Phase 18 — acausal component modelling
  Phase 19 — sensitivity analysis
  Phase 20 — hybrid ODE / event handling
"""

import pytest
from alkahest.alkahest import (
    DAE,
    ODE,
    AcausalSystem,
    Event,
    ExprPool,
    HybridODE,
    Matrix,
    SensitivitySystem,
    adjoint_system,
    cos,
    diff,
    exp,
    grad,
    jacobian,
    lower_to_first_order,
    pantelides,
    resistor,
    sensitivity_system,
    sin,
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 14: Reverse-mode AD
# ─────────────────────────────────────────────────────────────────────────────


class TestGrad:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")

    def test_grad_constant_is_zero(self):
        five = self.pool.integer(5)
        gs = grad(five, [self.x])
        assert str(gs[0]) == "0"

    def test_grad_identity(self):
        gs = grad(self.x, [self.x])
        assert str(gs[0]) == "1"

    def test_grad_unrelated_is_zero(self):
        gs = grad(self.x, [self.y])
        assert str(gs[0]) == "0"

    def test_grad_x_squared(self):
        x2 = self.x**2
        gs = grad(x2, [self.x])
        # ∂x²/∂x = 2x
        s = str(gs[0])
        assert "x" in s
        assert "2" in s

    def test_grad_multivariate_xy(self):
        # f = x*y → [∂/∂x = y, ∂/∂y = x]
        f = self.x * self.y
        gs = grad(f, [self.x, self.y])
        assert str(gs[0]) == str(self.y)
        assert str(gs[1]) == str(self.x)

    def test_grad_agrees_with_diff(self):
        # f = x³ + 2x  → diff and grad must agree
        two = self.pool.integer(2)
        x3 = self.x**3
        f = x3 + two * self.x
        sym = diff(f, self.x)
        rev = grad(f, [self.x])
        assert str(sym.value) == str(rev[0])

    def test_grad_sin(self):
        gs = grad(sin(self.x), [self.x])
        assert str(gs[0]) == str(cos(self.x))

    def test_grad_exp(self):
        gs = grad(exp(self.x), [self.x])
        assert str(gs[0]) == str(exp(self.x))

    def test_grad_empty_vars(self):
        gs = grad(self.x, [])
        assert gs == []

    def test_grad_two_vars_independent(self):
        # f = x² + y², ∂/∂x = 2x, ∂/∂y = 2y
        f = self.x**2 + self.y**2
        gs = grad(f, [self.x, self.y])
        dx_s = str(gs[0])
        dy_s = str(gs[1])
        assert "x" in dx_s
        assert "2" in dx_s
        assert "y" in dy_s
        assert "2" in dy_s


# ─────────────────────────────────────────────────────────────────────────────
# Phase 15: Symbolic matrices
# ─────────────────────────────────────────────────────────────────────────────


class TestMatrix:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")

    def _m(self, rows):
        return Matrix.from_rows(rows)

    def test_create_2x2(self):
        one = self.pool.integer(1)
        zero = self.pool.integer(0)
        m = self._m([[one, zero], [zero, one]])
        assert m.rows == 2
        assert m.cols == 2

    def test_get_entry(self):
        one = self.pool.integer(1)
        zero = self.pool.integer(0)
        m = self._m([[one, zero], [zero, one]])
        assert str(m.get(0, 0)) == "1"
        assert str(m.get(0, 1)) == "0"

    def test_transpose(self):
        one = self.pool.integer(1)
        two = self.pool.integer(2)
        three = self.pool.integer(3)
        four = self.pool.integer(4)
        m = self._m([[one, two], [three, four]])
        t = m.transpose()
        assert t.rows == 2
        assert t.cols == 2
        assert str(t.get(0, 1)) == str(m.get(1, 0))

    def test_add(self):
        one = self.pool.integer(1)
        two = self.pool.integer(2)
        a = self._m([[one, one]])
        b = self._m([[two, two]])
        result = (a + b).simplify()
        assert str(result.get(0, 0)) == "3"

    def test_matmul_identity(self):
        one = self.pool.integer(1)
        zero = self.pool.integer(0)
        identity = self._m([[one, zero], [zero, one]])
        m = self._m([[self.x, self.y], [self.y, self.x]])
        result = (identity @ m).simplify()
        assert str(result.get(0, 0)) == str(self.x)
        assert str(result.get(0, 1)) == str(self.y)

    def test_det_2x2(self):
        a = self.pool.symbol("a")
        b = self.pool.symbol("b")
        c = self.pool.symbol("c")
        d = self.pool.symbol("d")
        m = self._m([[a, b], [c, d]])
        det = m.det()
        s = str(det)
        assert "a" in s
        assert "d" in s

    def test_det_identity_3x3(self):
        one = self.pool.integer(1)
        zero = self.pool.integer(0)
        id3 = self._m(
            [
                [one, zero, zero],
                [zero, one, zero],
                [zero, zero, one],
            ]
        )
        assert str(id3.det()) == "1"

    def test_to_list(self):
        one = self.pool.integer(1)
        two = self.pool.integer(2)
        m = self._m([[one, two]])
        rows = m.to_list()
        assert len(rows) == 1
        assert len(rows[0]) == 2

    def test_dimension_mismatch_raises(self):
        one = self.pool.integer(1)
        a = self._m([[one, one]])  # 1×2
        b = self._m([[one], [one]])  # 2×1
        with pytest.raises(Exception):
            a + b


class TestJacobian:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")

    def test_jacobian_linear(self):
        # f = [x+y, x-y], vars = [x, y]
        # J = [[1,1],[1,-1]]
        neg_y = self.pool.integer(-1) * self.y
        f1 = self.x + self.y
        f2 = self.x + neg_y
        j = jacobian([f1, f2], [self.x, self.y])
        assert j.rows == 2
        assert j.cols == 2
        assert str(j.get(0, 0)) == "1"
        assert str(j.get(1, 1)) == "-1"

    def test_jacobian_quadratic(self):
        # f=[x², y²], J=[[2x,0],[0,2y]]
        f1 = self.x**2
        f2 = self.y**2
        j = jacobian([f1, f2], [self.x, self.y])
        assert str(j.get(0, 1)) == "0"
        assert str(j.get(1, 0)) == "0"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 16: ODE representation
# ─────────────────────────────────────────────────────────────────────────────


class TestODE:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.t = self.pool.symbol("t")

    def test_create_simple(self):
        # dx/dt = x
        ode = ODE.new([self.x], [self.x], self.t)
        assert ode.order() == 1

    def test_autonomous(self):
        ode = ODE.new([self.x], [self.x], self.t)
        assert ode.is_autonomous()

    def test_not_autonomous(self):
        tx = self.t * self.x
        ode = ODE.new([self.x], [tx], self.t)
        assert not ode.is_autonomous()

    def test_mismatch_raises(self):
        y = self.pool.symbol("y")
        with pytest.raises(Exception):
            ODE.new([self.x, y], [self.x], self.t)

    def test_with_ic(self):
        ode = ODE.new([self.x], [self.x], self.t)
        one = self.pool.integer(1)
        ode2 = ode.with_ic(self.x, one)
        # Should not raise; order unchanged
        assert ode2.order() == 1

    def test_state_vars(self):
        ode = ODE.new([self.x], [self.x], self.t)
        svs = ode.state_vars()
        assert len(svs) == 1
        assert str(svs[0]) == "x"

    def test_rhs(self):
        ode = ODE.new([self.x], [self.x], self.t)
        rhs = ode.rhs()
        assert str(rhs[0]) == "x"

    def test_simplify_rhs(self):
        zero = self.pool.integer(0)
        rhs = self.x + zero
        ode = ODE.new([self.x], [rhs], self.t)
        s = ode.simplify_rhs()
        assert str(s.rhs()[0]) == "x"


class TestLowerToFirstOrder:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.t = self.pool.symbol("t")

    def test_first_order_passthrough(self):
        ode = lower_to_first_order(self.x, self.x, 1, self.t)
        assert ode.order() == 1

    def test_second_order_harmonic_oscillator(self):
        # x'' = -x
        neg_x = self.pool.integer(-1) * self.x
        ode = lower_to_first_order(self.x, neg_x, 2, self.t)
        assert ode.order() == 2

    def test_third_order_produces_3_states(self):
        ode = lower_to_first_order(self.x, self.pool.integer(0), 3, self.t)
        assert ode.order() == 3


# ─────────────────────────────────────────────────────────────────────────────
# Phase 17: DAE structural analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestDAE:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.dx = self.pool.symbol("dx/dt")
        self.t = self.pool.symbol("t")

    def test_create_dae(self):
        neg_x = self.pool.integer(-1) * self.x
        eq = self.dx + neg_x  # dx/dt - x = 0
        dae = DAE.new([eq], [self.x], [self.dx], self.t)
        assert dae.n_equations() == 1
        assert dae.n_variables() == 1

    def test_pantelides_ode_no_steps(self):
        neg_x = self.pool.integer(-1) * self.x
        eq = self.dx + neg_x
        dae = DAE.new([eq], [self.x], [self.dx], self.t)
        reduced = pantelides(dae)
        # An ODE needs 0 differentiation steps
        assert reduced.n_equations() >= 1

    def test_pantelides_returns_dae(self):
        neg_x = self.pool.integer(-1) * self.x
        eq = self.dx + neg_x
        dae = DAE.new([eq], [self.x], [self.dx], self.t)
        result = pantelides(dae)
        assert isinstance(result, DAE)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 18: Acausal component modelling
# ─────────────────────────────────────────────────────────────────────────────


class TestAcausal:
    def setup_method(self):
        self.pool = ExprPool()
        self.t = self.pool.symbol("t")

    def test_resistor_returns_dict(self):
        R = self.pool.symbol("R")
        comp = resistor("R1", R)
        assert comp["n_equations"] == 1
        assert comp["n_ports"] == 2
        assert comp["name"] == "R1"

    def test_acausal_system_flatten(self):
        sys = AcausalSystem(self.pool)
        dae = sys.flatten(self.t)
        # Empty system → empty DAE
        assert dae.n_equations() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 19: Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestSensitivity:
    def setup_method(self):
        self.pool = ExprPool()
        self.y = self.pool.symbol("y")
        self.a = self.pool.symbol("a")
        self.t = self.pool.symbol("t")

    def test_sensitivity_linear(self):
        # dy/dt = a*y, param = a
        rhs = self.a * self.y
        ode = ODE.new([self.y], [rhs], self.t)
        sys = sensitivity_system(ode, [self.a])
        assert isinstance(sys, SensitivitySystem)
        assert sys.original_dim == 1
        assert sys.n_params == 1
        # Extended ODE has y + 1 sensitivity state
        assert sys.extended_ode.order() == 2

    def test_sensitivity_constant(self):
        # dy/dt = p, param = p → dS/dt = 1
        p = self.pool.symbol("p")
        ode = ODE.new([self.y], [p], self.t)
        sys = sensitivity_system(ode, [p])
        assert sys.extended_ode.order() == 2
        # The sensitivity RHS should be 1
        s_rhs = sys.extended_ode.rhs()[1]
        assert str(s_rhs) == "1"

    def test_sensitivity_two_params(self):
        b = self.pool.symbol("b")
        rhs = self.a * self.y + b
        ode = ODE.new([self.y], [rhs], self.t)
        sys = sensitivity_system(ode, [self.a, b])
        assert sys.n_params == 2
        assert sys.extended_ode.order() == 3

    def test_adjoint_system(self):
        # dy/dt = -y, objective ∂g/∂y = 1
        neg_y = self.pool.integer(-1) * self.y
        ode = ODE.new([self.y], [neg_y], self.t)
        one = self.pool.integer(1)
        adj = adjoint_system(ode, [one])
        assert adj.order() == 1
        # dλ/dt = λ (adjoint of -y system)
        lam = adj.state_vars()[0]
        assert str(adj.rhs()[0]) == str(lam)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 20: Hybrid ODE / Event handling
# ─────────────────────────────────────────────────────────────────────────────


class TestHybrid:
    def setup_method(self):
        self.pool = ExprPool()
        self.y = self.pool.symbol("y")
        self.v = self.pool.symbol("v")
        self.g = self.pool.symbol("g")
        self.t = self.pool.symbol("t")

    def _bouncing_ball(self):
        neg_g = self.pool.integer(-1) * self.g
        ode = ODE.new([self.y, self.v], [self.v, neg_g], self.t)
        neg_v = self.pool.integer(-1) * self.v
        bounce = Event.new("bounce", self.y, [(self.v, neg_v)])
        return HybridODE.new(ode).add_event(bounce)

    def test_create_hybrid(self):
        h = self._bouncing_ball()
        assert h.n_events() == 1

    def test_guards(self):
        h = self._bouncing_ball()
        guards = h.guards()
        assert len(guards) == 1
        assert str(guards[0]) == str(self.y)

    def test_event_creation(self):
        ev = Event.new("test", self.y, [])
        assert repr(ev) is not None

    def test_hybrid_repr(self):
        h = self._bouncing_ball()
        s = repr(h)
        assert "dx/dt" in s or "d" in s.lower()

    def test_event_direction(self):
        neg_v = self.pool.integer(-1) * self.v
        ev = Event.new("bounce", self.y, [(self.v, neg_v)])
        ev.falling()  # mark as falling edge
        # no error

    def test_no_events(self):
        ode = ODE.new([self.y], [self.v], self.t)
        h = HybridODE.new(ode)
        assert h.n_events() == 0
        assert h.guards() == []

    def test_multiple_events(self):
        ode = ODE.new([self.y, self.v], [self.v, self.y], self.t)
        ev1 = Event.new("ev1", self.y, [])
        ev2 = Event.new("ev2", self.v, [])
        h = HybridODE.new(ode).add_event(ev1).add_event(ev2)
        assert h.n_events() == 2
        guards = h.guards()
        assert str(guards[0]) == str(self.y)
        assert str(guards[1]) == str(self.v)
