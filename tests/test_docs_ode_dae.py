"""Every code block in ``docs/mdbook/src/ode-dae.md``, executed.

The page previously documented keyword constructors (``ODE(state=..., ...)``,
``DAE(equations=..., ...)``) and a one-argument ``lower_to_first_order`` that
never existed, plus ``pantelides(...).differentiated``, which the returned
object does not have.  Following it failed on the first line.  These tests are
the guard: each one is a doc snippet run verbatim, with the values the page
claims asserted.
"""

import alkahest
import pytest
from alkahest import (
    DAE,
    ODE,
    AcausalSystem,
    Event,
    ExprPool,
    HybridODE,
    adjoint_system,
    capacitor,
    lower_to_first_order,
    pantelides,
    resistor,
    sensitivity_system,
    voltage_source,
)


def test_ode_constructor():
    pool = ExprPool()
    t = pool.symbol("t")
    x = pool.symbol("x")
    v = pool.symbol("v")

    ode = ODE.new([x, v], [v, pool.integer(-1) * x], t)

    assert ode.order == 2
    assert [str(s) for s in ode.state_vars()] == ["x", "v"]
    assert [str(r) for r in ode.rhs()] == ["v", "(x * -1)"]
    assert ode.is_autonomous() is True

    # ODE(...) and ODE.new(...) agree.
    assert str(ODE([x, v], [v, pool.integer(-1) * x], t)) == str(ode)


def test_ode_with_ic_returns_a_new_ode():
    pool = ExprPool()
    t, x, v = pool.symbol("t"), pool.symbol("x"), pool.symbol("v")
    ode = ODE.new([x, v], [v, pool.integer(-1) * x], t)

    ode_with_ic = ode.with_ic(x, pool.integer(1)).with_ic(v, pool.integer(0))

    assert isinstance(ode_with_ic, ODE)
    assert ode_with_ic is not ode


def test_lower_to_first_order_takes_four_arguments():
    pool = ExprPool()
    t, x = pool.symbol("t"), pool.symbol("x")

    ode = lower_to_first_order(x, pool.integer(-4) * x, 2, t)

    assert [str(s) for s in ode.state_vars()] == ["x", "x_1"]
    assert [str(r) for r in ode.rhs()] == ["x_1", "(x * -4)"]

    # The documented one-argument form does not exist.
    with pytest.raises(TypeError):
        lower_to_first_order(ode)


def pendulum_dae():
    """The index-3 Cartesian pendulum from the DAE section of the page."""
    pool = ExprPool()
    t = pool.symbol("t")
    x, y, u, w = (pool.symbol(n) for n in ("x", "y", "u", "w"))
    lam = pool.symbol("lam")
    dx, dy, du, dw = (pool.symbol(n) for n in ("dx/dt", "dy/dt", "du/dt", "dw/dt"))
    one, two = pool.integer(1), pool.integer(2)

    dae = DAE.new(
        [dx - u, dy - w, du + lam * x, dw + lam * y + one, x**two + y**two - one],
        [x, y, u, w, lam],
        [dx, dy, du, dw],
        t,
    )
    return pool, t, dae


def test_dae_new_and_read_back():
    _pool, t, dae = pendulum_dae()

    assert dae.n_equations == 5
    assert dae.n_variables == 5
    assert len(dae.equations()) == 5
    assert [str(d) for d in dae.derivatives()] == ["dx/dt", "dy/dt", "du/dt", "dw/dt"]
    assert str(dae.time_var) == "t"
    assert dae.index is None

    # There is no keyword constructor.
    with pytest.raises(TypeError):
        DAE(equations=dae.equations(), variables=dae.variables(), independent=t)


def test_pantelides_returns_a_reduced_dae():
    _, _, dae = pendulum_dae()

    reduced = pantelides(dae)

    assert isinstance(reduced, DAE)
    assert reduced.index == 1
    assert reduced.n_equations == 6
    # The appended equation is the differentiated constraint 2x·x' + 2y·y' = 0.
    assert str(reduced.equations()[-1]) == "((x * dx/dt * 2) + (y * dy/dt * 2))"
    # Prolongation introduced second-order jets.
    assert "ddx/dt/dt" in [str(d) for d in reduced.derivatives()]

    # The page used to claim a `.differentiated` attribute; it never existed.
    assert not hasattr(reduced, "differentiated")


def test_pantelides_index_zero_when_already_matched():
    pool = ExprPool()
    t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")
    reduced = pantelides(DAE.new([dx - x], [x], [dx], t))
    assert reduced.index == 0


@pytest.mark.skipif(
    not hasattr(alkahest, "rosenfeld_groebner"),
    reason="native module built without groebner feature",
)
def test_rosenfeld_groebner_snippet():
    from alkahest import rosenfeld_groebner

    pool = ExprPool()
    t, x, dx = pool.symbol("t"), pool.symbol("x"), pool.symbol("dx/dt")

    dae = DAE.new([dx - x], [x], [dx], t)
    result = rosenfeld_groebner(dae, max_prolong_rounds=1)

    assert result.consistent is True
    assert result.truncated is True
    assert [str(v) for v in result.variables()] == ["t", "x", "dx/dt", "ddx/dt/dt"]
    assert [str(e) for e in result.final_basis().to_exprs()] == [
        "(x + (-1 * ddx/dt/dt))",
        "(dx/dt + (-1 * ddx/dt/dt))",
    ]


@pytest.mark.skipif(
    not hasattr(alkahest, "rosenfeld_groebner"),
    reason="native module built without groebner feature",
)
def test_rosenfeld_groebner_inconsistent_has_no_final_basis():
    from alkahest import rosenfeld_groebner

    pool = ExprPool()
    t, y, dy = pool.symbol("t"), pool.symbol("y"), pool.symbol("dy/dt")
    dae = DAE.new([dy - y, dy - y - pool.integer(1)], [y], [dy], t)

    result = rosenfeld_groebner(dae, max_prolong_rounds=1)

    assert result.consistent is False
    assert result.final_basis() is None


def test_sensitivity_and_adjoint_snippet():
    pool = ExprPool()
    t, y, k = pool.symbol("t"), pool.symbol("y"), pool.symbol("k")

    ode = ODE.new([y], [pool.integer(-1) * k * y], t)

    sens = sensitivity_system(ode, [k])
    assert sens.original_dim == 1
    assert sens.n_params == 1
    assert [str(s) for s in sens.extended_ode.state_vars()] == ["y", "dS_y_k"]

    adj = adjoint_system(ode, [pool.integer(2) * y])
    assert [str(s) for s in adj.state_vars()] == ["lambda_y"]
    assert [str(r) for r in adj.rhs()] == ["(k * lambda_y)"]


def test_acausal_snippet():
    pool = ExprPool()
    t = pool.symbol("t")

    src = voltage_source("V1", pool.symbol("Vs"))["component"]
    res = resistor("R1", pool.symbol("R"))["component"]
    cap = capacitor("C1", pool.symbol("C"))["component"]

    circuit = AcausalSystem(pool)
    circuit.add_component(src)
    circuit.add_component(res)
    circuit.add_component(cap)

    circuit.connect(src.port("V1.p"), res.port("R1.p"))
    circuit.connect(res.port("R1.n"), cap.port("C1.p"))
    circuit.connect(cap.port("C1.n"), src.port("V1.n"))

    dae = circuit.flatten(t)
    assert dae.n_equations == 10


def test_laplace_snippet():
    from alkahest.experimental import inverse_laplace_transform, laplace_transform

    pool = ExprPool()
    s, t = pool.symbol("s"), pool.symbol("t")

    F = laplace_transform(pool.integer(1), t, s)
    f = inverse_laplace_transform(F, s, t)

    assert str(F) == "s^-1"
    assert str(f) == "(1)"


def test_hybrid_snippet():
    pool = ExprPool()
    t, x, v = pool.symbol("t"), pool.symbol("x"), pool.symbol("v")

    base_ode = ODE.new([x, v], [v, pool.integer(-1)], t)
    bounce = Event.new("bounce", x, [(v, pool.integer(-1) * v)])

    hybrid = HybridODE.new(base_ode).add_event(bounce)
    assert hybrid.n_events == 1
    assert [str(g) for g in hybrid.guards()] == ["x"]


def test_documented_apis_carry_docstrings():
    """The page was the only documentation because none of these had one."""
    for obj in (
        DAE.new,
        DAE.equations,
        DAE.index,
        ODE.new,
        pantelides,
        lower_to_first_order,
    ):
        assert obj.__doc__, obj

    if hasattr(alkahest, "rosenfeld_groebner"):
        assert alkahest.rosenfeld_groebner.__doc__
        assert alkahest.RosenfeldGroebnerResult.final_basis.__doc__
        assert alkahest.RosenfeldGroebnerResult.variables.__doc__
