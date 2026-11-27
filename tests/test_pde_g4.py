"""PDE gap round G4: classification of PDE systems, the n-D coordinate layer
(plan decision D7), and closed-form characteristic quadratures in ``pdsolve``."""

from __future__ import annotations

import alkahest as ak
import alkahest.experimental as akx
import pytest
from alkahest import Domain
from alkahest.experimental import PDE, D, Function, PdeError


def code(e):
    return e.value.code


def zero(e):
    return ak.simplify(e).value == 0 or str(ak.simplify(e).value) == "0"


@pytest.fixture
def P():
    class _P:
        pass

    p = _P()
    p.pool = ak.ExprPool()
    p.x = p.pool.symbol("x")
    p.y = p.pool.symbol("y")
    p.t = p.pool.symbol("t")
    return p


# ---------------------------------------------------------------------------
# System classification
# ---------------------------------------------------------------------------


def test_cauchy_riemann_is_elliptic(P):
    u, v = Function("u")(P.x, P.y), Function("v")(P.x, P.y)
    info = akx.classify_pde([D(u, P.x) - D(v, P.y), D(u, P.y) + D(v, P.x)], [u, v])
    assert isinstance(info, akx.PdeSystemInfo)
    assert info.order == 1
    assert info.linearity == "linear"
    assert info.type == "elliptic"
    assert info.names == ["cauchy_riemann"]
    assert info.characteristic_determinant is not None


def test_shallow_water_hyperbolic_with_condition(P):
    g = P.pool.symbol("g")
    h, u = Function("h")(P.x, P.t), Function("u")(P.x, P.t)
    sw = PDE([D(h, P.t) + D(h * u, P.x), D(u, P.t) + u * D(u, P.x) + g * D(h, P.x)], [h, u])
    info = sw.classify()
    assert info.linearity == "quasilinear"
    assert info.type == "hyperbolic"
    assert info.strictly_hyperbolic is True
    assert str(info.evolution_variable) == "t"
    assert info.names == ["shallow_water"]
    assert any("g" in str(c) for c in info.type_conditions)
    speeds = [s for s, _ in info.characteristic_speeds]
    assert len(speeds) == 2
    assert zero(speeds[0] + speeds[1] - 2 * u)


def test_euler_1d_and_maxwell_1d(P):
    gamma = P.pool.symbol("gamma")
    rho, u, p = (Function(n)(P.x, P.t) for n in ("rho", "u", "p"))
    eqs = [
        D(rho, P.t) + D(rho * u, P.x),
        D(u, P.t) + u * D(u, P.x) + D(p, P.x) / rho,
        D(p, P.t) + u * D(p, P.x) + gamma * p * D(u, P.x),
    ]
    info = akx.classify_pde(eqs, [rho, u, p])
    assert info.type == "hyperbolic"
    assert info.names == ["euler_1d"]
    assert len(info.characteristic_speeds) == 3

    c = P.pool.symbol("c", Domain.Positive)
    E, B = Function("E")(P.x, P.t), Function("B")(P.x, P.t)
    info = akx.classify_pde([D(E, P.t) + c**2 * D(B, P.x), D(B, P.t) + D(E, P.x)], [E, B])
    assert info.type == "hyperbolic"
    assert info.names == ["maxwell_1d"]


def test_jordan_block_is_not_hyperbolic(P):
    u, v = Function("u")(P.x, P.t), Function("v")(P.x, P.t)
    info = akx.classify_pde([D(u, P.t) + D(u, P.x) + D(v, P.x), D(v, P.t) + D(v, P.x)], [u, v])
    assert info.type == "undetermined"
    info = akx.classify_pde([D(u, P.t) + D(u, P.x), D(v, P.t) + D(v, P.x)], [u, v])
    assert info.type == "hyperbolic"
    assert info.strictly_hyperbolic is False


def test_second_order_system_order_and_linearity(P):
    u, v = Function("u")(P.x, P.t), Function("v")(P.x, P.t)
    info = akx.classify_pde([D(u, P.t) - D(u, P.x, 2), D(v, P.t) - u * v], [u, v])
    assert info.order == 2
    assert info.orders == [2, 1]
    assert info.linearity == "semilinear"
    assert info.type is None


def test_scalar_still_returns_pde_info(P):
    u = Function("u")(P.x, P.t)
    info = akx.classify_pde(D(u, P.t) - D(u, P.x, 2), u)
    assert isinstance(info, akx.PdeInfo)
    assert info.names == ["heat"]


# ---------------------------------------------------------------------------
# n-D coordinates
# ---------------------------------------------------------------------------


def test_cartesian_n_laplacian_of_unknown(P):
    xs = [P.pool.symbol(f"x{i}") for i in range(1, 5)]
    c = akx.CoordinatesN.cartesian(*xs)
    assert c.dim == 4
    u = Function("u")(*xs)
    lap = c.laplacian(u)
    want = sum((D(u, x, 2) for x in xs[1:]), D(u, xs[0], 2))
    assert zero(lap - want)
    g = c.gradient(u)
    assert len(g) == 4
    assert zero(c.divergence(g) - lap)
    with pytest.raises(PdeError) as e:
        c.divergence(g[:3])
    assert code(e) == "E-PDE-041"


def test_hyperspherical_fundamental_solution(P):
    r = P.pool.symbol("r", Domain.Positive)
    phis = [P.pool.symbol(f"phi{i}") for i in range(1, 4)]
    c = akx.CoordinatesN.hyperspherical(r, *phis)
    assert c.dim == 4
    assert zero(c.laplacian(r ** (-2)))
    th = P.pool.symbol("theta")
    pol = akx.CoordinatesN.polar(r, th)
    assert zero(pol.laplacian(ak.log(r)))
    assert zero(pol.laplacian(r**2) - 4)


def test_embedding_orthogonality_is_proved(P):
    a, b = P.pool.symbol("a"), P.pool.symbol("b")
    with pytest.raises(PdeError) as e:
        akx.CoordinatesN.from_embedding([a, b], [a + b, b])
    assert code(e) == "E-PDE-041"
    r = P.pool.symbol("r", Domain.Positive)
    th = P.pool.symbol("theta")
    c = akx.CoordinatesN.from_embedding([r, th], [r * ak.cos(th), r * ak.sin(th)], "polar")
    assert zero(c.laplacian(r**2) - 4)


def test_symbolic_dimension(P):
    x = P.pool.symbol("x")
    d = P.pool.symbol("d", Domain.Positive)
    j = P.pool.symbol("j", Domain.Integer)
    c = akx.SymbolicCartesian(x, d)
    r2 = c.radius_squared()
    assert zero(c.laplacian(r2) - 2 * d)
    assert zero(c.gradient(r2, j) - 2 * c.coordinate(j))
    assert zero(c.divergence(c.coordinate(j), j) - d)
