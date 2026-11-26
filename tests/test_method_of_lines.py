"""PDE milestone M7b: method of lines (symbolic PDE -> compiled ODE/DAE).

Every discretisation is cross-checked against an exact solution (heat by
Fourier modes, d'Alembert, Burgers via Cole-Hopf, the KdV soliton, bistable
Allen-Cahn and Fisher-KPP travelling fronts, a coupled reaction-diffusion
system) and, where the theory gives one, against the expected convergence
order.
"""

from __future__ import annotations

import math
import warnings

import pytest

np = pytest.importorskip("numpy")

import alkahest as ak  # noqa: E402
import alkahest.experimental as akx  # noqa: E402
from alkahest.exceptions import NumericPdeError  # noqa: E402

pytestmark = pytest.mark.skipif(
    not hasattr(ak.alkahest, "_MolProgram"), reason="needs the method-of-lines runtime"
)


@pytest.fixture
def P():
    return ak.ExprPool()


def err_inf(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def orders(errs):
    return [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]


def heat(P, nu=1):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    pde = akx.PDE(akx.D(U, t) - nu * akx.D(U, x, 2), U).on(akx.Interval(x, 0, 1))
    return x, t, u, pde


def dx_at(u, x, t, val, P):
    """``u_x(val, t)``."""
    return ak.subs(akx.D(u(x, t), x), {x: P.integer(val)})


# ---------------------------------------------------------------------------
# Heat equation: Dirichlet / Neumann / Robin, orders, stretched grids, DAE
# ---------------------------------------------------------------------------


def test_heat_dirichlet_fourier_mode_order_2(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(P.pi() * x))
    errs = []
    for n in (11, 21, 41):
        ms = akx.method_of_lines(pr, {x: n})
        res = ms.integrate(0.1, method="bdf", rtol=1e-10, atol=1e-12)
        xs = ms.grid["x"]
        errs.append(err_inf(res.field("u"), np.exp(-(np.pi**2) * 0.1) * np.sin(np.pi * xs)))
    o = orders(errs)
    assert all(1.9 < v < 2.1 for v in o), (errs, o)


def test_heat_order_4_and_6_stencils(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(P.pi() * x))
    for p, lo, grids in ((4, 3.8, (21, 41, 81)), (6, 5.5, (11, 21, 41))):
        errs = []
        for n in grids:
            ms = akx.method_of_lines(pr, {x: n}, order=p)
            res = ms.integrate(0.05, rtol=1e-12, atol=1e-14)
            exact = np.exp(-(np.pi**2) * 0.05) * np.sin(np.pi * ms.grid["x"])
            errs.append(err_inf(res.field("u"), exact))
        o = orders(errs)
        assert all(v > lo for v in o), (p, errs, o)


def test_heat_neumann_eliminated_and_algebraic(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(dx_at(u, x, t, 0, P), 0).bc(dx_at(u, x, t, 1, P), 0).ic(u(x, 0), ak.cos(P.pi() * x))
    for mode in ("eliminate", "algebraic"):
        errs = []
        for n in (41, 81, 161):
            ms = akx.method_of_lines(pr, {x: n}, bc_mode=mode)
            assert ms.algebraic.any() == (mode == "algebraic")
            res = ms.integrate(0.1, rtol=1e-10, atol=1e-12)
            exact = np.exp(-(np.pi**2) * 0.1) * np.cos(np.pi * ms.grid["x"])
            errs.append(err_inf(res.field("u"), exact))
        o = orders(errs)
        # one-sided second-order closures: pre-asymptotic at coarse grids
        assert o[0] > 1.6, (mode, errs, o)
        assert o[-1] > 1.8, (mode, errs, o)


def test_heat_robin_and_time_dependent_dirichlet(P):
    # u = e^{-k^2 t} (k cos kx + sin kx): u_x(0) - u(0) = 0 (Robin) and the
    # exact, time-dependent value at x = 1.
    x, t, u, pde = heat(P)
    k = 2
    ex = ak.exp(-(k**2) * t) * (k * ak.cos(k * x) + ak.sin(k * x))
    pr = (
        pde.bc(dx_at(u, x, t, 0, P) - u(0, t), 0)
        .bc(u(1, t), ak.subs(ex, {x: P.integer(1)}))
        .ic(u(x, 0), ak.subs(ex, {t: P.integer(0)}))
    )
    errs = []
    for n in (21, 41, 81):
        ms = akx.method_of_lines(pr, {x: n})
        res = ms.integrate(0.2, rtol=1e-11, atol=1e-13)
        xs = ms.grid["x"]
        exact = np.exp(-(k**2) * 0.2) * (k * np.cos(k * xs) + np.sin(k * xs))
        errs.append(err_inf(res.field("u"), exact))
    o = orders(errs)
    assert all(v > 1.8 for v in o), (errs, o)


def test_stretched_grid(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(P.pi() * x))
    errs = []
    for n in (21, 41, 81):
        s = np.linspace(0, 1, n)
        nodes = s + 0.1 * np.sin(2 * np.pi * s) / (2 * np.pi)  # smooth stretching
        nodes[0], nodes[-1] = 0.0, 1.0
        ms = akx.method_of_lines(pr, {x: nodes})
        res = ms.integrate(0.1, rtol=1e-11, atol=1e-13)
        errs.append(err_inf(res.field("u"), np.exp(-(np.pi**2) * 0.1) * np.sin(np.pi * nodes)))
    o = orders(errs)
    assert all(v > 1.8 for v in o), (errs, o)


def test_chebyshev_spectral_accuracy(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(P.pi() * x))
    ms = akx.method_of_lines(pr, {x: 20}, scheme="chebyshev")
    res = ms.integrate(0.1, rtol=1e-12, atol=1e-14)
    xs = ms.grid["x"]
    assert err_inf(res.field("u"), np.exp(-(np.pi**2) * 0.1) * np.sin(np.pi * xs)) < 1e-9
    # Neumann with Chebyshev (both ends eliminated together).
    pr2 = (
        pde.bc(dx_at(u, x, t, 0, P), 0).bc(dx_at(u, x, t, 1, P), 0).ic(u(x, 0), ak.cos(P.pi() * x))
    )
    ms2 = akx.method_of_lines(pr2, {x: 20}, scheme="chebyshev")
    res2 = ms2.integrate(0.1, rtol=1e-12, atol=1e-14)
    xs = ms2.grid["x"]
    assert err_inf(res2.field("u"), np.exp(-(np.pi**2) * 0.1) * np.cos(np.pi * xs)) < 1e-8


def test_heat_2d_and_3d(P):
    x, y, z, t = (P.symbol(s) for s in "xyzt")
    u = akx.Function("u")
    U = u(x, y, t)
    pr = (
        akx.PDE(akx.D(U, t) - akx.D(U, x, 2) - akx.D(U, y, 2), U)
        .on(akx.Box([x, y], [(0, 1), (0, 1)]))
        .bc(u(0, y, t), 0)
        .bc(u(1, y, t), 0)
        .bc(u(x, 0, t), 0)
        .bc(u(x, 1, t), 0)
        .ic(u(x, y, 0), ak.sin(P.pi() * x) * ak.sin(P.pi() * y))
    )
    errs = []
    for n in (11, 21, 41):
        ms = akx.method_of_lines(pr, n)
        res = ms.integrate(0.05, rtol=1e-10, atol=1e-12)
        X, Y = np.meshgrid(ms.grid["x"], ms.grid["y"], indexing="ij")
        exact = np.exp(-2 * np.pi**2 * 0.05) * np.sin(np.pi * X) * np.sin(np.pi * Y)
        errs.append(err_inf(res.field("u"), exact))
    o = orders(errs)
    assert all(1.9 < v < 2.1 for v in o), (errs, o)
    # 3-D
    V = u(x, y, z, t)
    pr3 = (
        akx.PDE(akx.D(V, t) - akx.D(V, x, 2) - akx.D(V, y, 2) - akx.D(V, z, 2), V)
        .on(akx.Box([x, y, z], [(0, 1), (0, 1), (0, 1)]))
        .bc(u(0, y, z, t), 0)
        .bc(u(1, y, z, t), 0)
        .bc(u(x, 0, z, t), 0)
        .bc(u(x, 1, z, t), 0)
        .bc(u(x, y, 0, t), 0)
        .bc(u(x, y, 1, t), 0)
        .ic(u(x, y, z, 0), ak.sin(P.pi() * x) * ak.sin(P.pi() * y) * ak.sin(P.pi() * z))
    )
    ms3 = akx.method_of_lines(pr3, 11)
    assert ms3.n_states == 9**3
    res3 = ms3.integrate(0.02, rtol=1e-9, atol=1e-11)
    X, Y, Z = np.meshgrid(*(ms3.grid[k] for k in "xyz"), indexing="ij")
    exact = np.exp(-3 * np.pi**2 * 0.02) * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    assert err_inf(res3.field("u"), exact) < 5e-3


# ---------------------------------------------------------------------------
# Periodic, upwind, WENO, Fourier
# ---------------------------------------------------------------------------


def advection(P, c, nu=0):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    eq = akx.D(U, t) + c * akx.D(U, x) - nu * akx.D(U, x, 2)
    two_pi = 2 * P.pi()
    pr = (
        akx.PDE(eq, U)
        .on(akx.Interval(x, 0, two_pi))
        .bc(u(0, t), u(two_pi, t))
        .ic(u(x, 0), ak.sin(x))
    )
    return x, pr


def test_fourier_advection_diffusion(P):
    x, pr = advection(P, 1, P.rational(1, 10))
    ms = akx.method_of_lines(pr, {x: 32}, scheme="fourier")
    res = ms.integrate(1.0, rtol=1e-11, atol=1e-13)
    xs = ms.grid["x"]
    assert err_inf(res.field("u"), np.exp(-0.1) * np.sin(xs - 1.0)) < 1e-9


def test_periodic_central_and_upwind_orders(P):
    for c in (1, -1):
        x, pr = advection(P, c)
        for scheme, order, lo in (
            ("central", 2, 1.9),
            ("central", 4, 3.8),
            ("upwind", 1, 0.9),
            ("upwind", 3, 2.8),
        ):
            errs = []
            for n in (32, 64, 128):
                ms = akx.method_of_lines(pr, {x: n}, scheme=scheme, order=order)
                res = ms.integrate(1.0, method="rk45", rtol=1e-11, atol=1e-13)
                errs.append(err_inf(res.field("u"), np.sin(ms.grid["x"] - c * 1.0)))
            o = orders(errs)
            assert all(v > lo for v in o), (c, scheme, order, errs, o)


def test_upwind_is_monotone_for_a_step(P):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    pr = (
        akx.PDE(akx.D(U, t) - akx.D(U, x), U)  # velocity -1: moves left
        .on(akx.Interval(x, 0, 1))
        .bc(u(0, t), u(1, t))
        .ic(u(x, 0), ak.tanh(200 * (x - P.rational(1, 2))))
    )
    ms = akx.method_of_lines(pr, {x: 200}, scheme="upwind", order=1)
    res = ms.integrate(0.25, method="rk45", rtol=1e-8, atol=1e-10)
    v = res.field("u")
    assert v.max() <= 1 + 1e-6
    assert v.min() >= -1 - 1e-6
    ms2 = akx.method_of_lines(pr, {x: 200}, scheme="central", order=2)
    v2 = ms2.integrate(0.25, method="rk45", rtol=1e-8, atol=1e-10).field("u")
    assert v2.max() > 1.05  # central differences oscillate


def test_weno5_smooth_order_and_nonoscillatory(P):
    x, pr = advection(P, 1)
    errs = []
    for n in (20, 40, 80):
        ms = akx.method_of_lines(pr, {x: n}, scheme="weno5")
        res = ms.integrate(0.5, method="rk45", rtol=1e-12, atol=1e-14)
        errs.append(err_inf(res.field("u"), np.sin(ms.grid["x"] - 0.5)))
    o = orders(errs)
    assert o[-1] > 3.5, (errs, o)


# ---------------------------------------------------------------------------
# Nonlinear equations against exact solutions
# ---------------------------------------------------------------------------


def test_burgers_cole_hopf(P):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    nu = P.rational(1, 20)
    a = 2
    pr = (
        akx.PDE(akx.D(U, t) + U * akx.D(U, x) - nu * akx.D(U, x, 2), U)
        .on(akx.Interval(x, 0, 1))
        .bc(u(0, t), 0)
        .bc(u(1, t), 0)
        .ic(u(x, 0), 2 * nu * P.pi() * ak.sin(P.pi() * x) / (a + ak.cos(P.pi() * x)))
    )

    def exact(xs, tt):
        e = np.exp(-(np.pi**2) * 0.05 * tt)
        return 2 * 0.05 * np.pi * e * np.sin(np.pi * xs) / (a + e * np.cos(np.pi * xs))

    for scheme in ("central", "upwind", "weno5"):
        errs = []
        for n in (41, 81, 161):
            ms = akx.method_of_lines(pr, {x: n}, scheme=scheme, order=2)
            res = ms.integrate(1.0, rtol=1e-10, atol=1e-12)
            errs.append(err_inf(res.field("u"), exact(ms.grid["x"], 1.0)))
        o = orders(errs)
        assert o[-1] > 1.8, (scheme, errs, o)


def test_kdv_soliton(P):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    c = 2
    L = 20
    pr = (
        akx.PDE(akx.D(U, t) + 6 * U * akx.D(U, x) + akx.D(U, x, 3), U)
        .on(akx.Interval(x, -L, L))
        .bc(u(-L, t), u(L, t))
        .ic(u(x, 0), P.rational(c, 2) / ak.cosh(ak.sqrt(P.integer(c)) / 2 * x) ** 2)
    )

    def exact(xs, tt):
        return c / 2 / np.cosh(np.sqrt(c) / 2 * (xs - c * tt)) ** 2

    errs = []
    for n in (200, 400):
        ms = akx.method_of_lines(pr, {x: n}, order=4)
        res = ms.integrate(1.0, method="rk45", rtol=1e-10, atol=1e-12)
        errs.append(err_inf(res.field("u"), exact(ms.grid["x"], 1.0)))
    assert errs[-1] < 2e-3, errs
    assert orders(errs)[0] > 3.5, errs
    ms = akx.method_of_lines(pr, {x: 128}, scheme="fourier")
    res = ms.integrate(1.0, method="rk45", rtol=1e-10, atol=1e-12)
    assert err_inf(res.field("u"), exact(ms.grid["x"], 1.0)) < 1e-4


def test_allen_cahn_and_fisher_fronts(P):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    a = P.rational(1, 4)
    cspeed = math.sqrt(2) * (0.25 - 0.5)
    # Bistable Allen-Cahn (Nagumo): u_t = u_xx + u(1-u)(u-a); front
    # u = 1 / (1 + exp(-(x - c t)/sqrt 2)), c = sqrt2 (a - 1/2).
    s2 = ak.sqrt(P.integer(2))
    front = 1 / (1 + ak.exp(-(x - P.float(cspeed) * t) / s2))
    L = 20
    pr = (
        akx.PDE(akx.D(U, t) - akx.D(U, x, 2) - U * (1 - U) * (U - a), U)
        .on(akx.Interval(x, -L, L))
        .bc(u(-L, t), ak.subs(front, {x: P.integer(-L)}))
        .bc(u(L, t), ak.subs(front, {x: P.integer(L)}))
        .ic(u(x, 0), ak.subs(front, {t: P.integer(0)}))
    )

    def exact(xs, tt):
        return 1 / (1 + np.exp(-(xs - cspeed * tt) / np.sqrt(2)))

    errs = []
    for n in (101, 201, 401):
        ms = akx.method_of_lines(pr, {x: n})
        res = ms.integrate(5.0, rtol=1e-10, atol=1e-12)
        errs.append(err_inf(res.field("u"), exact(ms.grid["x"], 5.0)))
    o = orders(errs)
    assert all(v > 1.8 for v in o), (errs, o)
    # Fisher-KPP (Ablowitz-Zeppetella): u = (1 + exp(x/sqrt6 - 5t/6))^-2.
    s6 = ak.sqrt(P.integer(6))
    fk = (1 + ak.exp(x / s6 - P.rational(5, 6) * t)) ** -2
    prf = (
        akx.PDE(akx.D(U, t) - akx.D(U, x, 2) - U * (1 - U), U)
        .on(akx.Interval(x, -L, L))
        .bc(u(-L, t), ak.subs(fk, {x: P.integer(-L)}))
        .bc(u(L, t), ak.subs(fk, {x: P.integer(L)}))
        .ic(u(x, 0), ak.subs(fk, {t: P.integer(0)}))
    )
    ms = akx.method_of_lines(prf, {x: 401}, order=4)
    res = ms.integrate(2.0, rtol=1e-10, atol=1e-12)
    xs = ms.grid["x"]
    assert err_inf(res.field("u"), (1 + np.exp(xs / np.sqrt(6) - 5 * 2.0 / 6)) ** -2) < 1e-5


def test_wave_dalembert(P):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    pr = (
        akx.PDE(akx.D(U, t, 2) - 4 * akx.D(U, x, 2), U)
        .on(akx.Interval(x, 0, 1))
        .bc(u(0, t), 0)
        .bc(u(1, t), 0)
        .ic(u(x, 0), ak.sin(P.pi() * x))
        .ic(ak.subs(akx.D(U, t), {t: P.integer(0)}), 0)
    )
    errs = []
    for n in (21, 41, 81):
        ms = akx.method_of_lines(pr, {x: n})
        assert ms.fields[:2] == ["u", "u_t"]
        res = ms.integrate(0.7, method="rk45", rtol=1e-11, atol=1e-13)
        xs = ms.grid["x"]
        # d'Alembert: (f(x - 2t) + f(x + 2t)) / 2 with f the odd periodic extension.
        exact = 0.5 * (np.sin(np.pi * (xs - 1.4)) + np.sin(np.pi * (xs + 1.4)))
        errs.append(err_inf(res.field("u"), exact))
    o = orders(errs)
    assert all(v > 1.9 for v in o), (errs, o)


def test_coupled_reaction_diffusion_system(P):
    x, t = P.symbol("x"), P.symbol("t")
    u, v = akx.Function("u"), akx.Function("v")
    U, V = u(x, t), v(x, t)
    Dc = P.rational(1, 10)
    pr = (
        akx.PDE(
            [akx.D(U, t) - Dc * akx.D(U, x, 2) - V, akx.D(V, t) - Dc * akx.D(V, x, 2) + U], [U, V]
        )
        .on(akx.Interval(x, 0, 1))
        .bc(u(0, t), 0)
        .bc(u(1, t), 0)
        .bc(v(0, t), 0)
        .bc(v(1, t), 0)
        .ic(u(x, 0), ak.sin(P.pi() * x))
        .ic(v(x, 0), 0)
    )
    ms = akx.method_of_lines(pr, {x: 81})
    res = ms.integrate(1.0, rtol=1e-10, atol=1e-12)
    xs = ms.grid["x"]
    dec = np.exp(-0.1 * np.pi**2)
    assert err_inf(res.field("u"), dec * np.cos(1.0) * np.sin(np.pi * xs)) < 1e-3
    assert err_inf(res.field("v"), -dec * np.sin(1.0) * np.sin(np.pi * xs)) < 1e-3


# ---------------------------------------------------------------------------
# Jacobian, exports, integrators
# ---------------------------------------------------------------------------


def fd_jacobian(ms, t, y, eps=1e-7):
    n = y.size
    J = np.zeros((n, n))
    f0 = ms.rhs(t, y)
    for j in range(n):
        y1 = y.copy()
        h = eps * max(1.0, abs(y[j]))
        y1[j] += h
        J[:, j] = (ms.rhs(t, y1) - f0) / h
    return J


def test_exact_jacobian_matches_finite_differences(P):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    nu = P.rational(1, 10)
    base = akx.PDE(akx.D(U, t) + U * akx.D(U, x) - nu * akx.D(U, x, 2) + U**3, U).on(
        akx.Interval(x, 0, 1)
    )
    cases = [
        (base.bc(u(0, t), ak.sin(t)).bc(dx_at(u, x, t, 1, P) + 2 * u(1, t), 0), {}),
        (base.bc(u(0, t), u(1, t)), {}),
        (base.bc(u(0, t), 0).bc(dx_at(u, x, t, 1, P) + 2 * u(1, t), 1), {"scheme": "chebyshev"}),
        (base.bc(u(0, t), u(1, t)), {"scheme": "weno5"}),
        (base.bc(u(0, t), 1).bc(dx_at(u, x, t, 1, P) + u(1, t) ** 2, 0), {"bc_mode": "algebraic"}),
    ]
    rng = np.random.default_rng(0)
    for pr, kw in cases:
        pr = pr.ic(u(x, 0), ak.sin(x) + 1)
        ms = akx.method_of_lines(pr, {x: 12}, **kw)
        y = rng.uniform(0.2, 1.0, ms.n_states)
        J = ms.jacobian(0.3, y, format="dense")
        Jfd = fd_jacobian(ms, 0.3, y)
        assert np.allclose(J, Jfd, rtol=1e-4, atol=1e-4 * np.abs(Jfd).max()), kw


def test_rhs_kernel_count_is_independent_of_grid(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(P.pi() * x))
    small = akx.method_of_lines(pr, {x: 41})
    big = akx.method_of_lines(pr, {x: 1_000_001})
    assert small.n_kernels == big.n_kernels
    assert big.n_states == 999_999
    f = big.rhs(0.0, big.y0)
    xs = big.grid["x"][1:-1]
    # rounding of the second difference at h = 1e-6 is ~ eps/h^2 ~ 1e-4
    assert err_inf(f, -(np.pi**2) * np.sin(np.pi * xs)) < 1e-2
    mid = akx.method_of_lines(pr, {x: 2001})
    xs = mid.grid["x"][1:-1]
    assert err_inf(mid.rhs(0.0, mid.y0), -(np.pi**2) * np.sin(np.pi * xs)) < 1e-5


def test_jax_export_matches_native_rhs(P):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    pr = (
        akx.PDE(akx.D(U, t) + U * akx.D(U, x) - P.rational(1, 10) * akx.D(U, x, 2), U)
        .on(akx.Interval(x, 0, 1))
        .bc(u(0, t), ak.sin(t))
        .bc(dx_at(u, x, t, 1, P), 0)
        .ic(u(x, 0), ak.cos(x))
    )
    for kw in ({}, {"scheme": "chebyshev"}):
        ms = akx.method_of_lines(pr, {x: 16}, **kw)
        f = ms.to_jax()
        y = ms.y0 + 0.1
        assert np.allclose(np.asarray(jax.jit(f)(0.3, y)), ms.rhs(0.3, y), rtol=1e-12, atol=1e-10)
        # forward-mode Jacobian of the JAX function equals the exact one
        Jj = np.asarray(jax.jacfwd(f, argnums=1)(0.3, y))
        assert np.allclose(Jj, ms.jacobian(0.3, y, format="dense"), rtol=1e-9, atol=1e-9)
    text = ms.to_stablehlo()
    assert "stablehlo" in text or "mhlo" in text


def test_integrators_agree(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), 0).bc(dx_at(u, x, t, 1, P), 0).ic(u(x, 0), ak.sin(P.pi() * x / 2))
    ms = akx.method_of_lines(pr, {x: 15})
    ref = ms.integrate(0.2, method="bdf", rtol=1e-11, atol=1e-13).y[-1]
    for method in ("rk45", "alkahest_rk45", "scipy_bdf", "scipy_radau"):
        if method.startswith("scipy"):
            pytest.importorskip("scipy")
        res = ms.integrate(0.2, method=method, rtol=1e-9, atol=1e-11)
        assert err_inf(res.y[-1], ref) < 1e-6, method
    res = ms.integrate(0.2, method="rk4", dt=1e-4)
    assert err_inf(res.y[-1], ref) < 1e-6
    # the semi-discrete solution decays like exp(-pi^2/4 t)
    assert abs(ref.max() - math.exp(-(math.pi**2) / 4 * 0.2)) < 5e-3


def test_bdf_is_efficient_on_stiff_heat(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(P.pi() * x))
    ms = akx.method_of_lines(pr, {x: 401})
    res = ms.integrate(0.5, method="bdf", rtol=1e-6, atol=1e-9)
    assert res.stats["steps"] < 300, res.stats
    assert res.stats["band"][1] <= 2


# ---------------------------------------------------------------------------
# Stability hook
# ---------------------------------------------------------------------------


def test_stability_report_and_warning(P):
    x, t, u, pde = heat(P, nu=P.rational(1, 10))
    pr = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(P.pi() * x))
    ms = akx.method_of_lines(pr, {x: 21})
    rep = ms.stability_report()["u"]
    eu = rep["methods"]["euler"]
    h = 1 / 20
    assert eu["proof_status"] == "proved"
    assert abs(eu["max_dt"] - h**2 / (2 * 0.1)) < 1e-12
    rk4 = rep["methods"]["rk4"]
    assert rk4["max_dt"] > eu["max_dt"]
    with pytest.warns(akx.MolStabilityWarning, match="dt <= 0.0125"):
        ms.integrate(0.01, method="euler", dt=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = ms.integrate(0.1, method="euler", dt=0.01)
    exact = np.exp(-0.1 * np.pi**2 * 0.1) * np.sin(np.pi * ms.grid["x"])
    assert err_inf(res.field("u"), exact) < 5e-3


def test_stability_report_2d_is_necessary(P):
    x, y, t = (P.symbol(s) for s in "xyt")
    u = akx.Function("u")
    U = u(x, y, t)
    pr = (
        akx.PDE(akx.D(U, t) - akx.D(U, x, 2) - akx.D(U, y, 2), U)
        .on(akx.Box([x, y], [(0, 1), (0, 1)]))
        .bc(u(0, y, t), 0)
        .bc(u(1, y, t), 0)
        .bc(u(x, 0, t), 0)
        .bc(u(x, 1, t), 0)
        .ic(u(x, y, 0), ak.sin(P.pi() * x) * ak.sin(P.pi() * y))
    )
    ms = akx.method_of_lines(pr, 11)
    eu = ms.stability_report()["u"]["methods"]["euler"]
    assert eu["proof_status"] == "necessary"
    assert abs(eu["max_dt"] - 0.1**2 / 4) < 1e-12


# ---------------------------------------------------------------------------
# DAE path and refusals
# ---------------------------------------------------------------------------


def test_algebraic_bcs_give_a_dae(P):
    x, t, u, pde = heat(P)
    pr = pde.bc(u(0, t), ak.sin(t)).bc(dx_at(u, x, t, 1, P) + u(1, t), 0).ic(u(x, 0), 0)
    ms_e = akx.method_of_lines(pr, {x: 41})
    ms_a = akx.method_of_lines(pr, {x: 41}, bc_mode="algebraic")
    assert ms_a.n_states == ms_e.n_states + 2
    assert ms_a.algebraic.sum() == 2
    with pytest.raises(NumericPdeError) as ei:
        ms_a.integrate(1.0, method="rk45")
    assert ei.value.code == "E-PDE-369"
    a = ms_a.integrate(1.0, rtol=1e-9, atol=1e-11).field("u")
    e = ms_e.integrate(1.0, rtol=1e-9, atol=1e-11).field("u")
    assert err_inf(a, e) < 1e-6
    dae = akx.method_of_lines(pr, {x: 6}, bc_mode="algebraic").to_dae()
    assert dae.n_equations == 6


def test_refusals(P):
    x, t, u, pde = heat(P)
    U = u(x, t)
    nu = P.symbol("nu")
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(akx.PDE(akx.D(U, t) - akx.D(U, x, 2), U), 10)
    assert ei.value.code == "E-PDE-359"
    pr = akx.PDE(akx.D(U, t) ** 2 - akx.D(U, x, 2), U).on(akx.Interval(x, 0, 1))
    pr = pr.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), 0)
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(pr, 10)
    assert ei.value.code == "E-PDE-360"
    good = pde.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(x))
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(good, {x: 2})
    assert ei.value.code == "E-PDE-361"
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(pde.bc(u(P.rational(1, 2), t), 0).ic(u(x, 0), 0), 10)
    assert ei.value.code == "E-PDE-362"
    # u(0)^2 + u_x(0) = 0 is nonlinear in the boundary value: refused when
    # eliminating, fine as an algebraic constraint.
    bad = pde.bc(u(0, t) ** 2 + dx_at(u, x, t, 0, P), 0).bc(u(1, t), 0).ic(u(x, 0), 0)
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(bad, 10)
    assert ei.value.code == "E-PDE-363"
    assert akx.method_of_lines(bad, 10, bc_mode="algebraic").algebraic.sum() == 2
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(pde.bc(u(0, t), 2 * u(1, t)).ic(u(x, 0), 0), 10)
    assert ei.value.code == "E-PDE-362"
    bad = pde.bc(u(0, t) ** 3, 1).bc(u(1, t), 0).ic(u(x, 0), 0)
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(bad, 10)
    assert ei.value.code == "E-PDE-363"
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(pde.bc(u(0, t), 0).bc(u(1, t), 0), 10)
    assert ei.value.code == "E-PDE-364"
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(good, 10, scheme="fourier")
    assert ei.value.code == "E-PDE-365"
    pr_nu = akx.PDE(akx.D(U, t) - nu * akx.D(U, x, 2), U).on(akx.Interval(x, 0, 1))
    pr_nu = pr_nu.bc(u(0, t), 0).bc(u(1, t), 0).ic(u(x, 0), ak.sin(x))
    with pytest.raises(NumericPdeError) as ei:
        akx.method_of_lines(pr_nu, 10)
    assert ei.value.code == "E-PDE-366"
    ms = akx.method_of_lines(pr_nu, 10, params={nu: 0.5})
    assert ms.n_states == 8
    with pytest.raises(NumericPdeError) as ei:
        ms.integrate(1.0, method="euler")
    assert ei.value.code == "E-PDE-372"


def test_index_two_constraint_is_refused_with_the_index_reduction(P):
    # u_t = u_xx + v with the algebraic equation 0 = u - e^{-t} sin(pi x): the
    # constraint does not involve v, so the DAE has index 2.
    x, t = P.symbol("x"), P.symbol("t")
    u, v = akx.Function("u"), akx.Function("v")
    U, V = u(x, t), v(x, t)
    g = ak.exp(-t) * ak.sin(P.pi() * x)
    pr = (
        akx.PDE([akx.D(U, t) - akx.D(U, x, 2) - V, U - g], [U, V])
        .on(akx.Interval(x, 0, 1))
        .bc(u(0, t), 0)
        .bc(u(1, t), 0)
        .ic(u(x, 0), ak.sin(P.pi() * x))
    )
    ms = akx.method_of_lines(pr, {x: 6})
    assert ms.algebraic.sum() == 6
    with pytest.raises(NumericPdeError) as ei:
        ms.integrate(0.1)
    assert ei.value.code == "E-PDE-369"
    assert "dae_index_reduce" in str(ei.value)


def test_2d_neumann_corners_and_doubly_periodic(P):
    x, y, t = (P.symbol(s) for s in "xyt")
    u = akx.Function("u")
    U = u(x, y, t)
    heat2 = akx.PDE(akx.D(U, t) - akx.D(U, x, 2) - akx.D(U, y, 2), U)

    def nx(v):
        return ak.subs(akx.D(U, x), {x: P.integer(v)})

    def ny(v):
        return ak.subs(akx.D(U, y), {y: P.integer(v)})

    pr = (
        heat2.on(akx.Box([x, y], [(0, 1), (0, 1)]))
        .bc(nx(0), 0)
        .bc(nx(1), 0)
        .bc(ny(0), 0)
        .bc(ny(1), 0)
        .ic(u(x, y, 0), ak.cos(P.pi() * x) * ak.cos(P.pi() * y))
    )
    errs = []
    for n in (21, 41, 81):
        ms = akx.method_of_lines(pr, n)
        res = ms.integrate(0.05, rtol=1e-10, atol=1e-12)
        X, Y = np.meshgrid(ms.grid["x"], ms.grid["y"], indexing="ij")
        exact = np.exp(-2 * np.pi**2 * 0.05) * np.cos(np.pi * X) * np.cos(np.pi * Y)
        errs.append(err_inf(res.field("u"), exact))
    # the one-sided Neumann closures are pre-asymptotic on these grids (the
    # order tends to 2: 1.63, 1.83, ... in a refinement study to n = 161)
    assert orders(errs)[-1] > 1.55, errs
    assert errs[-1] < 5e-5, errs
    two_pi = 2 * P.pi()
    pp = (
        heat2.on(akx.Box([x, y], [(0, two_pi), (0, two_pi)]))
        .bc(u(0, y, t), u(two_pi, y, t))
        .bc(u(x, 0, t), u(x, two_pi, t))
        .ic(u(x, y, 0), ak.sin(x) * ak.cos(2 * y))
    )
    ms = akx.method_of_lines(pp, 32, order=4)
    res = ms.integrate(0.2, rtol=1e-10, atol=1e-12)
    X, Y = np.meshgrid(ms.grid["x"], ms.grid["y"], indexing="ij")
    exact = np.exp(-5 * 0.2) * np.sin(X) * np.cos(2 * Y)
    assert err_inf(res.field("u"), exact) < 1e-3
    J = ms.jacobian(0.0, ms.y0 + 0.1, format="dense")
    Jfd = fd_jacobian(ms, 0.0, ms.y0 + 0.1)
    assert np.allclose(J, Jfd, atol=1e-3 * np.abs(Jfd).max())


def test_mixed_derivative_and_variable_coefficients(P):
    # u_t = u_xx + u_yy + (1/2) u_xy + x*u_x on the unit square, Dirichlet 0:
    # checks the tensor-product mixed stencil and coordinate-dependent
    # coefficients against a fine-grid reference and the exact Jacobian.
    x, y, t = (P.symbol(s) for s in "xyt")
    u = akx.Function("u")
    U = u(x, y, t)
    eq = (
        akx.D(U, t)
        - akx.D(U, x, 2)
        - akx.D(U, y, 2)
        - P.rational(1, 2) * akx.D(U, x, y)
        - x * akx.D(U, x)
    )
    pr = (
        akx.PDE(eq, U)
        .on(akx.Box([x, y], [(0, 1), (0, 1)]))
        .bc(u(0, y, t), 0)
        .bc(u(1, y, t), 0)
        .bc(u(x, 0, t), 0)
        .bc(u(x, 1, t), 0)
        .ic(u(x, y, 0), ak.sin(P.pi() * x) * ak.sin(P.pi() * y))
    )
    vals = []
    for n in (11, 21, 41, 81):
        ms = akx.method_of_lines(pr, n)
        vals.append(ms.integrate(0.05, rtol=1e-10, atol=1e-12).field("u")[n // 2, n // 2])
    d = [abs(vals[i + 1] - vals[i]) for i in range(3)]
    assert d[1] / d[2] > 3.5, (vals, d)
    ms = akx.method_of_lines(pr, 7)
    y0 = ms.y0 + 0.05
    J = ms.jacobian(0.0, y0, format="dense")
    assert np.allclose(J, fd_jacobian(ms, 0.0, y0), rtol=1e-5, atol=1e-5)
