"""Gap G5 of the method-of-lines layer: the conservative WENO-5 flux
(``scheme="weno5_fv"``), the Krylov (GMRES + ILU(0)) linear solver of the
BDF integrator, and thread-count determinism.

The flux scheme is checked for its design order on smooth data (linear
advection, Burgers before the shock), for discrete conservation to rounding
(periodic Burgers through a shock; the Sod and Lax shock tubes, whose
totals change only by the boundary fluxes), and against the exact Riemann
solution of both shock tubes.
"""

from __future__ import annotations

import math

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


def periodic(P, eq_of, ic):
    x, t = P.symbol("x"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, t)
    two_pi = 2 * P.pi()
    pr = (
        akx.PDE(eq_of(U, x, t), U)
        .on(akx.Interval(x, 0, two_pi))
        .bc(u(0, t), u(two_pi, t))
        .ic(u(x, 0), ic(x))
    )
    return x, u, pr


# ---------------------------------------------------------------------------
# Design order on smooth data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("variant", "lo"), [("js", 4.4), ("z", 4.6)])
def test_linear_advection_design_order(P, variant, lo):
    x, _, pr = periodic(P, lambda U, x, t: akx.D(U, t) + akx.D(U, x), ak.sin)
    errs = []
    for n in (20, 40, 80, 160):
        ms = akx.method_of_lines(pr, {x: n}, scheme="weno5_fv", weno_variant=variant)
        res = ms.integrate(0.5, method="rk45", rtol=1e-12, atol=1e-14)
        errs.append(err_inf(res.field("u"), np.sin(ms.grid["x"] - 0.5)))
    o = orders(errs)
    assert o[-1] > lo, (errs, o)


def burgers_exact(xs, t):
    """``u = 1/2 + sin(x)/2`` advected by Burgers before the shock (t < 2)."""
    xi = xs.copy()
    for _ in range(60):
        g = xi + t * (0.5 + 0.5 * np.sin(xi)) - xs
        xi -= g / (1 + 0.5 * t * np.cos(xi))
    return 0.5 + 0.5 * np.sin(xi)


def test_burgers_auto_flux_design_order_before_the_shock(P):
    half = P.rational(1, 2)
    x, _, pr = periodic(
        P,
        lambda U, x, t: akx.D(U, t) + U * akx.D(U, x),
        lambda x: half + half * ak.sin(x),
    )
    errs = []
    for n in (40, 80, 160):
        # flux= omitted: F = u^2/2 is found from u u_x and checked.
        ms = akx.method_of_lines(pr, {x: n}, scheme="weno5_fv", weno_variant="z")
        res = ms.integrate(0.5, method="rk45", rtol=1e-12, atol=1e-14)
        errs.append(err_inf(res.field("u"), burgers_exact(ms.grid["x"], 0.5)))
    o = orders(errs)
    assert o[-1] > 4.5, (errs, o)


# ---------------------------------------------------------------------------
# Conservation and shocks
# ---------------------------------------------------------------------------


def test_periodic_burgers_conserves_mass_through_a_shock(P):
    half = P.rational(1, 2)
    x, u, pr = periodic(
        P,
        lambda U, x, t: akx.D(U, t) + U * akx.D(U, x),
        lambda x: half + ak.sin(x),
    )
    U = u(x, P.symbol("t"))
    ms = akx.method_of_lines(pr, {x: 200}, scheme="weno5_fv", flux={U: U**2 / 2})
    res = ms.integrate(3.0, method="rk45", rtol=1e-7, atol=1e-9)  # shock from t = 1
    v = res.field("u")
    m0 = ms.field(ms.y0).sum()
    assert abs(v.sum() - m0) <= 1e-12 * 200, abs(v.sum() - m0)
    # Essentially non-oscillatory: no new extrema beyond a small overshoot.
    assert v.max() < 1.5 + 2e-2, v.max()
    assert v.min() > -0.5 - 2e-2, v.min()
    # The shock is captured sharply: one cell carries most of the jump
    # (a smooth profile would change by ~0.03 per cell).
    assert np.max(np.abs(np.diff(v))) > 0.5


def test_global_lax_friedrichs_splitting(P):
    half = P.rational(1, 2)
    x, _u, pr = periodic(
        P,
        lambda U, x, t: akx.D(U, t) + U * akx.D(U, x),
        lambda x: half + half * ak.sin(x),
    )
    ms = akx.method_of_lines(
        pr, {x: 80}, scheme="weno5_fv", flux_splitting="lax_friedrichs", wave_speed=1
    )
    res = ms.integrate(0.5, method="rk45", rtol=1e-11, atol=1e-13)
    assert err_inf(res.field("u"), burgers_exact(ms.grid["x"], 0.5)) < 1e-4
    with pytest.raises(NumericPdeError) as e:
        akx.method_of_lines(pr, {x: 80}, scheme="weno5_fv", flux_splitting="lax_friedrichs")
    assert e.value.code == "E-PDE-373"


def riemann_exact(xs, t, L, R, g=1.4, x0=0.5):
    """Exact solution of the 1-D Euler Riemann problem (Toro, ch. 4):
    ``(rho, u, p)`` at every ``x``."""
    rl, ul, pl = L
    rr, ur, pr = R
    cl, cr = math.sqrt(g * pl / rl), math.sqrt(g * pr / rr)

    def fk(p, rk, pk, ck):
        if p > pk:
            a, b = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * math.sqrt(a / (p + b))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

    lo, hi = 1e-12, 100 * max(pl, pr)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if fk(mid, rl, pl, cl) + fk(mid, rr, pr, cr) + ur - ul > 0:
            hi = mid
        else:
            lo = mid
    ps = 0.5 * (lo + hi)
    us = 0.5 * (ul + ur) + 0.5 * (fk(ps, rr, pr, cr) - fk(ps, rl, pl, cl))
    out = np.zeros((3, len(xs)))
    for i, xv in enumerate(xs):
        s = (xv - x0) / t
        # Sample on the left of the contact, or mirror the right side.
        side = s <= us
        rk, uk, pk, ck = (rl, ul, pl, cl) if side else (rr, -ur, pr, cr)
        sk, usk = (s, us) if side else (-s, -us)
        if ps > pk:
            sh = uk - ck * math.sqrt((g + 1) / (2 * g) * ps / pk + (g - 1) / (2 * g))
            if sk <= sh:
                w = (rk, uk, pk)
            else:
                q = ps / pk
                w = (rk * (q + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * q + 1), usk, ps)
        else:
            head = uk - ck
            tail = usk - ck * (ps / pk) ** ((g - 1) / (2 * g))
            if sk <= head:
                w = (rk, uk, pk)
            elif sk > tail:
                w = (rk * (ps / pk) ** (1 / g), usk, ps)
            else:
                uu = 2 / (g + 1) * (ck + (g - 1) / 2 * uk + sk)
                c = 2 / (g + 1) * (ck + (g - 1) / 2 * (uk - sk))
                w = (rk * (c / ck) ** (2 / (g - 1)), uu, pk * (c / ck) ** (2 * g / (g - 1)))
        out[:, i] = (w[0], w[1] if side else -w[1], w[2])
    return out


def shock_tube(P, L, R):
    x, t = P.symbol("x"), P.symbol("t")
    rho, mom, ene = akx.Function("rho"), akx.Function("m"), akx.Function("E")
    r, m, E = rho(x, t), mom(x, t), ene(x, t)
    g = P.rational(7, 5)
    p = (g - 1) * (E - m**2 / (2 * r))
    F = [m, m**2 / r + p, (E + p) * m / r]
    eqs = [akx.D(U, t) + ak.diff(f, x).value for U, f in zip((r, m, E), F)]
    step = (1 + ak.tanh((x - P.rational(1, 2)) * 10**4)) / 2

    def ic(a, b):
        return P.float(a) + P.float(b - a) * step

    EL = L[2] / 0.4 + 0.5 * L[0] * L[1] ** 2
    ER = R[2] / 0.4 + 0.5 * R[0] * R[1] ** 2
    # No boundary conditions: the outer interfaces carry the physical flux
    # (transmissive), which the waves do not reach before the final time.
    pr = (
        akx.PDE(eqs, [r, m, E])
        .on(akx.Interval(x, 0, 1))
        .ic(rho(x, 0), ic(L[0], R[0]))
        .ic(mom(x, 0), ic(L[0] * L[1], R[0] * R[1]))
        .ic(ene(x, 0), ic(EL, ER))
    )
    speed = ak.abs(m / r) + ak.sqrt(g * p / r)
    return x, pr, {r: F[0], m: F[1], E: F[2]}, speed


def flux_of(w):
    r, u, p = w
    E = p / 0.4 + 0.5 * r * u**2
    return np.array([r * u, r * u**2 + p, (E + p) * u])


@pytest.mark.parametrize(
    ("name", "L", "R", "T", "tol", "ctol"),
    [
        ("sod", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2, 0.006, 1e-12),
        # The Lax rarefaction head comes within 13% of the left boundary:
        # the boundary cells move by ~1e-11 (numerical diffusion tails), so
        # the constant-rate identity holds to that, not to rounding.
        ("lax", (0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 0.14, 0.02, 1e-9),
    ],
)
def test_shock_tubes_against_the_exact_riemann_solution(P, name, L, R, T, tol, ctol):
    x, pr, flux, speed = shock_tube(P, L, R)
    n = 200
    ms = akx.method_of_lines(
        pr, {x: n}, scheme="weno5_fv", flux=flux, wave_speed=speed, jacobian=False
    )
    res = ms.integrate(T, method="rk45", rtol=1e-6, atol=1e-8)
    xs = ms.grid["x"]
    h = xs[1] - xs[0]
    ex = riemann_exact(xs, T, L, R)
    rho = res.field("rho")
    l1 = float(np.sum(np.abs(rho - ex[0])) * h)
    assert l1 < tol, (name, l1)
    assert rho.min() > 0
    # Discrete conservation: d/dt Σ h U = F(U_0) - F(U_{n-1}), constant while
    # the boundary states are untouched — exact up to rounding.
    tot0 = np.array([np.sum(ms.field(ms.y0, f)) * h for f in ("rho", "m", "E")])
    tot1 = np.array([np.sum(res.field(f)) * h for f in ("rho", "m", "E")])
    pred = tot0 + T * (flux_of(L) - flux_of(R))
    assert np.max(np.abs(tot1 - pred)) < ctol, tot1 - pred


def test_viscous_burgers_flux_plus_diffusion_with_bdf_and_exact_jacobian(P):
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
    ms = akx.method_of_lines(pr, {x: 81}, scheme="weno5_fv", order=4)
    y = ms.y0 + 0.01 * np.sin(np.arange(ms.n_states))
    J = ms.jacobian(0.0, y, format="dense")
    f0 = ms.rhs(0.0, y)
    eps = 1e-7
    fd = np.empty_like(J)
    for j in range(ms.n_states):
        yp = y.copy()
        yp[j] += eps
        fd[:, j] = (ms.rhs(0.0, yp) - f0) / eps
    assert np.max(np.abs(J - fd)) < 1e-4 * max(1.0, np.max(np.abs(J)))
    res = ms.integrate(1.0, rtol=1e-10, atol=1e-12)
    e = np.exp(-(np.pi**2) * 0.05)
    xs = ms.grid["x"]
    exact = 2 * 0.05 * np.pi * e * np.sin(np.pi * xs) / (a + e * np.cos(np.pi * xs))
    assert err_inf(res.field("u"), exact) < 1e-5


def test_2d_periodic_advection_conserves_and_converges(P):
    x, y, t = P.symbol("x"), P.symbol("y"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, y, t)
    two_pi = 2 * P.pi()
    pr = (
        akx.PDE(akx.D(U, t) + akx.D(U, x) + akx.D(U, y), U)
        .on(akx.Box([x, y], [(0, two_pi), (0, two_pi)]))
        .bc(u(0, y, t), u(two_pi, y, t))
        .bc(u(x, 0, t), u(x, two_pi, t))
        .ic(u(x, y, 0), ak.sin(x + y))
    )
    errs = []
    for n in (16, 32):
        ms = akx.method_of_lines(pr, n, scheme="weno5_fv", flux={U: {x: U, y: U}}, weno_variant="z")
        res = ms.integrate(0.5, method="rk45", rtol=1e-11, atol=1e-13)
        X, Y = np.meshgrid(ms.grid["x"], ms.grid["y"], indexing="ij")
        errs.append(err_inf(res.field("u"), np.sin(X + Y - 1.0)))
        assert abs(res.field("u").sum() - ms.field(ms.y0).sum()) < 1e-11
    assert orders(errs)[-1] > 4.0, errs


def test_flux_refusals(P):
    x, u, pr = periodic(P, lambda U, x, t: akx.D(U, t) + U * akx.D(U, x), ak.sin)
    U = u(x, P.symbol("t"))
    cases = [
        # The flux does not account for the advective term.
        {"flux": {U: U}},
        # A flux with a derivative in it.
        {"flux": {U: akx.D(U, x)}},
        # Not an unknown.
        {"flux": {"v": U}},
    ]
    for kw in cases:
        with pytest.raises(NumericPdeError) as e:
            akx.method_of_lines(pr, {x: 40}, scheme="weno5_fv", **kw)
        assert e.value.code == "E-PDE-373", kw
    with pytest.raises(NumericPdeError) as e:
        akx.method_of_lines(pr, {x: 40}, scheme="central", flux={U: U**2 / 2})
    assert e.value.code == "E-PDE-365"
    with pytest.raises(NumericPdeError) as e:
        akx.method_of_lines(pr, {x: 40}, scheme="weno5_fv", weno_variant="m")
    assert e.value.code == "E-PDE-365"
    ms = akx.method_of_lines(pr, {x: 40}, scheme="weno5_fv")
    assert "skipped" in ms.stability_report()["u"]


# ---------------------------------------------------------------------------
# Krylov linear solver and threads
# ---------------------------------------------------------------------------


def heat2d(P):
    x, y, t = P.symbol("x"), P.symbol("y"), P.symbol("t")
    u = akx.Function("u")
    U = u(x, y, t)
    return (
        akx.PDE(akx.D(U, t) - akx.D(U, x, 2) - akx.D(U, y, 2) + U**3, U)
        .on(akx.Box([x, y], [(0, 1), (0, 1)]))
        .bc(u(0, y, t), 0)
        .bc(u(1, y, t), 0)
        .bc(u(x, 0, t), 0)
        .bc(u(x, 1, t), 0)
        .ic(u(x, y, 0), ak.sin(P.pi() * x) * ak.sin(P.pi() * y))
    )


def test_krylov_bdf_agrees_with_the_direct_solver(P):
    ms = akx.method_of_lines(heat2d(P), 41)
    d = ms.integrate(0.05, rtol=1e-9, atol=1e-11, linear_solver="direct")
    k = ms.integrate(0.05, rtol=1e-9, atol=1e-11, linear_solver="krylov")
    assert d.stats["linear_solver"] == "direct"
    assert d.stats["krylov_iters"] == 0
    assert k.stats["linear_solver"] == "krylov"
    assert k.stats["krylov_iters"] > 0
    assert err_inf(d.y[-1], k.y[-1]) < 1e-7
    a = ms.integrate(0.05, rtol=1e-9, atol=1e-11)
    assert a.stats["linear_solver"] == "direct"  # 39² states: the band is cheap
    with pytest.raises(NumericPdeError) as e:
        ms.integrate(0.05, linear_solver="cholesky")
    assert e.value.code == "E-PDE-372"


def test_auto_chooses_krylov_on_a_large_3d_grid(P):
    x, y, z, t = (P.symbol(s) for s in "xyzt")
    u = akx.Function("u")
    V = u(x, y, z, t)
    pr = (
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
    ms = akx.method_of_lines(pr, 25)  # 23³ states, band ≈ 23²
    res = ms.integrate(0.02, rtol=1e-8, atol=1e-10)
    assert res.stats["linear_solver"] == "krylov", res.stats
    X, Y, Z = np.meshgrid(*(ms.grid[k] for k in "xyz"), indexing="ij")
    exact = np.exp(-3 * np.pi**2 * 0.02) * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    assert err_inf(res.field("u"), exact) < 2e-3


def test_results_are_bitwise_identical_across_thread_counts(P):
    ms = akx.method_of_lines(heat2d(P), 131)  # 129² states: chunked kernels
    runs = [
        ms.integrate(2e-3, rtol=1e-7, atol=1e-9, linear_solver="krylov", threads=k) for k in (1, 4)
    ]
    assert np.array_equal(runs[0].y, runs[1].y)
    assert runs[0].stats["krylov_iters"] == runs[1].stats["krylov_iters"]
    with pytest.raises(NumericPdeError):
        ms.integrate(1e-3, threads=0)
