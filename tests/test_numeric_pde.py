"""PDE milestone M7a: exact finite-difference weights, stencils and their
truncation errors, spectral differentiation matrices, and scheme analysis
(von Neumann stability, modified equation, consistency order).

Reference values are textbook ones: Fornberg, *Math. Comp.* 51 (1988),
Table 1 (centred) and Table 3 (one-sided); LeVeque, *Finite Difference
Methods for ODEs and PDEs* (2007), §1.5, §8–§10; Strikwerda, *Finite
Difference Schemes and PDEs* (2004), §2–§4, §8.2; Warming & Hyett,
*J. Comput. Phys.* 14 (1974); Trefethen, *Spectral Methods in MATLAB*
(2000), ch. 3 and 6.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import alkahest as ak
import numpy as np
import pytest
from alkahest.exceptions import NumericPdeError
from alkahest.experimental import numeric_pde as N


@pytest.fixture
def pool():
    return ak.ExprPool()


def _zero(e):
    n = ak.cancel(e).node()
    return n[0] == "integer" and n[1] == "0"


def _eq(a, b):
    return _zero(a - b)


# ---------------------------------------------------------------------------
# 1-D Fornberg weights
# ---------------------------------------------------------------------------

# Fornberg (1988) Table 1: centred first and second derivatives.
CENTRAL = {
    (1, 2): [F(-1, 2), 0, F(1, 2)],
    (1, 4): [F(1, 12), F(-2, 3), 0, F(2, 3), F(-1, 12)],
    (1, 6): [F(-1, 60), F(3, 20), F(-3, 4), 0, F(3, 4), F(-3, 20), F(1, 60)],
    (1, 8): [F(1, 280), F(-4, 105), F(1, 5), F(-4, 5), 0, F(4, 5), F(-1, 5), F(4, 105), F(-1, 280)],
    (2, 2): [1, -2, 1],
    (2, 4): [F(-1, 12), F(4, 3), F(-5, 2), F(4, 3), F(-1, 12)],
    (2, 6): [F(1, 90), F(-3, 20), F(3, 2), F(-49, 18), F(3, 2), F(-3, 20), F(1, 90)],
    (3, 2): [F(-1, 2), 1, 0, -1, F(1, 2)],
    (4, 2): [1, -4, 6, -4, 1],
}

# Fornberg (1988) Table 3: one-sided first derivative.
FORWARD = {
    1: [-1, 1],
    2: [F(-3, 2), 2, F(-1, 2)],
    3: [F(-11, 6), 3, F(-3, 2), F(1, 3)],
    4: [F(-25, 12), 4, -3, F(4, 3), F(-1, 4)],
    5: [F(-137, 60), 5, -5, F(10, 3), F(-5, 4), F(1, 5)],
}


@pytest.mark.parametrize("key", sorted(CENTRAL))
def test_fornberg_centred_table(key):
    m, p = key
    st = N.stencil(m, accuracy=p)
    assert list(st.weights) == [F(w) for w in CENTRAL[key]]
    assert st.error_order == p


@pytest.mark.parametrize("p", sorted(FORWARD))
def test_fornberg_one_sided_table(p):
    st = N.stencil(1, accuracy=p, kind="forward")
    assert list(st.weights) == [F(w) for w in FORWARD[p]]
    assert st.error_order == p
    back = N.stencil(1, accuracy=p, kind="backward")
    assert list(back.weights) == [-F(w) for w in FORWARD[p]][::-1]


def test_central_second_difference_error_is_minus_h2_over_12(pool):
    h = pool.symbol("h")
    x = pool.symbol("x")
    u = pool.func("u", [x])
    st = N.fd_weights(2, [-1, 0, 1], h=h)
    assert st.weights == (1, -2, 1)
    assert (st.error_order, st.error_constant) == (2, F(-1, 12))
    d4 = u
    for _ in range(4):
        d4 = ak.diff(d4, x).value
    assert _eq(st.truncation_error(u, x), pool.rational(-1, 12) * h**2 * d4)
    assert st.leading_error() == (F(-1, 12), 2, 4)


@pytest.mark.parametrize(
    ("m", "offsets", "p", "C"),
    [
        (1, [-2, -1, 0, 1, 2], 4, F(1, 30)),
        (2, [-2, -1, 0, 1, 2], 4, F(1, 90)),
        (1, [0, 1, 2], 2, F(1, 3)),
        (1, [-1, 1], 2, F(-1, 6)),
        (1, [0, 1], 1, F(-1, 2)),
        (4, [-2, -1, 0, 1, 2], 2, F(-1, 6)),
        (1, [F(-1, 2), F(1, 2)], 2, F(-1, 24)),  # staggered
        (0, [F(-1, 2), F(1, 2)], 2, F(-1, 8)),  # midpoint interpolation
    ],
)
def test_leading_error_constants(m, offsets, p, C):
    st = N.fd_weights(m, offsets)
    assert (st.error_order, st.error_constant) == (p, C)


@pytest.mark.parametrize(
    ("m", "offsets"),
    [
        (2, [-1, 0, 1]),
        (1, [0, 1, 2]),
        (1, [F(-1, 2), F(1, 2)]),
        (3, [-2, -1, 0, 1, 2]),
        (2, [0, 1, 3, 4]),
    ],
)
def test_error_term_sign_matches_numerics(m, offsets):
    """exact − approx ≈ C h^p f^{(m+p)} for f = exp(2x) at x = 0.3."""
    st = N.fd_weights(m, offsets)
    x0, k = 0.3, 2.0

    def deriv(n):
        return k**n * math.exp(k * x0)

    for h in (1e-2, 5e-3):
        approx = (
            sum(
                float(w) * math.exp(k * (x0 + float(s) * h)) for s, w in zip(st.offsets, st.weights)
            )
            / h**m
        )
        err = deriv(m) - approx
        pred = float(st.error_constant) * h**st.error_order * deriv(m + st.error_order)
        assert abs(err - pred) < 0.05 * abs(pred)


def test_staggered_fourth_order_first_derivative():
    st = N.fd_weights(1, [F(-3, 2), F(-1, 2), F(1, 2), F(3, 2)])
    assert list(st.weights) == [F(1, 24), F(-9, 8), F(9, 8), F(-1, 24)]
    assert (st.error_order, st.error_constant) == (4, F(3, 640))


def test_interpolation_at_a_node_is_exact():
    st = N.fd_weights(0, [0, 1, 2])
    assert st.weights == (1, 0, 0)
    assert st.error_order is None


def test_symbolic_nonuniform_grid(pool):
    s = pool.symbol("s")
    st = N.fd_weights(2, [-1, 0, s])
    w = st.weights
    assert _eq(w[0], 2 / (1 + s))
    assert _eq(w[1], pool.integer(-2) / s)
    assert _eq(w[2], 2 / (s * (1 + s)))
    assert st.error_order == 1
    assert _eq(st.error_constant, (1 - s) / 3)


def test_off_grid_evaluation_point():
    st = N.fd_weights(1, [0, 1, 2, 3], x0=F(1, 2))
    assert list(st.weights) == [F(-23, 24), F(7, 8), F(1, 8), F(-1, 24)]


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ((2, [0, 0, 1]), "E-PDE-350"),
        ((3, [0, 1, 2]), "E-PDE-350"),
        ((-1, [0, 1]), "E-PDE-350"),
        ((1, [0.0, 1.0]), "E-PDE-351"),
    ],
)
def test_fd_weights_refusals(args, code):
    with pytest.raises(NumericPdeError) as ei:
        N.fd_weights(*args)
    assert ei.value.code == code


def test_apply_and_kernel_1d(pool):
    j = pool.symbol("j")
    st = N.fd_weights(2, [-1, 0, 1], h=pool.symbol("h"))
    e = st.apply("v", j)
    want = (
        pool.func("v", [j - 1]) - 2 * pool.func("v", [j]) + pool.func("v", [j + 1])
    ) * pool.symbol("h") ** -2
    assert _eq(ak.simplify(e).value, ak.simplify(want).value)
    k = st.kernel()
    xs = np.linspace(0.0, 1.0, 21)
    out = k(xs**2, xs[1] - xs[0])
    assert out.shape == (19,)
    assert np.allclose(out, 2.0)
    src = st.emit_c("d2")
    assert "double d2(" in src


# ---------------------------------------------------------------------------
# multi-dimensional stencils
# ---------------------------------------------------------------------------


def test_laplacian_5_point():
    st = N.laplacian_stencil(2, 5)
    assert st.error_order == 2
    assert st.error_terms == {(4, 0): F(-1, 12), (0, 4): F(-1, 12)}


@pytest.mark.parametrize(("dim", "pts"), [(2, 9), (3, 19), (3, 27)])
def test_compact_laplacians_have_biharmonic_error(dim, pts):
    """Leading error −h²/12 Δ²u: the Mehrstellen property that makes
    ``L_h u = f + h²/12 Δf`` fourth-order accurate for Poisson."""
    st = N.laplacian_stencil(dim, pts)
    want = {}
    for i in range(dim):
        for j in range(dim):
            a = [0] * dim
            a[i] += 2
            a[j] += 2
            want[tuple(a)] = want.get(tuple(a), 0) + F(-1, 12)
    assert st.error_order == 2
    assert st.error_terms == want
    assert sum(st.weights.values()) == 0


def test_laplacian_7_point():
    st = N.laplacian_stencil(3, 7)
    assert st.error_terms == {(4, 0, 0): F(-1, 12), (0, 4, 0): F(-1, 12), (0, 0, 4): F(-1, 12)}


def test_mehrstellen_9pt_weights():
    st = N.laplacian_stencil(2, 9)
    assert st.weights[(0, 0)] == F(-20, 6)
    assert st.weights[(1, 0)] == F(4, 6)
    assert st.weights[(1, 1)] == F(1, 6)


def test_tensor_product_mixed_derivative(pool):
    st = N.stencil((1, 1))
    assert st.weights == {(-1, -1): F(1, 4), (-1, 1): F(-1, 4), (1, -1): F(-1, 4), (1, 1): F(1, 4)}
    assert st.error_terms == {(3, 1): F(-1, 6), (1, 3): F(-1, 6)}
    x, y, h = pool.symbol("x"), pool.symbol("y"), pool.symbol("h")
    st2 = N.stencil((1, 1), h=h)
    te = st2.truncation_error(pool.func("w", [x, y]))
    assert "Derivative(w(x, y)" in str(te)


def test_laplacian_kernel_is_exact_on_quadratics():
    k = N.laplacian_stencil(2, 9).kernel()
    x = np.linspace(0, 1, 9)
    X, Y = np.meshgrid(x, x, indexing="ij")
    out = k(3 * X**2 - Y**2 + X * Y, x[1] - x[0])
    assert out.shape == (7, 7)
    assert np.allclose(out, 4.0)


def test_unknown_laplacian_refused():
    with pytest.raises(NumericPdeError) as ei:
        N.laplacian_stencil(2, 7)
    assert ei.value.code == "E-PDE-350"


# ---------------------------------------------------------------------------
# spectral matrices
# ---------------------------------------------------------------------------


def _f(e):
    return float(ak.eval_expr(e, {}))


def test_cheb_small_exact_matrices():
    D, x = N.cheb_diff_matrix(1)
    assert [[str(e) for e in r] for r in D.to_list()] == [["1/2", "-1/2"], ["1/2", "-1/2"]]
    D, x = N.cheb_diff_matrix(2)
    want = [["3/2", "-2", "1/2"], ["1/2", "0", "-1/2"], ["-1/2", "2", "-3/2"]]
    assert [[str(e) for e in r] for r in D.to_list()] == want
    D, x = N.cheb_diff_matrix(3)
    assert [str(v) for v in x] == ["1", "1/2", "-1/2", "-1"]
    assert str(D.to_list()[0][0]) == "19/6"


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_cheb_exact_matches_numeric(n):
    D, x = N.cheb_diff_matrix(n)
    Dn, xn = N.cheb_diff_matrix(n, numeric=True)
    De = np.array([[_f(e) for e in r] for r in D.to_list()])
    assert np.allclose(De, Dn, atol=1e-12)
    assert np.allclose([_f(v) for v in x], xn)


def test_cheb_differentiates_polynomials_exactly():
    D, x = N.cheb_diff_matrix(12, numeric=True)
    f = x**7 - 3 * x**4 + x
    assert np.allclose(D @ f, 7 * x**6 - 12 * x**3 + 1, atol=1e-10)
    D, x = N.cheb_diff_matrix(24, numeric=True)
    assert np.allclose(D @ np.exp(x), np.exp(x), atol=1e-10)


def test_cheb_exact_rows_annihilate_constants():
    D, _ = N.cheb_diff_matrix(4)
    for row in D.to_list():
        acc = row[0]
        for e in row[1:]:
            acc = acc + e
        assert abs(_f(acc)) < 1e-14


@pytest.mark.parametrize("n", [4, 5, 6, 8, 9])
@pytest.mark.parametrize("order", [1, 2])
def test_fourier_exact_matches_numeric(n, order):
    D, _x = N.fourier_diff_matrix(n, order)
    Dn, _xn = N.fourier_diff_matrix(n, order, numeric=True)
    De = np.array([[_f(e) for e in r] for r in D.to_list()])
    assert np.allclose(De, Dn, atol=1e-12)


@pytest.mark.parametrize("n", [16, 17])
def test_fourier_differentiates_trig_polynomials(n):
    D, x = N.fourier_diff_matrix(n, numeric=True)
    D2, _ = N.fourier_diff_matrix(n, 2, numeric=True)
    f = np.sin(3 * x) + np.cos(5 * x)
    assert np.allclose(D @ f, 3 * np.cos(3 * x) - 5 * np.sin(5 * x), atol=1e-10)
    assert np.allclose(D2 @ f, -9 * np.sin(3 * x) - 25 * np.cos(5 * x), atol=1e-9)


def test_fourier_exact_n4():
    D, _ = N.fourier_diff_matrix(4)
    assert [str(e) for e in D.to_list()[0]] == ["0", "1/2", "0", "-1/2"]
    D2, _ = N.fourier_diff_matrix(4, 2)
    assert [str(e) for e in D2.to_list()[0]] == ["-3/2", "1", "-1/2", "1"]


def test_spectral_refusals():
    with pytest.raises(NumericPdeError):
        N.cheb_diff_matrix(0)
    with pytest.raises(NumericPdeError):
        N.fourier_diff_matrix(4, order=3)


# ---------------------------------------------------------------------------
# von Neumann analysis
# ---------------------------------------------------------------------------


def _schemes(p):
    r, c = p.symbol("r"), p.symbol("c")
    h = p.rational(1, 2)
    return (
        r,
        c,
        {
            "ftcs_heat": {(1, 0): 1, (0, 0): -1 + 2 * r, (0, 1): -r, (0, -1): -r},
            "btcs_heat": {(1, 0): 1 + 2 * r, (1, 1): -r, (1, -1): -r, (0, 0): -1},
            "cn_heat": {
                (1, 0): 1 + r,
                (1, 1): -r * h,
                (1, -1): -r * h,
                (0, 0): -1 + r,
                (0, 1): -r * h,
                (0, -1): -r * h,
            },
            "upwind": {(1, 0): 1, (0, 0): -1 + c, (0, -1): -c},
            "lax_friedrichs": {(1, 0): 1, (0, 1): -h + c * h, (0, -1): -h - c * h},
            "ftcs_advection": {(1, 0): 1, (0, 1): c * h, (0, -1): -c * h, (0, 0): -1},
            "lax_wendroff": {
                (1, 0): 1,
                (0, 0): -1 + c**2,
                (0, 1): c * h - c**2 * h,
                (0, -1): -c * h - c**2 * h,
            },
            "leapfrog_advection": {(2, 0): 1, (1, 1): c, (1, -1): -c, (0, 0): -1},
            "leapfrog_wave": {
                (2, 0): 1,
                (1, 0): -2 + 2 * c**2,
                (1, 1): -(c**2),
                (1, -1): -(c**2),
                (0, 0): 1,
            },
        },
    )


EXPECTED = {
    # name: (lo, hi, lo_closed, hi_closed) as one interval
    "ftcs_heat": [(0, F(1, 2), True, True)],
    "btcs_heat": [(0, None, True, False)],
    "cn_heat": [(0, None, True, False)],
    "upwind": [(0, 1, True, True)],
    "lax_friedrichs": [(-1, 1, True, True)],
    "ftcs_advection": [(0, 0, True, True)],
    "lax_wendroff": [(-1, 1, True, True)],
    "leapfrog_advection": [(-1, 1, False, False)],  # strict: |c| = 1 gives a double root on |g| = 1
    "leapfrog_wave": [(-1, 1, True, True)],  # second order in time: double roots allowed
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_von_neumann_textbook_conditions(pool, name):
    _, _, schemes = _schemes(pool)
    res = N.von_neumann(N.Scheme(schemes[name]))
    assert res["proof_status"] == "proved"
    got = [(lo, hi, a, b) for lo, hi, a, b in res["intervals"]]
    assert got == [
        tuple(F(v) if v is not None else None for v in t[:2]) + t[2:] for t in EXPECTED[name]
    ]
    assert res["stability_condition"] is not None


def test_von_neumann_harder_schemes(pool):
    r, c = pool.symbol("r"), pool.symbol("c")
    q, h = pool.rational, pool.rational(1, 2)
    # fourth-order-in-space FTCS for the heat equation: r ≤ 3/8
    s4 = {
        (1, 0): 1,
        (0, 0): -1 + r * q(5, 2),
        (0, 1): -r * q(4, 3),
        (0, -1): -r * q(4, 3),
        (0, 2): r * q(1, 12),
        (0, -2): r * q(1, 12),
    }
    assert N.von_neumann(N.Scheme(s4)).intervals == [(F(0), F(3, 8), True, True)]
    # Beam–Warming: 0 ≤ c ≤ 2
    bw = {
        (1, 0): 1,
        (0, 0): -1 + c * q(3, 2) - c**2 * h,
        (0, -1): -2 * c + c**2,
        (0, -2): c * h - c**2 * h,
    }
    assert N.von_neumann(N.Scheme(bw)).intervals == [(F(0), F(2), True, True)]
    # DuFort–Frankel: unconditionally stable; Richardson: unconditionally unstable
    df = {(2, 0): 1 + 2 * r, (1, 1): -2 * r, (1, -1): -2 * r, (0, 0): -1 + 2 * r}
    assert N.von_neumann(N.Scheme(df)).intervals == [(F(0), None, True, False)]
    ri = {(2, 0): 1, (1, 0): 4 * r, (1, 1): -2 * r, (1, -1): -2 * r, (0, 0): -1}
    assert N.von_neumann(N.Scheme(ri)).intervals == [(F(0), F(0), True, True)]


def test_von_neumann_irrational_boundary_is_decided_exactly(pool):
    """|G| ≤ 1 iff c² ≤ 1/2: the endpoints ±1/√2 are irrational, decided in Q(√2)."""
    c = pool.symbol("c")
    res = N.von_neumann(
        N.Scheme({(1, 0): 1, (0, 0): -1 + 2 * c**2, (0, 1): -(c**2), (0, -1): -(c**2)})
    )
    assert res.proof_status == "proved"
    assert not res.undecided_points
    ((lo, hi, a, b),) = res.intervals
    assert a
    assert b
    assert abs(_f(lo) + 2**-0.5) < 1e-12
    assert abs(_f(hi) - 2**-0.5) < 1e-12
    assert "RootOf" in str(res.stability_condition)


def test_von_neumann_condition_expressions(pool):
    r, _c, schemes = _schemes(pool)
    res = N.von_neumann(N.Scheme(schemes["ftcs_heat"]))
    assert str(res.stability_condition) == "((r ≥ 0) ∧ (r ≤ 1/2))"
    assert res.parameter == r
    res = N.von_neumann(N.Scheme(schemes["upwind"]))
    assert str(res.stability_condition) == "((c ≥ 0) ∧ (c ≤ 1))"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_von_neumann_agrees_with_sampling(pool, name):
    """A float cross-check of the *proved* intervals (never used to decide)."""
    r, c, schemes = _schemes(pool)
    sym = r if name.endswith("heat") else c
    res = N.von_neumann(N.Scheme(schemes[name]))
    thetas = np.linspace(-np.pi, np.pi, 721)

    def max_root(val):
        rows = {}
        for (lv, k), e in schemes[name].items():
            v = _f(ak.subs(e, {sym: val})) if hasattr(e, "node") else float(e)
            rows.setdefault(lv, []).append((k, v))
        L = max(rows)
        worst = 0.0
        for th in thetas:
            coeffs = [
                sum(v * np.exp(1j * k * th) for k, v in rows.get(lv, [])) for lv in range(L, -1, -1)
            ]
            worst = max(worst, max(abs(np.roots(coeffs))))
        return worst

    def inside(v):
        for lo, hi, a, b in res.intervals:
            if (lo is None or v > lo or (a and v == lo)) and (
                hi is None or v < hi or (b and v == hi)
            ):
                return True
        return False

    for v in (-1.5, -0.7, -0.2, 0.2, 0.45, 0.55, 0.9, 1.2, 3.0):
        stable = max_root(v) <= 1 + 1e-9
        assert stable == inside(v), (name, v)


def test_von_neumann_physical_symbols_merge_to_one_parameter(pool):
    n, j = pool.symbol("n"), pool.symbol("j")
    dt, dx, nu = pool.symbol("dt"), pool.symbol("dx"), pool.symbol("nu")

    def u(lv, k):
        return pool.func("u", [n + lv, j + k])

    e = (u(1, 0) - u(0, 0)) / dt - nu * (u(0, 1) - 2 * u(0, 0) + u(0, -1)) / dx**2
    S = N.Scheme.from_expr(e, "u", n, j, dt=dt, dx=dx)
    res = N.von_neumann(S)
    assert res.proof_status == "proved"
    assert _eq(res.parameter, nu * dt / dx**2)
    assert res.intervals == [(F(0), F(1, 2), True, True)]
    r = pool.symbol("r")
    res2 = N.von_neumann(S, subs={dt: r * dx**2 / nu})
    assert res2.parameter == r
    assert res2.intervals == [(F(0), F(1, 2), True, True)]


def test_von_neumann_parameter_free(pool):
    h = pool.rational(1, 4)
    ok = N.von_neumann(N.Scheme({(1, 0): 1, (0, 0): -1 + 2 * h, (0, 1): -h, (0, -1): -h}))
    assert ok.stable is True
    assert ok.proof_status == "proved"
    bad = N.von_neumann(N.Scheme({(1, 0): 1, (0, 0): -1 + 2, (0, 1): -1, (0, -1): -1}))
    assert bad.stable is False
    assert str(bad.stability_condition) == "False"


def test_von_neumann_amplification_factor(pool):
    r, _, schemes = _schemes(pool)
    res = N.von_neumann(N.Scheme(schemes["ftcs_heat"]))
    th = res.theta
    for rv, tv in [(0.3, 0.7), (0.5, 2.0)]:
        G = _f(ak.subs(res.G, {r: rv, th: tv}))
        assert abs(G - (1 - 4 * rv * math.sin(tv / 2) ** 2)) < 1e-12
        assert abs(_f(ak.subs(res.abs_G_squared, {r: rv, th: tv})) - G * G) < 1e-12
    assert res["amplification_polynomial"] is not None


def test_von_neumann_two_parameters_is_unknown(pool):
    c, d = pool.symbol("c"), pool.symbol("d")
    # upwind advection plus FTCS diffusion: two independent numbers
    S = N.Scheme({(1, 0): 1, (0, 0): -1 + c + 2 * d, (0, -1): -c - d, (0, 1): -d})
    res = N.von_neumann(S)
    assert res.proof_status == "unknown"
    assert res.stability_condition is None
    fixed = N.von_neumann(S, subs={d: pool.rational(1, 4)})
    assert fixed.proof_status == "proved"


def test_von_neumann_refuses_two_space_dimensions(pool):
    r = pool.symbol("r")
    S = N.Scheme(
        {
            (1, (0, 0)): 1,
            (0, (0, 0)): -1 + 4 * r,
            (0, (1, 0)): -r,
            (0, (-1, 0)): -r,
            (0, (0, 1)): -r,
            (0, (0, -1)): -r,
        }
    )
    with pytest.raises(NumericPdeError) as ei:
        N.von_neumann(S)
    assert ei.value.code == "E-PDE-353"


def test_scheme_from_expr_refuses_nonlinear(pool):
    n, j = pool.symbol("n"), pool.symbol("j")
    u0 = pool.func("u", [n, j])
    with pytest.raises(NumericPdeError) as ei:
        N.Scheme.from_expr(pool.func("u", [n + 1, j]) - u0 * u0, "u", n, j)
    assert ei.value.code == "E-PDE-352"


def test_scheme_refuses_float_coefficients(pool):
    with pytest.raises(NumericPdeError) as ei:
        N.von_neumann(N.Scheme({(1, 0): 1, (0, 0): -0.5, (0, 1): -0.5}))
    assert ei.value.code == "E-PDE-351"


# ---------------------------------------------------------------------------
# modified equation and consistency
# ---------------------------------------------------------------------------


def test_modified_equation_ftcs_heat(pool):
    n, j = pool.symbol("n"), pool.symbol("j")
    dt, dx, nu = pool.symbol("dt"), pool.symbol("dx"), pool.symbol("nu")

    def u(lv, k):
        return pool.func("u", [n + lv, j + k])

    e = (u(1, 0) - u(0, 0)) / dt - nu * (u(0, 1) - 2 * u(0, 0) + u(0, -1)) / dx**2
    S = N.Scheme.from_expr(e, "u", n, j, dt=dt, dx=dx)
    me = N.modified_equation(S, order=4)
    assert _eq(me.coefficients[2], nu)
    # u_t = ν u_xx + ν Δx²/12 (1 − 6 r) u_xxxx,  r = ν Δt/Δx²
    assert _eq(me.coefficients[4], nu * dx**2 / 12 - nu**2 * dt / 2)
    assert set(me.physical) == {2}
    assert me.leading_dissipation[0] == 4
    assert me.leading_dispersion is None


def test_modified_equation_upwind_and_lax_wendroff(pool):
    c, a, dt, dx = pool.symbol("c"), pool.symbol("a"), pool.symbol("dt"), pool.symbol("dx")
    h = pool.rational(1, 2)
    up = N.Scheme({(1, 0): 1, (0, 0): -1 + c, (0, -1): -c})
    me = N.modified_equation(up, order=3, dt=dt, dx=dx, subs={c: a * dt / dx})
    assert _eq(me.coefficients[1], -a)
    # numerical diffusion a Δx/2 (1 − c)
    assert _eq(me.leading_dissipation[1], a * dx / 2 * (1 - a * dt / dx))
    lw = N.Scheme(
        {(1, 0): 1, (0, 0): -1 + c**2, (0, 1): c * h - c**2 * h, (0, -1): -c * h - c**2 * h}
    )
    me = N.modified_equation(lw, order=4, dt=dt, dx=dx, subs={c: a * dt / dx})
    assert 2 not in me.coefficients
    # dispersion −a Δx²/6 (1 − c²)
    assert _eq(me.leading_dispersion[1], -a * dx**2 / 6 * (1 - (a * dt / dx) ** 2))
    assert me.leading_dispersion[0] == 3


def test_modified_equation_leapfrog_is_non_dissipative(pool):
    c, a, dt, dx = pool.symbol("c"), pool.symbol("a"), pool.symbol("dt"), pool.symbol("dx")
    lp = N.Scheme({(2, 0): 1, (1, 1): c, (1, -1): -c, (0, 0): -1})
    me = N.modified_equation(lp, order=6, dt=dt, dx=dx, subs={c: a * dt / dx})
    assert all(q % 2 for q in me.coefficients)
    assert me.leading_dissipation is None
    assert _eq(me.coefficients[3], -a * dx**2 / 6 * (1 - (a * dt / dx) ** 2))


def test_modified_equation_refuses_inconsistent(pool):
    with pytest.raises(NumericPdeError) as ei:
        N.modified_equation(N.Scheme({(1, 0): 1, (0, 0): -2}))
    assert ei.value.code == "E-PDE-354"


def test_consistency_orders(pool):
    r, c, schemes = _schemes(pool)
    dt, dx, nu, a = pool.symbol("dt"), pool.symbol("dx"), pool.symbol("nu"), pool.symbol("a")
    heat = {r: nu * dt / dx**2}
    adv = {c: a * dt / dx}
    cases = [
        ("ftcs_heat", {2: nu}, heat, (True, 1, 2)),
        ("btcs_heat", {2: nu}, heat, (True, 1, 2)),
        ("cn_heat", {2: nu}, heat, (True, 2, 2)),
        ("upwind", {1: -a}, adv, (True, 1, 1)),
        ("lax_wendroff", {1: -a}, adv, (True, 2, 2)),
        ("lax_friedrichs", {1: -a}, adv, (None, 1, 2)),  # needs Δx²/Δt → 0
    ]
    for name, pde, subs, want in cases:
        co = N.consistency(N.Scheme(schemes[name]), pde, dt=dt, dx=dx, subs=subs)
        assert (co.consistent, co.time_order, co.space_order) == want, name


def test_consistency_ftcs_truncation_term(pool):
    r, _, schemes = _schemes(pool)
    dt, dx, nu = pool.symbol("dt"), pool.symbol("dx"), pool.symbol("nu")
    co = N.consistency(
        N.Scheme(schemes["ftcs_heat"]), {2: nu}, dt=dt, dx=dx, subs={r: nu * dt / dx**2}
    )
    assert _eq(co.terms[4], nu**2 * dt / 2 - nu * dx**2 / 12)
    assert "Derivative" in str(co.truncation_error)


def test_exported_from_experimental():
    import alkahest.experimental as ex

    for name in (
        "fd_weights",
        "stencil",
        "von_neumann",
        "modified_equation",
        "cheb_diff_matrix",
        "fourier_diff_matrix",
    ):
        assert name in ex.__all__
        assert getattr(ex, name) is getattr(N, name)
