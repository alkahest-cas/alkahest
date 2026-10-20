"""F27 — tensor and index calculus (report 10-9 §4.3, §3 G10).

The report's physics probes built Christoffel/Ricci by hand-written loops plus
``cancel``; the Kerr metric's Ricci tensor (287 kB for R_00) could not be shown
to vanish symbolically, and FLRW needed a jet-symbol workaround for ``a(t)``.
``ak.Metric`` computes every curvature tensor in an exact canonical field, so
Kerr is *proved* Ricci-flat, and undefined functions are first-class.

Oracles: hard-coded closed forms (Schwarzschild Christoffels and Kretschmann,
Friedmann equations, de Sitter, the 2-sphere), an independent mpmath
finite-difference computation of the curvature from the metric at random
points, and SymPy for a Reissner–Nordström Einstein tensor.
"""

from __future__ import annotations

import random
import time
from fractions import Fraction

import alkahest as ak
import pytest
from alkahest import cos, exp, sin

mpmath = pytest.importorskip("mpmath")


def lit_zero(e) -> bool:
    return str(e) == "0"


def same(m: ak.Metric, a, b) -> bool:
    return m.is_zero(a - b)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def schwarzschild():
    p = ak.ExprPool()
    t, r, th, ph, M = (p.symbol(s) for s in ("t", "r", "theta", "phi", "M"))
    f = 1 - 2 * M / r
    g = [[-f, 0, 0, 0], [0, 1 / f, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2 * sin(th) ** 2]]
    return p, (t, r, th, ph, M), ak.Metric(g, [t, r, th, ph])


def kerr():
    p = ak.ExprPool()
    t, r, th, ph, M, a = (p.symbol(s) for s in ("t", "r", "theta", "phi", "M", "a"))
    sigma = r**2 + a**2 * cos(th) ** 2
    delta = r**2 - 2 * M * r + a**2
    s2 = sin(th) ** 2
    gtt = -(1 - 2 * M * r / sigma)
    gtp = -2 * M * r * a * s2 / sigma
    gpp = (r**2 + a**2 + 2 * M * r * a**2 * s2 / sigma) * s2
    g = [[gtt, 0, 0, gtp], [0, sigma / delta, 0, 0], [0, 0, sigma, 0], [gtp, 0, 0, gpp]]
    return p, (t, r, th, ph, M, a), g


# ---------------------------------------------------------------------------
# The report's repro: Kerr Ricci = 0, symbolically
# ---------------------------------------------------------------------------


def test_kerr_is_proved_ricci_flat():
    _p, (t, r, th, ph, _M, _a), g = kerr()
    t0 = time.perf_counter()
    m = ak.Metric(g, [t, r, th, ph])
    assert m.is_ricci_flat() is True
    ric = m.ricci().components
    assert all(lit_zero(ric[i][j]) for i in range(4) for j in range(4))
    assert lit_zero(m.ricci_scalar())
    assert all(lit_zero(x) for row in m.einstein().components for x in row)
    elapsed = time.perf_counter() - t0
    # The report's attempt ran > 12 s on R_00 alone without finishing.
    assert elapsed < 60, elapsed
    stages = m.timings()
    assert {"convert", "inverse", "christoffel", "riemann", "ricci"} <= set(stages)


def test_kerr_kretschmann_matches_closed_form():
    # K = 48 M² (r² − a²c²)(Σ² − 16 r² a² c²) / Σ⁶  with c = cos θ, Σ = r² + a²c².
    _p, (t, r, th, ph, M, a), g = kerr()
    m = ak.Metric(g, [t, r, th, ph])
    c = cos(th)
    sigma = r**2 + a**2 * c**2
    expected = 48 * M**2 * (r**2 - a**2 * c**2) * (sigma**2 - 16 * r**2 * a**2 * c**2) / sigma**6
    assert same(m, m.kretschmann(), expected)


# ---------------------------------------------------------------------------
# Schwarzschild: Christoffels, vacuum, Kretschmann, geodesics
# ---------------------------------------------------------------------------


def test_schwarzschild_christoffel_closed_forms():
    p, (_t, r, th, _ph, M), m = schwarzschild()
    G = m.christoffel()
    known = {
        (0, 0, 1): M / (r * (r - 2 * M)),
        (1, 0, 0): M * (r - 2 * M) / r**3,
        (1, 1, 1): -M / (r * (r - 2 * M)),
        (1, 2, 2): -(r - 2 * M),
        (1, 3, 3): -(r - 2 * M) * sin(th) ** 2,
        (2, 1, 2): 1 / r,
        (2, 3, 3): -sin(th) * cos(th),
        (3, 1, 3): 1 / r,
        (3, 2, 3): cos(th) / sin(th),
    }
    for a in range(4):
        for b in range(4):
            for c in range(4):
                key = (a, min(b, c), max(b, c))
                exp_ = known.get(key, 0)
                assert same(m, G[a][b][c], exp_ if exp_ != 0 else p.integer(0)), (
                    a,
                    b,
                    c,
                    G[a][b][c],
                )
    # cot θ comes back as cos/sin, not cos·sin/(1 − cos²)
    assert "+" not in str(G[3][2][3])


def test_schwarzschild_vacuum_and_kretschmann():
    _p, (_t, r, _th, _ph, M), m = schwarzschild()
    assert m.is_ricci_flat()
    assert str(m.kretschmann()) == str(m.simplify(48 * M**2 / r**6))
    assert same(m, m.kretschmann(), 48 * M**2 / r**6)
    # Ricci-flat ⇒ the Weyl tensor equals the Riemann tensor.
    W = m.weyl().components
    R = m.riemann_lower().components
    for idx in [(0, 1, 0, 1), (0, 2, 0, 2), (1, 2, 1, 2), (2, 3, 2, 3), (0, 3, 1, 3)]:
        a, b, c, d = idx
        assert same(m, W[a][b][c][d], R[a][b][c][d])
    # R^t_{rtr} = 2M / (r² (r − 2M))
    assert same(m, m.riemann()[0, 1, 0, 1], 2 * M / (r**2 * (r - 2 * M)))


def test_schwarzschild_geodesic_equations():
    p, (_t, r, th, _ph, M), m = schwarzschild()
    td, rd, thd, phd = (p.symbol(s) for s in ("td", "rd", "thd", "phd"))
    acc = m.geodesic_equations([td, rd, thd, phd])
    exp_r = (
        -M * (r - 2 * M) / r**3 * td**2
        + M / (r * (r - 2 * M)) * rd**2
        + (r - 2 * M) * (thd**2 + sin(th) ** 2 * phd**2)
    )
    assert same(m, acc[1], exp_r)
    exp_t = -2 * M / (r * (r - 2 * M)) * td * rd
    assert same(m, acc[0], exp_t)
    # default velocity symbols
    acc2 = m.geodesic_equations()
    assert "r_dot" in str(acc2[0])


# ---------------------------------------------------------------------------
# FLRW with an undefined a(t); de Sitter; the 2-sphere
# ---------------------------------------------------------------------------


def test_flrw_friedmann_equations():
    p = ak.ExprPool()
    t, r, th, ph, k = (p.symbol(s) for s in ("t", "r", "theta", "phi", "k"))
    a = p.func("a", [t])
    g = [
        [-1, 0, 0, 0],
        [0, a**2 / (1 - k * r**2), 0, 0],
        [0, 0, a**2 * r**2, 0],
        [0, 0, 0, a**2 * r**2 * sin(th) ** 2],
    ]
    m = ak.Metric(g, [t, r, th, ph])
    ad = ak.diff(a, t).value
    add = ak.diff(ad, t).value
    assert same(m, m.ricci_scalar(), 6 * (a * add + ad**2 + k) / a**2)
    G = m.einstein()
    # G_tt = 3(ȧ² + k)/a²  (first Friedmann equation with 8πGρ = G_tt)
    assert same(m, G[0, 0], 3 * (ad**2 + k) / a**2)
    # G^r_r = −(2aä + ȧ² + k)/a²  (second Friedmann equation, −8πG p)
    assert same(m, G[1, 1], -(2 * a * add + ad**2 + k) / (1 - k * r**2))
    for x in range(4):
        for y in range(4):
            if x != y:
                assert lit_zero(G[x, y])


def test_de_sitter_constant_curvature_and_conformal_flatness():
    p = ak.ExprPool()
    t, r, th, ph, ell = (p.symbol(s) for s in ("t", "r", "theta", "phi", "l"))
    f = 1 - r**2 / ell**2
    g = [[-f, 0, 0, 0], [0, 1 / f, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2 * sin(th) ** 2]]
    m = ak.Metric(g, [t, r, th, ph])
    assert same(m, m.ricci_scalar(), 12 / ell**2)
    a, b, c, d = ak.tensor_indices("a b c d")
    # R_ab = (3/ℓ²) g_ab and R_abcd = (g_ac g_bd − g_ad g_bc)/ℓ²
    diff = m.ricci()[-a, -b] - m.as_tensor()[-a, -b] * (3 / ell**2)
    assert diff.is_zero()
    gg = m.as_tensor()
    const = (gg[-a, -c] * gg[-b, -d] - gg[-a, -d] * gg[-b, -c]) / ell**2
    assert (m.riemann_lower()[-a, -b, -c, -d] - const).is_zero()
    assert all(lit_zero(x) for x in m.weyl().nonzero().values())
    assert not m.weyl().nonzero()


def test_two_sphere():
    p = ak.ExprPool()
    th, ph, rho = (p.symbol(s) for s in ("theta", "phi", "rho"))
    m = ak.Metric([[rho**2, 0], [0, rho**2 * sin(th) ** 2]], [th, ph])
    assert same(m, m.ricci_scalar(), 2 / rho**2)
    with pytest.raises(ak.TensorError) as ei:
        m.weyl()
    assert ei.value.code == "E-TENSOR-008"


# ---------------------------------------------------------------------------
# Independent numeric oracle: finite differences of the metric (mpmath)
# ---------------------------------------------------------------------------


def _numeric_curvature(gfun, x0, n):
    """Γ^a_bc and R^a_bcd at x0 from the metric function by mpmath.diff."""
    mp = mpmath.mp

    def ginv_at(x):
        return mpmath.matrix(gfun(x)) ** -1

    def dg_at(x, k):
        h = [0] * n
        h[k] = 1
        return [
            [mpmath.diff(lambda *y, i=i, j=j: gfun(list(y))[i][j], x, tuple(h)) for j in range(n)]
            for i in range(n)
        ]

    def gamma_at(x):
        gi = ginv_at(x)
        dg = [dg_at(x, k) for k in range(n)]
        G = [[[mp.zero] * n for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    G[a][b][c] = (
                        sum(gi[a, d] * (dg[b][d][c] + dg[c][d][b] - dg[d][b][c]) for d in range(n))
                        / 2
                    )
        return G

    G0 = gamma_at(x0)

    def dgamma(a, b, c, k):
        h = [0] * n
        h[k] = 1
        return mpmath.diff(lambda *y: gamma_at(list(y))[a][b][c], x0, tuple(h))

    R = {}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(c + 1, n):
                    v = dgamma(a, d, b, c) - dgamma(a, c, b, d)
                    v += sum(
                        G0[a][c][e] * G0[e][d][b] - G0[a][d][e] * G0[e][c][b] for e in range(n)
                    )
                    R[(a, b, c, d)] = v
    return G0, R


def test_curvature_matches_finite_differences_at_random_points():
    # A non-diagonal 3-D metric with trig, exp and a parameter.
    p = ak.ExprPool()
    x, y, z, q = (p.symbol(s) for s in ("x", "y", "z", "q"))
    g = [
        [2 + sin(x) ** 2, x * y / 5, 0],
        [x * y / 5, 3 + exp(z) / 4, cos(y) / 3],
        [0, cos(y) / 3, 1 + q * x**2],
    ]
    m = ak.Metric(g, [x, y, z])
    G = m.christoffel()
    R = m.riemann().components

    mpmath.mp.dps = 30
    rng = random.Random(27)
    for _ in range(2):
        pt = [Fraction(rng.randint(-60, 60), 100) for _ in range(3)]
        qv = Fraction(rng.randint(10, 90), 100)
        vals = {x: pt[0], y: pt[1], z: pt[2], q: qv}

        def gfun(X, qv=qv):
            X = [mpmath.mpf(v) for v in X]
            s, c = mpmath.sin, mpmath.cos
            return [
                [2 + s(X[0]) ** 2, X[0] * X[1] / 5, 0],
                [X[0] * X[1] / 5, 3 + mpmath.exp(X[2]) / 4, c(X[1]) / 3],
                [0, c(X[1]) / 3, 1 + mpmath.mpf(qv.numerator) / qv.denominator * X[0] ** 2],
            ]

        x0 = [mpmath.mpf(v.numerator) / v.denominator for v in pt]
        G0, R0 = _numeric_curvature(gfun, x0, 3)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    ours = ak.eval_expr(G[a][b][c], vals)
                    assert abs(ours - float(G0[a][b][c])) < 1e-9, (a, b, c)
        for (a, b, c, d), v in R0.items():
            ours = ak.eval_expr(R[a][b][c][d], vals)
            assert abs(ours - float(v)) < 1e-7, ((a, b, c, d), ours, v)


def test_reissner_nordstrom_einstein_vs_sympy():
    sympy = pytest.importorskip("sympy")
    p = ak.ExprPool()
    t, r, th, ph, M, Q = (p.symbol(s) for s in ("t", "r", "theta", "phi", "M", "Q"))
    f = 1 - 2 * M / r + Q**2 / r**2
    g = [[-f, 0, 0, 0], [0, 1 / f, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2 * sin(th) ** 2]]
    m = ak.Metric(g, [t, r, th, ph])
    ours = m.einstein().components

    # SymPy oracle: the same pipeline written out directly.
    T, R_, TH, PH, MM, QQ = sympy.symbols("t r theta phi M Q")
    X = [T, R_, TH, PH]
    F = 1 - 2 * MM / R_ + QQ**2 / R_**2
    gs = sympy.diag(-F, 1 / F, R_**2, R_**2 * sympy.sin(TH) ** 2)
    gi = gs.inv()
    Gam = [
        [
            [
                sum(
                    gi[a, d]
                    * (
                        sympy.diff(gs[d, b], X[c])
                        + sympy.diff(gs[d, c], X[b])
                        - sympy.diff(gs[b, c], X[d])
                    )
                    for d in range(4)
                )
                / 2
                for c in range(4)
            ]
            for b in range(4)
        ]
        for a in range(4)
    ]

    def ric(b, d):
        s = 0
        for a in range(4):
            s += sympy.diff(Gam[a][d][b], X[a]) - sympy.diff(Gam[a][a][b], X[d])
            for e in range(4):
                s += Gam[a][a][e] * Gam[e][d][b] - Gam[a][d][e] * Gam[e][a][b]
        return s

    Ric = sympy.Matrix(4, 4, lambda i, j: ric(i, j))
    Rs = sum(gi[i, j] * Ric[i, j] for i in range(4) for j in range(4))
    Gs = Ric - Rs * gs / 2
    rng = random.Random(9)
    for _ in range(3):
        v = {
            "t": Fraction(1, 3),
            "r": Fraction(rng.randint(30, 90), 10),
            "theta": Fraction(rng.randint(2, 28), 10),
            "phi": Fraction(1, 5),
            "M": Fraction(rng.randint(1, 9), 10),
            "Q": Fraction(rng.randint(1, 9), 20),
        }
        subs = {T: v["t"], R_: v["r"], TH: v["theta"], PH: v["phi"], MM: v["M"], QQ: v["Q"]}
        vals = {t: v["t"], r: v["r"], th: v["theta"], ph: v["phi"], M: v["M"], Q: v["Q"]}
        for i in range(4):
            for j in range(4):
                want = float(Gs[i, j].subs(subs).evalf(30))
                got = ak.eval_expr(ours[i][j], vals) if not lit_zero(ours[i][j]) else 0.0
                assert abs(want - got) < 1e-10, (i, j, want, got)
    # And the closed form: G_tt = Q² f / r⁴.
    assert same(m, ours[0][0], Q**2 * f / r**4)


# ---------------------------------------------------------------------------
# Abstract index layer
# ---------------------------------------------------------------------------


def test_index_layer_einstein_declaratively():
    p = ak.ExprPool()
    t, r, th, ph, k = (p.symbol(s) for s in ("t", "r", "theta", "phi", "k"))
    a_ = p.func("a", [t])
    g = [
        [-1, 0, 0, 0],
        [0, a_**2 / (1 - k * r**2), 0, 0],
        [0, 0, a_**2 * r**2, 0],
        [0, 0, 0, a_**2 * r**2 * sin(th) ** 2],
    ]
    m = ak.Metric(g, [t, r, th, ph])
    a, b, c, _d = ak.tensor_indices("a b c d")
    Ric = m.ricci()
    gt = m.as_tensor()
    R = Ric[a, -a]  # raise with g^{ab}, then contract
    assert R.free == ()
    assert same(m, R.components, m.ricci_scalar())
    G = Ric[-a, -b] - gt[-a, -b] * R / 2
    want = m.einstein().components
    got = G.components
    for i in range(4):
        for j in range(4):
            assert same(m, got[i][j], want[i][j])
    # R^c_{acb} contraction of the Riemann tensor is the Ricci tensor
    Ric2 = m.riemann()[c, -a, -c, -b]
    assert (Ric2 - Ric[-a, -b]).is_zero()
    # g^{ac} g_{cb} = δ^a_b
    delta = m.inverse()[a, c] * gt[-c, -b]
    comps = delta.components
    for i in range(4):
        for j in range(4):
            assert str(comps[i][j]) == ("1" if i == j else "0")
    # lowering then raising is the identity
    up = Ric[a, b]
    back = up * gt[-a, -c]  # R^{ab} g_{ac} = R_c^b, free indices (b, -c)
    assert (back - Ric[-c, b]).is_zero()
    assert not (back - Ric[-c, b] * 2).is_zero()


def test_index_layer_symmetrisation_and_errors():
    _p, (_t, _r, _th, _ph, _M), m = schwarzschild()
    a, b, c, d = ak.tensor_indices("a b c d")
    Rl = m.riemann_lower()[-a, -b, -c, -d]
    # antisymmetric in the first pair, so its symmetrisation vanishes
    assert Rl.symmetrize(a, b).is_zero()
    assert (Rl.antisymmetrize(a, b) - Rl).is_zero()
    # first Bianchi identity R_a[bcd] = 0
    assert Rl.antisymmetrize(b, c, d).is_zero()
    gt = m.as_tensor()
    with pytest.raises(ak.TensorError) as ei:
        _ = gt[-a, -b] * gt[-a, -c]  # index down twice
    assert ei.value.code == "E-TENSOR-009"
    with pytest.raises(ak.TensorError) as ei:
        _ = gt[-a, -b] + gt[-a, -c]  # free indices differ
    assert ei.value.code == "E-TENSOR-009"
    with pytest.raises(ak.TensorError):
        _ = gt[-a]  # wrong rank
    # a Tensor without a metric cannot change variance
    bare = ak.Tensor([[1, 0], [0, 1]], "dd")
    with pytest.raises(ak.TensorError):
        _ = bare[a, b]
    assert bare[0, 0] == 1


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_refusals_are_coded():
    p = ak.ExprPool()
    x, y = p.symbol("x"), p.symbol("y")
    with pytest.raises(ak.TensorError) as ei:
        ak.Metric([[1.5, 0], [0, 1]], [x, y])
    assert ei.value.code == "E-TENSOR-004"
    with pytest.raises(ak.TensorError) as ei:
        ak.Metric([[1, x], [y, 1]], [x, y])
    assert ei.value.code == "E-TENSOR-002"
    with pytest.raises(ak.TensorError) as ei:
        ak.Metric([[x, x], [x, x]], [x, y])
    assert ei.value.code == "E-TENSOR-003"
    with pytest.raises(ak.TensorError) as ei:
        ak.Metric([[1, 0], [0, 1]], [x, x])
    assert ei.value.code == "E-TENSOR-001"
    pw = ak.piecewise([(p.gt(x, p.integer(0)), x**2)], 1 + x**2)
    with pytest.raises(ak.TensorError) as ei:
        ak.Metric([[pw, 0], [0, 1]], [x, y])
    assert ei.value.code == "E-TENSOR-005"


def test_budget_is_honoured():
    _p, (t, r, th, ph, _M, _a), g = kerr()

    def build_and_curve():
        return ak.Metric(g, [t, r, th, ph]).ricci()

    with ak.context(budget=ak.Budget(max_steps=5)), pytest.raises(ak.BudgetExceededError):
        build_and_curve()


def test_exact_zero_test_of_the_field():
    p = ak.ExprPool()
    th, x = p.symbol("theta"), p.symbol("x")
    m = ak.Metric([[1, 0], [0, 1]], [th, x])
    assert m.is_zero(sin(th) ** 2 + cos(th) ** 2 - 1)
    assert m.is_zero(sin(2 * th) - 2 * sin(th) * cos(th))
    assert m.is_zero(exp(x) * exp(-x) - 1)
    assert not m.is_zero(sin(th) ** 2 - cos(th) ** 2)
    assert same(m, m.diff(sin(th) ** 3, th), 3 * sin(th) ** 2 * cos(th))


def test_undefined_profile_functions():
    # Static spherically symmetric: G_tt = e^{2ν} r⁻² d/dr[r (1 − e^{−2λ})]
    p = ak.ExprPool()
    t, r, th, ph = (p.symbol(s) for s in ("t", "r", "theta", "phi"))
    nu, lam = p.func("nu", [r]), p.func("lam", [r])
    g = [
        [-exp(2 * nu), 0, 0, 0],
        [0, exp(2 * lam), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sin(th) ** 2],
    ]
    m = ak.Metric(g, [t, r, th, ph])
    lamp = ak.diff(lam, r).value
    want = exp(2 * nu) / r**2 * (1 - exp(-2 * lam) + 2 * r * lamp * exp(-2 * lam))
    G = m.einstein()
    assert same(m, G[0, 0], want)
    # exp(v)^k is printed exp(k·v), not as a power of exp(v)
    assert "exp(nu(r))" not in str(G[0, 0])
    assert "exp(lam(r))" not in str(G[0, 0])
    # pp-wave: R_uu = −½ (H_xx + H_yy), every other component zero
    u, v, x, y = (p.symbol(s) for s in ("u", "v", "x", "y"))
    H = p.func("H", [u, x, y])
    pp = ak.Metric([[H, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [u, v, x, y])
    hxx = ak.diff(ak.diff(H, x).value, x).value
    hyy = ak.diff(ak.diff(H, y).value, y).value
    ric = pp.ricci()
    assert same(pp, ric[0, 0], -(hxx + hyy) / 2)
    assert set(ric.nonzero()) == {(0, 0)}
    assert lit_zero(pp.ricci_scalar())
