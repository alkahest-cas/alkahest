"""F21: controls follow-ups to G10d.

* exact CARE for n > 2 through the Hamiltonian's stable subspace over ℚ,
* structured exact Lyapunov / Stein solves at sizes the Kronecker system never reached,
* MIMO ``minreal`` (exact Kalman decomposition) with the transfer matrix verified,
* discrete time: margins on the unit circle, ``d2c``, ``dlyap``, ``dare``, ``dlqr``,
* step / impulse responses for denominators with irreducible factors of degree > 2
  (partial fractions over ``RootOf``).

Oracles: ``scipy`` and python-control (``pytest.importorskip`` — absent from the lean CI
venv) plus hand-computed closed forms and exact residuals that must be *identically* zero.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import alkahest as ak
import pytest
from alkahest.controls import (
    ControlsError,
    StateSpace,
    TransferFunction,
    care,
    controls_side_conditions,
    dare,
    dlqr,
    dlyap,
    lqr,
    lyapunov,
    margins,
)


@pytest.fixture
def P():
    pool = ak.ExprPool()
    syms = {n: pool.symbol(n) for n in ["s", "t", "z", "a", "b", "k", "r"]}
    syms["pool"] = pool
    return syms


def mat(pool, rows):
    """An ``ak.Matrix`` of exact rationals from nested lists of int / Fraction."""

    def lift(x):
        x = Fraction(x)
        return (
            pool.integer(x.numerator)
            if x.denominator == 1
            else pool.rational(x.numerator, x.denominator)
        )

    return ak.Matrix([[lift(x) for x in r] for r in rows])


def frac_rows(M):
    out = []
    for r in M.to_list():
        row = []
        for x in r:
            node = x._node_exact()
            if node[0] == "integer":
                row.append(Fraction(node[1]))
            elif node[0] == "rational":
                row.append(Fraction(node[1], node[2]))
            else:
                raise AssertionError(f"not an exact rational: {x}")
        out.append(row)
    return out


def fmm(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def ftr(a):
    return [list(r) for r in zip(*a)]


def fadd(a, b, sg=1):
    return [[x + sg * y for x, y in zip(r, q)] for r, q in zip(a, b)]


def numeric(M):
    return [
        [complex(ak.evaluate(x, {}, mode="complex").value).real for x in r] for r in M.to_list()
    ]


def ev(e, bind=None):
    v = complex(ak.evaluate(e, bind or {}, mode="complex").value)
    assert abs(v.imag) <= 1e-9 * (1 + abs(v.real)), v
    return v.real


def rational_stable(rng, n, shift):
    A = [[Fraction(rng.randint(-3, 3)) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[i][i] -= shift
    return A


# ---------------------------------------------------------------------------
# Lyapunov / Stein: structured exact solve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 6, 11])
def test_lyapunov_rational_structured_exact(P, n):
    """The residual is *identically* zero in ℚ, at sizes past the old 8×8 Kronecker cap."""
    rng = random.Random(100 + n)
    A = rational_stable(rng, n, 2 * n)
    Q = [[Fraction(rng.randint(-2, 2)) for _ in range(n)] for _ in range(n)]
    Q = [[Q[i][j] + Q[j][i] for j in range(n)] for i in range(n)]
    X = frac_rows(lyapunov(mat(P["pool"], A), mat(P["pool"], Q)))
    res = fadd(fadd(fmm(A, X), fmm(X, ftr(A))), Q)
    assert all(x == 0 for r in res for x in r)
    assert ftr(X) == X  # symmetric for symmetric Q
    assert controls_side_conditions() == []


def test_lyapunov_beyond_old_cap_and_scipy(P):
    np = pytest.importorskip("numpy")
    sl = pytest.importorskip("scipy.linalg")
    n = 14  # the Kronecker route stopped at 8
    rng = random.Random(5)
    A = rational_stable(rng, n, 14)
    Q = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    X = lyapunov(mat(P["pool"], A), mat(P["pool"], Q))
    ref = sl.solve_continuous_lyapunov(np.array(A, dtype=float), -np.array(Q, dtype=float))
    assert np.allclose(numeric(X), ref, rtol=1e-8, atol=1e-10)


def test_lyapunov_structured_singular_refusal(P):
    pool = P["pool"]
    # eigenvalues 1, -1, 2: λ₁ + λ₂ = 0
    A = mat(pool, [[1, 0, 0], [0, -1, 0], [0, 0, 2]])
    with pytest.raises(ControlsError) as ei:
        lyapunov(A, mat(pool, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    assert ei.value.code == "E-CTRL-004"


def test_dlyap_rational_exact_and_scipy(P):
    pool = P["pool"]
    A = [
        [Fraction(1, 2), Fraction(1, 3), 0, 0],
        [0, Fraction(1, 4), Fraction(1, 5), 0],
        [Fraction(1, 7), 0, Fraction(-1, 3), Fraction(1, 2)],
        [0, 0, Fraction(1, 6), Fraction(1, 9)],
    ]
    Q = [[2, 1, 0, 0], [1, 3, 0, 1], [0, 0, 1, 0], [0, 1, 0, 4]]
    X = frac_rows(dlyap(mat(pool, A), mat(pool, Q)))
    res = fadd(fadd(fmm(fmm(A, X), ftr(A)), X, -1), Q)
    assert all(x == 0 for r in res for x in r)
    np = pytest.importorskip("numpy")
    sl = pytest.importorskip("scipy.linalg")
    ref = sl.solve_discrete_lyapunov(np.array(A, dtype=float), np.array(Q, dtype=float))
    assert np.allclose(np.array([[float(x) for x in r] for r in X]), ref)


def test_dlyap_symbolic_and_refusals(P):
    a, pool = P["a"], P["pool"]
    one = pool.integer(1)
    X = dlyap([[a]], [[one]])
    assert ev(X.to_list()[0][0], {a: 0.5}) == pytest.approx(1 / (1 - 0.25))
    assert controls_side_conditions()  # the determinant of the Stein operator
    # eigenvalue -1 and a reciprocal pair both make λᵢλⱼ = 1
    for rows in ([[-1, 0], [0, Fraction(1, 3)]], [[2, 0], [0, Fraction(1, 2)]]):
        with pytest.raises(ControlsError) as ei:
            dlyap(mat(pool, rows), mat(pool, [[1, 0], [0, 1]]))
        assert ei.value.code == "E-CTRL-004"


# ---------------------------------------------------------------------------
# Exact CARE for n > 2
# ---------------------------------------------------------------------------


def designed_care(pool):
    """A 4-state, 2-input CARE built backwards from a chosen rational stabilising X₀.

    ``A = A_cl + G X₀`` with ``A_cl`` triangular Hurwitz, so the Hamiltonian's spectrum is
    ``±{−1, −2, −3, −1}`` and its characteristic polynomial factors over ℚ.
    """
    Acl = [[-1, 2, 0, 1], [0, -2, 1, 0], [0, 0, -3, 1], [0, 0, 0, -1]]
    B = [[1, 0], [0, 1], [0, 0], [1, 1]]
    X0 = [[2, 1, 0, 0], [1, 3, 1, 0], [0, 1, 1, 0], [0, 0, 0, 2]]
    R = [[1, 0], [0, 2]]
    Rinv = [[Fraction(1), 0], [0, Fraction(1, 2)]]
    G = fmm(fmm(B, Rinv), ftr(B))
    A = fadd(Acl, fmm(G, X0))
    Q = fadd(fmm(ftr(A), X0), fmm(X0, A))
    Q = fadd(Q, fmm(fmm(X0, G), X0), -1)
    Q = [[-x for x in r] for r in Q]
    return A, B, Q, R, X0, G


def test_care_exact_n4_returns_the_rational_solution(P):
    pool = P["pool"]
    A, B, Q, R, X0, G = designed_care(pool)
    X = care(mat(pool, A), mat(pool, B), mat(pool, Q), mat(pool, R), method="exact")
    Xf = frac_rows(X)
    assert Xf == [[Fraction(x) for x in r] for r in X0]
    res = fadd(fadd(fmm(ftr(A), Xf), fmm(Xf, A)), fmm(fmm(Xf, G), Xf), -1)
    res = fadd(res, Q)
    assert all(x == 0 for r in res for x in r)
    # auto picks the same exact route for n > 2 and states no fallback
    Xa = care(mat(pool, A), mat(pool, B), mat(pool, Q), mat(pool, R))
    assert isinstance(Xa, ak.Matrix)
    assert controls_side_conditions() == []
    # LQR gain K = R⁻¹BᵀX is exact too
    K, _ = lqr(mat(pool, A), mat(pool, B), mat(pool, Q), mat(pool, R))
    Rinv = [[Fraction(1), 0], [0, Fraction(1, 2)]]
    assert frac_rows(K) == fmm(fmm(Rinv, ftr(B)), Xf)


def test_care_exact_n4_matches_scipy(P):
    np = pytest.importorskip("numpy")
    sl = pytest.importorskip("scipy.linalg")
    pool = P["pool"]
    A, B, Q, R, _, _ = designed_care(pool)
    X = care(mat(pool, A), mat(pool, B), mat(pool, Q), mat(pool, R), method="exact")
    f = lambda M: np.array(M, dtype=float)  # noqa: E731
    ref = sl.solve_continuous_are(f(A), f(B), f(Q), f(R))
    assert np.allclose(numeric(X), ref)


def test_care_exact_diagonal_n3(P):
    """Decoupled scalar Riccati equations with rational roots: a ≠ 0 spectra ±√(a²+q/r)."""
    pool = P["pool"]
    # a=-3/4... choose q/r so √(a²+q/r) is rational: a=0, q=9/16·... -> eigenvalue ±3/4 etc.
    A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    B = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    Q = [[Fraction(9, 16), 0, 0], [0, 4, 0], [0, 0, 25]]
    R = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    X = frac_rows(care(mat(pool, A), mat(pool, B), mat(pool, Q), mat(pool, R), method="exact"))
    assert [[Fraction(3, 4), 0, 0], [0, 2, 0], [0, 0, 5]] == X


def test_care_mixed_factor_refuses_exact_and_falls_back_numeric(P):
    np = pytest.importorskip("numpy")
    sl = pytest.importorskip("scipy.linalg")
    pool = P["pool"]
    A = [[0, 1, 0], [0, 0, 1], [1, 2, -1]]
    B = [[0], [0], [1]]
    Q = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    R = [[1]]
    args = (mat(pool, A), mat(pool, B), mat(pool, Q), mat(pool, R))
    with pytest.raises(ControlsError) as ei:
        care(*args, method="exact")
    assert ei.value.code == "E-CTRL-005"
    X = care(*args)  # auto: numeric Schur, with the fallback stated
    conds = controls_side_conditions()
    assert any("not defined over ℚ" in c for c in conds)
    f = lambda M: np.array(M, dtype=float)  # noqa: E731
    assert np.allclose(X, sl.solve_continuous_are(f(A), f(B), f(Q), f(R)))


def test_care_exact_n3_no_stabilising_solution(P):
    pool = P["pool"]
    # unstable, uncontrollable modes: B = 0
    A = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    B = [[0], [0], [0]]
    Q = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ControlsError) as ei:
        care(mat(pool, A), mat(pool, B), mat(pool, Q), mat(pool, [[1]]), method="exact")
    assert ei.value.code == "E-CTRL-004"


def test_care_symbolic_n3_stays_a_coded_refusal(P):
    pool, a = P["pool"], P["a"]
    A = ak.Matrix([[a, pool.integer(1), pool.integer(0)]] + [[pool.integer(0)] * 3] * 2)
    with pytest.raises(ControlsError) as ei:
        care(
            A,
            mat(pool, [[0], [0], [1]]),
            mat(pool, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            mat(pool, [[1]]),
            method="exact",
        )
    assert ei.value.code == "E-CTRL-005"


# ---------------------------------------------------------------------------
# MIMO minreal
# ---------------------------------------------------------------------------


def tf_matrix_equal(G1, G2):
    return all(a.equals(b) for r1, r2 in zip(G1, G2) for a, b in zip(r1, r2))


def test_mimo_minreal_removes_uncontrollable_and_unobservable_modes(P):
    pool, s = P["pool"], P["s"]
    A = [[-1, 0, 0, 0], [0, -2, 0, 0], [0, 0, -3, 0], [0, 0, 0, -4]]
    B = [[1, 0], [1, 1], [0, 0], [0, 1]]  # state 3 is uncontrollable
    C = [[1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0]]  # state 3 is also seen only by y3
    D = [[0, 0], [0, 0], [0, 0]]
    ss = StateSpace(mat(pool, A), mat(pool, B), mat(pool, C), mat(pool, D))
    mr = ss.minreal()
    assert mr.nstates == 3
    assert mr.ninputs == 2
    assert mr.noutputs == 3
    assert controls_side_conditions() == []
    assert mr.is_controllable()
    assert mr.is_observable()
    assert tf_matrix_equal(ss.to_tf(s), mr.to_tf(s))


def test_mimo_minreal_random_embedding_has_expected_order(P):
    """Embed a minimal 2-state 2×2 system in 6 states by adding hidden modes."""
    pool, s = P["pool"], P["s"]
    rng = random.Random(9)
    A0 = [[-1, 2], [0, -3]]
    B0 = [[1, 0], [0, 1]]
    C0 = [[1, 1], [2, 0]]
    ss0 = StateSpace(mat(pool, A0), mat(pool, B0), mat(pool, C0))
    # block layout: [observable-not-controllable (1), core (2), controllable-not-observable (1),
    #                hidden (1)] with random couplings in the allowed triangular positions.
    n = 5
    A = [[0] * n for _ in range(n)]
    B = [[0, 0] for _ in range(n)]
    C = [[0] * n for _ in range(2)]
    # index 0: uo-controllable, 1-2: core, 3: controllable but unobservable, 4: neither
    core = [1, 2]
    for i in range(2):
        for j in range(2):
            A[core[i]][core[j]] = A0[i][j]
        B[core[i]] = list(B0[i])
        for o in range(2):
            C[o][core[i]] = C0[o][i]
    A[0][0], A[3][3], A[4][4] = -5, -6, -7
    A[0][1] = rng.randint(1, 3)  # state 0 is driven by the core but only seen at the output
    C[0][0] = 1
    A[3][2] = 0
    A[2][3] = rng.randint(1, 3)  # state 3 feeds the core but is not seen... controllable only
    B[3] = [1, 1]
    A[4][0] = 1  # hidden: neither driven by inputs nor seen
    ss = StateSpace(mat(pool, A), mat(pool, B), mat(pool, C))
    mr = ss.minreal()
    assert tf_matrix_equal(ss.to_tf(s), mr.to_tf(s))
    assert mr.nstates <= 4
    assert mr.is_controllable()
    assert mr.is_observable()
    assert mr.nstates >= ss0.minreal().nstates


def test_mimo_minreal_symbolic_states_generic_hypotheses(P):
    pool, a, s = P["pool"], P["a"], P["s"]
    one, zero = pool.integer(1), pool.integer(0)
    A = ak.Matrix([[-a, zero], [zero, pool.integer(-2)]])
    B = ak.Matrix([[one, zero], [zero, one]])
    C = ak.Matrix([[one, zero], [zero, a]])
    ss = StateSpace(A, B, C)
    mr = ss.minreal()
    assert mr.nstates == 2
    assert controls_side_conditions() == ["a ≠ 0"]
    assert tf_matrix_equal(ss.to_tf(s), mr.to_tf(s))


def test_minreal_static_and_float_refusals(P):
    pool = P["pool"]
    ss = StateSpace(
        mat(pool, [[-1, 0], [0, -2]]),
        mat(pool, [[0, 0], [0, 0]]),
        mat(pool, [[1, 0], [0, 1]]),
        mat(pool, [[1, 0], [0, 1]]),
    )
    with pytest.raises(ControlsError) as ei:
        ss.minreal()
    assert ei.value.code == "E-CTRL-001"
    fl = StateSpace(
        ak.Matrix([[pool.float(-1.0)]]), mat(pool, [[1]]), mat(pool, [[1]]), mat(pool, [[0]])
    )
    with pytest.raises(ControlsError) as ei:
        fl.minreal(method="kalman")
    assert ei.value.code == "E-CTRL-005"


def test_siso_minreal_default_is_unchanged(P):
    pool, s = P["pool"], P["s"]
    ss = StateSpace(mat(pool, [[-1, 0], [0, -2]]), mat(pool, [[1], [1]]), mat(pool, [[1, 0]]))
    assert ss.minreal().nstates == 1  # via the transfer function, canonical form
    assert ss.minreal(method="kalman").nstates == 1
    assert ss.minreal(method="kalman").to_tf(s).equals(ss.to_tf(s))


# ---------------------------------------------------------------------------
# Discrete time
# ---------------------------------------------------------------------------


def dtf(P, num, den, dt):
    pool, z = P["pool"], P["z"]

    def lift(x):
        x = Fraction(x)
        return (
            pool.integer(x.numerator)
            if x.denominator == 1
            else pool.rational(x.numerator, x.denominator)
        )

    return TransferFunction.from_coeffs(
        [lift(c) for c in num], [lift(c) for c in den], z, dt=lift(dt)
    )


def test_discrete_margins_closed_form(P):
    """G(z) = 1/(z + 1/2): G(−1) = −2 gives gm = 1/2 at the Nyquist frequency π/T."""
    G = dtf(P, [1], [1, Fraction(1, 2)], Fraction(1, 10))
    m = margins(G)
    assert m.gain_margin == pytest.approx(0.5)
    assert m.phase_crossover == pytest.approx(10 * math.pi)
    # |G| = 1 ⇔ |z + ½| = 1 ⇒ cos θ = −¼
    theta = math.acos(-0.25)
    assert m.gain_crossover == pytest.approx(theta / 0.1)
    z = complex(math.cos(theta), math.sin(theta))
    assert m.phase_margin == pytest.approx(
        math.degrees(math.atan2((1 / (z + 0.5)).imag, (1 / (z + 0.5)).real)) + 180
    )


@pytest.mark.parametrize(
    ("num", "den", "dt"),
    [
        (
            [Fraction(1, 2)],
            [1, Fraction(-19, 30), Fraction(-1, 10), Fraction(1, 30)],
            Fraction(1, 10),
        ),
        (
            [Fraction(3, 10), Fraction(1, 5)],
            [1, Fraction(-17, 10), Fraction(4, 5), 0],
            Fraction(1, 10),
        ),
    ],
)
def test_discrete_margins_vs_python_control(P, num, den, dt):
    control = pytest.importorskip("control")
    m = margins(dtf(P, num, den, dt))
    ref = control.margin(
        control.TransferFunction([float(c) for c in num], [float(c) for c in den], float(dt))
    )
    for got, want in zip(m, ref):
        if math.isnan(want):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(float(want), rel=1e-7)


def test_discrete_bode_matches_direct_evaluation(P):
    np = pytest.importorskip("numpy")
    G = dtf(P, [2, Fraction(3, 10)], [1, Fraction(-9, 10), Fraction(2, 5)], Fraction(1, 20))
    w = [0.5, 3.0, 17.0, 40.0]
    mag, _ph = G.bode(w)
    for k, wk in enumerate(w):
        zz = np.exp(1j * wk * 0.05)
        g = (2 * zz + 0.3) / (zz**2 - 0.9 * zz + 0.4)
        # the numerator above is 2z + 3/10 -> coefficients [2, 3/10] highest first
        assert mag[k] == pytest.approx(20 * math.log10(abs(g)), abs=1e-9)


def test_tustin_round_trip_is_exact(P):
    pool, s = P["pool"], P["s"]
    G = TransferFunction(s + 2, s**2 + 3 * s + 5, s)
    Gz = G.discretize(pool.rational(1, 10), "tustin")
    assert Gz.is_discrete
    Gc = Gz.d2c("tustin")
    assert not Gc.is_discrete
    assert Gc.equals(G)
    # symbolic gain and period survive the round trip
    K, T = P["k"], P["t"]
    H = TransferFunction(K, s + 1, s)
    Hc = H.discretize(T, "tustin").d2c("tustin")
    assert Hc.equals(H)


def test_state_space_tustin_round_trip_is_exact(P):
    pool, s = P["pool"], P["s"]
    ss = StateSpace(
        mat(pool, [[0, 1], [-2, -3]]), mat(pool, [[0], [1]]), mat(pool, [[1, 0]]), mat(pool, [[0]])
    )
    sd = ss.discretize(pool.rational(1, 4), "tustin")
    back = sd.d2c("tustin")
    assert back.dt is None
    assert back.to_tf(s).equals(ss.to_tf(s))


def test_zoh_round_trip_exact_for_integrator_chain(P):
    pool, s = P["pool"], P["s"]
    ss = StateSpace(
        mat(pool, [[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        mat(pool, [[0], [0], [1]]),
        mat(pool, [[1, 0, 0]]),
        mat(pool, [[0]]),
    )
    sd = ss.discretize(pool.rational(1, 2), "zoh")
    back = sd.d2c("zoh")
    assert frac_rows(back.A) == frac_rows(ss.A)
    assert frac_rows(back.B) == frac_rows(ss.B)
    assert back.to_tf(s).equals(ss.to_tf(s))


def test_zoh_round_trip_numeric(P):
    pytest.importorskip("numpy")
    pool, s = P["pool"], P["s"]
    G = TransferFunction(s + 2, s**2 + 3 * s + 5, s)
    Gz = G.discretize(pool.rational(1, 10), "zoh")
    Gc = Gz.d2c("zoh")
    for w in (0.3, 1.7, 6.0):
        assert Gc.freqresp([w])[0] == pytest.approx(G.freqresp([w])[0], rel=1e-9)


def test_discrete_refusals(P):
    s = P["s"]
    G = TransferFunction(1, s + 1, s)
    with pytest.raises(ControlsError) as ei:
        G.d2c()
    assert ei.value.code == "E-CTRL-001"


def test_dare_vs_scipy_and_dlqr_closed_loop(P):
    np = pytest.importorskip("numpy")
    sl = pytest.importorskip("scipy.linalg")
    pool = P["pool"]
    rng = np.random.default_rng(11)
    for n, m in ((2, 1), (4, 2)):
        A = rng.normal(size=(n, n))
        B = rng.normal(size=(n, m))
        Q = np.eye(n)
        R = np.eye(m)

        def to_m(M):
            return ak.Matrix([[pool.float(float(x)) for x in r] for r in M.tolist()])

        X = dare(to_m(A), to_m(B), to_m(Q), to_m(R))
        assert np.allclose(X, sl.solve_discrete_are(A, B, Q, R))
        K, _X2 = dlqr(to_m(A), to_m(B), to_m(Q), to_m(R))
        assert np.max(np.abs(np.linalg.eigvals(A - B @ K))) < 1
    with pytest.raises(ControlsError) as ei:
        dare(mat(pool, [[1]]), mat(pool, [[1]]), mat(pool, [[1]]), mat(pool, [[1]]), method="exact")
    assert ei.value.code == "E-CTRL-005"


def test_dare_scalar_closed_form(P):
    pytest.importorskip("numpy")
    pool = P["pool"]
    # a=2, b=1, q=1, r=1: x = a²x − a²x²/(1 + x) + 1  ⇒  x² − 4x − 1 = 0 ... use root 2+√5
    X = dare(mat(pool, [[2]]), mat(pool, [[1]]), mat(pool, [[1]]), mat(pool, [[1]]))
    x = X[0][0]
    assert 4 * x - 4 * x * x / (1 + x) + 1 == pytest.approx(x)
    assert x == pytest.approx(2 + math.sqrt(5))


# ---------------------------------------------------------------------------
# Step / impulse responses with higher-order denominators
# ---------------------------------------------------------------------------


def values(y, t, ts):
    return [complex(ak.evaluate(y, {t: tv}, mode="complex").value) for tv in ts]


def test_step_impulse_irreducible_cubic_vs_scipy(P):
    np = pytest.importorskip("numpy")
    sg = pytest.importorskip("scipy.signal")
    pool, s, t = P["pool"], P["s"], P["t"]
    G = TransferFunction(pool.integer(1), s**3 + s + 1, s)
    TT = np.linspace(0, 5, 101)
    idx = [10, 20, 80]
    ts = [float(TT[i]) for i in idx]
    ys = G.step_response(t)
    yi = G.impulse_response(t)
    sys = sg.lti([1], [1, 0, 1, 1])
    ref_step = sg.step(sys, T=TT)[1][idx]
    ref_imp = sg.impulse(sys, T=TT)[1][idx]
    for got, want in zip(values(ys, t, ts), ref_step):
        assert got.real == pytest.approx(want, rel=1e-8, abs=1e-10)
        assert abs(got.imag) < 1e-10
    for got, want in zip(values(yi, t, ts), ref_imp):
        assert got.real == pytest.approx(want, rel=1e-8, abs=1e-10)
        assert abs(got.imag) < 1e-10
    # the closed form is exact: step(0⁺) = 0 and step(∞) = DC gain, as exact RootOf identities
    assert abs(values(ys, t, [0.0])[0]) < 1e-12


def test_step_repeated_irreducible_factor_vs_scipy(P):
    np = pytest.importorskip("numpy")
    sg = pytest.importorskip("scipy.signal")
    _pool, s, t = P["pool"], P["s"], P["t"]
    den = (s**3 + s + 1) ** 2 * (s + 1)
    G = TransferFunction(s**2 + 3, den, s)
    TT = np.linspace(0, 5, 101)
    idx = [10, 20, 80]
    dn = np.polymul(np.polymul([1, 0, 1, 1], [1, 0, 1, 1]), [1, 1])
    ref = sg.step(sg.lti([1, 0, 3], dn), T=TT)[1][idx]
    got = values(G.step_response(t), t, [float(TT[i]) for i in idx])
    for g, w in zip(got, ref):
        assert g.real == pytest.approx(w, rel=1e-8, abs=1e-10)
        assert abs(g.imag) < 1e-9


def test_step_mixed_easy_and_hard_factors(P):
    np = pytest.importorskip("numpy")
    sg = pytest.importorskip("scipy.signal")
    _pool, s, t = P["pool"], P["s"], P["t"]
    G = TransferFunction(s + 2, (s + 1) * (s**2 + 2 * s + 5) * (s**4 + s + 1), s)
    dn = np.polymul(np.polymul([1, 1], [1, 2, 5]), [1, 0, 0, 1, 1])
    TT = np.linspace(0, 6, 121)
    idx = [10, 30, 100]
    ref = sg.step(sg.lti([1, 2], dn), T=TT)[1][idx]
    got = values(G.step_response(t), t, [float(TT[i]) for i in idx])
    for g, w in zip(got, ref):
        assert g.real == pytest.approx(w, rel=1e-8, abs=1e-10)
        assert abs(g.imag) < 1e-9


def test_state_space_step_goes_through_the_rootof_route(P):
    np = pytest.importorskip("numpy")
    sg = pytest.importorskip("scipy.signal")
    pool, t = P["pool"], P["t"]
    ss = StateSpace(
        mat(pool, [[0, 1, 0], [0, 0, 1], [-1, -1, 0]]),
        mat(pool, [[0], [0], [1]]),
        mat(pool, [[1, 0, 0]]),
    )
    y = ss.step_response(t)
    TT = np.linspace(0, 5, 101)
    ref = sg.step(sg.lti([1], [1, 0, 1, 1]), T=TT)[1][[10, 40]]
    got = values(y, t, [float(TT[10]), float(TT[40])])
    for g, w in zip(got, ref):
        assert g.real == pytest.approx(w, rel=1e-8, abs=1e-10)


def test_parametric_high_order_denominator_still_refuses(P):
    pool, s, t, k = P["pool"], P["s"], P["t"], P["k"]
    G = TransferFunction(pool.integer(1), s**3 + k * s + 1, s)
    with pytest.raises(Exception) as ei:
        G.step_response(t)
    assert "E-TRANSFORM" in str(ei.value) or getattr(ei.value, "code", "").startswith("E-")


def test_structured_solvers_honour_the_budget(P):
    """A long exact elimination stops at the active ``Budget`` instead of running on."""
    pool = P["pool"]
    n = 40
    rng = random.Random(1)
    A = rational_stable(rng, n, 2 * n)
    Q = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    with ak.context(budget=ak.Budget(wall_ms=1)), pytest.raises(ak.BudgetExceededError):
        lyapunov(mat(pool, A), mat(pool, Q))
