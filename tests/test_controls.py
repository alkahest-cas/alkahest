"""G10 controls layer: transfer functions, state space, Lyapunov / Riccati, Bode / margins.

Report 10-9 §4.2: state-space → TF worked through ``Matrix.inverse`` +
``cancel`` and Routh–Hurwitz worked, but there was no transfer-function / LTI
object, no Lyapunov or Riccati solver, no Kronecker product and no margins;
``Matrix([[a]]).rank()`` returned 1 with no ``a ≠ 0`` condition.

Oracles: closed forms computed by hand (always run), ``scipy.signal`` /
``scipy.linalg`` and python-control (``pytest.importorskip`` — the lean CI
venv has neither, the nightly extras shard and a local venv do).
"""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction

import alkahest as ak
import pytest
from alkahest.controls import (
    ControlsError,
    StateSpace,
    TransferFunction,
    bode,
    care,
    controls_side_conditions,
    kronecker,
    lqr,
    lyapunov,
    margins,
)


@pytest.fixture
def P():
    pool = ak.ExprPool()
    names = "s t k b m a K r T w z"
    syms = {n: pool.symbol(n) for n in names.split()}
    syms["pool"] = pool
    syms["I"] = pool.integer
    return syms


def ev(e, bind=None):
    """Numeric value of an expression (complex evaluation, real part checked)."""
    v = ak.evaluate(e, bind or {}, mode="complex").value
    v = complex(v)
    assert abs(v.imag) <= 1e-9 * (1 + abs(v.real)), v
    return v.real


def mat_vals(M, bind=None):
    return [[ev(x, bind) for x in row] for row in M.to_list()]


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * (1 + abs(b))


def all_close(A, B, tol=1e-9):
    return all(close(x, y, tol) for ra, rb in zip(A, B) for x, y in zip(ra, rb))


# ---------------------------------------------------------------------------
# Matrix.rank side conditions (report: rank() silently assumed a ≠ 0)
# ---------------------------------------------------------------------------


def test_symbolic_rank_states_generic_hypothesis(P):
    a = P["a"]
    assert ak.Matrix([[a]]).rank() == 1
    assert ak.matrix_rank_side_conditions() == ["a ≠ 0"]
    # A rational matrix carries no hypothesis, and does not inherit a stale one.
    assert ak.Matrix([[P["I"](2), P["I"](0)], [P["I"](0), P["I"](3)]]).rank() == 2
    assert ak.matrix_rank_side_conditions() == []
    # The rank really is only generic: at a = 0 it drops.
    assert ak.Matrix([[P["I"](0)]]).rank() == 0


def test_controllability_rank_report_repro(P):
    """Probe C6: [B AB] for A=[[0,1],[-k,-b]], B=[0,a]ᵀ has rank 2 *for a ≠ 0*."""
    N, k, b, a = P["I"], P["k"], P["b"], P["a"]
    sys = StateSpace([[N(0), N(1)], [-k, -b]], [[N(0)], [a]], [[N(1), N(0)]])
    assert sys.controllability_rank() == 2
    conds = controls_side_conditions()
    assert conds
    assert all("a" in c for c in conds)
    assert sys.is_observable()
    assert controls_side_conditions() == []  # C = [1 0]: observability needs nothing


# ---------------------------------------------------------------------------
# TransferFunction: poles / zeros / DC gain / minreal
# ---------------------------------------------------------------------------


def test_poles_zeros_report_repro(P):
    """Probe C5: poles of (s²+3s+2)/(s³+6s²+11s+6) — solve refused degree 3."""
    s, N = P["s"], P["I"]
    G = TransferFunction(s**2 + 3 * s + 2, s**3 + 6 * s**2 + 11 * s + 6, s)
    assert [ev(p) for p in G.poles()] == [-3, -2, -1]
    assert [ev(z) for z in G.zeros()] == [-2, -1]
    Gm = G.minreal()
    assert [ev(p) for p in Gm.poles()] == [-3]
    assert Gm.zeros() == []
    # The cancelled modes are stated.
    G.minreal()
    conds = controls_side_conditions()
    assert len(conds) == 1
    assert "s" in conds[0]
    assert str(G.dc_gain()) == "1/3"
    _ = N


def test_poles_have_multiplicity_and_rootof(P):
    s = P["s"]
    G = TransferFunction(1, (s + 1) ** 2 * (s + 2), s)
    assert [ev(p) for p in G.poles()] == [-2, -1, -1]
    H = TransferFunction(1, s**3 + 2 * s + 7, s)
    poles = H.poles()
    assert len(poles) == 3
    for p in poles:
        v = complex(ak.evaluate(p, {}, mode="complex").value)
        assert abs(v**3 + 2 * v + 7) < 1e-9


def test_parametric_poles_and_leading_coefficient_condition(P):
    s, m, b, k = P["s"], P["m"], P["b"], P["k"]
    G = TransferFunction(1, m * s**2 + b * s + k, s)
    poles = G.poles()
    assert len(poles) == 2
    assert "m ≠ 0" in controls_side_conditions()
    bind = {m: 2.0, b: 3.0, k: 5.0}
    for p in poles:
        v = complex(ak.evaluate(p, bind, mode="complex").value)
        assert abs(2 * v * v + 3 * v + 5) < 1e-9


def test_dc_gain_infinite_refuses(P):
    s = P["s"]
    G = TransferFunction(1, s * (s + 1), s)
    with pytest.raises(ControlsError) as ei:
        G.dc_gain()
    assert ei.value.code == "E-CTRL-004"
    # a cancelled integrator does not count
    assert ev(TransferFunction(s, s * (s + 2), s).dc_gain()) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Interconnection
# ---------------------------------------------------------------------------


def test_series_parallel_feedback_closed_forms(P):
    s, K = P["s"], P["K"]
    G = TransferFunction(1, s * (s + 1), s)
    Hc = TransferFunction(K, 1, s)
    cl = (Hc * G).feedback()
    # K / (s² + s + K)
    assert cl.equals(TransferFunction(K, s**2 + s + K, s))
    assert [str(c) for c in cl.den_coeffs] == ["1", "1", "K"]
    par = G + TransferFunction(1, s, s)
    assert par.equals(TransferFunction(s + 2, s * (s + 1), s))
    pos = G.feedback(1, sign=1)
    assert pos.equals(TransferFunction(1, s**2 + s - 1, s))
    with pytest.raises(ControlsError):
        G.feedback(1, sign=2)


def test_pid_closed_loop_char_poly_probe_c15(P):
    s = P["s"]
    pool = P["pool"]
    kp, ki, kd = pool.symbol("kp"), pool.symbol("ki"), pool.symbol("kd")
    C = TransferFunction(kd * s**2 + kp * s + ki, s, s)
    G = TransferFunction(1, s * (s + 1), s)
    cl = (C * G).feedback()
    char = cl.den
    expect = s**3 + (1 + kd) * s**2 + kp * s + ki
    assert str(ak.cancel(char - expect)) == "0"


# ---------------------------------------------------------------------------
# State space <-> transfer function
# ---------------------------------------------------------------------------


def test_tf2ss_controllable_canonical_form(P):
    s, m, b, k = P["s"], P["m"], P["b"], P["k"]
    sys = TransferFunction(1, m * s**2 + b * s + k, s).to_ss()
    assert "m ≠ 0" in controls_side_conditions()
    bind = {m: 2.0, b: 3.0, k: 5.0}
    assert all_close(mat_vals(sys.A, bind), [[-1.5, -2.5], [1.0, 0.0]])
    assert all_close(mat_vals(sys.B, bind), [[1.0], [0.0]])
    assert all_close(mat_vals(sys.C, bind), [[0.0, 0.5]])
    assert all_close(mat_vals(sys.D, bind), [[0.0]])
    back = sys.to_tf(s)
    assert back.equals(TransferFunction(1, m * s**2 + b * s + k, s))


def test_ss2tf_report_repro_mass_spring_damper(P):
    """Probe C1: C(sI−A)⁻¹B for the mass–spring–damper is 1/(m s² + b s + k)."""
    N, s, m, b, k = P["I"], P["s"], P["m"], P["b"], P["k"]
    sys = StateSpace([[N(0), N(1)], [-k / m, -b / m]], [[N(0)], [1 / m]], [[N(1), N(0)]])
    G = sys.to_tf(s)
    assert G.equals(TransferFunction(1, m * s**2 + b * s + k, s))
    poles = sys.poles()
    assert len(poles) == 2


def test_biproper_tf2ss_and_back(P):
    s = P["s"]
    G = TransferFunction(2 * s**2 + 3 * s + 1, s**2 + 5 * s + 6, s)
    sys = G.to_ss()
    assert mat_vals(sys.D) == [[2.0]]
    # C = [b1 - a1 b0, b2 - a2 b0] = [3 - 10, 1 - 12]
    assert mat_vals(sys.C) == [[-7.0, -11.0]]
    assert sys.to_tf(s).equals(G)


def test_improper_and_static_refusals(P):
    s = P["s"]
    with pytest.raises(ControlsError) as ei:
        TransferFunction(s**2, s + 1, s).to_ss()
    assert ei.value.code == "E-CTRL-002"
    with pytest.raises(ControlsError) as ei:
        TransferFunction(s, s + 1, s).impulse_response(P["t"])
    assert ei.value.code == "E-CTRL-002"
    with pytest.raises(ControlsError) as ei:
        TransferFunction(ak.sin(s), s + 1, s)
    assert ei.value.code == "E-CTRL-001"


def test_mimo_transfer_matrix(P):
    N, s = P["I"], P["s"]
    sys = StateSpace(
        [[N(-1), N(0)], [N(0), N(-2)]],
        [[N(1), N(0)], [N(0), N(1)]],
        [[N(1), N(1)]],
    )
    row = sys.to_tf(s)
    assert len(row) == 1
    assert len(row[0]) == 2
    assert row[0][0].minreal().equals(TransferFunction(1, s + 1, s))
    assert row[0][1].minreal().equals(TransferFunction(1, s + 2, s))
    dc = sys.dc_gain()
    assert mat_vals(dc) == [[1.0, 0.5]]


def test_state_space_interconnection_matches_tf_algebra(P):
    N, s = P["I"], P["s"]
    G1 = TransferFunction(s + 3, s**2 + 3 * s + 2, s)
    G2 = TransferFunction(2, s + 5, s)
    S1, S2 = G1.to_ss(), G2.to_ss()
    assert S1.series(S2).to_tf(s).equals(G1 * G2)
    assert S1.parallel(S2).to_tf(s).equals(G1 + G2)
    assert S1.feedback(S2).to_tf(s).equals(G1.feedback(G2))
    assert S1.feedback().to_tf(s).equals(G1.feedback(1))
    assert S1.feedback(S2, sign=1).to_tf(s).equals(G1.feedback(G2, sign=1))
    _ = N


# ---------------------------------------------------------------------------
# Time responses
# ---------------------------------------------------------------------------


def test_step_response_second_order_closed_form(P):
    """Probe C12: underdamped step response against the textbook formula."""
    s, t, w, z = P["s"], P["t"], P["w"], P["z"]
    G = TransferFunction(w**2, s**2 + 2 * z * w * s + w**2, s)
    y = G.step_response(t)
    for zz, ww, tt in [(0.3, 2.0, 1.1), (0.5, 1.0, 0.4), (0.1, 3.0, 2.5)]:
        wd = ww * math.sqrt(1 - zz * zz)
        ref = 1 - math.exp(-zz * ww * tt) * (math.cos(wd * tt) + zz * ww / wd * math.sin(wd * tt))
        assert ev(y, {z: zz, w: ww, t: tt}) == pytest.approx(ref, rel=1e-10)


def test_impulse_and_step_rational(P):
    s, t = P["s"], P["t"]
    G = TransferFunction(1, (s + 1) * (s + 2), s)
    y = G.impulse_response(t)
    st = G.step_response(t)
    for tt in (0.1, 0.7, 3.0):
        assert ev(y, {t: tt}) == pytest.approx(math.exp(-tt) - math.exp(-2 * tt))
        assert ev(st, {t: tt}) == pytest.approx(0.5 - math.exp(-tt) + 0.5 * math.exp(-2 * tt))


def test_state_space_step_response(P):
    N, t = P["I"], P["t"]
    sys = StateSpace([[N(0), N(1)], [N(-2), N(-3)]], [[N(0)], [N(1)]], [[N(1), N(0)]])
    y = sys.step_response(t)
    for tt in (0.2, 1.0, 4.0):
        assert ev(y, {t: tt}) == pytest.approx(0.5 - math.exp(-tt) + 0.5 * math.exp(-2 * tt))


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------


def test_tustin_report_repro(P):
    """Probe C11: w/(s+w) under s = 2/T (z−1)/(z+1)."""
    s, w, T = P["s"], P["w"], P["T"]
    Gd = TransferFunction(w, s + w, s).discretize(T, "tustin")
    zz = Gd.var
    expect = w * T * (zz + 1) / ((w * T + 2) * zz + (w * T - 2))
    bind = {w: 1.7, T: 0.1, zz: 0.37}
    assert ev(Gd.expr, bind) == pytest.approx(ev(expect, bind))
    assert Gd.is_discrete


def test_zoh_symbolic_first_order(P):
    """ZOH of a/(s+a): Ad = e^{−aT}, Bd = (1 − e^{−aT})/a, C = a."""
    s, a, T = P["s"], P["a"], P["T"]
    sys = TransferFunction(a, s + a, s).to_ss()
    d = sys.discretize(T, "zoh")
    bind = {a: 1.3, T: 0.25}
    assert ev(d.A.get(0, 0), bind) == pytest.approx(math.exp(-1.3 * 0.25))
    assert ev(d.B.get(0, 0), bind) == pytest.approx((1 - math.exp(-1.3 * 0.25)) / 1.3)


def test_zoh_double_integrator_exact(P):
    N, T = P["I"], P["T"]
    sys = StateSpace([[N(0), N(1)], [N(0), N(0)]], [[N(0)], [N(1)]], [[N(1), N(0)]])
    d = sys.discretize(T, "zoh")
    assert [[str(x) for x in r] for r in d.A.to_list()] == [["1", "T"], ["0", "1"]]
    assert ev(d.B.get(0, 0), {T: 0.3}) == pytest.approx(0.045)
    assert ev(d.B.get(1, 0), {T: 0.3}) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Kronecker / Lyapunov
# ---------------------------------------------------------------------------


def test_kronecker_definition(P):
    N, a, b = P["I"], P["a"], P["b"]
    A = ak.Matrix([[a, N(0)], [N(1), b]])
    B = ak.Matrix([[N(1), N(2)]])
    Kr = kronecker(A, B)
    assert Kr.shape() == (2, 4)
    assert mat_vals(Kr, {a: 3.0, b: 5.0}) == [[3, 6, 0, 0], [1, 2, 5, 10]]


def test_lyapunov_symbolic_report_repro(P):
    """Probe C7: A X + X Aᵀ + I = 0 for A = [[0,1],[-k,-b]] (python-control lyap)."""
    N, k, b = P["I"], P["k"], P["b"]
    A = ak.Matrix([[N(0), N(1)], [-k, -b]])
    X = lyapunov(A, [[N(1), N(0)], [N(0), N(1)]])
    conds = controls_side_conditions()
    assert conds
    assert all(("k" in c and "b" in c) for c in conds)
    # Closed form: X = [[(k²+k+b²)/(2kb)... ]] — check the residual and symmetry
    for kk, bb in [(2.0, 3.0), (0.5, 1.7), (5.0, 0.3)]:
        Xv = mat_vals(X, {k: kk, b: bb})
        An = [[0.0, 1.0], [-kk, -bb]]
        res = [
            [
                sum(An[i][q] * Xv[q][j] + Xv[i][q] * An[j][q] for q in range(2)) + (i == j)
                for j in range(2)
            ]
            for i in range(2)
        ]
        assert all(abs(x) < 1e-12 for r in res for x in r)
        assert Xv[0][1] == pytest.approx(Xv[1][0])


def test_lyapunov_rational_exact_and_singular_refusal(P):
    N = P["I"]
    A = ak.Matrix([[N(-1), N(2)], [N(0), N(-3)]])
    X = lyapunov(A, [[N(1), N(0)], [N(0), N(1)]])
    assert controls_side_conditions() == []
    # exact rationals
    # exact rationals: 2/3, 1/12, 1/6 (hand-solved 3×3 linear system)
    assert [[str(x) for x in r] for r in X.to_list()] == [["2/3", "1/12"], ["1/12", "1/6"]]
    # A with eigenvalues ±1: λ₁ + λ₂ = 0 → singular operator
    with pytest.raises(ControlsError) as ei:
        lyapunov([[N(0), N(1)], [N(1), N(0)]], [[N(1), N(0)], [N(0), N(1)]])
    assert ei.value.code == "E-CTRL-004"


# ---------------------------------------------------------------------------
# Riccati / LQR
# ---------------------------------------------------------------------------


def _double_integrator(P):
    N = P["I"]
    A = ak.Matrix([[N(0), N(1)], [N(0), N(0)]])
    B = ak.Matrix([[N(0)], [N(1)]])
    Q = ak.Matrix([[N(1), N(0)], [N(0), N(1)]])
    return A, B, Q


def test_lqr_double_integrator_textbook(P):
    """Probe C8: symbolic r. Textbook: X = [[√2 r^¼·… ]], K = [r^-½, √(2 r^-½ + …)].

    For A = [[0,1],[0,0]], B = [0,1]ᵀ, Q = I, R = r the stabilising solution is
    x12 = √r, x22 = √(r(1 + 2√r)), x11 = x22/√r; K = [1/√r, √((1 + 2√r)/r)].
    """
    r = P["r"]
    A, B, Q = _double_integrator(P)
    K, X = lqr(A, B, Q, [[r]])
    conds = controls_side_conditions()
    assert any("r" in c and ">" in c for c in conds)
    for rv in (0.25, 1.0, 2.0, 9.0):
        x12 = math.sqrt(rv)
        x22 = math.sqrt(rv * (1 + 2 * math.sqrt(rv)))
        x11 = x22 / math.sqrt(rv)
        Xv = mat_vals(X, {r: rv})
        assert all_close(Xv, [[x11, x12], [x12, x22]])
        Kv = mat_vals(K, {r: rv})
        assert all_close(Kv, [[1 / math.sqrt(rv), math.sqrt((1 + 2 * math.sqrt(rv)) / rv)]])


def test_care_rational_double_integrator_exact(P):
    N = P["I"]
    A, B, Q = _double_integrator(P)
    K, X = lqr(A, B, Q, [[N(1)]])
    s3 = math.sqrt(3)
    assert all_close(mat_vals(X), [[s3, 1.0], [1.0, s3]])
    assert all_close(mat_vals(K), [[1.0, s3]])
    assert [str(x) for x in K.to_list()[0]] == ["1", "sqrt(3)"]


def test_care_no_stabilising_solution(P):
    N = P["I"]
    # unstable and uncontrollable mode: (A, B) not stabilisable
    with pytest.raises(ControlsError) as ei:
        care(
            [[N(1), N(0)], [N(0), N(-1)]], [[N(0)], [N(1)]], [[N(1), N(0)], [N(0), N(1)]], [[N(1)]]
        )
    assert ei.value.code == "E-CTRL-004"


# ---------------------------------------------------------------------------
# Frequency domain
# ---------------------------------------------------------------------------


def test_margins_exact_crossovers(P):
    """Probe C14: K/(s(s+1)(s+5)), K = 10: phase crossover √5, GM = 30/K = 3."""
    s = P["s"]
    G = TransferFunction(10, s * (s + 1) * (s + 5), s)
    mg = margins(G)
    assert mg.phase_crossover == pytest.approx(math.sqrt(5), rel=1e-14)
    assert mg.gain_margin == pytest.approx(3.0, rel=1e-12)
    # gain crossover: |G(jω)| = 1 ⇔ ω²(ω²+1)(ω²+25) = 100
    wc = mg.gain_crossover
    assert wc * wc * (wc * wc + 1) * (wc * wc + 25) == pytest.approx(100.0, rel=1e-12)
    g = 10 / (1j * wc * (1j * wc + 1) * (1j * wc + 5))
    assert mg.phase_margin == pytest.approx(math.degrees(cmath.phase(g)) % 360 - 180)
    assert mg.gain_margin_db == pytest.approx(20 * math.log10(3))


def test_margins_no_crossing_is_inf(P):
    s = P["s"]
    mg = margins(TransferFunction(1, s + 2, s))
    assert mg.gain_margin == math.inf
    assert mg.phase_margin == math.inf
    assert math.isnan(mg.gain_crossover)


def test_margins_and_bode_refuse_free_parameters(P):
    s, K = P["s"], P["K"]
    G = TransferFunction(K, s * (s + 1), s)
    with pytest.raises(ControlsError) as ei:
        margins(G)
    assert ei.value.code == "E-CTRL-003"
    with pytest.raises(ControlsError) as ei:
        bode(G, [1.0])
    assert ei.value.code == "E-CTRL-003"
    mg = margins(G.subs({K: 2}))
    assert mg.gain_margin == math.inf  # a type-1 second-order loop never reaches -180°


def test_bode_matches_direct_evaluation(P):
    s = P["s"]
    G = TransferFunction(s + 3, s**3 + 2 * s**2 + 4 * s + 1, s)
    ws = [0.01, 0.3, 1.0, 2.0, 50.0]
    mag, ph = bode(G, ws)
    prev = None
    for k, wv in enumerate(ws):
        g = (1j * wv + 3) / ((1j * wv) ** 3 + 2 * (1j * wv) ** 2 + 4j * wv + 1)
        assert mag[k] == pytest.approx(20 * math.log10(abs(g)))
        # phase is the unwrapped angle: equal mod 360 and continuous
        d = (ph[k] - math.degrees(cmath.phase(g))) % 360
        assert min(d, 360 - d) == pytest.approx(0, abs=1e-9)
        if prev is not None:
            assert abs(ph[k] - prev) < 180
        prev = ph[k]


# ---------------------------------------------------------------------------
# Routh–Hurwitz vs pole real parts
# ---------------------------------------------------------------------------


def test_routh_condition_matches_pole_real_parts(P):
    """Probe C3b: s³+6s²+5s+K stable ⇔ 0 < K < 30; checked against the poles."""
    s, K = P["s"], P["K"]
    G = TransferFunction(K, s * (s + 1) * (s + 5), s).feedback()
    rh = ak.routh_hurwitz(G.den, s)
    first = rh["first_column"]
    for Kv in (0.5, 10.0, 29.0, 31.0, 100.0, -1.0):
        routh_stable = all(ev(c, {K: Kv}) > 0 for c in first)
        poles = G.subs({K: Fraction(Kv).limit_denominator(1000)}).poles()
        pole_stable = all(complex(ak.evaluate(p, {}, mode="complex").value).real < 0 for p in poles)
        assert routh_stable == pole_stable, Kv


# ---------------------------------------------------------------------------
# Oracle comparisons: scipy / python-control (skipped in the lean CI venv)
# ---------------------------------------------------------------------------


def _rand_stable_tf(rng, P, n):
    s = P["s"]
    N = P["I"]
    roots = [-rng.randint(1, 6) for _ in range(n)]
    den = [1]
    for r0 in roots:
        den = [a - r0 * b for a, b in zip([*den, 0], [0, *den])]
    num = [rng.randint(-4, 4) for _ in range(n)]
    num[0] = num[0] or 1
    G = TransferFunction.from_coeffs([N(c) for c in num], [N(c) for c in den], s)
    return G, num, den


def test_tf2ss_ss2tf_vs_scipy(P):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    rng = random.Random(7)
    s = P["s"]
    for n in (1, 2, 3, 4):
        G, num, den = _rand_stable_tf(rng, P, n)
        sys = G.to_ss()
        A, B, C, D = signal.tf2ss(num, den)
        assert np.allclose(mat_vals(sys.A), A)
        assert np.allclose(mat_vals(sys.B), B)
        assert np.allclose(mat_vals(sys.C), C)
        assert np.allclose(mat_vals(sys.D), D)
        back = sys.to_tf(s)
        bn, bd = signal.ss2tf(A, B, C, D)
        assert np.allclose([ev(c) for c in back.den_coeffs], bd)
        assert np.allclose(
            np.pad([ev(c) for c in back.num_coeffs], (len(bn[0]) - len(back.num_coeffs), 0)),
            bn[0],
        )
        poles = sorted(complex(ak.evaluate(p, {}, mode="complex").value).real for p in G.poles())
        assert np.allclose(poles, sorted(np.roots(den).real))


def test_step_impulse_vs_scipy(P):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    s, t, m, b, k = P["s"], P["t"], P["m"], P["b"], P["k"]
    G = TransferFunction(1, m * s**2 + b * s + k, s)
    y = G.step_response(t)
    h = G.impulse_response(t)
    ts = np.linspace(0.0, 6, 9)  # scipy simulates from T[0] as t = 0
    for mv, bv, kv in [(1.0, 0.6, 2.0), (2.0, 3.0, 1.0)]:
        bind = {m: mv, b: bv, k: kv}
        _, ys = signal.step(([1.0], [mv, bv, kv]), T=ts)
        _, hs = signal.impulse(([1.0], [mv, bv, kv]), T=ts)
        got = [ev(y, {**bind, t: float(tt)}) for tt in ts[1:]]
        goth = [ev(h, {**bind, t: float(tt)}) for tt in ts[1:]]
        assert np.allclose(got, ys[1:], rtol=1e-6, atol=1e-9)
        assert np.allclose(goth, hs[1:], rtol=1e-6, atol=1e-9)


def test_discretize_vs_scipy(P):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    N = P["I"]
    A = [[0, 1], [-2, -3]]
    B = [[0], [1]]
    C = [[1, 0]]
    sys = StateSpace(
        [[N(x) for x in r] for r in A],
        [[N(x) for x in r] for r in B],
        [[N(x) for x in r] for r in C],
    )
    for method in ("zoh", "bilinear"):
        d = sys.discretize(P["pool"].rational(1, 10), method)
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete(
            (np.array(A, float), np.array(B, float), np.array(C, float), np.zeros((1, 1))),
            0.1,
            method=method,
        )
        assert np.allclose(mat_vals(d.A), Ad)
        assert np.allclose(mat_vals(d.B), Bd)
        assert np.allclose(mat_vals(d.C), Cd)
        assert np.allclose(mat_vals(d.D), Dd)


def test_bode_vs_scipy(P):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    s = P["s"]
    G = TransferFunction(10 * (s + 2), s * (s + 1) * (s**2 + s + 9), s)
    ws = np.logspace(-2, 2, 40)
    mag, ph = bode(G, ws)
    _, smag, sph = signal.bode(([10, 20], [1, 2, 10, 9, 0]), w=ws)
    assert np.allclose(mag, smag)
    assert np.allclose(ph, sph)


def test_margins_vs_python_control(P):
    np = pytest.importorskip("numpy")
    control = pytest.importorskip("control")
    s, N = P["s"], P["I"]
    cases = [
        ([10], [1, 6, 5, 0]),
        ([4, 8], [1, 3, 2, 0.5]),
        ([1, 2, 5], [1, 4, 6, 4, 1]),
        ([50], [1, 9, 23, 15]),
    ]
    for num, den in cases:
        G = TransferFunction.from_coeffs(
            [
                N(int(c))
                if float(c).is_integer()
                else P["pool"].rational(*Fraction(c).as_integer_ratio())
                for c in num
            ],
            [
                N(int(c))
                if float(c).is_integer()
                else P["pool"].rational(*Fraction(c).as_integer_ratio())
                for c in den
            ],
            s,
        )
        gm, pm, wcg, wcp = control.margin(control.tf(num, den))
        mg = margins(G)
        for got, ref in zip(mg, (gm, pm, wcg, wcp)):
            if np.isinf(ref) or np.isnan(ref):
                assert (math.isinf(got) and np.isinf(ref)) or (math.isnan(got) and np.isnan(ref))
            else:
                assert got == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_lyapunov_vs_scipy_random_points(P):
    np = pytest.importorskip("numpy")
    sl = pytest.importorskip("scipy.linalg")
    N, a, b, k = P["I"], P["a"], P["b"], P["k"]
    A = ak.Matrix([[-a, N(1), N(0)], [N(0), -b, N(1)], [-k, N(0), -a - b]])
    Q = ak.Matrix([[N(2), N(1), N(0)], [N(1), N(3), N(0)], [N(0), N(0), N(1)]])
    X = lyapunov(A, Q)
    rng = random.Random(3)
    for _ in range(4):
        bind = {a: rng.uniform(0.5, 3), b: rng.uniform(0.5, 3), k: rng.uniform(0.1, 2)}
        An = np.array(mat_vals(A, bind))
        Qn = np.array(mat_vals(Q))
        ref = sl.solve_continuous_lyapunov(An, -Qn)
        assert np.allclose(mat_vals(X, bind), ref)
    # numeric path
    An = np.array([[-1.0, 2.0], [0.5, -3.0]])
    Xn = lyapunov(
        ak.Matrix([[P["pool"].float(x) for x in r] for r in An.tolist()]),
        [[N(1), N(0)], [N(0), N(1)]],
    )
    assert np.allclose(Xn, sl.solve_continuous_lyapunov(An, -np.eye(2)))


def test_care_vs_scipy(P):
    np = pytest.importorskip("numpy")
    sl = pytest.importorskip("scipy.linalg")
    r = P["r"]
    A, B, Q = _double_integrator(P)
    X = care(A, B, Q, [[r]])
    for rv in (0.3, 2.0, 7.5):
        ref = sl.solve_continuous_are(
            np.array([[0, 1], [0, 0.0]]), np.array([[0], [1.0]]), np.eye(2), np.array([[rv]])
        )
        assert np.allclose(mat_vals(X, {r: rv}), ref)
    # rational 2×2 with a coupled A: exact radicals, same X as scipy
    N = P["I"]
    Xr = care(
        [[N(1), N(2)], [N(-1), N(-3)]],
        [[N(0)], [N(1)]],
        [[N(2), N(0)], [N(0), N(1)]],
        [[N(3)]],
    )
    ref = sl.solve_continuous_are(
        np.array([[1, 2], [-1, -3.0]]), np.array([[0], [1.0]]), np.diag([2, 1.0]), np.array([[3.0]])
    )
    assert np.allclose(mat_vals(Xr), ref)
    # numeric path on a random 4-state system
    rng = np.random.default_rng(5)
    An = rng.normal(size=(4, 4))
    Bn = rng.normal(size=(4, 2))
    Qn = np.eye(4)
    Rn = np.eye(2)
    flt = P["pool"].float

    def m(M):
        return ak.Matrix([[flt(float(x)) for x in row] for row in M])

    Xn = care(m(An), m(Bn), m(Qn), m(Rn))
    assert np.allclose(Xn, sl.solve_continuous_are(An, Bn, Qn, Rn), atol=1e-8)
    Kn, _ = lqr(m(An), m(Bn), m(Qn), m(Rn))
    assert np.all(np.linalg.eigvals(An - Bn @ Kn).real < 0)


def test_kronecker_vs_numpy(P):
    np = pytest.importorskip("numpy")
    N = P["I"]
    A = [[1, 2], [3, 4], [5, 6]]
    B = [[0, 1, 2], [-1, 3, 1]]
    Kr = kronecker([[N(x) for x in r] for r in A], [[N(x) for x in r] for r in B])
    assert np.allclose(mat_vals(Kr), np.kron(A, B))


def test_minreal_states_parameter_content(P):
    """``m·s/(m·s²)`` is ``1/s`` only for ``m ≠ 0`` — the hypothesis is stated."""
    s, m = P["s"], P["m"]
    G = TransferFunction(m * s, m * s**2, s).minreal()
    assert G.equals(TransferFunction(1, s, s))
    conds = controls_side_conditions()
    assert any("m" in c for c in conds)
    assert any("s" in c for c in conds)


def test_care_scalar_symbolic_branch_with_sample(P):
    """Scalar ARE 2aX − X² + 1 = 0: stabilising X = a + √(a² + 1) for every real a."""
    N, a = P["I"], P["a"]
    for sample in (None, {a: -3}):
        X = care([[a]], [[N(1)]], [[N(1)]], [[N(1)]], sample=sample)
        for av in (-3.0, 0.5, 2.0):
            assert ev(X.get(0, 0), {a: av}) == pytest.approx(av + math.sqrt(av * av + 1))


def test_tf_zoh_discretize_vs_scipy(P):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    s = P["s"]
    G = TransferFunction(3, s**2 + 4 * s + 3, s)
    Gd = G.discretize(P["pool"].rational(1, 5), "zoh")
    num, den, _ = signal.cont2discrete(([3.0], [1.0, 4.0, 3.0]), 0.2, method="zoh")
    lc = ev(Gd.den_coeffs[0])
    assert np.allclose([ev(c) / lc for c in Gd.den_coeffs], den)
    got_num = [ev(c) / lc for c in Gd.num_coeffs]
    ref_num = np.trim_zeros(np.ravel(num), "f")
    assert np.allclose(got_num, ref_num)
