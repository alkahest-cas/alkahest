"""G9 — real-form output.

Oscillatory ``matrix_exp``, ``dsolve`` and inverse-Laplace results used to come
back written through complex eigenvalues — ``exp(√(−4w²t²)/2)``,
``cos(t·√(k₁₀k₂₁ − (k₁₀+k₁₂+k₂₁)²/4))`` — which real-mode ``eval_expr``
refused (E-EVAL-009) and the JIT evaluated to NaN.  They now come back in real
form, and ``to_real_form`` exposes the rewrite.

Every closed form is checked numerically against an independent oracle
(``scipy.linalg.expm``, textbook oscillator formulas, SymPy) at random
parameters, and against complex-mode evaluation of the result itself.
"""

from __future__ import annotations

import math
import random

import alkahest as ak
import pytest
from alkahest import experimental as ex

np = pytest.importorskip("numpy")
sl = pytest.importorskip("scipy.linalg")

RNG = random.Random(9)


def _no_imaginary(e) -> bool:
    s = str(e)
    return "I" not in s.replace("Integer", "")


def _real_and_complex_agree(e, env, tol=1e-10):
    """Real-mode value equals complex-mode value, whose imaginary part is ~0."""
    r = ak.eval_expr(e, env)
    c = ak.evaluate(e, env, mode="complex").value
    c = complex(c)
    scale = 1.0 + abs(r)
    assert abs(c.imag) < tol * scale, (e, env, c)
    assert abs(c.real - r) < tol * scale, (e, env, c, r)
    return r


# ---------------------------------------------------------------------------
# matrix_exp
# ---------------------------------------------------------------------------


def test_matrix_exp_oscillator_is_real():
    p = ak.ExprPool()
    t, w = p.symbol("t"), p.symbol("w")
    i0 = p.integer(0)
    E = ak.Matrix([[i0, t], [-(w**2) * t, i0]]).matrix_exp()
    for i in range(2):
        for j in range(2):
            assert _no_imaginary(E.get(i, j)), E
            assert "sqrt" not in str(E.get(i, j)), E
    for _ in range(4):
        wv, tv = RNG.uniform(-3, 3), RNG.uniform(0.05, 2)
        ref = sl.expm(np.array([[0.0, tv], [-(wv**2) * tv, 0.0]]))
        env = {w: wv, t: tv}
        got = [[_real_and_complex_agree(E.get(i, j), env) for j in range(2)] for i in range(2)]
        assert np.allclose(got, ref, atol=1e-10), (got, ref)
    # The JIT agrees too (no NaN).
    f = ak.compile_expr(E.get(0, 1), [w, t])
    assert abs(f([2.0, 0.7]) - sl.expm(np.array([[0, 0.7], [-2.8, 0]]))[0, 1]) < 1e-10


def test_matrix_exp_oscillator_closed_form_under_positivity():
    p = ak.ExprPool()
    t = p.symbol("t", ak.Domain.Positive)
    w = p.symbol("w", ak.Domain.Positive)
    i0 = p.integer(0)
    E = ak.Matrix([[i0, t], [-(w**2) * t, i0]]).matrix_exp()
    wt = w * t
    expected = [[ak.cos(wt), ak.sin(wt) / w], [-w * ak.sin(wt), ak.cos(wt)]]
    for i in range(2):
        for j in range(2):
            assert ak.equals(E.get(i, j), expected[i][j]).value is True, (i, j, E.get(i, j))


# ---------------------------------------------------------------------------
# Inverse Laplace: two-compartment PK, step responses
# ---------------------------------------------------------------------------


def _pk_pool():
    pool = ak.ExprPool()
    P = ak.Domain.Positive
    t, s = pool.symbol("t"), pool.symbol("s")
    k12, k21, k10 = (pool.symbol(n, P) for n in ("k12", "k21", "k10"))
    return pool, t, s, k10, k12, k21


def test_two_compartment_iv_is_biexponential():
    _pool, t, s, k10, k12, k21 = _pk_pool()
    G = (s + k21) / ((s + k10 + k12) * (s + k21) - k12 * k21)
    g = ex.inverse_laplace_transform(G, s, t)
    assert _no_imaginary(g)
    assert "cos(" not in str(g), g
    assert "sin(" not in str(g), g
    # No hypothesis left: the discriminant is proved positive.
    assert not any(">" in c for c in ex.transform_side_conditions())
    f = ak.compile_expr(g, [k10, k12, k21, t])
    for _ in range(5):
        a, b, c, tv = (RNG.uniform(0.05, 2) for _ in range(4))
        ref = sl.expm(np.array([[-(a + b), c], [b, -c]]) * tv)[0, 0]
        env = {k10: a, k12: b, k21: c, t: tv}
        assert abs(_real_and_complex_agree(g, env) - ref) < 1e-10
        assert abs(f([a, b, c, tv]) - ref) < 1e-10
    code = ak.emit_c_expr(g, [k10, k12, k21, t])
    # The function itself; a complex-fallback runtime (N11) may precede it.
    assert "cos(" not in code.split("double f(")[-1]


def test_two_compartment_oral_is_real():
    pool, t, s, k10, k12, k21 = _pk_pool()
    ka = pool.symbol("ka", ak.Domain.Positive)
    G = (s + k21) / ((s + k10 + k12) * (s + k21) - k12 * k21)
    g = ex.inverse_laplace_transform(ka * G / (s + ka), s, t)
    assert _no_imaginary(g)
    env = {k10: 0.37, k12: 0.81, k21: 0.23, ka: 1.1, t: 1.7}
    A = np.array([[-1.1, 0, 0], [1.1, -(0.37 + 0.81), 0.23], [0, 0.81, -0.23]])
    ref = sl.expm(A * 1.7)[1, 0]
    assert abs(_real_and_complex_agree(g, env) - ref) < 1e-10
    f = ak.compile_expr(g, [k10, k12, k21, ka, t])
    assert abs(f([0.37, 0.81, 0.23, 1.1, 1.7]) - ref) < 1e-10


def _second_order_step(assume):
    p = ak.ExprPool()
    s, t, z, w = (p.symbol(n) for n in ("s", "t", "z", "w"))
    A = ak.Assumptions(p)
    for pred in assume(p, z, w):
        A.refine(pred)
    F = w**2 / (s * (s**2 + 2 * z * w * s + w**2))
    with ak.context(assumptions=A):
        y = ex.inverse_laplace_transform(F, s, t)
    return y, z, w, t


def test_step_response_underdamped():
    y, z, w, t = _second_order_step(
        lambda p, z, w: [p.gt(z, p.integer(0)), p.lt(z, p.integer(1)), p.gt(w, p.integer(0))]
    )
    assert _no_imaginary(y)
    for _ in range(4):
        zv, wv, tv = RNG.uniform(0.05, 0.95), RNG.uniform(0.2, 3), RNG.uniform(0, 4)
        wd = wv * math.sqrt(1 - zv * zv)
        ref = 1 - math.exp(-zv * wv * tv) * (math.cos(wd * tv) + zv * wv / wd * math.sin(wd * tv))
        assert abs(_real_and_complex_agree(y, {z: zv, w: wv, t: tv}) - ref) < 1e-10


def test_step_response_overdamped():
    y, z, w, t = _second_order_step(lambda p, z, w: [p.gt(z, p.integer(1)), p.gt(w, p.integer(0))])
    assert _no_imaginary(y)
    assert "cosh" in str(y) or "exp" in str(y)
    for _ in range(4):
        zv, wv, tv = RNG.uniform(1.05, 3), RNG.uniform(0.2, 3), RNG.uniform(0, 4)
        wd = wv * math.sqrt(zv * zv - 1)
        ref = 1 - math.exp(-zv * wv * tv) * (math.cosh(wd * tv) + zv * wv / wd * math.sinh(wd * tv))
        assert abs(_real_and_complex_agree(y, {z: zv, w: wv, t: tv}) - ref) < 1e-9


def test_step_response_unknown_damping_keeps_side_condition():
    # Neither branch can be chosen: the oscillatory form is returned with the
    # hypothesis that makes it the real-valued answer, as before.
    p = ak.ExprPool()
    s, t, z, w = (p.symbol(n) for n in ("s", "t", "z", "w"))
    ex.inverse_laplace_transform(w**2 / (s * (s**2 + 2 * z * w * s + w**2)), s, t)
    assert any(">" in c for c in ex.transform_side_conditions())


def test_controls_step_response_is_real():
    p = ak.ExprPool()
    s, t = p.symbol("s"), p.symbol("t")
    k = p.symbol("k", ak.Domain.Positive)
    G = ak.controls.TransferFunction(k**2, s**2 + k**2, s)
    y = G.step_response(t)
    assert _no_imaginary(y)
    for kv, tv in [(1.3, 0.7), (0.4, 2.2)]:
        ref = 1 - math.cos(kv * tv)
        assert abs(_real_and_complex_agree(y, {k: kv, t: tv}) - ref) < 1e-10


# ---------------------------------------------------------------------------
# dsolve
# ---------------------------------------------------------------------------


def _dsolve(eq_fn, assume=None):
    q = ak.ExprPool()
    x, k = q.symbol("x"), q.symbol("k")
    y, y1, y2 = (q.symbol(n) for n in ("y", "y1", "y2"))
    args = [eq_fn(q, y, y1, y2, k), x, y, [y1, y2]]
    if assume is not None:
        A = ak.Assumptions(q)
        for pred in assume(q, k):
            A.refine(pred)
        args.append(A)
    return ex.dsolve(*args)[0], x, k


def test_dsolve_buckling_with_positive_k():
    sol, x, k = _dsolve(lambda q, y, y1, y2, k: y2 + k**2 * y, lambda q, k: [q.gt(k, q.integer(0))])
    yx = sol["y_of_x"]
    s = str(yx)
    for part in ("cos", "sin"):
        assert part in s, s
    for part in ("sqrt", "exp"):
        assert part not in s, s
    assert not sol["side_conditions"], sol
    C1, C2 = sol["constants"]
    kv, xv = 1.7, 0.6
    ref = 0.3 * math.cos(kv * xv) - 1.1 * math.sin(kv * xv)
    v = ak.eval_expr(yx, {k: kv, x: xv, C1: 0.3, C2: -1.1})
    # The sin coefficient's sign is the solver's choice; accept either basis sign.
    alt = 0.3 * math.cos(kv * xv) + 1.1 * math.sin(kv * xv)
    assert min(abs(v - ref), abs(v - alt)) < 1e-12


def test_dsolve_harmonic_real_k_is_oscillatory_with_condition():
    sol, _x, _k = _dsolve(lambda q, y, y1, y2, k: y2 + k**2 * y)
    s = str(sol["y_of_x"])
    assert "cos" in s, s
    assert "sin" in s, s
    assert "exp" not in s, s
    assert sol["side_conditions"], sol  # k² ≠ 0 (k = 0 is the double root)


def _damped(zeta_assume):
    q = ak.ExprPool()
    x, z, w = q.symbol("x"), q.symbol("z"), q.symbol("w")
    y, y1, y2 = (q.symbol(n) for n in ("y", "y1", "y2"))
    A = ak.Assumptions(q)
    for pred in zeta_assume(q, z, w):
        A.refine(pred)
    sol = ex.dsolve(y2 + 2 * z * w * y1 + w**2 * y, x, y, [y1, y2], A)[0]
    return sol, x, z, w


@pytest.mark.parametrize(
    "case",
    ["under", "over"],
)
def test_dsolve_damped_oscillator_families(case):
    if case == "under":
        sol, x, z, w = _damped(
            lambda q, z, w: [q.gt(z, q.integer(0)), q.lt(z, q.integer(1)), q.gt(w, q.integer(0))]
        )
        assert "cos" in str(sol["y_of_x"])
        assert "sin" in str(sol["y_of_x"])
        zv = 0.3
    else:
        sol, x, z, w = _damped(lambda q, z, w: [q.gt(z, q.integer(1)), q.gt(w, q.integer(0))])
        assert "cos(" not in str(sol["y_of_x"])
        zv = 1.8
    assert _no_imaginary(sol["y_of_x"])
    assert not sol["side_conditions"], sol
    C1, C2 = sol["constants"]
    wv, xv = 1.4, 0.9
    # Check the ODE numerically by finite differences of the real value.
    env = {z: zv, w: wv, C1: 0.7, C2: -0.4}

    def Y(xx):
        return ak.eval_expr(sol["y_of_x"], {**env, x: xx})

    h = 1e-4
    ypp = (Y(xv + h) - 2 * Y(xv) + Y(xv - h)) / h**2
    yp = (Y(xv + h) - Y(xv - h)) / (2 * h)
    assert abs(ypp + 2 * zv * wv * yp + wv**2 * Y(xv)) < 1e-5


def test_dsolve_critically_damped():
    q = ak.ExprPool()
    x, w = q.symbol("x"), q.symbol("w", ak.Domain.Positive)
    y, y1, y2 = (q.symbol(n) for n in ("y", "y1", "y2"))
    sol = ex.dsolve(y2 + 2 * w * y1 + w**2 * y, x, y, [y1, y2])[0]
    assert "repeated" in sol["method"], sol


# ---------------------------------------------------------------------------
# to_real_form directly
# ---------------------------------------------------------------------------


def test_to_real_form_sqrt_of_square():
    p = ak.ExprPool()
    a = p.symbol("a")
    r = ak.to_real_form(ak.sqrt(4 * a**2))
    assert ak.equals(r, 2 * ak.abs(a)).value is True, r
    A = ak.Assumptions(p)
    A.refine(p.ge(a, p.integer(0)))
    assert ak.equals(ak.to_real_form(ak.sqrt(4 * a**2), A), 2 * a).value is True


def test_to_real_form_two_dof_radicand():
    sympy = pytest.importorskip("sympy")
    p = ak.ExprPool()
    k, m = p.symbol("k", ak.Domain.Positive), p.symbol("m", ak.Domain.Positive)
    r = ak.to_real_form(ak.sqrt(-4 * k**2 * m**2 + 9 * (m * k) ** 2))
    assert "abs" not in str(r)
    K, Mm = sympy.symbols("k m", positive=True)
    ref = sympy.sqrt(-4 * K**2 * Mm**2 + 9 * (Mm * K) ** 2)
    for kv, mv in [(1.3, 0.4), (2.0, 5.0)]:
        assert abs(ak.eval_expr(r, {k: kv, m: mv}) - float(ref.subs({K: kv, Mm: mv}))) < 1e-12


def test_to_real_form_trig_of_imaginary():
    p = ak.ExprPool()
    x = p.symbol("x")
    i = p.imaginary_unit()
    assert ak.equals(ak.to_real_form(ak.cosh(i * x)), ak.cos(x)).value is True
    # e^{ix} is genuinely complex: returned unchanged, never truncated.
    e = ak.exp(i * x)
    assert ak.to_real_form(e) == e


def test_to_real_form_conjugate_pair():
    p = ak.ExprPool()
    w, t = p.symbol("w"), p.symbol("t")
    rad = -4 * w**2 * t**2
    e = (ak.exp(ak.sqrt(rad) / 2) + ak.exp(-ak.sqrt(rad) / 2)) / 2
    r = ak.to_real_form(e)
    assert ak.equals(r, ak.cos(w * t)).value is True, r
