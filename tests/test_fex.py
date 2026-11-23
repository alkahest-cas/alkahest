"""``alkahest.fex`` — the Finite Expression Method evaluation engine.

Checks, from Python:

* the fused jets ``u, grad u, lap u`` and the residual Jacobian against
  symbolic :func:`alkahest.diff` of :meth:`Template.instantiate` and against
  finite differences;
* batch evaluation against single-candidate calls;
* canonical keys (commutativity, id chains, 0-padding) and their soundness;
* overflow / division guards (``+inf``, never NaN);
* Levenberg–Marquardt recovery of the FEX paper's Poisson and conservation
  law solutions to float64 precision, and an end-to-end ``search``;
* ``E-FEX-*`` error codes.
"""

from __future__ import annotations

import math

import alkahest as ak
import numpy as np
import pytest
from alkahest import fex


def _rng(seed=0):
    return np.random.default_rng(seed)


def _poisson(d):
    return fex.Problem.poisson(
        d,
        rhs=-d,
        bc=lambda x: 0.5 * sum(xi**2 for xi in x),
        exact=lambda p: 0.5 * (p**2).sum(axis=1),
    )


def _evaluator(d, n=200, nb=100, seed=0, shape="depth2_sub"):
    rng = _rng(seed)
    tpl = fex.Template(shape, dim=d)
    return fex.Evaluator(
        tpl,
        _poisson(d),
        fex.sample_box(n, d, rng=rng),
        fex.sample_box_boundary(nb, d, rng=rng),
    )


def test_template_basics():
    tpl = fex.Template("depth2_sub", dim=4)
    assert tpl.shape == "U(B(L,L))"
    assert tpl.slot_kinds == ["leaf", "binary", "leaf", "unary"]
    assert tpl.n_params == 12
    assert tpl.slot_arities == [9, 3, 9, 9]
    assert tpl.n_skeletons == 2187
    assert tpl.describe([3, 0, 0, 2]) == "id((L[sq] + L[0]))"
    assert fex.Template.binary_tree(2, dim=3).shape == "B(U(B(L,L)),U(B(L,L)))"


def test_jets_match_symbolic_diff_of_instantiate():
    d = 3
    rng = _rng(1)
    tpl = fex.Template("depth2", dim=d)
    ev = fex.Evaluator(tpl, _poisson(d), fex.sample_box(4, d, rng=rng))
    xs = tpl.coords()
    pts = fex.sample_box(3, d, rng=rng)
    for _ in range(5):
        e = tpl.random_skeletons(1, rng)[0]
        theta = rng.uniform(-0.6, 0.6, tpl.n_params)
        u_expr = tpl.instantiate(e, theta, xs)
        u, g, lap = ev.jets(e, theta, pts)
        grads = [ak.diff(u_expr, x).value for x in xs]
        lap_expr = sum(ak.diff(gi, x).value for gi, x in zip(grads, xs))
        for p in range(len(pts)):
            b = {x: float(v) for x, v in zip(xs, pts[p])}
            assert math.isclose(u[p], ak.eval_expr(u_expr, b), rel_tol=1e-10, abs_tol=1e-12)
            for i in range(d):
                assert math.isclose(
                    g[p, i], ak.eval_expr(grads[i], b), rel_tol=1e-9, abs_tol=1e-10
                )
            assert math.isclose(lap[p], ak.eval_expr(lap_expr, b), rel_tol=1e-9, abs_tol=1e-9)


def test_jacobian_matches_finite_differences():
    ev = _evaluator(4, n=40, nb=20)
    tpl = ev.template
    rng = _rng(2)
    e = [7, 1, 3, 8]
    theta = rng.uniform(-0.5, 0.5, tpl.n_params)
    r, j = ev.residual_and_jacobian(e, theta)
    assert r.shape == (60,) and j.shape == (60, tpl.n_params)
    h = 1e-6
    for k in range(tpl.n_params):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        fd = (ev.residual_and_jacobian(e, tp, False)[0] - ev.residual_and_jacobian(e, tm, False)[0]) / (
            2 * h
        )
        np.testing.assert_allclose(j[:, k], fd, rtol=1e-6, atol=1e-7)
    assert math.isclose(ev.loss(e, theta), float(r @ r), rel_tol=1e-12)
    loss, g = ev.loss_and_grad(e, theta)
    np.testing.assert_allclose(g, 2 * j.T @ r, rtol=1e-10, atol=1e-12)


def test_batch_matches_single_and_accepts_dlpack():
    ev = _evaluator(3, n=150, nb=40, shape="depth2")
    tpl = ev.template
    rng = _rng(3)
    E = tpl.random_skeletons(6, rng)
    Th = tpl.init_params(6, rng)
    R, J = ev.residual_and_jacobian(E, Th)
    assert R.shape == (6, 190) and J.shape == (6, 190, tpl.n_params)
    for c in range(6):
        r, j = ev.residual_and_jacobian(E[c], Th[c])
        np.testing.assert_array_equal(R[c], r)
        np.testing.assert_array_equal(J[c], j)
    # numpy arrays export __dlpack__; a DLPack producer is accepted directly
    losses = ev.loss(E, Th)
    np.testing.assert_allclose(losses, (R**2).sum(axis=1), rtol=1e-12)


def test_canonical_keys():
    tpl = fex.Template("depth2_sub", dim=3)
    k = fex.canonical_key
    assert k(tpl, [3, 0, 7, 6]) == k(tpl, [7, 0, 3, 6])  # + commutes
    assert k(tpl, [3, 1, 7, 6]) == k(tpl, [7, 1, 3, 6])  # * commutes
    assert k(tpl, [3, 0, 0, 2]) == "L[sq]"  # id chain + 0 padding
    assert k(tpl, [0, 1, 1, 6]) == "c"
    assert k(tpl, [2, 0, 0, 8]) == k(tpl, [2, 0, 0, 7])  # cos(shift-closed) = sin
    assert k(tpl, [3, 1, 7, 2]) != k(tpl, [3, 0, 7, 2])
    reps = tpl.canonical_skeletons()
    assert len(reps) < 700
    assert len({key for key, _ in reps}) == len(reps)


def test_same_function_is_proved_exactly():
    tpl = fex.Template("depth2_sub", dim=2)
    th1 = [0.5, 0.5, 0, 0, 0, 0, 1, 0]
    th2 = [0, 0, 0, 0.5, 0.5, 0, 1, 0]
    assert tpl.same_function([3, 0, 0, 2], th1, [0, 0, 3, 2], th2) is True
    assert tpl.same_function([3, 0, 0, 2], th1, [0, 0, 3, 2], [0, 0, 0, 0.25, 0.5, 0, 1, 0]) is not True


def test_overflow_guards_give_inf_not_nan():
    d = 2
    rng = _rng(4)
    tpl = fex.Template("U(B(L,L))", unary=["0", "id", "exp", "log"], binary=["+", "/"], dim=d)
    ev = fex.Evaluator(tpl, _poisson(d), fex.sample_box(20, d, rng=rng), fex.sample_box_boundary(5, d, rng=rng))
    cases = [
        ([2, 0, 0, 2], [500.0, 500.0, 0, 0, 0, 0, 1, 0]),
        ([0, 1, 0, 1], [0.0] * 8),
        ([0, 0, 0, 3], [0, 0, -1.0, 0, 0, -1.0, 1, 0]),
    ]
    for e, th in cases:
        r, j = ev.residual_and_jacobian(e, th)
        assert np.all(np.isposinf(r))
        assert np.all(j == 0)
        assert ev.loss(e, th) == math.inf
        res = fex.fit(ev, e, th, rng=rng)
        assert not math.isnan(res.loss)


def test_lm_recovers_poisson_to_float64_precision():
    d = 10
    ev = _evaluator(d, n=400, nb=200, seed=5)
    res = fex.fit(ev, [3, 0, 0, 2], rng=_rng(5))
    assert res.loss < 1e-24
    test = fex.sample_box(2000, d, rng=_rng(6))
    assert ev.relative_l2(res.e, res.theta, points=test) < 1e-11


def test_lm_recovers_conservation_law():
    d = 5
    c = math.pi / 4
    coef = [(d - 1) * c] + [-1.0] * (d - 1)
    rng = _rng(7)
    prob = fex.Problem.transport(
        coef,
        bc=lambda x: ak.sin(c * sum(x[1:])),
        time=0,
        exact=lambda p: np.sin(p[:, 0] + c * p[:, 1:].sum(axis=1)),
    )
    tpl = fex.Template("depth2_sub", dim=d)
    ev = fex.Evaluator(
        tpl,
        prob,
        fex.sample_box(400, d, 0.0, 1.0, rng=rng),
        fex.sample_initial_slice(200, d, 0.0, 1.0, rng=rng),
    )
    res = fex.fit(ev, [2, 0, 0, 7], restarts=8, rng=rng)
    assert res.loss < 1e-22
    assert ev.relative_l2(res.e, res.theta) < 1e-10


def test_expression_operator_and_rayleigh():
    d = 3
    jet = fex.Jet(d)
    prob = fex.Problem.from_operator(jet, lambda j: -j.lap, rhs=-d, bc=lambda x: 0.5 * sum(v**2 for v in x))
    assert prob.operator_kind == "linear"
    nls = fex.Problem.from_operator(
        jet, lambda j: -j.lap + j.u**3 + sum(ak.cos(x) for x in j.x) * j.u, bc_weight=0.0
    )
    assert nls.operator_kind == "compiled" and nls.order == 2
    rng = _rng(8)
    tpl = fex.Template("depth2_sub", dim=d)
    pts = fex.sample_box(100, d, rng=rng)
    builtin = fex.Problem.cubic_schrodinger(
        d, potential=lambda x: sum(ak.cos(v) for v in x), loss="least_squares", bc_weight=0.0, jet=jet
    )
    e1 = fex.Evaluator(tpl, nls, pts)
    e2 = fex.Evaluator(tpl, builtin, pts)
    th = tpl.init_params(1, rng)[0]
    r1, j1 = e1.residual_and_jacobian([7, 1, 3, 6], th)
    r2, j2 = e2.residual_and_jacobian([7, 1, 3, 6], th)
    np.testing.assert_allclose(r1, r2, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(j1, j2, rtol=1e-11, atol=1e-13)
    eig = fex.Problem.eigen(d, potential=lambda x: sum(ak.cos(v) for v in x))
    ev = fex.Evaluator(tpl, eig, pts, fex.sample_box_boundary(20, d, rng=rng))
    loss, g = ev.loss_and_grad([8, 0, 3, 6], th)
    assert math.isfinite(loss) and g.shape == (tpl.n_params,)
    with pytest.raises(fex.FexError) as ei:
        ev.residual_and_jacobian([8, 0, 3, 6], th)
    assert ei.value.code == "E-FEX-008"


def test_search_recovers_poisson_structure():
    d = 5
    rng = _rng(9)
    tpl = fex.Template("depth2_sub", dim=d)
    res = fex.search(
        tpl,
        _poisson(d),
        fex.sample_box(300, d, rng=rng),
        fex.sample_box_boundary(150, d, rng=rng),
        epochs=60,
        target_loss=1e-20,
        rng=rng,
    )
    assert res.loss < 1e-20
    # The fitted function is exactly ½Σx² once its parameters are snapped.
    true_theta = [0.5] * d + [0.0] + [0.0] * (d + 1) + [1.0, 0.0]
    assert tpl.same_function(res.e, res.theta, [3, 0, 0, 2], true_theta) is True
    assert res.unique_skeletons <= res.candidates_scored


def test_error_codes():
    with pytest.raises(fex.FexError) as ei:
        fex.Template("U(B(L,L)", dim=2)
    assert ei.value.code == "E-FEX-001"
    with pytest.raises(fex.FexError) as ei:
        fex.Template("L", unary=["tanh"], dim=2)
    assert ei.value.code == "E-FEX-002"
    tpl = fex.Template("depth2_sub", dim=2)
    with pytest.raises(fex.FexError) as ei:
        tpl.canonical_key([0, 5, 0, 0])
    assert ei.value.code == "E-FEX-003"
    with pytest.raises(fex.FexError) as ei:
        tpl.instantiate([0, 0, 0, 0], [0.0])
    assert ei.value.code == "E-FEX-004"
    with pytest.raises(fex.FexError) as ei:
        fex.Evaluator(tpl, _poisson(2), np.zeros((4, 3)))
    assert ei.value.code == "E-FEX-005"
    with pytest.raises(fex.FexError) as ei:
        fex.Evaluator(tpl, _poisson(3), np.zeros((4, 2)))
    assert ei.value.code in ("E-FEX-005", "E-FEX-007")
    jet = fex.Jet(2)
    y = jet.pool.symbol("y")
    with pytest.raises(fex.FexError) as ei:
        fex.Problem.from_operator(jet, y * jet.u)
    assert ei.value.code == "E-FEX-006"
