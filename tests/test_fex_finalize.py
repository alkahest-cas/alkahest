"""``alkahest.fex.finalize`` — exact, verified closed forms from FEX fits (F3).

Checks:

* coefficient processing: redundant pins, filtering, tied leaf weights,
  degenerate scales, snapping to rationals and to ``π/4`` (with PSLQ / LLL
  candidates), each kept only under the loss gate;
* lifting to symbolic ``d`` and exact verification for every ``d``:
  ``u = ½Σxᵢ²`` with ``−Δu = −d`` (and the boundary condition on every face of
  ``[−1, 1]^d``), ``u = sin(t + π/4·Σxᵢ)`` with ``(πd/4)·u_t − Σu_{xᵢ} = 0``,
  the cubic Schrödinger solution ``⅓·exp(Σcos xᵢ / d)``, and the harmonic
  ground state with eigenvalue ``λ = d``;
* explicit families (``E-FEX-010`` when one does not match the problem);
* the interval enclosure when nothing exact exists (never a proof);
* Lean certificate emission;
* :func:`recovery_metric` (skeleton equivalence);
* ``E-FEX-009`` … ``E-FEX-011``.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")

from alkahest import fex  # noqa: E402  (needs numpy)
from alkahest.experimental import Sum, laplacian_indexed  # noqa: E402

C = math.pi / 4


def _poisson(d):
    return fex.Problem.poisson(d, rhs=-d, bc=lambda x: 0.5 * sum(xi**2 for xi in x))


def _conservation(d):
    coef = [(d - 1) * C] + [-1.0] * (d - 1)
    return fex.Problem.transport(coef, bc=lambda x: ak.sin(C * sum(x[1:])), time=0)


def _fit(tpl, prob, e, theta, pts, bd, rng):
    ev = fex.Evaluator(tpl, prob, pts, bd)
    method = "lm" if prob.loss == "least_squares" else "bfgs"
    return fex.fit(ev, e, np.asarray(theta), method=method, restarts=1, max_iter=200, rng=rng)


def test_poisson_symbolic_d():
    d = 4
    rng = np.random.default_rng(0)
    tpl = fex.Template("depth2_sub", dim=d)
    prob = _poisson(d)
    pts = fex.sample_box(400, d, rng=rng)
    bd = fex.sample_box_boundary(200, d, rng=rng)
    # A degenerate representative: outer scale 2, leaf weights 1/4 (+ noise).
    theta = [0.25 + 1e-7 * k for k in range(d)] + [0.3] + [0.01] * (d + 1) + [2.0, 0.0]
    res = _fit(tpl, prob, [3, 0, 0, 2], theta, pts, bd, rng)
    out = fex.finalize(res, prob, template=tpl, points=pts, boundary=bd)
    assert out["exact"]
    x = prob.x
    assert ak.is_zero(out["expr"] - sum(xi**2 for xi in x) / 2) is True
    conc = out["verification"]["concrete"]
    assert conc["status"] == "exactly_verified"
    assert len(conc["bc"]) == 2 * d  # every face
    sym = out["verification"]["symbolic_d"]
    assert sym["status"] == "exactly_verified" and sym["verified"]
    assert sym["family"]["rhs"] == "(-1 * d)"
    assert "Sum(Indexed(x, i)^2, i, 1, d)" in str(out["expr_symbolic_d"])
    # The lifted solution really has Laplacian d (independently of finalize).
    fs = fex.family_symbols(prob)
    assert laplacian_indexed(out["expr_symbolic_d"], fs.x, 1, fs.d) == fs.d
    how = {p["how"] for p in out["snapped_params"]}
    assert "redundant" in how and "rational" in how
    assert out["certificate"]["status"] == "emitted"
    lean = out["certificate"]["lean"]
    assert "theorem fex_pde" in lean and "sorry" not in lean
    assert out["certificate"]["externally_verified"] is False


def test_conservation_snaps_pi_over_4_and_lifts():
    d = 3
    rng = np.random.default_rng(1)
    tpl = fex.Template("depth2_sub", dim=d)
    prob = _conservation(d)
    pts = fex.sample_box(600, d, 0.0, 1.0, rng)
    bd = fex.sample_initial_slice(200, d, 0.0, 1.0, 0, 0.0, rng)
    theta = [1.0] + [C + 1e-8] * (d - 1) + [0.0] + [0.0] * (d + 1) + [1.0, 0.0]
    res = _fit(tpl, prob, [2, 0, 0, 7], theta, pts, bd, rng)
    out = fex.finalize(res, prob, template=tpl, points=pts, boundary=bd)
    assert out["exact"]
    assert any(p["exact"] == "1/4*pi" for p in out["snapped_params"])
    t, xs = prob.x[0], prob.x[1:]
    assert ak.is_zero(out["expr"] - ak.sin(t + sum(xs) * C)) is not False
    assert out["verification"]["concrete"]["status"] == "exactly_verified"
    sym = out["verification"]["symbolic_d"]
    assert sym["verified"] and sym["family_inferred"]
    # The paper's family, given explicitly: (πd/4) u_t − Σ u_x = 0.
    fs = fex.family_symbols(prob)
    pi = fs.pool.pi()
    fam = {
        "time_coeff": pi * fs.d / 4,
        "space_coeff": -1,
        "bc": ak.sin(pi / 4 * fs.sum(lambda xi: xi)),
    }
    out2 = fex.finalize(res, prob, template=tpl, points=pts, boundary=bd, family=fam)
    sym2 = out2["verification"]["symbolic_d"]
    assert sym2["verified"] and not sym2["family_inferred"]
    assert "Sum(Indexed(x, i), i, 1, d)" in str(out2["expr_symbolic_d"])
    assert "d" not in str(out2["expr_symbolic_d"]).replace("Indexed", "").replace(", d)", "")


def test_wrong_family_is_refused():
    d = 3
    rng = np.random.default_rng(2)
    tpl = fex.Template("depth2_sub", dim=d)
    prob = _poisson(d)
    theta = [0.5] * d + [0.0] + [0.0] * (d + 1) + [1.0, 0.0]
    res = fex.FitResult([3, 0, 0, 2], np.asarray(theta), 0.0)
    fs = fex.family_symbols(prob)
    with pytest.raises(fex.FexError) as ei:
        fex.finalize(res, prob, template=tpl, family={"rhs": -2 * fs.d}, rng=rng)
    assert ei.value.code == "E-FEX-010"
    good = fex.finalize(
        res,
        prob,
        template=tpl,
        family={"rhs": -fs.d, "bc": fs.sum(lambda xi: xi**2 / 2)},
        rng=rng,
    )
    assert good["verification"]["symbolic_d"]["verified"]


def test_cubic_schrodinger_lifts_with_one_over_d():
    d = 3
    rng = np.random.default_rng(3)
    tpl = fex.Template("depth2_sub", dim=d)

    def V(x):
        S = sum(ak.cos(v) for v in x)
        return -ak.exp(2.0 / d * S) / 9 + sum(ak.sin(v) ** 2 for v in x) / d**2 - S / d

    prob = fex.Problem.cubic_schrodinger(d, potential=V)
    pts = fex.sample_box(400, d, rng=rng)
    cos = fex.PAPER_UNARY.index("cos")
    exp = fex.PAPER_UNARY.index("exp")
    theta = [1 / d + 1e-6] * d + [0.0] + [0.0] * (d + 1) + [1 / 3, 0.0]
    res = _fit(tpl, prob, [cos, 0, 0, exp], theta, pts, None, rng)
    out = fex.finalize(res, prob, template=tpl, points=pts)
    assert out["exact"]
    assert out["verification"]["concrete"]["status"] == "exactly_verified"
    sym = out["verification"]["symbolic_d"]
    assert sym["verified"]
    assert "d^-1" in str(out["expr_symbolic_d"])
    assert "d^-2" in sym["family"]["potential"]


def test_eigen_ground_state_eigenvalue_d():
    d = 2
    rng = np.random.default_rng(4)
    tpl = fex.Template("depth2_sub", dim=d)
    prob = fex.Problem.eigen(d, potential=lambda x: sum(v**2 for v in x))
    pts = fex.sample_box(1500, d, -3, 3, rng)
    bd = fex.sample_box_boundary(300, d, -3, 3, rng)
    sq, exp = fex.PAPER_UNARY.index("sq"), fex.PAPER_UNARY.index("exp")
    theta = [-0.52, -0.48] + [0.0] + [0.0] * (d + 1) + [1.0, 0.0]
    res = _fit(tpl, prob, [sq, 0, 0, exp], theta, pts, bd, rng)
    out = fex.finalize(res, prob, template=tpl, points=pts, boundary=bd)
    assert out["eigenvalue"] == "2"
    conc = out["verification"]["concrete"]
    assert conc["interior_verified"]
    # u = exp(-|x|²/2) is not 0 on the faces of the truncated box [-3, 3]^d.
    assert conc["status"] == "refuted"
    sym = out["verification"]["symbolic_d"]
    assert sym["interior_verified"]
    assert sym["family"]["eigenvalue"] == "d"


def test_unsnappable_candidate_gets_an_enclosure_only():
    d = 2
    rng = np.random.default_rng(5)
    tpl = fex.Template("depth2_sub", dim=d)
    prob = _poisson(d)
    pts = fex.sample_box(200, d, rng=rng)
    bd = fex.sample_box_boundary(100, d, rng=rng)
    cos = fex.PAPER_UNARY.index("cos")
    res = _fit(tpl, prob, [cos, 0, 0, 2], tpl.init_params(1, rng)[0], pts, bd, rng)
    out = fex.finalize(res, prob, template=tpl, points=pts, boundary=bd, certificate=True)
    v = out["verification"]
    assert not out["exact"]
    assert v["concrete"] is None or not v["concrete"]["verified"]
    assert v["interval"]["status"] == "enclosure"
    assert v["interval"]["verified"] is False
    assert v["interval"]["max_abs_residual_upper"] > 0
    assert out["certificate"]["status"] == "withheld"
    assert v["symbolic_d"]["verified"] is False


def test_snap_candidates():
    best = fex.snap_candidates(math.pi / 4)[0]
    assert str(best[0]) == "1/4*pi" and best[1] == "constant"
    assert str(fex.snap_candidates(0.5)[0][0]) == "1/2"
    two = fex.snap_candidates(1 + math.pi / 4, rel_tol=1e-6)
    assert any(str(c) == "1 + 1/4*pi^1" or c.value() == pytest.approx(1 + math.pi / 4) for c, _ in two)
    assert any(how in ("pslq", "lll") for _, how in two)
    # A "random" number does not snap to a complicated constant.
    assert all(c.cost() <= 12 for c, _ in fex.snap_candidates(0.7390851332151607 + 1e-9))


def test_recovery_metric():
    pool = ak.ExprPool()
    x = [pool.symbol(f"x{i}") for i in range(3)]
    truth = sum(v**2 for v in x) / 2
    near = sum(pool.float(0.49999, 53) * v**2 for v in x)
    r = fex.recovery_metric(near, truth)
    assert r["skeleton_equivalent"] is True
    assert r["exact"] is not True  # floats: never a proof of equality
    r2 = fex.recovery_metric(sum(v**2 for v in x) * pool.rational(1, 2), truth)
    assert r2["skeleton_equivalent"] and r2["exact"] is True
    r3 = fex.recovery_metric(ak.sin(sum(x)), truth)
    assert r3["skeleton_equivalent"] is False
    with pytest.raises(fex.FexError) as ei:
        fex.recovery_metric("x", truth)
    assert ei.value.code == "E-FEX-011"


def test_finalize_input_errors():
    d = 2
    prob = _poisson(d)
    res = fex.FitResult([3, 0, 0, 2], np.zeros(8), 0.0)
    with pytest.raises(fex.FexError) as ei:
        fex.finalize(res, prob)
    assert ei.value.code == "E-FEX-009"
    tpl3 = fex.Template("depth2_sub", dim=3)
    with pytest.raises(fex.FexError) as ei:
        fex.finalize(res, prob, template=tpl3)
    assert ei.value.code == "E-FEX-009"
    assert "Remediation" in str(ei.value)


def test_search_result_carries_what_finalize_needs():
    d = 3
    rng = np.random.default_rng(6)
    tpl = fex.Template("depth2_sub", dim=d)
    prob = _poisson(d)
    res = fex.search(
        tpl,
        prob,
        fex.sample_box(300, d, rng=rng),
        fex.sample_box_boundary(150, d, rng=rng),
        epochs=60,
        target_loss=1e-20,
        rng=rng,
    )
    assert res.template is tpl and res.points is not None
    out = fex.finalize(res, prob)
    assert out["verification"]["symbolic_d"]["verified"]
    assert fex.recovery_metric(out, sum(v**2 for v in prob.x) / 2)["exact"] is True


def test_search_with_a_normalised_loss_runs():
    d = 2
    rng = np.random.default_rng(7)
    tpl = fex.Template("depth1", dim=d)
    prob = fex.Problem.cubic_schrodinger(d, potential=lambda x: sum(ak.cos(v) for v in x))
    res = fex.search(tpl, prob, fex.sample_box(100, d, rng=rng), epochs=3, rng=rng)
    assert math.isfinite(res.loss) or res.loss == math.inf


def test_symbolic_heads_used_are_m0s():
    pool = ak.ExprPool()
    i, d = pool.symbol("i", "integer"), pool.symbol("d", "positive")
    from alkahest.experimental import IndexedBase

    X = IndexedBase(pool.symbol("x"))
    assert laplacian_indexed(Sum(X[i] ** 2 / 2, i, 1, d), X, 1, d) == d
