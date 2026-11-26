"""FEX milestone F4: symbolic pruning (``analyze`` / ``solve``), the
dependent-variable substitution ``u = e^v`` and variable reduction, the
configurable search controller, the centred loss, ``tanh``, and the PyTorch
evaluator.

Checks:

* :func:`fex.analyze` on the FEX paper's problems: vocabulary, separability,
  radial symmetry, dispersion (characteristics), scaling degree, the
  ``u = e^v`` proposal for the eigenproblem and the cubic Schrödinger
  equation, variable dependence for the committor;
* :func:`fex.transform_problem` (derived through the jet layer) and
  :func:`fex.reduce_problem`, and their refusals (``E-FEX-012``);
* :func:`fex.lift_candidate` reproduces ``e^v`` exactly (folded and wrapped)
  and places reduced weights;
* :func:`fex.solve` recovers NLS (the search failure F3 reported at
  ``d = 12``) and the harmonic ground state (the failure at ``d = 10``,
  here at a smaller ``d``), and :func:`fex.finalize` proves them;
* controller options: slot restrictions are respected, the variants run end
  to end, malformed options raise ``E-FEX-013``;
* the PyTorch evaluator agrees with the Rust one (skipped without torch).
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")

from alkahest import fex  # noqa: E402  (needs numpy)

C = math.pi / 4


def _rng(seed=0):
    return np.random.default_rng(seed)


def _poisson(d):
    return fex.Problem.poisson(d, rhs=-d, bc=lambda x: 0.5 * sum(v**2 for v in x))


def _conservation(d):
    coef = [(d - 1) * C] + [-1.0] * (d - 1)
    return fex.Problem.transport(coef, bc=lambda x: ak.sin(C * sum(x[1:])), time=0)


def _nls(d):
    def V(x):
        S = sum(ak.cos(v) for v in x)
        return -ak.exp(2.0 / d * S) / 9 + sum(ak.sin(v) ** 2 for v in x) / d**2 - S / d

    return fex.Problem.cubic_schrodinger(d, potential=V)


def _eigen(d, delta=0.0):
    def V(x):
        out = sum(v**2 for v in x)
        return out + delta * sum(v**4 for v in x) if delta else out

    return fex.Problem.eigen(d, potential=V)


def _committor(d, beta=5.0):
    jet = fex.Jet(d)
    x = jet.x
    dV = [4 * x[0] * (x[0] ** 2 - 1)] + [0.6 * v for v in x[1:]]
    op = jet.lap / beta - sum(a * g for a, g in zip(dV, jet.grad))
    return fex.Problem.from_operator(jet, op, rhs=0.0, bc=(1 + x[0]) / 2)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_paper_problems():
    a = fex.analyze(_poisson(6))
    assert a.linear
    assert not a.homogeneous
    assert a.separable
    assert a.radial
    assert a.scaling_degree == 2
    assert a.leaf_vocabulary == ["id", "sq"]
    assert a.proposals[0].name == "scaling"
    assert a.proposals[0].slot_ops["leaf"] == ["0", "sq"]
    assert a.proposals[-1].name == "paper"

    a = fex.analyze(_conservation(5))
    assert a.dispersion is not None
    assert a.dispersion["first_order"]
    assert a.inner_vocabulary == ["sin"]
    first = a.proposals[0]
    assert first.name == "characteristics"
    assert first.slot_ops["leaf"] == ["0", "id"]

    a = fex.analyze(_nls(6))
    assert a.linear is False
    assert a.homogeneous
    assert a.transforms == ["exp"]
    assert a.proposals[0].transform == "exp"
    assert "cos" in a.proposals[0].slot_ops["leaf"]

    a = fex.analyze(_eigen(5))
    assert a.eigen
    assert a.separable
    assert a.radial
    assert a.proposals[0].name == "exp-separable"
    assert a.proposals[0].loss == "centered"
    assert a.proposals[0].binary == ["+"]

    a = fex.analyze(_committor(6), unary=[*fex.PAPER_UNARY, "tanh"])
    assert a.depends_on == [0]
    reduced = [p for p in a.proposals if p.name == "reduced"]
    assert reduced
    assert reduced[0].variables == [0]
    assert "tanh" in reduced[0].unary


# ---------------------------------------------------------------------------
# transforms, reductions, lifting
# ---------------------------------------------------------------------------


def test_transform_eigenproblem_to_centred_riccati():
    d = 3
    tp = fex.transform_problem(_eigen(d), "exp")
    assert tp.loss == "centered"
    assert tp.spec["transform"] == "exp"
    # v = −½|x|² solves −Δv − |∇v|² + |x|² = λ (λ = d) exactly: centred loss 0.
    rng = _rng(1)
    tpl = fex.Template("depth2_sub", dim=d)
    ev = fex.Evaluator(tpl, tp, fex.sample_box(200, d, -3, 3, rng))
    th = np.zeros(tpl.n_params)
    th[:d] = -0.5
    th[tpl.param_offsets[-1]] = 1.0
    assert ev.loss([3, 0, 0, 2], th) < 1e-26
    th[0] = 0.5  # one growing coordinate: still a solution of the transformed PDE
    assert ev.loss([3, 0, 0, 2], th) < 1e-26


def test_transform_and_reduce_refusals():
    with pytest.raises(fex.FexError) as ei:
        fex.transform_problem(_poisson(3), "exp")
    assert ei.value.code == "E-FEX-012"
    with pytest.raises(fex.FexError) as ei:
        fex.transform_problem(_nls(3), "sqrt")
    assert ei.value.code == "E-FEX-012"
    with pytest.raises(fex.FexError) as ei:
        fex.reduce_problem(_poisson(3), [0])  # bc depends on x1, x2
    assert ei.value.code == "E-FEX-012"
    red = fex.reduce_problem(_committor(4), [0])
    assert red.dim == 1


def test_lift_candidate_is_exp_of_v():
    d = 3
    rng = _rng(2)
    pts = fex.sample_box(50, d, -1, 1, rng)
    tpl = fex.Template("depth2_sub", unary=["0", "id", "sq", "cos"], dim=d)
    prob = _poisson(d)
    for e in ([2, 0, 3, 1], [2, 1, 3, 2], [3, 0, 0, 3]):
        th = tpl.init_params(1, rng, scale=0.4)[0]
        v, _, _ = fex.Evaluator(tpl, prob, pts).jets(e, th, pts)
        tu, eu, thu = fex.lift_candidate(tpl, e, th, transform="exp")
        u, _, _ = fex.Evaluator(tu, prob, pts).jets(eu, thu, pts)
        assert np.allclose(u, np.exp(v), rtol=1e-12, atol=0)
    # Placing a reduced (1-D) candidate at coordinate 2 of a 4-D template.
    t1 = fex.Template("depth2_sub", unary=["0", "id", "sq"], dim=1)
    th1 = t1.init_params(1, rng)[0]
    tu, eu, thu = fex.lift_candidate(t1, [2, 0, 1, 1], th1, variables=[2], dim=4)
    p4 = fex.sample_box(20, 4, -1, 1, rng)
    u4, _, _ = fex.Evaluator(tu, _poisson(4), p4).jets(eu, thu, p4)
    u1, _, _ = fex.Evaluator(t1, _poisson(1), p4[:, 2:3]).jets([2, 0, 1, 1], th1, p4[:, 2:3])
    assert np.allclose(u4, u1, rtol=1e-12)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------


def test_solve_nls_through_exp_substitution_and_finalize():
    d = 6
    rng = _rng(3)
    prob = _nls(d)
    res = fex.solve(prob, fex.sample_box(800, d, -1, 1, rng), None, epochs=40, rng=rng)
    assert res.proposal.transform == "exp"
    assert res.loss < 1e-20
    assert res.seconds_to_target is not None
    out = fex.finalize(res, prob)
    assert out["exact"]
    truth = ak.exp(sum(ak.cos(v) for v in prob.x) / d) / 3
    assert fex.recovery_metric(out["expr"], truth)["exact"] is True
    assert out["verification"]["symbolic_d"]["status"] == "exactly_verified"


def test_solve_eigen_picks_the_decaying_ground_state():
    d = 4
    rng = _rng(4)
    prob = _eigen(d)
    res = fex.solve(
        prob,
        fex.sample_box(800, d, -3, 3, rng),
        fex.sample_box_boundary(200, d, -3, 3, rng),
        epochs=40,
        rng=rng,
    )
    assert res.proposal.name == "exp-separable"
    # u = c·exp(−½|x|²): every coordinate decays.
    pts = fex.sample_box(200, d, -1, 1, _rng(5))
    ev = fex.Evaluator(res.template, prob, pts)
    u, _, _ = ev.jets(res.e, res.theta, pts)
    ref = np.exp(-0.5 * (pts**2).sum(1))
    c = float(u @ ref) / float(u @ u)
    assert np.linalg.norm(c * u - ref) / np.linalg.norm(ref) < 1e-10
    out = fex.finalize(res, prob)
    assert out["exact"]
    assert out["eigenvalue"] == "4"


def test_solve_poisson_and_conservation_pruned():
    rng = _rng(6)
    d = 5
    res = fex.solve(
        _poisson(d),
        fex.sample_box(400, d, rng=rng),
        fex.sample_box_boundary(150, d, rng=rng),
        epochs=30,
        rng=rng,
    )
    assert res.key == "L[sq]"
    assert res.loss < 1e-20
    assert res.attempts[0]["proposal"] == "scaling"
    res = fex.solve(
        _conservation(4),
        fex.sample_box(600, 4, 0, 1, rng),
        fex.sample_initial_slice(200, 4, 0, 1, 0, 0.0, rng),
        epochs=30,
        rng=rng,
    )
    assert res.loss < 1e-20
    assert res.proposal.name == "characteristics"


def test_solve_with_the_analysis_as_a_soft_prior():
    d = 5
    rng = _rng(14)
    prob = _nls(d)
    res = fex.solve(
        prob, fex.sample_box(600, d, -1, 1, rng), None, epochs=60, restrict="soft", rng=rng
    )
    assert res.loss < 1e-20
    # The full table is searched (the proposal only biases the sampling).
    assert "quartic" in res.sub_result.template.unary
    with pytest.raises(fex.FexError) as ei:
        fex.solve(prob, fex.sample_box(50, d, rng=rng), restrict="medium")
    assert ei.value.code == "E-FEX-013"


def test_solve_without_analysis_and_with_explicit_proposals():
    rng = _rng(7)
    d = 3
    prob = _poisson(d)
    xi = fex.sample_box(300, d, rng=rng)
    xb = fex.sample_box_boundary(100, d, rng=rng)
    res = fex.solve(prob, xi, xb, analysis=False, epochs=30, rng=rng)
    assert res.proposal.name == "paper"
    assert res.analysis is None
    a = fex.analyze(prob)
    res = fex.solve(prob, xi, xb, proposals=a.proposals[:1], epochs=30, rng=rng)
    assert len(res.attempts) == 1


# ---------------------------------------------------------------------------
# controller options
# ---------------------------------------------------------------------------


def test_slot_ops_restrict_every_sample():
    d = 3
    rng = _rng(8)
    tpl = fex.Template("depth2_sub", dim=d)
    res = fex.search(
        tpl,
        _poisson(d),
        fex.sample_box(200, d, rng=rng),
        fex.sample_box_boundary(80, d, rng=rng),
        epochs=8,
        slot_ops={"leaf": ["sq", "cos"], "unary": ["id"], "binary": ["+"]},
        prune=False,
        rng=rng,
    )
    for r in res.pool:
        names = [
            (tpl.binary if k == "binary" else tpl.unary)[v] for v, k in zip(r.e, tpl.slot_kinds)
        ]
        assert names[0] in ("sq", "cos")
        assert names[2] in ("sq", "cos")
        assert names[1] == "+"
        assert names[3] == "id"


@pytest.mark.parametrize(
    "opts",
    [
        {"entropy_weight": 0.01},
        {"pqt_size": 5, "pqt_weight": 0.5},
        {"population": 3, "batch": 4, "restart_patience": 3},
        {"curriculum": 3},
        {"baseline": "ewma"},
        {"baseline": "mean"},
        {"prior": {"leaf": {"sq": 2.0}}},
        {"seeds": [[3, 0, 0, 2]]},
        {"stall_epochs": 2},
    ],
)
def test_controller_variants_run(opts):
    d = 3
    rng = _rng(9)
    tpl = fex.Template("depth2_sub", dim=d)
    res = fex.search(
        tpl,
        _poisson(d),
        fex.sample_box(200, d, rng=rng),
        fex.sample_box_boundary(80, d, rng=rng),
        epochs=10,
        target_loss=1e-20,
        rng=rng,
        **opts,
    )
    assert math.isfinite(res.loss)
    assert res.epochs >= 1
    if "seeds" in opts:
        assert res.loss < 1e-20
        assert res.epochs == 1


def test_default_search_is_deterministic():
    d = 3

    def run():
        rng = _rng(10)
        tpl = fex.Template("depth2_sub", dim=d)
        return fex.search(
            tpl,
            _poisson(d),
            fex.sample_box(150, d, rng=rng),
            fex.sample_box_boundary(60, d, rng=rng),
            epochs=5,
            rng=rng,
        )

    a, b = run(), run()
    assert a.key == b.key
    assert a.history == b.history


def test_controller_option_errors():
    d = 2
    rng = _rng(11)
    tpl = fex.Template("depth2_sub", dim=d)
    xi = fex.sample_box(50, d, rng=rng)
    for kw in (
        {"slot_ops": {"leaf": ["tanh"]}},
        {"slot_ops": {"leaf": []}},
        {"slot_ops": {"root": ["id"]}},
        {"slot_ops": [["sq"]]},
        {"baseline": "median"},
        {"population": 0},
    ):
        with pytest.raises(fex.FexError) as ei:
            fex.search(tpl, _poisson(d), xi, epochs=1, rng=rng, **kw)
        assert ei.value.code == "E-FEX-013"


# ---------------------------------------------------------------------------
# tanh and the PyTorch evaluator
# ---------------------------------------------------------------------------


def test_tanh_operator_instantiates_and_fits():
    d = 1
    tpl = fex.Template("depth2_sub", unary=["0", "id", "tanh"], dim=d)
    e = [2, 0, 0, 1]  # id(L[tanh] + L[0])
    th = np.array([0.7, 0.1, 0.0, 0.0, 1.0, 0.0])
    expr = tpl.instantiate(e, th)
    assert "tanh" in str(expr)
    jet = fex.Jet(d)
    x = jet.x[0]
    # u = tanh(x): u'' = −2 tanh(x)(1 − tanh²x) = −2u(1 − u²).
    prob = fex.Problem.from_operator(jet, jet.lap + 2 * jet.u * (1 - jet.u**2), bc=ak.tanh(x))
    rng = _rng(12)
    ev = fex.Evaluator(tpl, prob, fex.sample_box(100, 1, rng=rng), np.array([[-1.0], [1.0]]))
    fit = fex.fit(ev, [2, 0, 0, 1], restarts=4, rng=rng)
    assert fit.loss < 1e-18


def test_torch_evaluator_matches_rust():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = _rng(13)
    for prob, lo, hi, bd in (
        (_poisson(4), -1, 1, True),
        (_conservation(4), 0, 1, True),
        (_nls(4), -1, 1, False),
        (_eigen(3), -3, 3, True),
    ):
        d = prob.dim
        tpl = fex.Template("depth2", dim=d)
        xi = fex.sample_box(200, d, lo, hi, rng)
        xb = fex.sample_box_boundary(60, d, lo, hi, rng) if bd else None
        ev = fex.Evaluator(tpl, prob, xi, xb)
        tev = fex.TorchEvaluator(tpl, prob, xi, xb, device=device)
        es = tpl.random_skeletons(32, rng)
        th = tpl.init_params(32, rng, scale=0.5)
        l1, g1 = ev.loss_and_grad(es, th)
        l2, g2 = tev.loss_and_grad(es, th)
        fin = np.isfinite(l1)
        assert (fin == np.isfinite(l2)).all()
        assert np.allclose(l1[fin], l2[fin], rtol=1e-10, atol=1e-12)
        assert np.allclose(g1[fin], g2[fin], rtol=1e-8, atol=1e-10)
        fits = tev.fit(es[:4], th[:4], max_iter=5)
        assert all(f.loss <= start for f, start in zip(fits, l2[:4]))
    # A gradient inside a non-polynomial function stays on the CPU evaluator
    # (expression operators of the form F0 + a·∇u + b|∇u|² are covered:
    # tests/test_fex_g6.py).
    jet = fex.Jet(2)
    bad = fex.Problem.from_operator(jet, ak.sin(jet.grad[0]) + jet.lap)
    with pytest.raises(fex.FexError) as ei:
        fex.TorchEvaluator(fex.Template("depth1", dim=2), bad, np.zeros((4, 2)))
    assert ei.value.code == "E-FEX-014"
