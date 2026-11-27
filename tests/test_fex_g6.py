"""FEX gap round G6: PIDEs (FEX-PG), fast finalize for normalised losses,
reflective-kernel Lean certificates for PDE residuals, and the PyTorch
evaluator for expression operators.

Checks:

* the interpreter's nonlocal terms and second-order forms through
  :meth:`fex.Problem.pide`: the Taylor-moment replacement is exact on the
  quadratic solutions of Hardwick–Liang–Yang's high-dimensional examples
  (loss at rounding level), and their residual Jacobian matches finite
  differences;
* :func:`fex.gaussian_expectation` against closed forms (polynomials,
  ``exp``, ``cos``) and its refusal outside its class;
* :func:`fex.solve` + :func:`fex.finalize` on both FEX-PG examples:
  recovered exactly, proved against the **exact** jump integral at the
  fitted ``d`` and for symbolic ``d``; the paper's printed right-hand side
  of its eq. (33) is refuted for the solution it states;
* finalize on the cubic Schrödinger equation: Levenberg–Marquardt refits on
  the normalised residual, fitted-``d`` verdict now exact (``exp`` split
  over its argument), and a kernel certificate;
* kernel certificates for Poisson, the eigenproblem and transport (emitted,
  no ``sorry``), and withheld reasons;
* ``TorchEvaluator`` on expression operators, the centred loss and PIDE
  nonlocal terms against the Rust evaluator (skipped without torch);
* :func:`fex.group_weights`, ``E-FEX-015``.
"""

from __future__ import annotations

from fractions import Fraction as Q

import alkahest as ak
import pytest

np = pytest.importorskip("numpy")

from alkahest import fex  # noqa: E402  (needs numpy)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _pide4(d):
    """FEX-PG eq. (32): u_t + ½θ²Δu + jump = λ(μ² + σ²) + θ², u(1, x) = |x|²/d."""
    jet = fex.Jet(d + 1, time=0)
    pool = jet.pool
    lam, mu, theta, var = Q(3, 10), Q(1), Q(3, 10), Q(1, 10000)
    c = lam * (mu**2 + var) + theta**2
    r2 = sum(v**2 for v in jet.x[1:])
    return fex.Problem.pide(
        d + 1,
        time=0,
        diffusion=theta,
        jump=fex.GaussianJump(lam, mu, var),
        rhs=pool.rational(c.numerator, c.denominator),
        terminal=r2 / d,
        jet=jet,
    )


def _pide5(d, *, paper_rhs=False):
    """FEX-PG eq. (33), with the right-hand side its solution satisfies
    (``(2d−1)/d·θ²``) or, with ``paper_rhs``, the printed ``(2d−2)/d·θ²``."""
    jet = fex.Jet(d + 1, time=0)
    pool = jet.pool
    lam, mu, theta, var, eps = Q(3, 10), Q(1), Q(1, 5), Q(1, 10**8), Q(1, 20)
    k = 2 * d - 2 if paper_rhs else 2 * d - 1
    c0 = lam * (mu**2 + var) + Q(k, d) * theta**2
    r2 = sum(v**2 for v in jet.x[1:])
    rhs = pool.rational(c0.numerator, c0.denominator) + pool.rational(
        1, 20 * d
    ) * r2 ** pool.rational(3, 2)
    he = eps / 2
    return fex.Problem.pide(
        d + 1,
        time=0,
        drift=fex.RadialDrift(lambda r: pool.rational(he.numerator, he.denominator) * r),
        diffusion=fex.BidiagonalDiffusion(theta),
        jump=fex.GaussianJump(lam, mu, var),
        rhs=rhs,
        terminal=r2 / d,
        jet=jet,
    )


def _points(d, rng, n=600, nb=150):
    xi = fex.sample_box(n, d + 1, -1.0, 1.0, rng)
    xi[:, 0] = rng.uniform(0.0, 1.0, n)
    xb = fex.sample_initial_slice(nb, d + 1, -1.0, 1.0, 0, 1.0, rng)
    return xi, xb


def _exact_theta(tpl, d):
    """``id(L[sq] + L[0])`` with spatial weights ``1/d``: ``|x|²/d``."""
    e = [tpl.unary.index("sq"), tpl.binary.index("+"), tpl.unary.index("0"), tpl.unary.index("id")]
    th = np.zeros(tpl.n_params)
    o = tpl.param_offsets
    th[o[0] + 1 : o[0] + d + 1] = 1.0 / d
    th[o[3]] = 1.0
    return e, th


# ---------------------------------------------------------------------------
# The interpreter's PIDE terms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", [_pide4, _pide5])
def test_pide_taylor_replacement_is_exact_on_the_quadratic_solution(make):
    d = 5
    prob = make(d)
    assert prob._native.n_nonlocal == 1
    tpl = fex.Template("depth2_sub", dim=d + 1)
    assert tpl.slot_kinds == ["leaf", "binary", "leaf", "unary"]
    xi, xb = _points(d, _rng(1))
    ev = fex.Evaluator(tpl, prob, xi, xb)
    e, th = _exact_theta(tpl, d)
    assert ev.loss(e, th) < 1e-26
    # A perturbed candidate has a large loss, and an exact Jacobian.
    rng = _rng(2)
    th2 = th + 0.05 * rng.standard_normal(len(th))
    r, J = ev.residual_and_jacobian(e, th2)
    assert float(r @ r) > 1e-6
    h = 1e-6
    for k in range(0, len(th), 3):
        tp, tm = th2.copy(), th2.copy()
        tp[k] += h
        tm[k] -= h
        fd = (ev.residual_and_jacobian(e, tp)[0] - ev.residual_and_jacobian(e, tm)[0]) / (2 * h)
        assert np.allclose(fd, J[:, k], rtol=1e-6, atol=1e-7)


def test_gaussian_expectation_closed_forms():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    mu, var = pool.rational(1, 3), pool.rational(1, 5)
    cases = [
        (x**2, x**2 + 2 * mu * x + mu**2 + var),
        (x**3, x**3 + 3 * x**2 * mu + 3 * x * (mu**2 + var) + mu**3 + 3 * mu * var),
        (ak.exp(2 * x), ak.exp(2 * x + 2 * mu + 2 * var)),
        (x * ak.exp(x), (x + mu + var) * ak.exp(x + mu + var / 2)),
        (ak.cos(3 * x + 1), ak.exp(-9 * var / 2) * ak.cos(3 * x + 1 + 3 * mu)),
    ]
    for f, want in cases:
        got = fex.gaussian_expectation(f, [x], mu, var)
        assert ak.is_zero(ak.simplify(got - want).value) is True or ak.is_zero(
            ak.expand(ak.simplify(got - want).value)
        ), (f, got)
    # Multiplicative jumps: E[(x e^z)²] = x² e^{2μ + 2σ²}.
    got = fex.gaussian_expectation(x**2, [x], mu, var, multiplicative=True)
    assert ak.is_zero(ak.simplify(got - x**2 * ak.exp(2 * mu + 2 * var)).value) is True
    from alkahest._fex_pide import NoClosedForm

    with pytest.raises(NoClosedForm):
        fex.gaussian_expectation(ak.exp(x**2), [x], mu, var)


def test_taylor_moment_formula():
    jet = fex.Jet(3, time=0)
    tm = fex.taylor_moment(fex.GaussianJump(0.3, 1, 0.0001), jet, [1, 2])
    (scale, shift, cu, clap) = tm["nonlocal"][0]
    assert scale == []
    assert shift == [0.0, 1.0, 1.0]
    assert cu == pytest.approx(0.3)
    assert clap == pytest.approx(0.5 * 0.3 * 0.0001)
    assert "Δu(x + μ)" in tm["formula"]
    # The local part is −λu − λμ Σ ∂ᵢu over the spatial coordinates.
    want = -Q(3, 10) * jet.u - Q(3, 10) * (jet.grad[1] + jet.grad[2])
    assert ak.is_zero(ak.simplify(tm["local"] - want).value) is True


def test_group_weights_matches_threshold_clustering():
    w = [0.1, 0.1004, 0.25, 0.2502, 0.9]
    assert fex.group_weights(w, 0.01) == [0, 0, 1, 1, 2]
    assert fex.group_weights(w, 1.0) == [0, 0, 0, 0, 0]


def test_pide_errors():
    with pytest.raises(fex.FexError) as ei:
        fex.GaussianJump(0.3, 1.0, -1.0)
    assert ei.value.code == "E-FEX-015"
    from alkahest.alkahest import _FexProblem

    pool = ak.ExprPool()
    xs = [pool.symbol(f"x{i}") for i in range(2)]
    with pytest.raises(fex.FexError) as ei:
        _FexProblem(2, "poisson", coords=xs, nonlocal_terms=[([], [0.1], 1.0, 0.0)])
    assert ei.value.code == "E-FEX-015"
    p = _FexProblem(
        2, "poisson", coords=xs, nonlocal_terms=[([], [0.1, 0.0], 1.0, 0.0)], loss="normalized"
    )
    tpl = fex.Template("depth1", dim=2)
    from alkahest.alkahest import _FexEvaluator

    with pytest.raises(fex.FexError) as ei:
        _FexEvaluator(tpl._native, p, np.zeros((3, 2)), None)
    assert ei.value.code == "E-FEX-015"


# ---------------------------------------------------------------------------
# Solve + finalize on FEX-PG
# ---------------------------------------------------------------------------


def test_pide4_solved_and_proved_for_every_d():
    d = 4
    rng = _rng(3)
    prob = _pide4(d)
    xi, xb = _points(d, rng)
    res = fex.solve(prob, xi, xb, epochs=60, rng=rng, stall_epochs=20)
    assert res.loss < 1e-24
    out = fex.finalize(res, prob)
    assert out["exact"]
    truth = sum(v**2 for v in prob.x[1:]) / d
    assert fex.recovery_metric(out["expr"], truth)["exact"] is True
    conc = out["verification"]["concrete"]
    assert conc["status"] == "exactly_verified"
    assert conc["jump"].startswith("exact")
    assert conc["taylor_exact_on_solution"] is True
    sym = out["verification"]["symbolic_d"]
    assert sym["status"] == "exactly_verified"
    assert "Sum" in str(sym["expr"])


def test_pide5_needs_its_family_and_refutes_the_printed_rhs():
    d = 3
    rng = _rng(4)
    prob = _pide5(d)
    xi, xb = _points(d, rng)
    tpl = fex.Template("depth2_sub", dim=d + 1)
    e, th = _exact_theta(tpl, d)
    res = fex.FitResult(e, th, 0.0)
    fs = fex.family_symbols(prob)
    pool = fs.pool
    r2 = fs.sum(lambda xi_: xi_**2)
    rhs = (
        pool.rational(3, 10) * (1 + pool.rational(1, 10**8))
        + (2 * fs.d - 1) / fs.d * pool.rational(1, 25)
        + pool.rational(1, 20) / fs.d * r2 ** pool.rational(3, 2)
    )
    out = fex.finalize(
        res, prob, template=tpl, points=xi, boundary=xb, family={"rhs": rhs, "bc": r2 / fs.d}
    )
    assert out["verification"]["concrete"]["status"] == "exactly_verified"
    assert out["verification"]["symbolic_d"]["status"] == "exactly_verified"
    # With the right-hand side as printed in the paper the stated solution
    # is not a solution: the residual is θ²/d ≠ 0, a proved nonzero.
    from alkahest._fex_finalize import _exact_sources, _verify_concrete_pide

    bad = _pide5(d, paper_rhs=True)
    u = sum(v**2 for v in bad.x[1:]) / d
    src, _ = _exact_sources(bad, "pide", 0)
    v = _verify_concrete_pide(u, bad, src, bad.jet.pool)
    assert v["status"] == "refuted"
    assert v["pde"][0]["status"] == "proved_nonzero"


# ---------------------------------------------------------------------------
# Finalize speed and the cubic Schrödinger certificate
# ---------------------------------------------------------------------------


def _nls(d):
    def V(x):
        S = sum(ak.cos(v) for v in x)
        return -ak.exp(2.0 / d * S) / 9 + sum(ak.sin(v) ** 2 for v in x) / d**2 - S / d

    return fex.Problem.cubic_schrodinger(d, potential=V)


def test_nls_finalize_refits_by_lm_and_certifies():
    import time

    d = 8
    rng = _rng(5)
    prob = _nls(d)
    xi = fex.sample_box(1500, d, -1, 1, rng)
    res = fex.solve(prob, xi, None, epochs=40, rng=rng)
    assert res.loss < 1e-20
    t = time.perf_counter()
    out = fex.finalize(res, prob)
    took = time.perf_counter() - t
    assert out["exact"]
    assert took < 60.0
    assert any("no refit" in ln for ln in out["log"])
    assert out["verification"]["concrete"]["status"] == "exactly_verified"
    assert out["verification"]["symbolic_d"]["status"] == "exactly_verified"
    cert = out["certificate"]
    assert cert["status"] == "emitted"
    assert cert["path"] == "kernel"
    assert "pde_sound" in cert["lean"]
    assert "sorry" not in cert["lean"]
    assert "fexU" in cert["lean"]


def test_kernel_certificates_for_poisson_eigen_and_transport():
    from alkahest._fex_lean import kernel_certificate

    pool = ak.ExprPool()
    xs = [pool.symbol(f"x{i}") for i in range(3)]
    r2 = sum(v**2 for v in xs)
    half = pool.rational(1, 2)
    # Poisson: −Δ(½|x|²) = −3.
    c = kernel_certificate("poisson", half * r2, xs, None, {"rhs": pool.integer(-3)})
    assert c["status"] == "emitted", c
    # Eigen: exp(−½|x|²), λ = 3; a wrong λ is not certified.
    u = ak.exp(-half * r2)
    c = kernel_certificate("eigen", u, xs, None, {"potential": r2, "eigenvalue": pool.integer(3)})
    assert c["status"] == "emitted", c
    c = kernel_certificate("eigen", u, xs, None, {"potential": r2, "eigenvalue": pool.integer(2)})
    assert c["status"] == "withheld"
    # Transport: u = sin(t + x1 + x2) solves 2u_t − u_x1 − u_x2 = 0.
    u = ak.sin(xs[0] + xs[1] + xs[2])
    coeffs = [pool.integer(2), pool.integer(-1), pool.integer(-1)]
    c = kernel_certificate("transport", u, xs, 0, {"coeffs": coeffs})
    assert c["status"] == "emitted", c
    # Not differentiable everywhere: withheld with the kernel's reason.
    c = kernel_certificate("poisson", ak.log(xs[0]), xs, None, {"rhs": None})
    assert c["status"] == "withheld"


# ---------------------------------------------------------------------------
# GPU evaluator: expression operators, centred loss, PIDE
# ---------------------------------------------------------------------------


def test_torch_evaluator_expression_operators():
    torch = pytest.importorskip("torch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = _rng(6)
    jet = fex.Jet(4)
    x = jet.x
    dV = [4 * x[0] * (x[0] ** 2 - 1)] + [0.6 * v for v in x[1:]]
    committor = fex.Problem.from_operator(
        jet, jet.lap / 5 - sum(a * g for a, g in zip(dV, jet.grad)), bc=(1 + x[0]) / 2
    )

    def V(xx):
        return sum(v**2 for v in xx)

    eig = fex.transform_problem(fex.Problem.eigen(3, potential=V), "exp")
    assert eig.loss == "centered"
    cases = [
        (committor, fex.sample_box(200, 4, -1, 1, rng), fex.sample_box_boundary(50, 4, -1, 1, rng)),
        (eig, fex.sample_box(200, 3, -2, 2, rng), None),
        (_pide4(3), *_points(3, rng, 200, 50)),
    ]
    for prob, xi, xb in cases:
        tpl = fex.Template("depth2", dim=prob.dim)
        ev = fex.Evaluator(tpl, prob, xi, xb)
        tev = fex.TorchEvaluator(tpl, prob, xi, xb, device=device)
        es = tpl.random_skeletons(48, rng)
        th = tpl.init_params(48, rng, scale=0.5)
        l1, g1 = ev.loss_and_grad(es, th)
        l2, g2 = tev.loss_and_grad(es, th)
        fin = np.isfinite(l1) & (l1 < 1e100)
        assert (np.isfinite(l1) == np.isfinite(l2)).all()
        assert np.allclose(l1[fin], l2[fin], rtol=1e-12, atol=1e-14)
        assert np.allclose(g1[fin], g2[fin], rtol=1e-9, atol=1e-11)


# ---------------------------------------------------------------------------
# Variational (Ritz) energy
# ---------------------------------------------------------------------------


def test_dirichlet_energy_matches_its_definition_and_reduces():
    rng = _rng(7)
    d = 3
    jet = fex.Jet(d)
    x = jet.x
    w = ak.exp(-((x[0] ** 2 - 1) ** 2))
    prob = fex.Problem.dirichlet(jet, weight=w, bc=(1 + x[0]) / 2)
    assert prob.loss == "dirichlet"
    xi = fex.sample_box(300, d, -1, 1, rng)
    xb = fex.sample_box(40, d, -1, 1, rng)
    tpl = fex.Template("depth2_sub", dim=d)
    ev = fex.Evaluator(tpl, prob, xi, xb)
    es = tpl.random_skeletons(8, rng)
    th = tpl.init_params(8, rng, scale=0.5)
    losses = ev.loss(es, th)
    for e, t, lo in zip(es, th, losses):
        if not np.isfinite(lo):
            continue
        u, g, _ = ev.jets(e, t, xi)
        wv = np.exp(-((xi[:, 0] ** 2 - 1) ** 2))
        ub, _, _ = ev.jets(e, t, xb)
        want = np.mean(wv * (g**2).sum(1)) + 100.0 * np.mean((ub - (1 + xb[:, 0]) / 2) ** 2)
        assert lo == pytest.approx(want, rel=1e-12)
    # The weight depends on x₀ only, so the reduction to x₀ keeps it.
    red = fex.reduce_problem(prob, [0])
    assert red.loss == "dirichlet"
    assert red.spec["potential"] is not None
