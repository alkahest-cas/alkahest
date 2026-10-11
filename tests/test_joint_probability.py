"""Joint distributions and the probability gaps of report 10-9 §3 G10/G11, §4.4.

Covers: multivariate normal (density, MGF, linear maps, marginals, Schur
conditionals, quadratic forms), independent-block joint laws with covariance and
correlated variance, sums of independent variates, symbolic-``n`` Binomial and
NegativeBinomial, closed-form expectations under Normal / Gamma / Beta /
LogNormal, order statistics with symbolic ``n``, and simplified Normal skewness
and kurtosis.

Oracles: SymPy ``sympy.stats`` where it has the quantity in closed form, scipy /
numpy formulas, and Monte-Carlo with a fixed seed and a tolerance set from the
CLT (several standard errors). Each symbolic answer is evaluated at random
admissible parameters, and differentiated where that is part of the claim.
"""

from __future__ import annotations

import math

import alkahest as ak
import pytest
from alkahest import experimental as ex

np = pytest.importorskip("numpy")

POOL = ak.ExprPool()
S = POOL.symbol


def num(expr, env: dict) -> float:
    return float(ak.eval_expr(expr, {S(k): float(v) for k, v in env.items()}))


def code_of(excinfo) -> str:
    return getattr(excinfo.value, "code", "")


RNG = np.random.default_rng(20261011)


def mc_close(sample: np.ndarray, value: float, k: float = 6.0) -> None:
    """|mean(sample) - value| within k standard errors (CLT)."""
    se = sample.std(ddof=1) / math.sqrt(sample.size)
    assert abs(sample.mean() - value) <= k * se + 1e-12, (sample.mean(), value, se)


# ---------------------------------------------------------------------------
# The report's repros
# ---------------------------------------------------------------------------


def test_binomial_with_symbolic_n_is_accepted_and_has_moments():
    n, p = S("n"), S("p")
    b = ex.Binomial(n, p)
    assert b.support()[0] == "integers_up_to"
    env = {"n": 9, "p": 0.35}
    assert num(b.mean(), env) == pytest.approx(9 * 0.35, rel=1e-12)
    assert num(b.variance(), env) == pytest.approx(9 * 0.35 * 0.65, rel=1e-12)
    assert any("Integer" in c for c in ex.prob_side_conditions())
    # SymPy has the same symbolic moments
    sp = pytest.importorskip("sympy")
    from sympy.stats import Binomial as SBinomial
    from sympy.stats import E as SE

    for k in (3, 4):
        want = float(SE(SBinomial("B", 9, sp.Rational(7, 20)) ** k))
        assert num(b.moment(k), env) == pytest.approx(want, rel=1e-12)
    # The mean is n*p as an expression: differentiate it in n.
    assert num(ak.diff(b.mean(), n).value, env) == pytest.approx(0.35, rel=1e-12)


def test_binomial_with_symbolic_n_refuses_enumeration_routes():
    n, p, x = S("n"), S("p"), S("x")
    b = ex.Binomial(n, p)
    with pytest.raises(ak.AlkahestError) as e:
        b.entropy()
    assert code_of(e) == "E-PROB-004"
    with pytest.raises(ak.AlkahestError) as e:
        ex.expectation(ak.exp(x), x, b)
    assert code_of(e) == "E-PROB-002"
    with pytest.raises(ak.AlkahestError) as e:
        ex.Binomial(POOL.rational(5, 2), p)
    assert code_of(e) == "E-PROB-001"


def test_normal_skewness_and_kurtosis_simplify_to_zero():
    n = ex.Normal(S("mu"), S("sigma"))
    assert str(n.skewness()) == "0"
    assert str(n.excess_kurtosis()) == "0"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_normal_expectations_of_common_functions(seed):
    rng = np.random.default_rng(seed)
    mu, sigma = float(rng.uniform(-1, 1)), float(rng.uniform(0.3, 1.2))
    t = float(rng.uniform(-1, 1))
    x = S("x")
    n = ex.Normal(S("mu"), S("sigma"))
    env = {"mu": mu, "sigma": sigma, "t": t}
    m = mu + t * sigma**2
    cases = {
        ak.exp(S("t") * x): math.exp(mu * t + sigma**2 * t**2 / 2),
        x**2 * ak.exp(S("t") * x): math.exp(mu * t + sigma**2 * t**2 / 2) * (m * m + sigma**2),
        ak.cos(S("t") * x): math.exp(-(t**2) * sigma**2 / 2) * math.cos(t * mu),
        ak.sin(S("t") * x): math.exp(-(t**2) * sigma**2 / 2) * math.sin(t * mu),
    }
    for f, want in cases.items():
        got = num(ex.expectation(f, x, n), env)
        assert got == pytest.approx(want, rel=1e-10, abs=1e-12)
    # Monte-Carlo cross-check of the tilted moment, independent of the formula.
    draws = rng.normal(mu, sigma, 400_000)
    mc_close(
        draws**2 * np.exp(t * draws), num(ex.expectation(x**2 * ak.exp(S("t") * x), x, n), env)
    )


def test_gamma_expectations_of_power_log_and_reciprocal():
    sp = pytest.importorskip("sympy")
    from sympy.stats import E as SE
    from sympy.stats import Gamma as SGamma

    k, th, x = S("k"), S("theta"), S("x")
    g = ex.Gamma(k, th)
    env = {"k": 2.75, "theta": 0.8}
    G = SGamma("G", sp.Rational(11, 4), sp.Rational(4, 5))
    X = sp.Symbol("X", positive=True)
    pdf = sp.stats.density(G)(X)
    assert num(ex.expectation(x**2, x, g), env) == pytest.approx(float(SE(G**2)), rel=1e-12)
    want_log = float(sp.integrate(sp.log(X) * pdf, (X, 0, sp.oo)).evalf(30))
    assert num(ex.expectation(ak.log(x), x, g), env) == pytest.approx(want_log, rel=1e-10)
    assert num(ex.expectation(1 / x, x, g), env) == pytest.approx(float(SE(1 / G)), rel=1e-12)
    assert any("k" in c for c in ex.prob_side_conditions())
    # E[X^s] with symbolic s: theta^s Gamma(k+s)/Gamma(k); check d/ds at s=0 = E[log X].
    s = S("s")
    es = ex.expectation(x**s, x, g)
    assert num(es, {**env, "s": 0.6}) == pytest.approx(
        0.8**0.6 * math.gamma(3.35) / math.gamma(2.75), rel=1e-10
    )
    d = ak.diff(es, s).value
    assert num(d, {**env, "s": 0.0}) == pytest.approx(want_log, rel=1e-9)


def test_gamma_reciprocal_moment_diverges_at_shape_one():
    x = S("x")
    with pytest.raises(ak.AlkahestError) as e:
        ex.expectation(1 / x, x, ex.Gamma(POOL.integer(1), POOL.integer(2)))
    assert code_of(e) == "E-PROB-006"


# ---------------------------------------------------------------------------
# Multivariate normal
# ---------------------------------------------------------------------------


def _mvn_symbolic():
    mu = [S("m1"), S("m2"), S("m3")]
    s = {(i, j): S(f"s{min(i, j)}{max(i, j)}") for i in range(1, 4) for j in range(1, 4)}
    cov = [[s[(i, j)] for j in range(1, 4)] for i in range(1, 4)]
    return ex.MultivariateNormal(mu, cov)


def _random_spd(rng, d=3):
    a = rng.normal(size=(d, d))
    return a @ a.T + d * np.eye(d)


def _env(mu, cov):
    env = {f"m{i + 1}": mu[i] for i in range(3)}
    for i in range(3):
        for j in range(i, 3):
            env[f"s{i + 1}{j + 1}"] = cov[i, j]
    return env


@pytest.mark.parametrize("seed", [3, 4])
def test_mvn_conditional_matches_numpy_schur_complement(seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=3)
    cov = _random_spd(rng)
    mvn = _mvn_symbolic()
    a = S("a")
    c = mvn.conditional([2], [a])
    env = {**_env(mu, cov), "a": 0.7}
    s12 = cov[:2, 2:]
    s22 = cov[2:, 2:]
    want_mean = mu[:2] + (s12 @ np.linalg.inv(s22) @ (np.array([0.7]) - mu[2:]))
    want_cov = cov[:2, :2] - s12 @ np.linalg.inv(s22) @ s12.T
    got_mean = [num(e, env) for e in c.mean()]
    got_cov = [[num(e, env) for e in row] for row in c.covariance()]
    assert np.allclose(got_mean, want_mean, rtol=1e-11)
    assert np.allclose(got_cov, want_cov, rtol=1e-11)
    # two given, one left
    c2 = mvn.conditional([0, 2], [a, S("b")])
    env2 = {**env, "b": -0.4}
    idx, rest = [0, 2], [1]
    s12 = cov[np.ix_(rest, idx)]
    s22 = cov[np.ix_(idx, idx)]
    w = s12 @ np.linalg.inv(s22)
    want = mu[rest] + w @ (np.array([0.7, -0.4]) - mu[idx])
    assert num(c2.mean()[0], env2) == pytest.approx(float(want[0]), rel=1e-11)


def test_mvn_moments_covariance_and_quadratic_form_vs_monte_carlo():
    rng = np.random.default_rng(5)
    mu = rng.normal(size=3)
    cov = _random_spd(rng)
    env = _env(mu, cov)
    mvn = _mvn_symbolic()
    X, Y, Z = S("X"), S("Y"), S("Z")
    blocks = [([X, Y, Z], mvn)]
    draws = rng.multivariate_normal(mu, cov, size=600_000)
    x, y, z = draws.T
    # E[XY], E[X^2 Z], E[X^2 Y^2]
    for expr, sample in (
        (X * Y, x * y),
        (X**2 * Z, x**2 * z),
        (X**2 * Y**2, x**2 * y**2),
    ):
        mc_close(sample, num(ex.joint_expectation(expr, blocks), env))
    # exact second moments
    assert num(ex.joint_expectation(X * Y, blocks), env) == pytest.approx(
        cov[0, 1] + mu[0] * mu[1], rel=1e-12
    )
    cov_xy = num(ex.covariance(X + 2 * Y, Y - Z, blocks), env)
    a = np.array([1, 2, 0])
    b = np.array([0, 1, -1])
    assert cov_xy == pytest.approx(float(a @ cov @ b), rel=1e-11)
    var = num(ex.joint_variance(X - 3 * Z, blocks), env)
    c = np.array([1, 0, -3])
    assert var == pytest.approx(float(c @ cov @ c), rel=1e-11)
    # E[x^T A x] = tr(A S) + mu^T A mu
    A = [[1, 2, 0], [0, 3, -1], [4, 0, 1]]
    An = np.array(A, dtype=float)
    q = num(mvn.quadratic_form_expectation([[POOL.integer(v) for v in r] for r in A]), env)
    assert q == pytest.approx(float(np.trace(An @ cov) + mu @ An @ mu), rel=1e-11)
    # Cov of two quadratic forms: 2 tr(A S B S) + 4 mu^T A S B mu (A, B symmetric)
    Aq = X**2 + X * Y
    Bq = Y * Z
    As = np.array([[1, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])
    Bs = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]])
    want = 2 * np.trace(As @ cov @ Bs @ cov) + 4 * mu @ As @ cov @ Bs @ mu
    assert num(ex.covariance(Aq, Bq, blocks), env) == pytest.approx(float(want), rel=1e-10)


def test_mvn_pdf_mgf_and_linear_transform():
    sp = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(6)
    mu = rng.normal(size=3)
    cov = _random_spd(rng)
    env = _env(mu, cov)
    mvn = _mvn_symbolic()
    pt = [0.3, -0.2, 1.1]
    got = num(mvn.pdf([POOL.rational(3, 10), POOL.rational(-1, 5), POOL.rational(11, 10)]), env)
    assert got == pytest.approx(sp.multivariate_normal(mu, cov).pdf(pt), rel=1e-10)
    t = [S("t1"), S("t2"), S("t3")]
    tv = np.array([0.2, -0.1, 0.05])
    mgf = num(mvn.moment_generating_function(t), {**env, "t1": 0.2, "t2": -0.1, "t3": 0.05})
    assert mgf == pytest.approx(math.exp(mu @ tv + tv @ cov @ tv / 2), rel=1e-12)
    A = [[1, 1, 0], [0, 2, -1]]
    lt = mvn.linear_transform([[POOL.integer(v) for v in r] for r in A], [S("b1"), 0])
    An = np.array(A, dtype=float)
    want_cov = An @ cov @ An.T
    got_cov = [[num(e, env) for e in r] for r in lt.covariance()]
    assert np.allclose(got_cov, want_cov, rtol=1e-11)
    assert num(lt.mean()[0], {**env, "b1": 0.5}) == pytest.approx(mu[0] + mu[1] + 0.5)
    marg = mvn.marginal([2, 0])
    assert num(marg.covariance()[0][1], env) == pytest.approx(cov[2, 0])


def test_mvn_refusals():
    z = POOL.integer(0)
    with pytest.raises(ak.AlkahestError) as e:
        ex.MultivariateNormal([z, z], [[POOL.integer(1), S("a")], [S("b"), POOL.integer(1)]])
    assert code_of(e) == "E-PROB-001"
    with pytest.raises(ak.AlkahestError) as e:
        ex.MultivariateNormal([z, z], [[POOL.integer(1), POOL.integer(2)], [2, 1]])
    assert code_of(e) == "E-PROB-001"
    mvn = ex.MultivariateNormal([z, z], [[POOL.integer(2), 1], [1, 2]])
    with pytest.raises(ak.AlkahestError) as e:
        mvn.linear_transform([[1, POOL.integer(1)], [1, 1]])
    assert code_of(e) == "E-PROB-001"
    X, Y = S("X"), S("Y")
    with pytest.raises(ak.AlkahestError) as e:
        ex.joint_expectation(ak.sin(X), [([X, Y], mvn)])
    assert code_of(e) == "E-PROB-002"


def test_independent_blocks_and_univariate_products():
    X, Y = S("X"), S("Y")
    blocks = [(X, ex.Normal(S("mu"), S("sigma"))), (Y, ex.Exponential(S("lam")))]
    env = {"mu": 0.4, "sigma": 1.3, "lam": 2.5}
    assert num(ex.joint_expectation(X * Y, blocks), env) == pytest.approx(0.4 / 2.5)
    assert num(ex.covariance(X, Y, blocks), env) == pytest.approx(0.0, abs=1e-14)
    v = num(ex.joint_variance(X * Y, blocks), env)
    # Var(XY) = E[X^2]E[Y^2] - (E[X]E[Y])^2
    want = (1.3**2 + 0.4**2) * (2 / 2.5**2) - (0.4 / 2.5) ** 2
    assert v == pytest.approx(want, rel=1e-12)
    rng = np.random.default_rng(7)
    xs = rng.normal(0.4, 1.3, 500_000)
    ys = rng.exponential(1 / 2.5, 500_000)
    mc_close(xs * ys * ys, num(ex.joint_expectation(X * Y**2, blocks), env))


def test_sum_independent_closed_families_vs_monte_carlo():
    rng = np.random.default_rng(8)
    s = ex.sum_independent([ex.Normal(S("m1"), S("s1")), ex.Normal(S("m2"), S("s2"))])
    assert s.kind == "Normal"
    env = {"m1": 0.5, "m2": -1.0, "s1": 0.7, "s2": 1.4}
    assert num(s.variance(), env) == pytest.approx(0.49 + 1.96)
    g = ex.sum_independent([ex.Gamma(POOL.integer(2), S("th")), ex.Gamma(S("k"), S("th"))])
    assert g.kind == "Gamma"
    draws = rng.gamma(2, 0.6, 400_000) + rng.gamma(1.3, 0.6, 400_000)
    mc_close(draws, num(g.mean(), {"k": 1.3, "th": 0.6}))
    po = ex.sum_independent([ex.Poisson(S("a")), ex.Poisson(S("b"))])
    assert po.kind == "Poisson"
    bi = ex.sum_independent([ex.Binomial(S("n1"), S("q")), ex.Bernoulli(S("q"))])
    assert bi.kind == "Binomial"
    assert num(bi.mean(), {"n1": 6, "q": 0.25}) == pytest.approx(7 * 0.25)
    with pytest.raises(ak.AlkahestError) as e:
        ex.sum_independent([ex.Normal(0 * S("u"), S("s1")), ex.Poisson(S("a"))])
    assert code_of(e) == "E-PROB-002"


def test_negative_binomial_vs_scipy():
    st = pytest.importorskip("scipy.stats")
    nb = ex.NegativeBinomial(S("r"), S("p"))
    env = {"r": 2.5, "p": 0.35}
    ref = st.nbinom(2.5, 0.35)
    assert num(nb.mean(), env) == pytest.approx(ref.mean(), rel=1e-12)
    assert num(nb.variance(), env) == pytest.approx(ref.var(), rel=1e-12)
    assert num(nb.skewness(), env) == pytest.approx(float(ref.stats(moments="s")), rel=1e-10)
    assert num(nb.pdf(POOL.integer(4)), env) == pytest.approx(ref.pmf(4), rel=1e-12)
    with pytest.raises(ak.AlkahestError) as e:
        ex.NegativeBinomial(S("r"), POOL.integer(0))
    assert code_of(e) == "E-PROB-001"


# ---------------------------------------------------------------------------
# Order statistics
# ---------------------------------------------------------------------------


def test_order_statistics_with_symbolic_n():
    n = S("n")
    u = ex.order_statistic(ex.Uniform(POOL.integer(0), POOL.integer(1)), n)
    assert str(u.mean()) in {"n/(n + 1)", "n*(n + 1)^-1", "n/(1 + n)"} or num(
        u.mean(), {"n": 5}
    ) == pytest.approx(5 / 6)
    assert u.distribution().kind == "Beta"
    lam = S("lam")
    emax = ex.order_statistic(ex.Exponential(lam), n, "max")
    rng = np.random.default_rng(9)
    for nn in (2, 5):
        draws = rng.exponential(1 / 1.7, size=(300_000, nn)).max(axis=1)
        mc_close(draws, num(emax.mean(), {"n": nn, "lam": 1.7}))
        h = sum(1 / k for k in range(1, nn + 1))
        assert num(emax.mean(), {"n": nn, "lam": 1.7}) == pytest.approx(h / 1.7, rel=1e-12)
        h2 = sum(1 / k**2 for k in range(1, nn + 1))
        assert num(emax.variance(), {"n": nn, "lam": 1.7}) == pytest.approx(h2 / 1.7**2, rel=1e-10)
    emin = ex.order_statistic(ex.Exponential(lam), n, "min")
    assert emin.distribution().kind == "Exponential"
    # the density integrates the CDF: d/dx F_max = pdf
    x = S("x")
    cdf = emax.cdf(x)
    pdf = emax.pdf(x)
    env = {"n": 4, "lam": 1.7, "x": 0.9}
    assert num(ak.diff(cdf, x).value, env) == pytest.approx(num(pdf, env), rel=1e-10)
    with pytest.raises(ak.AlkahestError) as e:
        ex.order_statistic(ex.Normal(0 * n, POOL.integer(1)), n)
    assert code_of(e) == "E-PROB-002"
    with pytest.raises(ak.AlkahestError) as e:
        ex.order_statistic(ex.Exponential(lam), POOL.rational(3, 2))
    assert code_of(e) == "E-PROB-001"
