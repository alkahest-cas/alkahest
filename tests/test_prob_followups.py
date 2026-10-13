"""Probability follow-ups to the joint-law work (F23).

* general ``k``-th order statistics for any continuous base law with a
  closed-form CDF (density, CDF, moments via the integrator where it closes);
* the large-``n`` asymptotic entropy of ``Binomial`` / ``NegativeBinomial`` /
  ``Poisson``, returned by its own function and never by ``entropy``;
* ``E[x**k cos(w x + phi)]`` and ``sin`` under a ``Normal``;
* entropy, KL divergence and mutual information of multivariate normals.

Oracles: scipy / numpy formulas, mpmath quadrature, exact enumeration of the
defining sums, and Monte-Carlo with a fixed seed and CLT tolerances.
"""

from __future__ import annotations

import math
from fractions import Fraction

import alkahest as ak
import pytest
from alkahest import experimental as ex

np = pytest.importorskip("numpy")
sps = pytest.importorskip("scipy.stats")
mpmath = pytest.importorskip("mpmath")

POOL = ak.ExprPool()
S = POOL.symbol
RNG = np.random.default_rng(20261012)


def E(v):
    """An exact constant Expr from an int / Fraction / short decimal."""
    f = Fraction(v).limit_denominator(10**6)
    return POOL.rational(f.numerator, f.denominator)


def num(expr, env: dict | None = None) -> float:
    return float(ak.eval_expr(expr, {S(k): float(v) for k, v in (env or {}).items()}))


def code_of(excinfo) -> str:
    return getattr(excinfo.value, "code", "")


def mc_close(sample, value: float, k: float = 6.0) -> None:
    se = sample.std(ddof=1) / math.sqrt(sample.size)
    assert abs(sample.mean() - value) <= k * se + 1e-12, (sample.mean(), value, se)


# ---------------------------------------------------------------------------
# (1) k-th order statistics
# ---------------------------------------------------------------------------


def _base_laws():
    """(name, alkahest law, scipy frozen law) -- continuous, closed-form CDF."""
    return [
        ("normal", ex.Normal(E(1), E(Fraction(3, 2))), sps.norm(1.0, 1.5)),
        ("gamma", ex.Gamma(POOL.integer(3), POOL.integer(2)), sps.gamma(3.0, scale=2.0)),
        ("beta", ex.Beta(POOL.integer(2), POOL.integer(3)), sps.beta(2.0, 3.0)),
        ("lognormal", ex.LogNormal(E(0), E(Fraction(1, 2))), sps.lognorm(0.5)),
        ("uniform", ex.Uniform(POOL.integer(-1), POOL.integer(3)), sps.uniform(-1.0, 4.0)),
        ("exponential", ex.Exponential(E(Fraction(7, 5))), sps.expon(scale=1 / 1.4)),
    ]


@pytest.mark.parametrize(("n", "k"), [(5, 2), (4, 4), (6, 3)])
def test_kth_order_statistic_density_and_cdf_vs_scipy_and_simulation(n, k):
    x = S("x")
    for name, law, ref in _base_laws():
        os_ = ex.order_statistic(law, n, k=k)
        # scipy order-statistic density from the definition
        pts = [float(ref.ppf(q)) for q in (0.2, 0.5, 0.8)]
        coeff = math.factorial(n) / (math.factorial(k - 1) * math.factorial(n - k))
        for t in pts:
            F, f = ref.cdf(t), ref.pdf(t)
            want_pdf = coeff * F ** (k - 1) * (1 - F) ** (n - k) * f
            assert num(os_.pdf(x), {"x": t}) == pytest.approx(want_pdf, rel=1e-9), name
            want_cdf = sum(math.comb(n, j) * F**j * (1 - F) ** (n - j) for j in range(k, n + 1))
            assert num(os_.cdf(x), {"x": t}) == pytest.approx(want_cdf, rel=1e-9), name
        # simulation: P(X_(k) <= t) is a Bernoulli mean
        draws = np.sort(ref.rvs(size=(120_000, n), random_state=RNG), axis=1)[:, k - 1]
        t = pts[1]
        mc_close((draws <= t).astype(float), num(os_.cdf(x), {"x": t}))


def test_kth_order_statistic_density_integrates_to_one_and_matches_cdf_derivative():
    x = S("x")
    law = ex.Normal(E(0), E(1))
    os_ = ex.order_statistic(law, 5, k=2)
    pdf = os_.pdf(x)
    cdf = os_.cdf(x)
    total = mpmath.quad(lambda t: num(pdf, {"x": float(t)}), [-8, 0, 8])
    assert float(total) == pytest.approx(1.0, abs=1e-8)
    d = ak.diff(cdf, x).value
    for t in (-0.7, 0.1, 0.9):
        assert num(d, {"x": t}) == pytest.approx(num(pdf, {"x": t}), rel=1e-9)


def _os_moments(cdf, pdf, n, k, lo, hi):
    c = mpmath.factorial(n) / (mpmath.factorial(k - 1) * mpmath.factorial(n - k))

    def w(t):
        return c * cdf(t) ** (k - 1) * (1 - cdf(t)) ** (n - k) * pdf(t)

    m1 = mpmath.quad(lambda t: t * w(t), [lo, hi])
    m2 = mpmath.quad(lambda t: t * t * w(t), [lo, hi])
    return float(m1), float(m2 - m1 * m1)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_kth_order_statistic_moments_vs_mpmath_where_the_integrator_closes(k):
    n = 3
    cases = [
        (
            ex.Beta(POOL.integer(2), POOL.integer(3)),
            lambda t: mpmath.betainc(2, 3, 0, t, regularized=True),
            lambda t: t * (1 - t) ** 2 / mpmath.beta(2, 3),
            0,
            1,
        ),
        (
            ex.Gamma(POOL.integer(2), POOL.integer(1)),
            lambda t: 1 - mpmath.e ** (-t) * (1 + t),
            lambda t: t * mpmath.e ** (-t),
            0,
            mpmath.inf,
        ),
        (
            ex.Uniform(POOL.integer(-1), POOL.integer(3)),
            lambda t: (t + 1) / 4,
            lambda t: mpmath.mpf(1) / 4,
            -1,
            3,
        ),
    ]
    for law, cdf, pdf, lo, hi in cases:
        os_ = ex.order_statistic(law, n, k=k)
        m, v = _os_moments(cdf, pdf, n, k, lo, hi)
        assert num(os_.mean()) == pytest.approx(m, rel=1e-9)
        assert num(os_.variance()) == pytest.approx(v, rel=1e-8)


def test_kth_order_statistic_exponential_is_renyi_with_symbolic_rate():
    lam = S("lam")
    for k, mean, var in [(1, 1 / 3, 1 / 9), (2, 5 / 6, 13 / 36), (3, 11 / 6, 49 / 36)]:
        os_ = ex.order_statistic(ex.Exponential(lam), 3, k=k)
        assert num(os_.mean(), {"lam": 2.0}) == pytest.approx(mean / 2.0, rel=1e-12)
        assert num(os_.variance(), {"lam": 2.0}) == pytest.approx(var / 4.0, rel=1e-12)


def test_kth_order_statistic_monte_carlo_moments_for_gamma_and_beta():
    for law, ref in [
        (ex.Gamma(POOL.integer(2), POOL.integer(1)), sps.gamma(2.0)),
        (ex.Beta(POOL.integer(2), POOL.integer(3)), sps.beta(2.0, 3.0)),
    ]:
        os_ = ex.order_statistic(law, 4, k=2)
        draws = np.sort(ref.rvs(size=(200_000, 4), random_state=RNG), axis=1)[:, 1]
        mc_close(draws, num(os_.mean()))
        mc_close(draws**2, num(os_.variance()) + num(os_.mean()) ** 2)


def test_kth_order_statistic_extremes_are_normalised_and_uniform_is_beta():
    law = ex.Uniform(E(0), E(1))
    assert ex.order_statistic(law, 5, k=1).kind == "min"
    assert ex.order_statistic(law, 5, k=5).kind == "max"
    mid = ex.order_statistic(law, 5, k=2)
    assert mid.kind == "kth"
    beta = mid.distribution()
    assert beta.kind == "Beta"
    # U_(2:5) ~ Beta(2, 4): mean 1/3
    assert num(beta.mean()) == pytest.approx(1 / 3)
    assert num(mid.mean()) == pytest.approx(1 / 3)


def test_kth_order_statistic_symbolic_n_and_k_pdf_matches_literal():
    n, k, x = S("n"), S("k"), S("x")
    law = ex.Normal(E(0), E(1))
    sym = ex.order_statistic(law, n, k=k)
    lit = ex.order_statistic(law, 5, k=2)
    sym_pdf = sym.pdf(x)
    conds = " ".join(str(c) for c in ex.prob_side_conditions())
    got = num(sym_pdf, {"n": 5, "k": 2, "x": 0.3})
    assert got == pytest.approx(num(lit.pdf(x), {"x": 0.3}), rel=1e-12)
    assert got == pytest.approx(0.26291489935968493, rel=1e-10)
    assert "k" in conds
    assert "Integer" in conds
    with pytest.raises(ak.AlkahestError) as e:
        sym.cdf(x)
    assert code_of(e) == "E-PROB-004"
    with pytest.raises(ak.AlkahestError) as e:
        sym.mean()
    assert code_of(e) == "E-PROB-002"


def test_kth_order_statistic_refusals():
    x = S("x")
    with pytest.raises(ak.AlkahestError) as e:
        ex.order_statistic(ex.Normal(E(0), E(1)), 3, k=4)
    assert code_of(e) == "E-PROB-001"
    with pytest.raises(ak.AlkahestError) as e:
        ex.order_statistic(ex.Normal(E(0), E(1)), 3, k=0)
    assert code_of(e) == "E-PROB-001"
    with pytest.raises(ak.AlkahestError) as e:
        ex.order_statistic(ex.Poisson(E(2)), 3, k=2)
    assert code_of(e) == "E-PROB-002"
    with pytest.raises(ValueError):
        ex.order_statistic(ex.Normal(E(0), E(1)), 3, "min", k=2)
    # a Normal's erf CDF is outside the integrator: the moment is refused, not guessed
    with pytest.raises(ak.AlkahestError) as e:
        ex.order_statistic(ex.Normal(E(0), E(1)), 4, k=2).mean()
    assert code_of(e) in {"E-PROB-003", "E-PROB-006"}
    # Gamma with a non-integer shape has no closed-form CDF
    with pytest.raises(ak.AlkahestError) as e:
        ex.order_statistic(ex.Gamma(E(Fraction(5, 2)), E(1)), 3, k=2).cdf(x)
    assert code_of(e) == "E-PROB-004"


# ---------------------------------------------------------------------------
# (2) asymptotic entropy
# ---------------------------------------------------------------------------


def _discrete_entropy(pmf):
    pmf = pmf[pmf > 0]
    return float(-(pmf * np.log(pmf)).sum())


def test_entropy_asymptotic_binomial_symbolic_n_vs_exact_sum():
    n, p = S("n"), S("p")
    h = ex.entropy_asymptotic(ex.Binomial(n, p))
    for nn, pp in [(200, 1 / 3), (400, 0.5), (1000, 0.1)]:
        exact = _discrete_entropy(sps.binom.pmf(np.arange(nn + 1), nn, pp))
        got = num(h, {"n": nn, "p": pp})
        var = nn * pp * (1 - pp)
        assert abs(got - exact) < 2.0 / var**2, (nn, pp, got, exact)
        # strictly better than the leading Gaussian term alone
        gauss = 0.5 * math.log(2 * math.pi * math.e * var)
        if pp != 0.5:  # the skewness term vanishes at p = 1/2
            assert abs(got - exact) < abs(gauss - exact)


def test_entropy_asymptotic_negative_binomial_and_poisson_vs_exact_sum():
    r, p, lam = S("r"), S("p"), S("lam")
    hn = ex.entropy_asymptotic(ex.NegativeBinomial(r, p))
    for rr, pp in [(80, 1 / 3), (200, 0.2)]:
        ks = np.arange(0, 4000)
        exact = _discrete_entropy(sps.nbinom.pmf(ks, rr, pp))
        # the error is O(1/r^2) at fixed p
        assert abs(num(hn, {"r": rr, "p": pp}) - exact) < 1.0 / rr**2
    hp = ex.entropy_asymptotic(ex.Poisson(lam))
    for ll in (40.0, 160.0):
        exact = _discrete_entropy(sps.poisson.pmf(np.arange(0, 1500), ll))
        assert abs(num(hp, {"lam": ll}) - exact) < 1.0 / ll**2
    # base conversion: bits
    hb = ex.entropy_asymptotic(ex.Poisson(lam), POOL.integer(2))
    assert num(hb, {"lam": 40.0}) == pytest.approx(num(hp, {"lam": 40.0}) / math.log(2), rel=1e-12)


def test_exact_entropy_still_refuses_and_points_to_the_asymptotic():
    n, p, r = S("n"), S("p"), S("r")
    for d in (ex.Binomial(n, p), ex.NegativeBinomial(r, p)):
        with pytest.raises(ak.AlkahestError) as e:
            d.entropy()
        assert code_of(e) == "E-PROB-004"
        assert "entropy_asymptotic" in str(e.value)
    with pytest.raises(ak.AlkahestError) as e:
        ex.entropy_asymptotic(ex.Normal(E(0), E(1)))
    assert code_of(e) == "E-PROB-002"


# ---------------------------------------------------------------------------
# (3) E[x^k cos(wx + phi)] under a Normal
# ---------------------------------------------------------------------------


def _trig_oracle(k, a, w, phi, mu, sigma, kind):
    trig = mpmath.cos if kind == "cos" else mpmath.sin
    f = lambda t: (  # noqa: E731
        t**k * mpmath.exp(a * t) * trig(w * t + phi) * mpmath.npdf(t, mu, sigma)
    )
    return float(mpmath.quad(f, [-mpmath.inf, mu, mpmath.inf]))


@pytest.mark.parametrize(
    ("k", "a", "w", "phi", "mu", "sigma", "kind"),
    [
        (3, 0.0, 2.0, 0.0, 0.3, 0.7, "cos"),
        (2, 0.5, 1.5, 0.2, -0.4, 1.2, "sin"),
        (4, 0.0, 1.0, 0.0, 0.0, 1.0, "cos"),
        (1, 0.0, 1.0, 0.0, 0.0, 1.0, "sin"),
        (5, 0.0, 3.0, 0.7, 1.1, 0.5, "cos"),
        (0, 0.0, 2.0, 0.0, 0.3, 0.7, "sin"),
    ],
)
def test_polynomial_times_trig_under_normal_vs_mpmath(k, a, w, phi, mu, sigma, kind):
    x = S("x")
    law = ex.Normal(E(mu), E(sigma))
    trig = ak.cos if kind == "cos" else ak.sin
    f = x**k * ak.exp(E(a) * x) * trig(E(w) * x + E(phi))
    got = num(ex.expectation(f, x, law))
    want = _trig_oracle(k, a, w, phi, mu, sigma, kind)
    assert got == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_x_to_the_k_cos_x_closed_form_standard_normal():
    # E[x^4 cos x] = e^{-1/2} (He_4-type polynomial at the tilted mean): -2 e^{-1/2}
    x = S("x")
    got = num(ex.expectation(x**4 * ak.cos(x), x, ex.Normal(E(0), E(1))))
    assert got == pytest.approx(-2 * math.exp(-0.5), rel=1e-12)


def test_trig_moment_with_symbolic_parameters_matches_numeric():
    x, mu, sg, w = S("x"), S("mu"), S("sigma"), S("w")
    e = ex.expectation(x**2 * ak.cos(w * x), x, ex.Normal(mu, sg))
    env = {"mu": 0.4, "sigma": 0.9, "w": 1.3}
    want = _trig_oracle(2, 0.0, 1.3, 0.0, 0.4, 0.9, "cos")
    assert num(e, env) == pytest.approx(want, rel=1e-9)


# ---------------------------------------------------------------------------
# (4) multivariate-normal information measures
# ---------------------------------------------------------------------------

M0 = np.array([0, 1, -1.0])
S0 = np.array([[2, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 1.5]])
M1 = np.array([0.5, 0, 1.0])
S1 = np.array([[1, 0.2, 0], [0.2, 3.0, 0.5], [0, 0.5, 1.0]])


def _mvn(m, c):
    return ex.MultivariateNormal(
        [E(v) for v in m],
        [[E(v) for v in row] for row in c],
    )


def test_mvn_entropy_vs_scipy_and_is_mean_free():
    for m, c in [(M0, S0), (M1, S1)]:
        got = num(_mvn(m, c).entropy())
        assert got == pytest.approx(sps.multivariate_normal(m, c).entropy(), rel=1e-12)
    # bits
    got2 = num(_mvn(M0, S0).entropy(POOL.integer(2)))
    assert got2 == pytest.approx(sps.multivariate_normal(M0, S0).entropy() / math.log(2), rel=1e-12)
    # symbolic: shift-invariant, scales as log of the determinant
    mu1, mu2, s1, s2, r = S("mu1"), S("mu2"), S("s1"), S("s2"), S("r")
    sym = ex.MultivariateNormal([mu1, mu2], [[s1, r], [r, s2]])
    env = {"mu1": 5.0, "mu2": -2.0, "s1": 2.0, "s2": 1.0, "r": 0.5}
    want = sps.multivariate_normal([0, 0], [[2, 0.5], [0.5, 1]]).entropy()
    assert num(sym.entropy(), env) == pytest.approx(want, rel=1e-12)


def test_mvn_kl_divergence_vs_closed_form_numeric_integration_and_univariate():
    p, q = _mvn(M0, S0), _mvn(M1, S1)
    inv = np.linalg.inv(S1)
    dm = M1 - M0
    want = 0.5 * (
        np.trace(inv @ S0) + dm @ inv @ dm - 3 + math.log(np.linalg.det(S1) / np.linalg.det(S0))
    )
    assert num(p.kl_divergence(q)) == pytest.approx(want, rel=1e-12)
    assert num(p.kl_divergence(q)) == pytest.approx(3.4159208491228323, rel=1e-12)
    assert num(p.kl_divergence(p)) == pytest.approx(0.0, abs=1e-12)
    # not symmetric
    assert num(q.kl_divergence(p)) != pytest.approx(num(p.kl_divergence(q)), rel=1e-3)
    # Monte-Carlo: E_p[log p/q]
    rng = np.random.default_rng(1)
    z = rng.multivariate_normal(M0, S0, size=400_000)
    lp = sps.multivariate_normal(M0, S0).logpdf(z)
    lq = sps.multivariate_normal(M1, S1).logpdf(z)
    mc_close(lp - lq, num(p.kl_divergence(q)))
    # d = 1 agrees with the (quadrature-gated) univariate KL
    one_p = ex.MultivariateNormal([E(Fraction(1, 2))], [[E(Fraction(9, 4))]])
    one_q = ex.MultivariateNormal([E(-1)], [[E(Fraction(1, 4))]])
    uni = ex.kl_divergence(
        ex.Normal(E(Fraction(1, 2)), E(Fraction(3, 2))), ex.Normal(E(-1), E(Fraction(1, 2)))
    )
    assert num(one_p.kl_divergence(one_q)) == pytest.approx(num(uni), rel=1e-10)


def test_bivariate_normal_mutual_information_is_minus_half_log_one_minus_rho_squared():
    rho = S("rho")
    mvn = ex.MultivariateNormal([E(0), E(0)], [[E(1), rho], [rho, E(1)]])
    mi = mvn.mutual_information([0], [1])
    for r in (0.6, -0.3, 0.95):
        assert num(mi, {"rho": r}) == pytest.approx(-0.5 * math.log(1 - r * r), rel=1e-12)
    # with general variances the correlation is s12/sqrt(s11 s22) and the variances cancel
    s11, s22, s12 = S("s11"), S("s22"), S("s12")
    gen = ex.MultivariateNormal([S("m1"), S("m2")], [[s11, s12], [s12, s22]])
    env = {"s11": 3.0, "s22": 0.5, "s12": 0.7, "m1": 1.0, "m2": 2.0}
    r2 = 0.7**2 / (3.0 * 0.5)
    assert num(gen.mutual_information([0], [1]), env) == pytest.approx(
        -0.5 * math.log(1 - r2), rel=1e-12
    )
    # sigma-parametrised: the standard deviations drop out symbolically
    sa, sb = S("sa"), S("sb")
    par = ex.MultivariateNormal([E(0), E(0)], [[sa**2, rho * sa * sb], [rho * sa * sb, sb**2]])
    got = par.mutual_information([0], [1])
    assert num(got, {"rho": 0.6, "sa": 2.0, "sb": 0.3}) == pytest.approx(0.22314355131420974)
    assert "sa" not in str(got)
    assert "sb" not in str(got)
    # 0.6 -> 0.2231435513142097 (scipy/numpy value)
    assert -0.5 * math.log(1 - 0.36) == pytest.approx(0.22314355131420974)


def test_bivariate_normal_mutual_information_vs_numeric_integration():
    sint = pytest.importorskip("scipy.integrate")
    c = np.array([[1.5, 0.9], [0.9, 1.0]])
    mvn = _mvn([0, 0], c)
    joint = sps.multivariate_normal([0, 0], c)
    m1, m2 = sps.norm(0, math.sqrt(c[0, 0])), sps.norm(0, math.sqrt(c[1, 1]))

    def integrand(y, x):
        pj = joint.pdf([x, y])
        return pj * math.log(pj / (m1.pdf(x) * m2.pdf(y)))

    val, _ = sint.dblquad(integrand, -9, 9, -9, 9, epsabs=1e-10, epsrel=1e-10)
    assert num(mvn.mutual_information([0], [1])) == pytest.approx(val, rel=1e-7)


def test_mvn_mutual_information_between_groups_vs_numpy():
    mvn = _mvn(M0, S0)

    def mi(a, b):
        det = lambda ix: np.linalg.det(S0[np.ix_(ix, ix)])  # noqa: E731
        return 0.5 * math.log(det(a) * det(b) / det(a + b))

    assert num(mvn.mutual_information([0], [1, 2])) == pytest.approx(mi([0], [1, 2]), rel=1e-11)
    assert num(mvn.mutual_information([0, 1], [2])) == pytest.approx(mi([0, 1], [2]), rel=1e-11)
    assert num(mvn.mutual_information([0], [2])) == pytest.approx(mi([0], [2]), rel=1e-11)
    # symmetric in its arguments, and zero for a block-diagonal covariance
    assert num(mvn.mutual_information([1, 2], [0])) == pytest.approx(
        num(mvn.mutual_information([0], [1, 2])), rel=1e-12
    )
    diag = _mvn([0, 0, 0], np.diag([1.0, 2.0, 3.0]))
    assert num(diag.mutual_information([0], [1, 2])) == pytest.approx(0.0, abs=1e-14)
    # the data-processing check: I(X0; X1,X2) >= I(X0; X1)
    assert num(mvn.mutual_information([0], [1, 2])) >= num(mvn.mutual_information([0], [1]))


def test_mvn_information_refusals_and_hypotheses():
    mvn = _mvn(M0, S0)
    for a, b in [([0], [0, 1]), ([], [1]), ([0], [5])]:
        with pytest.raises(ak.AlkahestError) as e:
            mvn.mutual_information(a, b)
        assert code_of(e) == "E-PROB-002"
    with pytest.raises(ak.AlkahestError) as e:
        mvn.kl_divergence(_mvn([0, 0], np.eye(2)))
    assert code_of(e) == "E-PROB-002"
    with pytest.raises(ak.AlkahestError) as e:
        mvn.entropy(POOL.integer(1))
    assert code_of(e) == "E-PROB-001"
    # a symbolic covariance carries its positive-definiteness hypotheses
    r = S("r")
    sym = ex.MultivariateNormal([E(0), E(0)], [[E(1), r], [r, E(1)]])
    sym.entropy()
    assert ex.prob_side_conditions()
