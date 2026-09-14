"""Silent-error cases for probability.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.
"""

from __future__ import annotations

from typing import Any, Callable

import alkahest as ak
import alkahest.experimental as ex
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import POOL, _int, _rat

PROB_T = POOL.symbol("prob_t")
PROB_X = POOL.symbol("prob_x")
PROB_Z = POOL.symbol("prob_z")


def _prob_value(build: Callable[[], Any], env: dict | None = None) -> Callable[[], float]:
    """Answer = a probability closed form reduced to a number.

    The Gaussian normalising constant is written with the interned π symbol,
    which the evaluators resolve without a binding.
    """

    def op() -> float:
        return float(ak.eval_expr(build(), (env or {})))

    return op


def _prob_phi(build: Callable[[], Any], env: dict | None = None) -> Callable[[], tuple]:
    """Answer = ``(Re φ(t), Im φ(t))``.

    ``φ`` is complex-valued, so the real evaluator refuses it with ``E-EVAL-009``
    and ``evaluate(..., mode="complex")`` is the route.  Reading that refusal as
    a failure would make every characteristic-function case vacuously "refused".
    """

    def op() -> tuple:
        v = ak.evaluate(build(), (env or {}), mode="complex").value
        return (float(v.real), float(v.imag))

    return op


def _prob_unconditional(build: Callable[[], Any], env: dict) -> Callable[[], float]:
    """Answer = a probability closed form's value, but only if it was *unconditional*.

    ``experimental.prob_side_conditions()`` reports the hypotheses the call that
    just ran had to assume.  A number returned under an undischarged hypothesis
    is not an unconditional answer — the caller was told — so it is surfaced as
    a refusal rather than scored as a stated value.  That is the whole point of
    the channel: without it, ``F(x)`` for a symbolic ``x`` and ``F(x)`` for an
    ``x`` known to be in the support are indistinguishable at the call site.
    """

    def op() -> float:
        out = build()
        conds = ex.prob_side_conditions()
        if conds:
            raise ValueError(f"answer holds only under {conds}")
        return float(ak.eval_expr(out, env))

    return op


def _phi_real_mode(build: Callable[[], Any], env: dict) -> Callable[[], float]:
    """Answer = ``phi(t)`` forced through a **real** evaluator.

    ``phi`` is complex.  A real evaluator that quietly hands back ``Re phi`` is
    the silent error this case exists to catch: it is a clean number of the
    right magnitude, and nothing in the return value says the imaginary part —
    which for a non-symmetric law carries the whole of the mean — was dropped.
    Refusing is the only acceptable outcome.
    """

    def op() -> float:
        r = ak.evaluate(build(), env, mode="f64")
        if r.value is None:
            raise ValueError(f"real evaluation declined: {r.status} {r.reason}")
        return float(r.value)

    return op


def _moment_via_charfun(build: Callable[[], Any], n: int) -> Callable[[], float]:
    """Answer = ``φ⁽ⁿ⁾(0) / iⁿ`` — i.e. ``E[Xⁿ]`` reached by differentiating the
    characteristic function at the origin instead of by the moment table.

    ``φ⁽ⁿ⁾(0) = iⁿ·E[Xⁿ]``, so the n-th derivative is real for even ``n`` and
    purely imaginary for odd ``n``; this reads off whichever part carries it.
    The two routes share no closed form, which is what makes the agreement worth
    asserting.
    """

    def op() -> float:
        e = build().characteristic_function(PROB_T)
        for _ in range(n):
            e = ak.simplify(ak.diff(e, PROB_T))
            if isinstance(e, ak.DerivedResult):
                e = e.value
        v = ak.evaluate(e, {PROB_T: 0.0}, mode="complex").value
        return [v.real, v.imag, -v.real, -v.imag][n % 4]

    return op


CASES: list[Case] = [
    Case(
        id="prob_lognormal_mean_is_not_exp_mu",
        subsystem="probability",
        statement="E[X] for X ~ LogNormal(1/5, 1/2) is e^{mu+sigma^2/2}, not e^{mu}",
        op=_prob_value(lambda: ex.LogNormal(_rat(1, 5), _rat(1, 2)).mean()),
        contract=Returns(1.3840306459807514, tol=1e-12),
        verified_by="mpmath 50 dps: quad(x·lognormal_pdf(x, 0.2, 0.5), [0, 1, 10, inf]) = "
        "1.3840306459807514212, equal to exp(0.2 + 0.125). The trap is that "
        "mu is *not* the mean of X — reading it as one gives e^0.2 = 1.2214, "
        "which is 12% low and entirely plausible.",
    ),
    Case(
        id="prob_lognormal_variance_is_not_sigma_squared",
        subsystem="probability",
        statement="Var[X] for X ~ LogNormal(1/5, 1/2) is (e^{s^2}-1)e^{2m+s^2}, not s^2",
        op=_prob_value(lambda: ex.LogNormal(_rat(1, 5), _rat(1, 2)).variance()),
        contract=Returns(0.5440622821430536, tol=1e-12),
        verified_by="mpmath 50 dps: quad((x-m)^2·lognormal_pdf(x, 0.2, 0.5), [0, 1, 10, inf]) "
        "with m = e^{0.325} gives 0.54406228214305359365. Reading sigma as the "
        "standard deviation of X gives 0.25 — a factor of 2.2 out.",
    ),
    Case(
        id="prob_gamma_is_shape_scale_not_shape_rate",
        subsystem="probability",
        statement="E[X] for X ~ Gamma(7/2, 4/5) under the shape-scale convention is k*theta",
        op=_prob_value(lambda: ex.Gamma(_rat(7, 2), _rat(4, 5)).mean()),
        contract=Returns(2.8, tol=1e-12),
        verified_by="mpmath: quad(x·gamma_pdf(x, k=3.5, theta=0.8), [0, 1, 10, inf]) = 2.8 = "
        "k·theta. The shape-*rate* convention — which SciPy's `gamma(a, scale=)` "
        "and R's `rgamma(rate=)` both also expose — would give k/theta = 4.375. "
        "Both are clean numbers and neither looks wrong on its own.",
    ),
    Case(
        id="prob_normal_fourth_moment_is_three_sigma_fourth",
        subsystem="probability",
        statement="E[X^4] for X ~ Normal(0, 3/2) is 3*sigma^4, not sigma^4",
        op=_prob_value(lambda: ex.Normal(_int(0), _rat(3, 2)).moment(4)),
        contract=Returns(15.1875, tol=1e-12),
        verified_by="Isserlis/Wick: E[X^4] = 3 sigma^4 for a centred Gaussian, = 3·(1.5)^4 = "
        "15.1875; confirmed by mpmath quad(x^4·normal_pdf(x, 0, 1.5), "
        "[-inf, 0, inf]). Dropping the double-factorial gives 5.0625.",
    ),
    Case(
        id="prob_poisson_third_moment_is_the_touchard_polynomial",
        subsystem="probability",
        statement="E[X^3] for X ~ Poisson(12/5) is lam^3 + 3 lam^2 + lam, not lam^3",
        op=_prob_value(lambda: ex.Poisson(_rat(12, 5)).moment(3)),
        contract=Returns(33.504, tol=1e-12),
        verified_by="Touchard: E[X^n] = sum_r S(n,r) lam^r, so E[X^3] = lam^3+3lam^2+lam = "
        "33.504 at lam = 2.4; confirmed by mpmath nsum(k^3·poisson_pmf(k), "
        "[0, inf]) = 33.504. A Poisson has all its cumulants equal to lam, which "
        "makes 'the third moment is lam^3' a tempting and wrong shortcut (13.824).",
    ),
    Case(
        id="prob_binomial_second_moment_is_not_the_mean_squared",
        subsystem="probability",
        statement="E[X^2] for X ~ Binomial(5, 7/20) is np(1-p) + (np)^2, not (np)^2",
        op=_prob_value(lambda: ex.Binomial(_int(5), _rat(7, 20)).moment(2)),
        contract=Returns(4.2, tol=1e-12),
        verified_by="Exact sum over the support: sum_k k^2 C(5,k) p^k (1-p)^{5-k} at p = 0.35 "
        "is 4.2, and Var + mean^2 = 5(0.35)(0.65) + (1.75)^2 = 1.1375 + 3.0625 = "
        "4.2. Confusing E[X^2] with E[X]^2 gives 3.0625.",
    ),
    Case(
        id="prob_uniform_third_moment_is_not_the_cube_of_the_mean",
        subsystem="probability",
        statement="E[X^3] for X ~ Uniform(-2, 3) is (b^4-a^4)/(4(b-a)), not ((a+b)/2)^3",
        op=_prob_value(lambda: ex.Uniform(_int(-2), _int(3)).moment(3)),
        contract=Returns(3.25, tol=1e-12),
        verified_by="(b^4 - a^4)/(4(b-a)) = (81-16)/20 = 3.25, matching mpmath "
        "quad(x^3/(b-a), [-2, 3]). E[X]^3 = 0.5^3 = 0.125 — an asymmetric "
        "interval makes the two differ by a factor of 26.",
    ),
    Case(
        id="prob_control_erlang_cdf_closes",
        subsystem="probability",
        statement="P(X <= 12/5) for X ~ Gamma(3, 4/5) — integer shape, so the Erlang sum closes",
        op=_prob_value(lambda: ex.Gamma(_int(3), _rat(4, 5)).cdf(PROB_X), {PROB_X: 2.4}),
        contract=Returns(0.5768099188731565, tol=1e-12),
        verified_by="mpmath 50 dps: quad(gamma_pdf(t, k=3, theta=0.8), [0, 2.4]) = "
        "0.57680991887315648468, equal to the Erlang form "
        "1 - e^{-u} sum_{j<3} u^j/j! at u = 3. The control for the "
        "non-integer-shape refusal below: the family that *does* close must not "
        "be swept up in it.",
    ),
    Case(
        id="prob_gamma_cdf_at_a_non_integer_shape_refuses",
        subsystem="probability",
        statement="P(X <= x) for Gamma with a symbolic shape needs the incomplete gamma",
        op=lambda: ex.Gamma(POOL.symbol("kshape"), POOL.symbol("kscale")).cdf(PROB_X),
        contract=Raises("E-PROB-004"),
        verified_by="F(x) = P(k, x/theta), the regularised lower incomplete gamma "
        "(DLMF 8.2.4). It is not elementary for general k, and alkahest has no "
        "primitive for it, so there is nothing to return. Emitting the Erlang "
        "finite sum anyway would be wrong for every non-integer k.",
    ),
    Case(
        id="prob_cdf_below_the_support_is_zero_not_the_formula",
        subsystem="probability",
        statement="P(X <= -5) for X ~ Gamma(3, 4/5) is 0; the Erlang formula there gives -7396.87",
        op=_prob_value(lambda: ex.Gamma(_int(3), _rat(4, 5)).cdf(_int(-5))),
        contract=Returns(0.0, tol=1e-15),
        verified_by="A Gamma puts no mass below 0, so P(X <= -5) = 0 by definition. The "
        "in-support closed form 1 - e^{-u} sum_{j<3} u^j/j! at u = -6.25 "
        "evaluates to -7396.8706522947593 — a *negative probability*, returned "
        "with no complaint. Value computed by hand from the Erlang sum and "
        "confirmed with mpmath.",
    ),
    Case(
        id="prob_cdf_above_the_support_is_one_not_the_formula",
        subsystem="probability",
        statement="P(X <= 7) for X ~ Uniform(-2, 3) is 1; the linear formula there gives 1.8",
        op=_prob_value(lambda: ex.Uniform(_int(-2), _int(3)).cdf(_int(7))),
        contract=Returns(1.0, tol=1e-15),
        verified_by="All of a Uniform(-2, 3)'s mass is below 7, so the probability is 1. "
        "(x-a)/(b-a) at x = 7 is 9/5 = 1.8 — a probability greater than one. "
        "Both values are one line of arithmetic from the definition.",
    ),
    Case(
        id="prob_control_cdf_inside_the_support_is_the_formula",
        subsystem="probability",
        statement="P(X <= 1/2) for X ~ Uniform(-2, 3) is 1/2 — the control for the two clamps",
        op=_prob_value(lambda: ex.Uniform(_int(-2), _int(3)).cdf(_rat(1, 2))),
        contract=Returns(0.5, tol=1e-14),
        verified_by="(0.5 - (-2))/(3 - (-2)) = 2.5/5 = 0.5. Without this the two clamp cases "
        "would be passed by an implementation that returned 0 or 1 for every "
        "argument.",
    ),
    Case(
        id="prob_quantile_outside_zero_one_is_refused",
        subsystem="probability",
        statement="Uniform(-2, 3).quantile(2) has no meaning; the formula returns 8",
        op=lambda: ex.Uniform(_int(-2), _int(3)).quantile(_int(2)),
        contract=Raises("E-PROB-001"),
        verified_by="F^-1 is defined on [0, 1]. a + p(b-a) at p = 2 is -2 + 2·5 = 8, a point "
        "outside the support [−2, 3] that the quantile is supposed to name. "
        "There is no probability 2, so no value is correct here.",
    ),
    Case(
        id="prob_control_exponential_quantile_closes",
        subsystem="probability",
        statement="Exponential(7/4).quantile(9/10) = -log(1-p)/lambda",
        op=_prob_value(lambda: ex.Exponential(_rat(7, 4)).quantile(_rat(9, 10))),
        contract=Returns(1.3157629102823118, tol=1e-12),
        verified_by="-log(1 - 0.9)/1.75 = log(10)/1.75 = 1.3157629102823118194 (mpmath, "
        "50 dps), and substituting it back gives 1 - e^{-1.75 q} = 0.9 exactly. "
        "The control for the out-of-range refusal above.",
    ),
    Case(
        id="prob_normal_quantile_refuses_for_want_of_erfinv",
        subsystem="probability",
        statement="The normal quantile needs erf^-1, which is not a registered primitive",
        op=lambda: ex.Normal(_int(0), _int(1)).quantile(_rat(9, 10)),
        contract=Raises("E-PROB-004"),
        verified_by="Phi^-1(p) = sqrt(2)·erf^-1(2p-1) (DLMF 7.17). erf^-1 has no elementary "
        "closed form and alkahest does not implement it, so the honest answer "
        "is that there is none to give — not a series truncation dressed as one.",
    ),
    Case(
        id="prob_characteristic_function_convention_is_e_itx",
        subsystem="probability",
        statement="phi_Normal(t) = e^{i mu t - sigma^2 t^2/2}; a 2pi or a sign slip is invisible",
        op=_prob_phi(
            lambda: ex.Normal(_rat(7, 10), _rat(11, 10)).characteristic_function(_rat(13, 10))
        ),
        contract=Returns((0.2207720571448093, 0.2839944144298271), tol=1e-12),
        verified_by="mpmath 50 dps, from the definition phi(t) = int p(x)e^{itx}dx: "
        "quad(normal_pdf(x, 0.7, 1.1)·cos(1.3x), [-inf, 0, inf]) = "
        "0.22077205714480930348 and the sin integral = 0.28399441442982708525, "
        "equal to e^{i(0.7)(1.3) - (1.21)(1.69)/2}. alkahest's `fourier_transform` "
        "is unitary ordinary-frequency (e^{-2 pi i x xi}), so phi(t) = F{p}(-t/2pi): "
        "a 2 pi *and* a sign apart. Both near-misses were computed and are "
        "detectably different — dropping the 2 pi gives Re = 2.49e-18 rather than "
        "0.2208, and flipping the sign negates the imaginary part — so this "
        "case can actually fail.",
    ),
    Case(
        id="prob_moment_from_the_characteristic_function_agrees_with_the_table",
        subsystem="probability",
        statement="phi''(0) = -E[X^2] for Binomial(5, 7/20): two routes, no shared closed form",
        op=_moment_via_charfun(lambda: ex.Binomial(_int(5), _rat(7, 20)), 2),
        contract=Returns(4.2, tol=1e-10),
        verified_by="Exact sum over the support: sum_k k^2 C(5,k) p^k (1-p)^{5-k} at p = 0.35 "
        "is 4.2 = np(1-p) + (np)^2. Reached here the other way — differentiate "
        "phi(t) = (1-p+p e^{it})^5 twice and evaluate at t = 0, using "
        "phi^(n)(0) = i^n E[X^n] (Feller II, XV.4) — so the moment table and the "
        "phi table have to agree without sharing a line of code. It is "
        "deliberately the same number as "
        "`prob_binomial_second_moment_is_not_the_mean_squared`: that case pins "
        "the table, this one pins the two routes to each other, and a "
        "convention slip in phi shows up here as a wrong moment.",
    ),
    Case(
        id="prob_lognormal_characteristic_function_refuses",
        subsystem="probability",
        statement="phi for a log-normal has no closed form at all",
        op=lambda: ex.LogNormal(_int(0), _int(1)).characteristic_function(PROB_T),
        contract=Raises("E-PROB-004"),
        verified_by="The log-normal moment series sum (it)^n e^{n mu + n^2 sigma^2/2}/n! "
        "diverges for every t != 0 (the moments grow like e^{n^2 sigma^2/2}, "
        "faster than n!), and the log-normal is the standard example of a "
        "distribution not determined by its moments — Heyde 1963, J. London "
        "Math. Soc. 38. There is no elementary or standard-special-function "
        "expression to return.",
    ),
    Case(
        id="prob_control_normal_characteristic_function_closes",
        subsystem="probability",
        statement="phi for a normal is e^{i mu t - sigma^2 t^2/2} — the control for the refusal",
        op=_prob_phi(lambda: ex.Normal(_int(0), _int(1)).characteristic_function(_int(1))),
        contract=Returns((0.6065306597126334, 0.0), tol=1e-12),
        verified_by="phi_{N(0,1)}(1) = e^{-1/2} = 0.60653065971263342360, real because a "
        "centred normal is symmetric; mpmath quad(normal_pdf(x)·cos x, "
        "[-inf, 0, inf]) agrees and the sin integral is 0. A gate made only of "
        "refusals is passed by a module that refuses every characteristic "
        "function.",
    ),
    Case(
        id="prob_divergent_expectation_is_refused_not_evaluated",
        subsystem="probability",
        statement="E[e^{X^2}] under Normal(0,1) diverges: the integrand is e^{x^2/2}",
        op=lambda: ex.expectation(
            POOL.func("exp", [PROB_X**2]), PROB_X, ex.Normal(_int(0), _int(1))
        ),
        contract=Raises("E-PROB-006"),
        verified_by="The integrand is e^{x^2/2}/sqrt(2 pi), which grows without bound, so the "
        "integral diverges and E[e^{X^2}] does not exist. Formally completing "
        "the square gives -1/sqrt(-1) style nonsense that simplifies to a clean "
        "finite number, which is exactly the trap: the MGF of a chi-squared "
        "exists only for t < 1/2 and this is t = 1.",
    ),
    Case(
        id="prob_control_normal_mgf_at_one_converges",
        subsystem="probability",
        statement="E[e^X] under Normal(0,1) is e^{1/2} — the convergent neighbour of E[e^{X^2}]",
        op=_prob_value(
            lambda: ex.expectation(POOL.func("exp", [PROB_X]), PROB_X, ex.Normal(_int(0), _int(1)))
        ),
        contract=Returns(1.6487212707001282, tol=1e-11),
        verified_by="The Gaussian MGF is E[e^{tX}] = e^{mu t + sigma^2 t^2/2}, so at t = 1 it "
        "is sqrt(e) = 1.6487212707001281468; mpmath quad(e^x·normal_pdf(x), "
        "[-inf, 0, inf]) agrees to 20 digits. Without this the divergence case "
        "would be passed by a module that refused every exponential "
        "expectation.",
    ),
    Case(
        id="prob_product_of_two_variates_needs_a_joint_law",
        subsystem="probability",
        statement="E[A*B] is not determined by the marginals of A and B",
        op=lambda: ex.expectation_affine(
            POOL.symbol("pv_a") * POOL.symbol("pv_b"),
            [
                (POOL.symbol("pv_a"), ex.Normal(_int(0), _int(1))),
                (POOL.symbol("pv_b"), ex.Normal(_int(0), _int(1))),
            ],
        ),
        contract=Raises("E-PROB-002"),
        verified_by="E[AB] = E[A]E[B] + Cov(A,B), and the covariance is not a function of the "
        "two marginals: A ~ N(0,1) with B = A gives E[AB] = 1, while A ~ N(0,1) "
        "with B = -A gives -1, and both have the same marginals. Answering 0 "
        "(the independent case) would be right for one joint law and wrong for "
        "uncountably many others, with nothing in the call to say which was "
        "assumed.",
    ),
    Case(
        id="prob_control_linearity_of_expectation_needs_no_independence",
        subsystem="probability",
        statement="E[3X + 2Y + 5] = 3E[X] + 2E[Y] + 5 for any joint law",
        op=_prob_value(
            lambda: ex.expectation_affine(
                3 * POOL.symbol("pv_a") + 2 * POOL.symbol("pv_b") + _int(5),
                [
                    (POOL.symbol("pv_a"), ex.Normal(_int(1), _int(2))),
                    (POOL.symbol("pv_b"), ex.Exponential(_int(4))),
                ],
            )
        ),
        contract=Returns(8.5, tol=1e-12),
        verified_by="Linearity of expectation holds for every joint law, dependent or not: "
        "3(1) + 2(1/4) + 5 = 8.5, using E[Normal(1,2)] = 1 and "
        "E[Exponential(4)] = 1/4 (rate parametrisation). The control for the "
        "product refusal — the distinction being that linearity needs no "
        "independence and a covariance does.",
    ),
    Case(
        id="prob_a_negative_scale_is_not_a_distribution",
        subsystem="probability",
        statement="Normal(mu, -1) does not exist; sigma > 0 is part of the definition",
        op=lambda: ex.Normal(POOL.symbol("pmu"), _int(-1)),
        contract=Raises("E-PROB-001"),
        verified_by="The density e^{-(x-mu)^2/(2 sigma^2)}/(sigma sqrt(2 pi)) integrates to "
        "-1 at sigma = -1: sigma^2 is blind to the sign but the normalising "
        "constant is not. Every moment then comes back with the wrong sign and "
        "nothing in the arithmetic complains.",
    ),
    Case(
        id="prob_control_a_symbolic_scale_is_carried_not_decided",
        subsystem="probability",
        statement="Normal(mu, sigma) with symbolic sigma is accepted, carrying sigma > 0",
        op=lambda: len(ex.Normal(POOL.symbol("pmu"), POOL.symbol("psigma")).constraints()),
        contract=Returns(1),
        verified_by="A symbolic sigma cannot be decided either way, so refusing it would be a "
        "false refusal — the distribution is perfectly well defined for every "
        "sigma > 0. The constraint is returned as a predicate for the caller to "
        "discharge; there is exactly one of them for a Normal. The control that "
        "the constructor check above is a *decision* and not a blanket refusal.",
    ),
    Case(
        id="prob_symbolic_cdf_discloses_that_it_is_the_in_support_branch",
        subsystem="probability",
        statement="F(x) for symbolic x is the in-support branch; at x = -5 it gives -7396.87",
        op=_prob_unconditional(lambda: ex.Gamma(_int(3), _rat(4, 5)).cdf(PROB_X), {PROB_X: -5.0}),
        contract=RefusesOr(0.0),
        verified_by="A Gamma puts no mass below 0, so P(X <= -5) = 0 by definition. The "
        "Erlang closed form 1 - e^{-u} sum_{j<3} u^j/j! at u = -6.25 is "
        "-7396.8706522947593 — a negative probability. With a symbolic "
        "argument alkahest cannot place x, so it publishes the hypothesis "
        "'x > 0' on `prob_side_conditions()`; this case scores a disclosed "
        "answer as a refusal, which is what makes it able to fail if the "
        "disclosure is ever dropped.",
        note="The companion `prob_cdf_below_the_support_is_zero_not_the_formula` covers "
        "the decidable argument, where alkahest returns the exact 0 instead of "
        "disclosing a hypothesis. Both routes have to hold.",
    ),
    Case(
        id="prob_characteristic_function_is_not_silently_realified",
        subsystem="probability",
        statement="phi_Normal(1.3) is complex; a real evaluator must refuse, not return Re",
        op=_phi_real_mode(
            lambda: ex.Normal(_rat(7, 10), _rat(11, 10)).characteristic_function(_rat(13, 10)),
            {},
        ),
        contract=RefusesOr(),
        verified_by="phi(t) = e^{i mu t - sigma^2 t^2/2} = 0.22077205714480930348 + "
        "0.28399441442982708525i (mpmath 50 dps, from int p(x)e^{itx}dx). No "
        "real number is its value. Handing back the real part alone would be a "
        "clean number of the right magnitude that has silently discarded the "
        "imaginary part — which for a non-symmetric law carries the mean, since "
        "phi'(0) = i·E[X].",
        note="The control is `prob_characteristic_function_convention_is_e_itx`, which "
        "asks the same question through `evaluate(..., mode='complex')` and gets "
        "both components. A gate made only of this case would be passed by a "
        "module whose characteristic functions could not be evaluated at all.",
    ),
    Case(
        id="prob_lognormal_mgf_diverges_and_is_refused",
        subsystem="probability",
        statement="M_X(t) = E[e^{tX}] for a LogNormal is +infinity for every t > 0",
        op=lambda: ex.LogNormal(_int(0), _int(1)).moment_generating_function(_int(1)),
        contract=Raises("E-PROB-006"),
        verified_by="int_0^inf e^{tx} e^{-(log x)^2/2}/(x sqrt(2 pi)) dx diverges for every "
        "t > 0: substituting x = e^u gives int e^{t e^u - u^2/2} du/sqrt(2 pi), "
        "and t e^u outgrows u^2/2 without bound. mpmath quad over [0, inf] at "
        "t = 1 fails to converge and nsum of the moment series "
        "sum t^n e^{n^2/2}/n! diverges (the terms grow: e^{n^2/2}/n! -> inf). "
        "Aitchison & Brown, *The Lognormal Distribution* §2.4; the same fact "
        "underlies Heyde 1963 (J. London Math. Soc. 38) on the log-normal not "
        "being determined by its moments. A CAS that completes the square in "
        "the exponent returns a clean e^{...} that is the value of no integral.",
        note="The control is `prob_control_normal_mgf_closes_and_is_returned`: a gate "
        "made only of this case is passed by a library with no MGF at all.",
    ),
    Case(
        id="prob_control_normal_mgf_closes_and_is_returned",
        subsystem="probability",
        statement="M_X(13/10) for X ~ Normal(7/10, 11/10) is e^{mu t + sigma^2 t^2/2}",
        op=_prob_value(
            lambda: ex.Normal(_rat(7, 10), _rat(11, 10)).moment_generating_function(_rat(13, 10))
        ),
        contract=Returns(6.906410235712461, tol=1e-11),
        verified_by="mpmath 40 dps, from the definition M(t) = int e^{tx} p(x) dx: "
        "quad(e^{1.3x}·normal_pdf(x, 0.7, 1.1), [-inf, 0.7, inf]) = "
        "6.9064102357124608639, equal to exp(0.7·1.3 + 1.21·1.69/2). The "
        "control for the log-normal refusal: a Gaussian MGF is entire, so "
        "there is no strip to report and the answer is unconditional.",
    ),
    Case(
        id="prob_exponential_mgf_outside_its_strip_is_refused",
        subsystem="probability",
        statement="lambda/(lambda - t) is the Exponential MGF only for t < lambda",
        op=lambda: ex.Exponential(_rat(7, 4)).moment_generating_function(_rat(7, 2)),
        contract=Raises("E-PROB-006"),
        verified_by="E[e^{tX}] = int_0^inf e^{tx} lambda e^{-lambda x} dx = "
        "lambda/(lambda - t) *only* where lambda - t > 0; at t = 2 lambda the "
        "integrand is lambda e^{+lambda x}, which grows without bound, so the "
        "expectation is +infinity (Feller II, XIII.2). The closed form there "
        "evaluates to 1.75/(1.75 - 3.5) = -1 — a *negative* moment generating "
        "function, when E[e^{tX}] > 0 for every t at which it exists. Both "
        "numbers are one line of arithmetic; only one of them is an answer.",
        note="The control is `prob_control_exponential_mgf_inside_its_strip`.",
    ),
    Case(
        id="prob_control_exponential_mgf_inside_its_strip",
        subsystem="probability",
        statement="M_X(1/2) for X ~ Exponential(7/4) is 1.4, unconditionally",
        op=_prob_unconditional(
            lambda: ex.Exponential(_rat(7, 4)).moment_generating_function(_rat(1, 2)), {}
        ),
        contract=Returns(1.4, tol=1e-12),
        verified_by="mpmath 40 dps: quad(e^{x/2}·1.75 e^{-1.75x}, [0, inf]) = 1.4 exactly, "
        "= 1.75/(1.75 - 0.5) = 7/5. The rate is numeric and 1/2 < 7/4, so the "
        "convergence condition is *decided* rather than carried, and this case "
        "scores the answer only if `prob_side_conditions()` comes back empty. "
        "Without it the out-of-strip refusal would be passed by a library that "
        "refused every exponential MGF.",
    ),
    Case(
        id="prob_symbolic_mgf_argument_reports_its_strip_out_of_band",
        subsystem="probability",
        statement="M(t) for Exponential(lam) with symbolic t must publish lam - t > 0",
        op=_prob_unconditional(
            lambda: ex.Exponential(POOL.symbol("gf_lam")).moment_generating_function(PROB_T),
            {POOL.symbol("gf_lam"): 1.75, PROB_T: 3.5},
        ),
        contract=RefusesOr(),
        verified_by="With both lambda and t symbolic the strip lambda - t > 0 cannot be "
        "decided, so the closed form is conditional and the hypothesis is the "
        "whole of what distinguishes it from a theorem. Evaluated at the "
        "parameter point used here — lambda = 1.75, t = 3.5, outside the strip "
        "— it gives -1, while E[e^{3.5X}] = +infinity (the integrand is "
        "1.75 e^{1.75x}). This case scores a *disclosed* answer as a refusal, "
        "so it fails if the disclosure is ever dropped.",
        note="`prob_control_exponential_mgf_inside_its_strip` is the companion where "
        "the condition is decidable and the channel is correctly empty.",
    ),
    Case(
        id="prob_pgf_of_a_continuous_law_is_a_category_error",
        subsystem="probability",
        statement="G_X(z) = sum_k z^k P(X = k) is meaningless for a Normal",
        op=lambda: ex.Normal(_int(0), _int(1)).probability_generating_function(PROB_Z),
        contract=Raises("E-PROB-002"),
        verified_by="A probability generating function is a power series whose coefficients "
        "are P(X = k) for k = 0, 1, 2, ... (Feller I, XI.1). A normal is "
        "continuous, so P(X = k) = 0 for every k and the series is identically "
        "0 — it is not e^{mu log z + sigma^2 log^2 z/2}. That expression is "
        "what the formal rewrite E[z^X] = E[e^{X log z}] produces, and it is "
        "the *moment* generating function at log z: a clean, finite, plausible "
        "answer to a different question.",
        note="The control is `prob_control_poisson_pgf_closes`.",
    ),
    Case(
        id="prob_control_poisson_pgf_closes",
        subsystem="probability",
        statement="G_X(1/2) for X ~ Poisson(12/5) is e^{lam(z-1)} = e^{-6/5}",
        op=_prob_value(
            lambda: ex.Poisson(_rat(12, 5)).probability_generating_function(PROB_Z),
            {PROB_Z: 0.5},
        ),
        contract=Returns(0.3011942119122021, tol=1e-12),
        verified_by="mpmath 40 dps, summing the definition: "
        "nsum(0.5^k e^{-2.4} 2.4^k / k!, [0, inf]) = "
        "0.30119421191220209664, equal to exp(2.4·(0.5 - 1)) = e^{-1.2}. The "
        "control for the category-error refusal above — a gate made only of "
        "refusals is passed by a library with no PGF at all.",
    ),
    Case(
        id="prob_lognormal_has_no_cumulants",
        subsystem="probability",
        statement="kappa_n = K^{(n)}(0) needs K = log M, and a LogNormal has no M",
        op=lambda: ex.LogNormal(_rat(1, 5), _rat(1, 2)).cumulant(3),
        contract=Raises("E-PROB-006"),
        verified_by="kappa_n is by definition the n-th derivative of log E[e^{tX}] at the "
        "origin, and E[e^{tX}] = +infinity for every t > 0 for a log-normal "
        "(see `prob_lognormal_mgf_diverges_and_is_refused`), so K exists on no "
        "neighbourhood of 0. The moment-cumulant recursion nevertheless runs "
        "on the log-normal's moments e^{n mu + n^2 sigma^2/2} and produces "
        "finite numbers — the coefficients of a divergent series. That is "
        "exactly the failure mode: arithmetic that completes and means "
        "nothing.",
        note="The control is `prob_control_lognormal_skewness_still_exists`: the "
        "refusal must not sweep up the shape statistics, which are defined "
        "from central moments and are finite.",
    ),
    Case(
        id="prob_control_lognormal_skewness_still_exists",
        subsystem="probability",
        statement="gamma_1 for LogNormal(0, 1) is (e^{s^2}+2)sqrt(e^{s^2}-1), not a refusal",
        op=_prob_value(lambda: ex.LogNormal(_int(0), _int(1)).skewness()),
        contract=Returns(6.184877138632555, tol=1e-9),
        verified_by="Skewness is E[((X-mu)/sigma)^3], a ratio of *central moments*, and a "
        "log-normal has moments of every order. At sigma = 1 the standard "
        "closed form (e^{s^2}+2)sqrt(e^{s^2}-1) (Aitchison & Brown §2.3) is "
        "(e+2)sqrt(e-1) = 6.1848771386325547948 to 20 digits by mpmath. "
        "Refusing it because the cumulants do not exist would be a false "
        "refusal — the quantity is standard, finite and printed in every "
        "reference.",
    ),
    Case(
        id="prob_excess_kurtosis_of_a_normal_is_zero_not_three",
        subsystem="probability",
        statement="excess kurtosis subtracts the 3; a Normal has gamma_2 = 0",
        op=_prob_value(lambda: ex.Normal(_rat(7, 10), _rat(11, 10)).excess_kurtosis()),
        contract=Returns(0.0, tol=1e-9),
        verified_by="E[(X-mu)^4] = 3 sigma^4 for any Gaussian (Isserlis), so the fourth "
        "standardised moment is 3 and the *excess* kurtosis is 3 - 3 = 0. The "
        "two conventions differ by exactly the constant a reader is least "
        "likely to notice: both 0 and 3 are plausible for a bell curve, and a "
        "library that returned the raw kurtosis under this name would look "
        "right for every heavy-tailed law. Confirmed by mpmath "
        "quad(((x-0.7)/1.1)^4·normal_pdf(x, 0.7, 1.1), [-inf, 0.7, inf]) = 3.",
    ),
]
