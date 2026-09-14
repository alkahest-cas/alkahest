"""Silent-error cases for entropy, divergence and cross-entropy.

One subsystem per module: see :mod:`corpus` for how the modules are
discovered, concatenated and checked.

Differential entropy is unusually rich in *plausible* wrong answers, which is
why it is worth its own subsystem.  Three families of trap live here:

* ``h`` is not ``H``.  It is not non-negative, it is not the limit of Shannon
  entropy under refinement, and it is not invariant under a change of
  variables.  Every one of those mistakes produces a clean number.
* ``D(P||Q)`` is ``+inf`` off a nested support, and the closed forms do not
  fail there — they return a finite, negative "divergence", which Gibbs'
  inequality forbids.
* ``D`` is not symmetric, so an implementation that sorts its arguments, caches
  on an unordered key, or reads a reference's ``(P, Q)`` the wrong way round
  returns the other branch's number with nothing to mark it.

Each refusal case is paired with its control: the nearest neighbour where the
same formula *does* apply and a value must come back.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import alkahest as ak
import alkahest.experimental as ex
from contracts import Case, Raises, RefusesOr, Returns

from ._shared import POOL, _int, _rat

IT_A = POOL.symbol("it_a")
IT_B = POOL.symbol("it_b")


def _it_value(build: Callable[[], Any]) -> Callable[[], float]:
    """Answer = an information-theoretic closed form reduced to a number.

    The Gaussian entropy is written with the interned pi symbol, which
    ``eval_expr`` resolves without a binding.
    """

    def op() -> float:
        return float(ak.eval_expr(build(), {}))

    return op


def _it_unconditional(build: Callable[[], Any], env: dict) -> Callable[[], float]:
    """Answer = the value, but only if it was returned *unconditionally*.

    ``experimental.prob_side_conditions()`` reports the hypotheses the call that
    just ran had to assume.  A number returned under an undischarged
    containment hypothesis is not an unconditional answer, so it is surfaced as
    a refusal rather than scored as a stated value.
    """

    def op() -> float:
        out = build()
        conds = ex.prob_side_conditions()
        if conds:
            raise ValueError(f"answer holds only under {conds}")
        return float(ak.eval_expr(out, env))

    return op


def _asymmetry_gap() -> Callable[[], float]:
    """Answer = ``D(P||Q) - D(Q||P)`` for a pair where the gap is large.

    Zero would mean the implementation has symmetrised the divergence — by
    sorting its arguments, by caching on an unordered key, or by transcribing a
    reference's ``(P, Q)`` the wrong way round.  Each of those is a silent
    error: the wrong branch is still a positive number of the right order of
    magnitude, and ``D`` really *is* symmetric for some pairs (two Bernoullis
    at ``p`` and ``1-p``), so a spot check can easily land on one.
    """

    def op() -> float:
        p, q = ex.Poisson(_rat(12, 5)), ex.Poisson(_int(1))
        forward = float(ak.eval_expr(ex.kl_divergence(p, q), {}))
        backward = float(ak.eval_expr(ex.kl_divergence(q, p), {}))
        return forward - backward

    return op


CASES: list[Case] = [
    # -- differential entropy is not Shannon entropy ------------------------
    Case(
        id="it_differential_entropy_of_a_narrow_uniform_is_negative",
        subsystem="information_theory",
        statement="h(Uniform(0, 1/2)) = log(1/2) < 0; clamping it at zero is the trap",
        op=_it_value(lambda: ex.Uniform(_int(0), _rat(1, 2)).entropy()),
        contract=Returns(-0.6931471805599453, tol=1e-14),
        verified_by="By hand from the definition: f = 2 on [0, 1/2], so "
        "-int f log f = -(1/2)(2)(log 2) = -log 2 = -0.69314718055994530942. "
        "scipy.stats.uniform(0, 0.5).entropy() agrees. Differential entropy is "
        "not an information content and is not bounded below by 0; a "
        "max(h, 0) anywhere in the pipeline, or a Shannon formula reused for "
        "the continuous case, returns 0 here and is indistinguishable from a "
        "correct answer for any *wide* uniform.",
    ),
    Case(
        id="it_control_differential_entropy_of_a_wide_uniform_is_positive",
        subsystem="information_theory",
        statement="h(Uniform(-2, 3)) = log 5 > 0 — the control for the negative case",
        op=_it_value(lambda: ex.Uniform(_int(-2), _int(3)).entropy()),
        contract=Returns(1.6094379124341003, tol=1e-14),
        verified_by="h = log(b - a) = log 5 = 1.6094379124341003746 (math.log(5)), "
        "confirmed by scipy.stats.uniform(-2, 5).entropy(). Same formula, same "
        "code path, opposite sign: without this control the negative case would "
        "be passed by an implementation that always returned a negative number, "
        "or that refused every continuous entropy.",
    ),
    Case(
        id="it_differential_entropy_is_not_invariant_under_a_change_of_variables",
        subsystem="information_theory",
        statement="h(LogNormal(mu, s)) - h(Normal(mu, s)) = mu, not 0",
        op=_it_value(
            lambda: (
                ex.LogNormal(_rat(7, 10), _rat(1, 2)).entropy()
                - ex.Normal(_rat(7, 10), _rat(1, 2)).entropy()
            )
        ),
        contract=Returns(0.7, tol=1e-13),
        verified_by="X = e^Y with Y ~ Normal(mu, s) is a smooth bijection onto (0, inf), "
        "and h shifts by E[log|dx/dy|] = E[Y] = mu = 0.7. Independently: "
        "scipy.stats.lognorm(s=0.5, scale=exp(0.7)).entropy() = 1.4257913526447273 "
        "and scipy.stats.norm(0.7, 0.5).entropy() = 0.7257913526447274, whose "
        "difference is 0.7 to 15 digits. Shannon entropy *is* relabelling-"
        "invariant; treating h the same way makes this difference 0, which is "
        "the classic conflation of the two quantities.",
    ),
    Case(
        id="it_poisson_entropy_has_no_closed_form",
        subsystem="information_theory",
        statement="H(Poisson(lam)) needs e^{-lam} sum lam^k log(k!)/k!, which is not closed",
        op=lambda: ex.Poisson(_rat(12, 5)).entropy(),
        contract=Raises("E-PROB-004"),
        verified_by="H = lam(1 - log lam) + e^{-lam} sum_k lam^k log(k!)/k! (Evans, Boersma "
        "1988). The residual sum is not elementary and has only asymptotic "
        "expansions; the standard asymptotic "
        "(1/2)log(2 pi e lam) - 1/(12 lam) - ... is an *approximation*, and at "
        "lam = 2.4 it gives 1.8748 against the true 1.8563 — wrong in the third "
        "digit, which no caller would notice.",
    ),
    Case(
        id="it_control_poissons_divergence_does_close",
        subsystem="information_theory",
        statement="D(Poisson(12/5)||Poisson(1)) = l1 log(l1/l2) + l2 - l1, even though H does not",
        op=_it_value(lambda: ex.kl_divergence(ex.Poisson(_rat(12, 5)), ex.Poisson(_int(1)))),
        contract=Returns(0.7011249696493598, tol=1e-13),
        verified_by="log(dP/dQ) = k log(l1/l2) - (l1 - l2), so the log(k!) that makes the "
        "entropy intractable cancels between the two densities. "
        "2.4*log(2.4) + 1 - 2.4 = 0.70112496964935984551 (mpmath, 30 dps — a "
        "float evaluation of the same expression loses three digits to "
        "cancellation). Confirmed by "
        "summing p(k)(log p(k) - log q(k)) over k = 0..200 in mpmath. The "
        "control for the entropy refusal above: refusing the divergence too "
        "would be a false refusal, and a gate of refusals alone proves nothing.",
    ),
    # -- KL off a nested support --------------------------------------------
    Case(
        id="it_kl_with_non_nested_support_is_infinite_not_the_formula",
        subsystem="information_theory",
        statement="D(Uniform(0,1) || Uniform(0,1/2)) = +inf; the formula there gives -log 2",
        op=lambda: ex.kl_divergence(ex.Uniform(_int(0), _int(1)), ex.Uniform(_int(0), _rat(1, 2))),
        contract=Raises("E-PROB-006"),
        verified_by="P gives mass 1/2 to (1/2, 1], where Q has density 0, so log(dP/dQ) is "
        "+inf on a set of P-measure 1/2 and D = +inf by definition (Cover & "
        "Thomas 2.3, with the 0 log(0/0) = 0 and p log(p/0) = inf conventions). "
        "Substituting into the same-family closed form log((b2-a2)/(b1-a1)) "
        "gives log(1/2) = -0.693: finite, plausible, and a *negative* KL "
        "divergence, which Gibbs' inequality forbids. Nothing in that "
        "arithmetic complains.",
    ),
    Case(
        id="it_control_kl_with_nested_support_is_the_formula",
        subsystem="information_theory",
        statement="D(Uniform(0,1) || Uniform(-2,3)) = log 5 — same formula, nested support",
        op=_it_value(
            lambda: ex.kl_divergence(ex.Uniform(_int(0), _int(1)), ex.Uniform(_int(-2), _int(3)))
        ),
        contract=Returns(1.6094379124341003, tol=1e-13),
        verified_by="[0,1] is inside [-2,3], so D = int_0^1 1*log(1/(1/5)) dx = log 5 = "
        "1.6094379124341003746 (math.log(5)); mpmath quad of p log(p/q) over "
        "[0, 1] agrees. The control for the +inf verdict: the gate must "
        "distinguish the nested case from the unnested one, not refuse both "
        "Uniform pairs.",
    ),
    Case(
        id="it_kl_between_bernoullis_with_a_degenerate_q_is_infinite",
        subsystem="information_theory",
        statement="D(Bernoulli(1/2) || Bernoulli(1)) = +inf: Q gives the atom 0 no mass",
        op=lambda: ex.kl_divergence(ex.Bernoulli(_rat(1, 2)), ex.Bernoulli(_int(1))),
        contract=Raises("E-PROB-006"),
        verified_by="P puts 1/2 on the atom {0} and Q puts 0 there, so the k = 0 term is "
        "(1/2) log((1/2)/0) = +inf. The closed form "
        "p1 log(p1/p2) + (1-p1) log((1-p1)/(1-p2)) has a log(0) in its second "
        "term, which in floating point is -inf times (1-p1) > 0 and in exact "
        "arithmetic is undefined; a library that drops the term as 'the zero "
        "case' returns (1/2)log(1/2) = -0.347, a negative divergence.",
    ),
    Case(
        id="it_control_kl_between_interior_bernoullis_closes",
        subsystem="information_theory",
        statement="D(Bernoulli(1/2) || Bernoulli(7/10)) is finite — the control for the atom gate",
        op=_it_value(lambda: ex.kl_divergence(ex.Bernoulli(_rat(1, 2)), ex.Bernoulli(_rat(7, 10)))),
        contract=Returns(0.08717669357238888, tol=1e-13),
        verified_by="0.5*log(0.5/0.7) + 0.5*log(0.5/0.3) = 0.087176693572388876 (mpmath, "
        "30 dps); scipy.stats.entropy([0.5, 0.5], [0.3, 0.7]) = "
        "0.08717669357238894 agrees. Moving q off the boundary must restore a "
        "value, or the atom gate would be a blanket refusal of every Bernoulli "
        "pair.",
    ),
    Case(
        id="it_symbolic_containment_is_disclosed_not_assumed",
        subsystem="information_theory",
        statement="D(Uniform(a,b)||Uniform(-2,3)) for symbolic a, b holds only if [a,b] is inside",
        op=_it_unconditional(
            lambda: ex.kl_divergence(ex.Uniform(IT_A, IT_B), ex.Uniform(_int(-2), _int(3))),
            {IT_A: -5.0, IT_B: 10.0},
        ),
        contract=RefusesOr(),
        verified_by="The closed form log((b2-a2)/(b1-a1)) is derived under [a1,b1] inside "
        "[a2,b2] and says nothing outside it. At a = -5, b = 10 the containment "
        "fails in both directions and the true value is +inf, while the formula "
        "gives log(5/15) = -1.0986122886681098 (math.log) — a negative KL "
        "divergence, which Gibbs' inequality forbids. With symbolic endpoints "
        "alkahest cannot decide the containment, so it publishes 'a + 2 >= 0' "
        "and '3 - b >= 0' on prob_side_conditions(); this case scores a "
        "disclosed answer as a refusal, which is what lets it fail if the "
        "disclosure is ever dropped.",
        note="The companion it_kl_with_non_nested_support_is_infinite_not_the_formula "
        "covers the decidable case, where the verdict is E-PROB-006 rather than a "
        "hypothesis. Both routes have to hold.",
    ),
    # -- asymmetry -----------------------------------------------------------
    Case(
        id="it_kl_is_not_symmetric",
        subsystem="information_theory",
        statement="D(Pois(12/5)||Pois(1)) - D(Pois(1)||Pois(12/5)) = 3.4 log 2.4 - 2.8, not 0",
        op=_asymmetry_gap(),
        contract=Returns(0.17659370700325978, tol=1e-12),
        verified_by="D(Pois(a)||Pois(b)) = a log(a/b) + b - a, so the two directions are "
        "2.4 log 2.4 - 1.4 = 0.70112496964935985 and log(1/2.4) + 1.4 = "
        "0.52453126264610006 (mpmath 30 dps), and their difference collapses to "
        "3.4 log 2.4 - 2.8 = 0.17659370700325978114. Each direction was also "
        "confirmed by summing p(k)(log p(k) - log q(k)) over k = 0..200. A "
        "divergence that sorts its arguments or caches on an unordered pair "
        "returns the same value twice and makes this exactly 0 — which is not "
        "an obviously wrong answer, because D genuinely is symmetric for some "
        "pairs (two Bernoullis at p and 1-p).",
    ),
    # -- the identity, and Gibbs --------------------------------------------
    Case(
        id="it_cross_entropy_is_entropy_plus_divergence",
        subsystem="information_theory",
        statement="H(P,Q) - H(P) - D(P||Q) = 0 for Normal(0,1) against Normal(2,3/2)",
        op=_it_value(
            lambda: (
                ex.cross_entropy(ex.Normal(_int(0), _int(1)), ex.Normal(_int(2), _rat(3, 2)))
                - ex.Normal(_int(0), _int(1)).entropy()
                - ex.kl_divergence(ex.Normal(_int(0), _int(1)), ex.Normal(_int(2), _rat(3, 2)))
            )
        ),
        contract=Returns(0.0, tol=1e-12),
        verified_by="-int p log q = -int p log p + int p log(p/q) is an identity, not an "
        "approximation. By hand: H(P,Q) = 0.5*log(2 pi * 2.25) + 5/4.5 = "
        "2.4355147524239482349, H(P) = 0.5*log(2 pi e) = 1.4189385332046727418 "
        "and "
        "D = log(1.5) + 5/4.5 - 0.5 = 1.0165762192192754931, whose sum is "
        "H(P,Q) to the last bit. "
        "Two of the three come from different closed-form tables here, so a "
        "transcription error in either shows up as a non-zero difference.",
    ),
    Case(
        id="it_gibbs_kl_is_non_negative",
        subsystem="information_theory",
        statement="min over 81 Bernoulli pairs of D(P||Q) is >= 0 (Gibbs), and 0 only at P = Q",
        op=lambda: _gibbs_min(),
        contract=Returns(0.0, tol=1e-12),
        verified_by="Gibbs' inequality: D(P||Q) >= 0 with equality iff P = Q almost "
        "everywhere (Cover & Thomas 2.6.3, from Jensen). The sweep covers "
        "p, q in {1/10, ..., 9/10} and includes the diagonal, where the true "
        "minimum 0 is attained — so the answer must be exactly 0, never "
        "negative. Every one of the 81 values was also computed directly as "
        "p log(p/q) + (1-p) log((1-p)/(1-q)) with math.log. A sign slip in a "
        "closed form typically stays positive near the diagonal and only goes "
        "negative further out, which is why this sweeps rather than samples.",
    ),
    # -- units ---------------------------------------------------------------
    Case(
        id="it_entropy_in_bits_is_nats_over_log_two",
        subsystem="information_theory",
        statement="H(Bernoulli(1/2)) is 1 bit and log 2 nats; reporting nats as bits is the trap",
        op=_it_value(lambda: ex.Bernoulli(_rat(1, 2)).entropy(base=_int(2))),
        contract=Returns(1.0, tol=1e-14),
        verified_by="A fair coin carries exactly one bit, by the definition of a bit. In "
        "nats the same quantity is log 2 = 0.6931471805599453; "
        "scipy.stats.entropy([0.5, 0.5], base=2) = 1.0 and the same call "
        "without base= gives 0.6931471805599453. The two differ by a factor of "
        "1.44, which is small enough to look like a modelling choice and large "
        "enough to be wrong.",
    ),
    Case(
        id="it_base_one_is_not_a_unit",
        subsystem="information_theory",
        statement="entropy(base=1) is a division by log 1 = 0, not a unit",
        op=lambda: ex.Bernoulli(_rat(1, 2)).entropy(base=_int(1)),
        contract=Raises("E-PROB-001"),
        verified_by="Converting nats to base b divides by log b, and log 1 = 0. There is no "
        "base-1 logarithm and no base-1 entropy. Carrying the division through "
        "gives inf (or a ZeroDivisionError deep in an evaluator), and inf is "
        "not the entropy of a fair coin under any convention.",
    ),
    # -- the pair quantities need a joint law -------------------------------
    Case(
        id="it_cross_family_divergence_is_refused",
        subsystem="information_theory",
        statement="D(Normal || Exponential) is not the same-family formula with the names swapped",
        op=lambda: ex.kl_divergence(ex.Normal(_int(0), _int(1)), ex.Exponential(_int(1))),
        contract=Raises("E-PROB-002"),
        verified_by="A Normal puts mass on (-inf, 0), where an Exponential has none, so this "
        "particular pair is in fact +inf; more generally int p log(p/q) across "
        "two families has no elementary value and the same-family closed forms "
        "are not it. Reaching for the nearest table entry and substituting "
        "(mu, sigma) for (lambda) would return a number with no relation to the "
        "integral.",
    ),
]


def _gibbs_min() -> float:
    """Minimum of ``D(P||Q)`` over every Bernoulli pair on a 9x9 grid.

    Gibbs' inequality says the answer is ``0``: non-negative everywhere, and
    attained exactly on the diagonal.  Sweeping rather than sampling one point
    is what makes a sign error in a closed form visible — a wrong sign is
    typically still positive near the diagonal and only goes negative further
    out.  Bernoulli because the whole sweep then costs a few milliseconds; the
    same sweep over the continuous families, where each value carries a
    verifying quadrature, lives in the Rust test suite instead.
    """
    laws = [ex.Bernoulli(_rat(k, 10)) for k in range(1, 10)]
    worst = math.inf
    for p in laws:
        for q in laws:
            v = float(ak.eval_expr(ex.kl_divergence(p, q), {}))
            worst = min(worst, v)
    return worst
