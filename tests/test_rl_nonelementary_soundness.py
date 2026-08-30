"""Soundness guards for the RL "honest refusal" reward (directional issue #3).

The RL integration environment rewards a model for refusing to integrate a
*non-elementary* integrand. That reward is only trustworthy if the label
"non-elementary" is itself sound. Two independent properties must hold:

1. The engine must NEVER classify an actually-ELEMENTARY integrand as
   ``NonElementary`` (error code ``E-INT-004``). A false ``NonElementary``
   verdict (bug B2, fixed in #219) would let a training pipeline mint a
   hard-negative from an elementary integrand and reward a *wrong* refusal.

2. The RL hard-negative labels must come from a curated known-non-elementary
   corpus, not from trusting the engine's own verdict — because that verdict is
   sound-but-incomplete (it gives up with ``E-INT-001`` on genuinely
   non-elementary integrands, and historically could emit a false ``E-INT-004``).

These tests need only ``alkahest`` (no ``verifiers`` / ``datasets`` extra).
"""

from __future__ import annotations

import alkahest as ak
from alkahest.alkahest import ArbBall, interval_eval

# Known-ELEMENTARY integrands. Each has a genuine elementary antiderivative, so
# integrate() must NOT return the E-INT-004 (NonElementary) verdict. It is
# acceptable for the engine to *succeed* or to decline with a non-E-INT-004
# code (e.g. E-INT-001 "not implemented"); the only forbidden outcome is a
# false NonElementary claim.
KNOWN_ELEMENTARY = [
    # B2 regression: ∫ (exp(x)·log(x) + exp(x)/x) dx = exp(x)·log(x).
    # Before #219 the engine falsely reported this as NonElementary.
    "exp(x)*log(x) + exp(x)/x",
    # Rational / polynomial.
    "2*x",
    "x^3 - 2*x + 1",
    "1/(1+x^2)",  # atan(x)
    "1/x",  # log(x)
    # Trigonometric.
    "cos(x)",
    "sin(x)*cos(x)",
    # By parts.
    "x*exp(x)",
    # Algebraic.
    "1/sqrt(1-x^2)",  # asin(x)
    "exp(x)",
    # Risch Gap F regression: an exp tower whose exponent has a logarithmic
    # part gives the Risch DE coefficient f = k·η' a simple pole with a
    # *positive integer* residue.  The solution v then has a pole that the
    # right-hand side c does not predict, and the old `E = gcd(D, D')`
    # denominator bound could not represent it — so the solver returned "no
    # rational solution" and the exp-tower caller minted E-INT-004 for these
    # perfectly elementary integrands.
    "exp(x + log(x))",  # = x·eˣ,        ∫ = (x−1)·eˣ
    "exp(x + 2*log(x))",  # = x²·eˣ,       ∫ = (x²−2x+2)·eˣ
    "exp(x + log(x))/x",  # = eˣ,          ∫ = eˣ
    "exp(x + log(x^2 + 1))",  # = (x²+1)·eˣ,   ∫ = (x²−2x+3)·eˣ
    "x*exp(x + log(x))",  # = x²·eˣ
    # The residue here (4000) is past the solver's internal resonance-search
    # cap, so the denominator bound is not provably complete.  Declining
    # (E-INT-001) is the honest answer; E-INT-004 would be a wrong theorem.
    "exp(x + 4000*log(x))",  # = x⁴⁰⁰⁰·eˣ
]


def _integrate_code(f_str: str) -> str:
    """Return "OK" if integrate() succeeds, else the E-INT-* error code."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.parse(f_str, pool, {"x": x})
    try:
        ak.integrate(f, x)
        return "OK"
    except ak.IntegrationError as exc:
        return getattr(exc, "code", "E-INT-?")


def test_elementary_integrands_never_classified_nonelementary():
    """No known-elementary integrand may receive the E-INT-004 verdict."""
    offenders = {s: code for s in KNOWN_ELEMENTARY if (code := _integrate_code(s)) == "E-INT-004"}
    assert not offenders, (
        f"engine falsely classified elementary integrands as NonElementary (E-INT-004): {offenders}"
    )


def test_b2_case_integrates_to_exp_times_log():
    """B2: ∫ (exp(x)·log(x) + exp(x)/x) dx must succeed and equal exp(x)·log(x)."""
    pool = ak.ExprPool()
    x = pool.symbol("x")
    f = ak.parse("exp(x)*log(x) + exp(x)/x", pool, {"x": x})

    # Must NOT raise (and in particular must not be E-INT-004).
    result = ak.integrate(f, x)

    # Verify by differentiating the returned antiderivative back to f.
    d = ak.diff(result.value, x)
    residual = ak.simplify(ak.simplify(d.value).value - f).value
    if str(residual).strip() not in ("0", "0/1"):
        reduced = ak.simplify_egraph(residual)
        residual = getattr(reduced, "value", reduced)
    assert str(residual).strip() in ("0", "0/1"), (
        f"d/dx of returned antiderivative != integrand; residual={residual}"
    )


def test_curated_nonelementary_labels_are_genuinely_nonelementary():
    """The RL curated hard-negatives must not be integrable to an elementary form.

    Guards against a future edit that slips an elementary integrand into the
    curated corpus (which would reward a *wrong* refusal). We assert the engine
    never *succeeds* on them; whether it reports E-INT-004 or E-INT-001 is
    irrelevant to the label's correctness, which is established by the curation.
    """
    from alkahest.rl.envs.integration.env import _known_nonelementary_forms

    pool = ak.ExprPool()
    x = pool.symbol("x")
    # An antiderivative naming one of these is, by construction, *not* an
    # elementary function — so it does not contradict the label, it spells the
    # label out.  `∫sin(x)/x dx = Si(x)` and `∫eˣ/x dx = Ei(x)` are curated
    # hard-negatives that the integrator now answers; what would break the label
    # is an *elementary* answer.
    basis = ("Ei", "li", "Si", "Ci", "Shi", "Chi", "erf", "fresnels", "fresnelc", "dilog")
    integrable = []
    for f in _known_nonelementary_forms(pool, x):
        try:
            cap = str(ak.integrate(f, x).value)
        except ak.IntegrationError:
            continue
        if not any(name in cap for name in basis):
            integrable.append(f"{f} -> {cap}")
    assert not integrable, (
        f"curated 'non-elementary' forms were integrated to an ELEMENTARY form: {integrable}"
    )


def test_reward_never_reads_engine_verdict_for_labels():
    """The RL row label must be the curated flag, independent of the engine.

    A hard-negative row is stamped ``is_elementary=False`` purely from the
    curated corpus; an elementary row is stamped ``is_elementary=True``. This
    documents/locks the invariant that labels are curated, not engine-derived.
    """
    import random

    from alkahest.rl.envs.integration.env import _make_row

    neg = _make_row(tier=0, nonelementary=True, rng=random.Random(0))
    assert neg["is_elementary"] is False

    pos = _make_row(tier=0, nonelementary=False, rng=random.Random(0))
    assert pos["is_elementary"] is True


def test_exp_tower_with_logarithmic_exponent_is_not_falsely_nonelementary():
    """∫ exp(x + log x) dx = (x−1)·eˣ — verified by differentiating back.

    ``exp(x + log x)`` is just ``x·eˣ`` written through the exp tower.  Its
    Risch DE is ``v' + (1/x + 1)·v = 1``, whose solution ``v = (x−1)/x`` has a
    simple pole at 0 forced by the *coefficient*'s residue 1 — not by the
    right-hand side, which is the constant 1.  A denominator bound read off the
    right-hand side alone misses it, and the miss used to be reported as
    ``E-INT-004``: a false claim of non-elementarity about an elementary
    integrand.
    """
    for src in ("exp(x + log(x))", "exp(x + 2*log(x))", "exp(x + log(x))/x"):
        pool = ak.ExprPool()
        x = pool.symbol("x")
        f = ak.parse(src, pool, {"x": x})
        result = ak.integrate(f, x)  # must not raise, and must not be E-INT-004

        # Verify numerically by differentiating the answer back.  (An exact
        # symbolic residual would need exp(log u) = u inside the tower
        # generator, which the simplifier does not apply here.)
        d = ak.simplify(ak.diff(result.value, x).value).value
        checked = 0
        for pt in (1.3, 2.1, 3.4):
            bindings = {x: ArbBall(pt)}
            lhs = interval_eval(d, bindings).mid
            rhs = interval_eval(f, bindings).mid
            if lhs != lhs or rhs != rhs:
                continue
            assert abs(lhs - rhs) < 1e-8, (
                f"{src}: d/dx F({pt}) = {lhs} != f({pt}) = {rhs}\n  F = {result.value}"
            )
            checked += 1
        assert checked >= 2, f"{src}: too few usable sample points"


def test_resonance_search_cap_declines_rather_than_certifying():
    """A bound the solver cannot prove complete must be E-INT-001, not E-INT-004.

    ``exp(x + 4000·log x)`` is ``x⁴⁰⁰⁰·eˣ`` — elementary.  The residue 4000 is
    past the internal resonance-search cap, so the solver cannot prove its
    denominator bound complete.  Succeeding is fine, declining is fine; the one
    forbidden outcome is a non-elementarity certificate.
    """
    assert _integrate_code("exp(x + 4000*log(x))") != "E-INT-004"
