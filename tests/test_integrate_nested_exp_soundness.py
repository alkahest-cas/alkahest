"""The nested-exp Risch tower must never certify an elementary integrand.

Background — the bug this pins.  ``decompose_wrt_exp`` writes an integrand as a
Laurent polynomial ``Σ cₖ·tᵏ`` in the outer generator ``t = exp(exp(x))``, but it
peeled powers of ``t`` off a product and dumped *everything else* into the
coefficient without checking that what was left is free of ``t``.  So

    eˣ·e^(eˣ)/(e^(eˣ)+1)      →      coefficient  eˣ/(e^(eˣ)+1),  k = 1

looked like a monomial whose coefficient lives in the base field, when it is in
fact a genuine rational function of ``t``.  The exp case then applied the
Laurent theorem ("for k ≠ 0, ∫cₖtᵏ is elementary iff the Risch DE has a solution
in K"), found no solution, and returned a *certified* ``E-INT-004``.  But the
antiderivative here is ``log(e^(eˣ)+1)`` — a **new logarithm over the tower**,
which no Risch DE can produce.  Ruling out the rational part says nothing about
the logarithmic part.

The trigger was ``simplify``: the raw parse fell through to u-substitution and
solved, while the simplified spelling (folded ``^-1``) entered the tower and got
a false proof.  Both spellings are pinned below, and every answer is checked by
differentiating it back — never by matching a display string.
"""

import alkahest as ak
import pytest
from alkahest.alkahest import ArbBall, ExprPool, diff, integrate, interval_eval

_TEST_POINTS = (0.2, 0.8, 1.3, 1.9)


def _spellings(src, pool):
    """``(label, expr)`` for the raw parse and its ``simplify``d form.

    They denote the same function, so ``integrate`` must give them the same
    verdict — that is the contract ``test_integrate_form_robustness`` already
    pins for other families.
    """
    raw = ak.parse(src, pool)
    return [("raw", raw), ("simplified", ak.simplify(raw).value)]


def _check_antiderivative(x, f, cap, label):
    """``d/dx F(x) == f(x)`` at several real points."""
    d = diff(cap, x).value
    checked = 0
    for pt in _TEST_POINTS:
        bindings = {x: ArbBall(pt)}
        lhs = interval_eval(d, bindings).mid
        rhs = interval_eval(f, bindings).mid
        if lhs != lhs or rhs != rhs:  # NaN at a singularity — skip
            continue
        assert abs(lhs - rhs) < 1e-8, (
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n"
            f"  F = {cap}\n  f = {f}"
        )
        checked += 1
    assert checked >= 2, f"{label}: too few usable sample points"


# ---------------------------------------------------------------------------
# The reported false-certificate family: elementary, and must integrate.
# ---------------------------------------------------------------------------

#: ``(integrand, witness antiderivative)``.  Each witness is elementary by
#: construction; the test verifies the *engine's* answer numerically, so a
#: different-but-equivalent antiderivative is fine.
_ELEMENTARY_NESTED_EXP = [
    ("exp(x)*exp(exp(x))/(exp(exp(x))+1)", "log(exp(exp(x))+1)"),
    ("exp(x)*exp(exp(x))/(exp(exp(x))+1)^2", "-1/(exp(exp(x))+1)"),
    ("-3*exp(x)*exp(exp(x))/(1+exp(exp(x)))^2", "3/(1+exp(exp(x)))"),
    ("-4*exp(x)*exp(exp(x))/(exp(exp(x))-2)^2", "4/(exp(exp(x))-2)"),
    # Same shape, a denominator of degree 2 in the outer generator
    # (∫ = atan(e^(eˣ)) and ½log(e^(2eˣ)+1) respectively).
    ("exp(x)*exp(exp(x))/(exp(exp(x))^2+1)", None),
    ("exp(x)*exp(exp(x))^2/(exp(exp(x))^2+1)", None),
]


@pytest.mark.parametrize(("src", "_witness"), _ELEMENTARY_NESTED_EXP)
def test_nested_exp_rational_in_generator_integrates(src, _witness):
    """Both spellings integrate, and both answers differentiate back to ``f``."""
    pool = ExprPool()
    x = pool.symbol("x")
    for label, f in _spellings(src, pool):
        result = integrate(f, x)
        _check_antiderivative(x, f, result.value, f"{src} [{label}]")


@pytest.mark.parametrize(("src", "_witness"), _ELEMENTARY_NESTED_EXP)
def test_simplify_does_not_change_the_verdict(src, _witness):
    """``simplify`` before ``integrate`` must not turn an answer into a proof.

    This is the regression proper: on the buggy build the raw parse solved and
    the simplified spelling returned ``E-INT-004``.
    """
    pool = ExprPool()
    x = pool.symbol("x")
    verdicts = []
    for _label, f in _spellings(src, pool):
        try:
            integrate(f, x)
            verdicts.append("ok")
        except ak.IntegrationError as exc:
            verdicts.append(getattr(exc, "code", "?"))
    assert verdicts[0] == verdicts[1], (
        f"{src}: raw and simplified spellings disagree: {verdicts}"
    )
    assert "E-INT-004" not in verdicts, f"{src}: false NonElementary certificate"


# ---------------------------------------------------------------------------
# The guard: "no rational solution" is not by itself a proof.
# ---------------------------------------------------------------------------

#: Integrands that are genuine *rational functions* of the outer generator.
#: Whatever the engine does with these, an ``E-INT-004`` is unjustified without
#: a Rothstein–Trager residue reduction over ``K[t]``, which is not implemented:
#: the logarithmic part of the antiderivative is never examined.  A decline
#: (``E-INT-001``) is the honest verdict; a certificate is a bug.
_UNDECIDED_RATIONAL_IN_GENERATOR = [
    "exp(x)*exp(exp(x))/(exp(exp(x))+1)",
    "exp(x)*exp(exp(x))/(exp(exp(x))+1)^2",
    "exp(x)*exp(exp(x))/(exp(exp(x))^2+1)",
    "exp(x)*exp(exp(x))/(exp(exp(x))+1)^3",
    "x*exp(x)*exp(exp(x))/(exp(exp(x))+1)",
    "exp(x)*exp(exp(x))^2/(exp(exp(x))+1)",
    "exp(exp(x))/(exp(exp(x))+1)",
    "1/(exp(exp(x))+1)",
]


@pytest.mark.parametrize("src", _UNDECIDED_RATIONAL_IN_GENERATOR)
def test_rational_in_generator_is_never_certified(src):
    pool = ExprPool()
    x = pool.symbol("x")
    for label, f in _spellings(src, pool):
        try:
            result = integrate(f, x)
        except ak.IntegrationError as exc:
            assert getattr(exc, "code", "") != "E-INT-004", (
                f"{src} [{label}]: certified non-elementary without ruling out the "
                f"logarithmic part over the tower — {exc}"
            )
            continue
        # If it *did* solve, the answer must be correct.
        _check_antiderivative(x, f, result.value, f"{src} [{label}]")


# ---------------------------------------------------------------------------
# Genuinely non-elementary nested-exp integrands must stay certified.
# ---------------------------------------------------------------------------

#: Each of these is non-elementary for a reason the implementation now actually
#: *proves*, rather than assumes:
#:
#: * ``exp(exp(x))`` and ``x·exp(exp(x))`` — the coefficient is ``c ∈ ℚ(x)`` with
#:   no inner-generator factor, so the forced Laurent recursion leaves the
#:   residual ``c/k ≠ 0``: the Risch DE has no solution in ``ℚ(x)(eˣ)``.
#: * ``x·eˣ·e^(eˣ)`` — cascade gives ``v₀ = x`` and residual ``−1 ≠ 0``.
#: * ``eˣ/(eˣ+1)·e^(eˣ)`` — the coefficient has a *simple* pole in the inner
#:   generator, so any solution would have to be regular there while ``c`` is
#:   not (Bronstein §6.1).
#: * ``e^(−x)·e^(eˣ)`` — pure negative Laurent index, residual ``1/k ≠ 0``.
#:
#: With ``k ≠ 0`` no new logarithm can arise either (Bronstein §5.3), so "no
#: solution to the Risch DE" really is non-elementarity for these.
_GENUINELY_NONELEMENTARY = [
    "exp(exp(x))",
    "x*exp(exp(x))",
    "x^2*exp(exp(x))",
    "x*exp(x)*exp(exp(x))",
    "exp(x)/(exp(x)+1)*exp(exp(x))",
]


@pytest.mark.parametrize("src", _GENUINELY_NONELEMENTARY)
def test_genuinely_nonelementary_nested_exp_stays_certified(src):
    pool = ExprPool()
    x = pool.symbol("x")
    for label, f in _spellings(src, pool):
        with pytest.raises(ak.IntegrationError) as excinfo:
            integrate(f, x)
        assert excinfo.value.code == "E-INT-004", (
            f"{src} [{label}]: should stay certified non-elementary, got "
            f"{excinfo.value.code}: {excinfo.value}"
        )


# ---------------------------------------------------------------------------
# The `li` pre-check must not swallow the elementary h'/h · log(h)^-n family.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "-1/(x*log(x)^2)",  # = 1/log(x)   — the shape `known_nonelementary` ate
        "1/(x*log(x))",
        "1/(x*log(x)^2)",
        "-2/(x*log(x)^3)",
        "3/(x*log(x))",
    ],
)
def test_log_derivative_family_is_not_certified_li(src):
    """``∫ c·(h'/h)·log(h)^(-n) dx`` is elementary — never the ``li`` integral."""
    pool = ExprPool()
    x = pool.symbol("x")
    for label, f in _spellings(src, pool):
        try:
            result = integrate(f, x)
        except ak.IntegrationError as exc:
            assert getattr(exc, "code", "") != "E-INT-004", (
                f"{src} [{label}]: elementary integrand certified as li — {exc}"
            )
            continue
        _check_antiderivative(x, f, result.value, f"{src} [{label}]")


@pytest.mark.parametrize(
    "src",
    [
        "1/log(x)",
        # Q(x) is *not* a constant multiple of the log's argument, so the
        # `h'/h` cancellation cannot happen and `li` really is the answer.
        "(x+1)^(-1)*log(x)^(-1)",
        "x^(-2)*log(x)^(-1)",
    ],
)
def test_real_li_family_stays_certified(src):
    """The genuine logarithmic-integral shapes keep their certificate."""
    pool = ExprPool()
    x = pool.symbol("x")
    f = ak.simplify(ak.parse(src, pool)).value
    with pytest.raises(ak.IntegrationError) as excinfo:
        integrate(f, x)
    assert excinfo.value.code == "E-INT-004", (
        f"∫ {src} dx should certify non-elementary, got {excinfo.value.code}"
    )
