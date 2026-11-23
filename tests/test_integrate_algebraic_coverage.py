"""Algebraic-integrand coverage: Chebyshev/Trager, radical normalisation,
reciprocal and binomial pullbacks, rationalised cofactors.

Every answer is checked by differentiating it and comparing with the integrand
at points on each real component of the domain (both signs of x where the
integrand is real there), so a branch-one-sided answer fails.
"""

import cmath

import alkahest as ak
import pytest


def _x():
    return ak.symbol("x")


def _check(src, points, tol=1e-7):
    x = _x()
    f = ak.parse(src)
    r = ak.integrate(f, x)
    F = r.value
    dF = ak.diff(F, x)
    dF = dF.value if hasattr(dF, "value") else dF
    for p in points:
        lhs = complex(ak.eval_expr(dF, {x: p}))
        rhs = complex(ak.eval_expr(f, {x: p}))
        assert cmath.isfinite(lhs), f"{src}: F'({p}) = {lhs}\n  F = {F}"
        assert abs(lhs - rhs) <= tol * max(1.0, abs(rhs)), (
            f"{src}: F'({p}) = {lhs}, f({p}) = {rhs}\n  F = {F}"
        )
    return F


def test_chebyshev_trager_quartic_is_elementary():
    # Trager 1984 / Bronstein §10: residues at the two places over ∞, torsion of order 8.
    F = _check("x/sqrt(x^4+10*x^2-96*x-71)", [5.0, 6.5, 9.0, -3.0, -5.0])
    assert "log" in str(F)


def test_chebyshev_twin_is_certified_non_elementary():
    with pytest.raises(Exception) as e:
        ak.integrate(ak.parse("x/sqrt(x^4+10*x^2-96*x-72)"), _x())
    assert "E-INT-004" in str(e.value)


@pytest.mark.parametrize(
    ("src", "points"),
    [
        # reciprocal pullback t = x - 1/x  (Charlwood #38, #39)
        ("(x^2+1)/((1-x^2)*sqrt(x^4+1))", [0.3, 0.7, 2.0, -0.4, -2.5]),
        ("(1-x^2)/((x^2+1)*sqrt(x^4+1))", [0.3, 0.7, 2.0, -0.4, -2.5]),
        # two square roots of linear radicands (Moses #38)
        ("sqrt(x)*(x+1)^(5/2)", [0.5, 1.4, 3.0]),
        # x^(1/3) and sqrt(x) share s = x^(1/6) (Moses #35)
        ("1/(x^(1/3)+sqrt(x))", [0.5, 1.4, 3.0]),
        # two radicals merged in Q(x)(sqrt P, sqrt Q)
        ("x/(sqrt(1-x^2)*sqrt(1+x^2))", [0.2, 0.5, -0.6]),
        # by parts after rationalising the algebraic cofactor (Charlwood #14, #15, #17)
        ("asin(x)/(sqrt(1-x^2)+1)", [0.2, 0.5, -0.6]),
        ("log(x+sqrt(x^2+1))/(1-x^2)^(3/2)", [0.2, 0.5, -0.6]),
        ("log(x+sqrt(x^2-1))/(x^2+1)^(3/2)", [1.4, 1.9, 2.6]),
        # Charlwood #13
        ("x*atan(x+sqrt(1-x^2))/sqrt(1-x^2)", [0.2, 0.5, -0.6]),
        # asin part kept as asin when the weight is rational
        ("sqrt(1-x^2)/x^2", [0.2, 0.5, -0.6]),
        # binomial differentials, Chebyshev's elementary cases
        ("x^(-2/3)*(1+x^(1/3))^(1/2)", [0.5, 1.4, 3.0]),
        ("1/(x^2*(1+x^4)^(3/4))", [0.5, 1.4, 3.0, -0.7, -2.0]),
    ],
)
def test_algebraic_integrals_verify(src, points):
    _check(src, points)


def test_binomial_outside_chebyshev_is_non_elementary():
    # m = 1/2, n = 1/3, p = 1/3: none of p, (m+1)/n, (m+1)/n + p is an integer.
    with pytest.raises(Exception) as e:
        ak.integrate(ak.parse("x^(1/2)*(1+x^(1/3))^(1/3)"), _x())
    assert "E-INT-004" in str(e.value)
