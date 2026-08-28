"""Improper integrals over the whole real line: right, or refused.

Two separate guarantees are pinned here.

**Nothing wrong comes back.**  ``integrate(f, x, -oo, oo)`` used to return a
confident ``0`` for ``1/(x^4+1)`` and ``x^2/(x^4+1)``.  The antiderivative for
those is a ``RootSum`` (Lazard-Rioboo-Trager), the limit engine has no rule for
it and handed the expression back *unchanged*, and ``F(+oo) - F(-oo)`` then
cancelled syntactically.  Every assertion below that expects a value checks the
value; every assertion that expects a refusal checks that no number is
produced -- never that a particular error string appears.

**Divergence is reported as divergence.**  A divergent integral must not be
given a finite value, even when a Cauchy principal value exists
(``x/(x^2+1)``) and even when the antiderivative happens to be well behaved.
"""

import math

import alkahest as ak
import pytest
from alkahest import ExprPool, eval_expr


@pytest.fixture
def pool():
    return ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def _line(pool, x, src):
    """``integrate(src, x, -oo, oo)`` -- the DerivedResult, or the exception."""
    f = ak.parse(src, pool)
    pos = pool.pos_infinity()
    neg = pool.integer(-1) * pos
    return ak.integrate(f, x, neg, pos)


def _num(pool, expr):
    """Numeric value of a closed form, binding the ``pi`` symbol."""
    return float(eval_expr(expr, {pool.symbol("pi"): math.pi}))


def _value(pool, x, src):
    return _num(pool, _line(pool, x, src).value)


# ---------------------------------------------------------------------------
# Values.  Cross-checked against composite Simpson on a tanh-substituted
# integral, so the expected numbers are not taken on faith either.
# ---------------------------------------------------------------------------


def _quadrature(f, n=20001):
    """int_{-oo}^{oo} f dx via x = tan(t), t in (-pi/2, pi/2), Simpson."""
    a, b = -math.pi / 2 + 1e-9, math.pi / 2 - 1e-9
    h = (b - a) / (n - 1)
    total = 0.0
    for i in range(n):
        t = a + i * h
        w = 1 if i in (0, n - 1) else (4 if i % 2 else 2)
        total += w * f(math.tan(t)) / math.cos(t) ** 2
    return total * h / 3


CONVERGENT = [
    ("1/(x^2+1)", math.pi, lambda v: 1 / (v * v + 1)),
    ("1/(x^2+4)", math.pi / 2, lambda v: 1 / (v * v + 4)),
    ("1/(x^4+1)", math.pi / math.sqrt(2), lambda v: 1 / (v**4 + 1)),
    ("x^2/(x^4+1)", math.pi / math.sqrt(2), lambda v: v * v / (v**4 + 1)),
    ("1/(x^6+1)", 2 * math.pi / 3, lambda v: 1 / (v**6 + 1)),
    ("1/(x^4+x^2+1)", math.pi / math.sqrt(3), lambda v: 1 / (v**4 + v * v + 1)),
    ("1/(x^2+1)^2", math.pi / 2, lambda v: 1 / (v * v + 1) ** 2),
    ("1/(x^2+2*x+2)", math.pi, lambda v: 1 / (v * v + 2 * v + 2)),
]


@pytest.mark.parametrize(("src", "expected", "f"), CONVERGENT, ids=[c[0] for c in CONVERGENT])
def test_expected_values_agree_with_quadrature(src, expected, f):
    """The closed forms this file asserts are themselves the right numbers."""
    assert abs(_quadrature(f) - expected) < 1e-6


@pytest.mark.parametrize(("src", "expected", "f"), CONVERGENT, ids=[c[0] for c in CONVERGENT])
def test_residue_route_returns_the_exact_value(pool, x, src, expected, f):
    got = _value(pool, x, src)
    assert abs(got - expected) < 1e-9 * (1 + abs(expected)), (
        f"int_-oo^oo {src} dx = {expected}, got {got}"
    )


def test_the_two_regressions_are_not_zero(pool, x):
    """The exact shape of the original bug: a silent ``0``."""
    for src in ("1/(x^4+1)", "x^2/(x^4+1)"):
        assert _value(pool, x, src) > 2.0, f"{src} regressed to a near-zero value"


def test_odd_integrand_is_genuinely_zero(pool, x):
    """``x/(x^4+1)`` converges (deg 4 >= deg 1 + 2) and is odd, so it *is* 0."""
    assert abs(_value(pool, x, "x/(x^4+1)")) < 1e-12


# ---------------------------------------------------------------------------
# Divergence.  Never a finite value.
# ---------------------------------------------------------------------------


DIVERGENT = [
    "1/(x^2-1)",  # real poles at +-1
    "1/(x-1)",  # real pole, and the degree condition fails too
    "x/(x^2+1)",  # deg Q = deg P + 1: PV is 0, but it does not converge
    "x^2/(x^2+1)",  # does not even tend to 0
    "1/(x^2-4)",
]


@pytest.mark.parametrize("src", DIVERGENT)
def test_divergent_integrals_are_refused(pool, x, src):
    with pytest.raises(Exception) as excinfo:
        _line(pool, x, src)
    assert getattr(excinfo.value, "code", "E-INT-001").startswith("E-")


def test_principal_value_is_not_passed_off_as_the_integral(pool, x):
    """``int x/(x^2+1)`` has PV 0; returning 0 would be wrong."""
    with pytest.raises(Exception):
        _line(pool, x, "x/(x^2+1)")


# ---------------------------------------------------------------------------
# The task-1 guard, from Python: an unestablished endpoint value must not
# become a number, on a *finite* interval either.
# ---------------------------------------------------------------------------


def test_finite_interval_root_sum_is_not_silently_zero(pool, x):
    """``int_0^1 dx/(x^4+1) ~ 0.8669`` -- and must never come back as ``0``.

    ``kernel::subs`` does not descend into a ``RootSum``, so substituting the
    bounds into that antiderivative is a no-op and the difference cancels.  The
    engine now refuses rather than returning the cancellation.
    """
    f = ak.parse("1/(x^4+1)", pool)
    try:
        r = ak.integrate(f, x, pool.integer(0), pool.integer(1))
    except Exception:
        return  # an honest decline is the accepted outcome
    got = _num(pool, r.value)
    assert abs(got - 0.86697) < 1e-4, f"int_0^1 dx/(x^4+1) = 0.86697, got {got}"


def test_divergent_with_an_infinite_bound_and_an_interior_pole(pool, x):
    """``int_0^oo dx/(x-3)^2`` diverges; it used to return ``-1/3``."""
    f = ak.parse("1/(x-3)^2", pool)
    with pytest.raises(Exception):
        ak.integrate(f, x, pool.integer(0), pool.pos_infinity())
