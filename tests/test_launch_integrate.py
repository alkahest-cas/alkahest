"""Launch blockers in ``integrate``: false certificates, false divergence claims.

Three separate guarantees are pinned here, each against an independent oracle
(``mpmath.quad`` to 30 digits for values, central differences for
antiderivatives):

* **E-INT-004 is a proof about the caller's integrand.**  The derivative of an
  elementary function never earns it, whatever the order of its summands.
  ``∫ d/dx[log x·log(log x)/x] dx`` used to be certified non-elementary because
  the Risch log tower was entered at ``log x`` instead of ``log(log x)``, which
  made ``log(log x)`` part of a coefficient whose own integral is non-elementary.
* **The certificate names the integrand it is about**, and the Risch-DE wording
  is readable.
* **A "does not converge" claim is true.**  A zero of the denominator is a pole
  only when the numerator does not vanish there: ``sin(x)/x`` at 0 is
  removable, ``√x/x`` at 0 is integrable.
"""

import math
import random

import alkahest as ak
import mpmath as mp
import pytest
from alkahest import ExprPool, eval_expr

mp.mp.dps = 40


@pytest.fixture
def pool():
    return ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


def _num(expr):
    return float(eval_expr(expr, {}))


def _code(exc):
    return getattr(exc, "code", None)


def _check_antiderivative(F, f, x, points=(1.7, 2.3, 3.1, 4.4)):
    """``d/dx F == f`` by central differences at a few points (x > 1)."""
    for t in points:
        h = 1e-5 * t
        fp = eval_expr(F, {x: t + h})
        fm = eval_expr(F, {x: t - h})
        num = (fp - fm) / (2 * h)
        want = eval_expr(f, {x: t})
        assert abs(num - want) <= 1e-5 * (1 + abs(want)), (str(F), t, num, want)


# ---------------------------------------------------------------------------
# 1. No false E-INT-004
# ---------------------------------------------------------------------------


def test_derivative_of_log_loglog_over_x_integrates(pool, x):
    L = ak.log(x)
    LL = ak.log(L)
    f = ak.diff(L * LL / x, x).value
    F = ak.integrate(f, x).value
    _check_antiderivative(F, f, x)


def test_two_term_sum_integrates(pool, x):
    L = ak.log(x)
    LL = ak.log(L)
    f = LL / x**2 - L * LL / x**2
    F = ak.integrate(f, x).value
    _check_antiderivative(F, f, x)


@pytest.mark.parametrize("perm", [(0, 1, 2), (1, 0, 2), (2, 1, 0), (1, 2, 0), (0, 2, 1), (2, 0, 1)])
def test_verdict_does_not_depend_on_term_order(pool, x, perm):
    L = ak.log(x)
    LL = ak.log(L)
    terms = [x**-2, -1 * L * LL * x**-2, LL * x**-2]
    f = terms[perm[0]] + terms[perm[1]] + terms[perm[2]]
    F = ak.integrate(f, x).value
    _check_antiderivative(F, f, x)


def _random_elementary(pool, x, rng):
    """A random elementary F from log/exp/rational/algebraic pieces."""
    k = lambda: pool.integer(rng.randint(1, 3))  # noqa: E731
    pieces = [
        lambda: ak.log(x),
        lambda: ak.log(ak.log(x)),
        lambda: ak.log(x + k()),
        lambda: ak.exp(x / k()),
        lambda: ak.exp(-x),
        lambda: x ** rng.randint(1, 3),
        lambda: 1 / (x + k()),
        lambda: 1 / x,
        lambda: ak.sqrt(x + k()),
    ]

    def product():
        e = rng.choice(pieces)()
        for _ in range(rng.randint(0, 2)):
            e = e * rng.choice(pieces)()
        return e

    F = product()
    if rng.random() < 0.5:
        F = F + product()
    return F


def test_fuzz_derivatives_never_certified_nonelementary(pool, x):
    """∫ F′ dx for random elementary F: never E-INT-004; any answer verifies."""
    rng = random.Random(20261024)
    certified = []
    for _ in range(120):
        F = _random_elementary(pool, x, rng)
        try:
            f = ak.diff(F, x).value
        except Exception:
            continue
        try:
            with ak.context(budget=ak.Budget(wall_ms=4000)):
                G = ak.integrate(f, x).value
        except Exception as e:
            if _code(e) == "E-INT-004":
                certified.append((str(F), str(e)[:200]))
            continue
        _check_antiderivative(G, f, x)
    assert not certified, certified


# ---------------------------------------------------------------------------
# 2. Messages
# ---------------------------------------------------------------------------


def test_certificate_names_the_users_integrand(pool, x):
    with pytest.raises(Exception) as info:
        ak.integrate(ak.sin(x) / ak.log(x), x)
    assert _code(info.value) == "E-INT-004"
    msg = str(info.value)
    assert "sin(x)" in msg, msg
    assert "1/log(x) is the logarithmic integral" not in msg


def test_risch_de_message_is_readable(pool, x):
    with pytest.raises(Exception) as info:
        ak.integrate(ak.exp(x**2), x)
    msg = str(info.value)
    assert "(x * 2)(x)" not in msg, msg
    assert "= 1(x)" not in msg, msg


def test_exp_x2_certificate_wording(pool, x):
    with pytest.raises(Exception) as info:
        ak.integrate(ak.exp(x**2), x)
    assert _code(info.value) == "E-INT-004"
    assert "v′" in str(info.value)


def test_not_implemented_remediation_is_current(pool, x):
    with pytest.raises(Exception) as info:
        ak.integrate(ak.sin(ak.sin(x)), x)
    rem = str(getattr(info.value, "remediation", "")) + str(info.value)
    assert "only power, linearity" not in rem


# ---------------------------------------------------------------------------
# 3. Definite integrals
# ---------------------------------------------------------------------------


def _bounds(pool, lo, hi):
    oo = pool.pos_infinity()
    conv = {"oo": oo, "-oo": pool.integer(-1) * oo}
    pick = lambda b: conv[b] if isinstance(b, str) else pool.integer(b)  # noqa: E731
    return pick(lo), pick(hi)


CASES = [
    # (label, integrand builder, lower, upper, mpmath oracle)
    ("sin(x)/x", lambda x: ak.sin(x) / x, 0, "oo", lambda: mp.pi / 2),
    (
        "log(1+x)/x",
        lambda x: ak.log(1 + x) / x,
        0,
        1,
        lambda: mp.quad(lambda t: mp.log(1 + t) / t, [0, 1]),
    ),
    (
        "1/(1+x^2) half-line",
        lambda x: 1 / (1 + x**2),
        0,
        "oo",
        lambda: mp.quad(lambda t: 1 / (1 + t**2), [0, mp.inf]),
    ),
    (
        "1/(x^4+1) half-line",
        lambda x: 1 / (x**4 + 1),
        0,
        "oo",
        lambda: mp.quad(lambda t: 1 / (t**4 + 1), [0, mp.inf]),
    ),
    (
        "1/(1+x^3) half-line",
        lambda x: 1 / (1 + x**3),
        0,
        "oo",
        lambda: mp.quad(lambda t: 1 / (1 + t**3), [0, mp.inf]),
    ),
    (
        "cos(x)/(1+x^2)",
        lambda x: ak.cos(x) / (1 + x**2),
        "-oo",
        "oo",
        lambda: mp.quadosc(lambda t: mp.cos(t) / (1 + t**2), [-mp.inf, mp.inf], omega=1),
    ),
    (
        "cos(2x)/(x^2+x+1)",
        lambda x: ak.cos(2 * x) / (x**2 + x + 1),
        "-oo",
        "oo",
        lambda: mp.quadosc(lambda t: mp.cos(2 * t) / (t**2 + t + 1), [-mp.inf, mp.inf], omega=2),
    ),
    (
        "x*sin(3x)/((x^2+1)(x^2+4))",
        lambda x: x * ak.sin(3 * x) / (x**4 + 5 * x**2 + 4),
        "-oo",
        "oo",
        lambda: mp.quadosc(
            lambda t: t * mp.sin(3 * t) / (t**4 + 5 * t**2 + 4), [-mp.inf, mp.inf], omega=3
        ),
    ),
    ("sqrt(x)/x", lambda x: ak.sqrt(x) / x, 0, 1, lambda: mp.mpf(2)),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_definite_values(pool, x, case):
    label, build, lo, hi, oracle = case
    a, b = _bounds(pool, lo, hi)
    r = ak.integrate(build(x), x, a, b)
    got = _num(r.value)
    want = float(oracle())
    assert abs(got - want) <= 1e-12 * (1 + abs(want)), (label, str(r.value), got, want)


def test_closed_forms_match_mpmath_to_30_digits():
    """The textbook values the cases above are expected to produce, to 30 digits."""
    tol = mp.mpf(10) ** -30
    assert abs(mp.quad(lambda t: mp.log(1 + t) / t, [0, 1]) - mp.pi**2 / 12) < tol
    assert abs(mp.quad(lambda t: mp.atan(t) / t, [0, 1]) - mp.catalan) < tol
    assert abs(mp.quad(lambda t: 1 / (1 + t**2), [0, mp.inf]) - mp.pi / 2) < tol
    assert abs(mp.quad(lambda t: 1 / (t**4 + 1), [0, mp.inf]) - mp.pi / (2 * mp.sqrt(2))) < tol
    assert abs(mp.quad(lambda t: 1 / (1 + t**3), [0, mp.inf]) - 2 * mp.pi / (3 * mp.sqrt(3))) < tol
    assert (
        abs(mp.quadosc(lambda t: mp.cos(t) / (1 + t**2), [-mp.inf, mp.inf], omega=1) - mp.pi / mp.e)
        < tol
    )


REMOVABLE = [
    ("atan(x)/x", lambda x: ak.atan(x) / x, 0, 1, lambda: mp.catalan),
    (
        "sin(x)^2/x^2",
        lambda x: ak.sin(x) ** 2 / x**2,
        "-oo",
        "oo",
        lambda: mp.pi,
    ),
]


@pytest.mark.parametrize("case", REMOVABLE, ids=[c[0] for c in REMOVABLE])
def test_removable_singularity_is_not_called_divergent(pool, x, case):
    """Either the right value or a refusal that does not claim divergence."""
    label, build, lo, hi, oracle = case
    a, b = _bounds(pool, lo, hi)
    try:
        r = ak.integrate(build(x), x, a, b)
    except Exception as e:
        msg = str(e)
        assert "does not converge" not in msg, msg
        assert "has a pole" not in msg, msg
        return
    got = _num(r.value)
    want = float(oracle())
    assert abs(got - want) <= 1e-12 * (1 + abs(want)), (label, str(r.value), got, want)


@pytest.mark.parametrize(
    ("build", "lo", "hi"),
    [
        (lambda x: 1 / x, -1, 1),
        (lambda x: x**-2, 0, 1),
        (lambda x: ak.cos(x) / x, 0, 1),
        (lambda x: ak.exp(x) / x, -1, 1),
        (lambda x: ak.sin(x) / x**2, 0, 1),
    ],
    ids=["1/x", "1/x^2", "cos/x", "exp/x", "sin/x^2"],
)
def test_genuine_divergence_is_still_refused(pool, x, build, lo, hi):
    a, b = _bounds(pool, lo, hi)
    with pytest.raises(Exception):
        ak.integrate(build(x), x, a, b)


def test_limit_atan_at_infinity(pool, x):
    oo = pool.pos_infinity()
    assert abs(_num(ak.limit(ak.atan(x), x, oo).value) - math.pi / 2) < 1e-15
    assert abs(_num(ak.limit(ak.atan(x), x, pool.integer(-1) * oo).value) + math.pi / 2) < 1e-15
    assert abs(_num(ak.limit(ak.atan(2 * x + 1), x, oo).value) - math.pi / 2) < 1e-15
