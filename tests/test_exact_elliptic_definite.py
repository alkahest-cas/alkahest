"""Exact definite elliptic integrals and the degree-15 Lazard–Rioboo–Trager log part.

Report 10-9 / F10.

* ``∫_a^b R(x)/√P(x) dx`` with ``P`` a cubic or quartic whose roots are rational
  used to reach the numeric elliptic reduction, whose float-lifted constants
  make the definite path refuse (W3).  These now come back as exact
  combinations of ``EllipticF``/``EllipticE``/``EllipticPi``/``EllipticK`` with
  rational parameters and square roots of rationals.  Every value is compared
  with ``mpmath.quad`` and must contain no binary fraction lifted from a float.
* ``∫ (x⁷+3x²−1)/(x¹⁵+x⁴+2x+1)² dx`` spent minutes in a Euclidean gcd over a
  degree-15 number field; the log argument now comes from the subresultant
  chain over ``ℚ[t]``.  The answer must differentiate back to the integrand.
"""

import re

import alkahest as ak
import pytest
from alkahest import diff, eval_expr, integrate, parse

mp = pytest.importorskip("mpmath")

_FRACTION = re.compile(r"(\d+)/(\d+)")


def _float_lifted(expr) -> list:
    out = []
    for _, d in _FRACTION.findall(str(expr)):
        d = int(d)
        if d >= 2**32 and d & (d - 1) == 0:
            out.append(d)
    return out


# (integrand, lower, upper) — every radicand splits over ℚ.
CASES = [
    ("1/sqrt(x*(1-x)*(2-x))", "0", "1"),
    ("1/sqrt(x*(1-x)*(2-x))", "1/4", "3/4"),
    ("x/sqrt(x*(1-x)*(2-x))", "0", "1"),
    ("x^2/sqrt(x*(1-x)*(2-x))", "0", "1"),
    ("x^3/sqrt(x*(1-x)*(2-x))", "1/3", "1"),
    ("1/sqrt(x^3-x)", "2", "3"),
    ("1/sqrt(x^3-x)", "1", "oo"),
    ("1/sqrt(x^3-x)", "-1", "0"),
    ("1/sqrt((x-1)*(x-2)*(x-3))", "3", "5"),
    ("1/sqrt((x-1)*(x-2)*(x-3))", "1", "2"),
    ("x/sqrt((x-1)*(x-2)*(x-3))", "4", "7"),
    ("1/sqrt(x*(x+1)*(x+3))", "1", "2"),
    ("1/sqrt((1-x^2)*(4-x^2))", "0", "1"),
    ("1/sqrt((1-x^2)*(4-x^2))", "0", "1/2"),
    ("x^2/sqrt((1-x^2)*(4-x^2))", "-1", "1"),
    ("1/sqrt((x-1)*(x-2)*(x-3)*(x-4))", "2", "3"),
    ("x/sqrt((x-1)*(x-2)*(x-3)*(x-4))", "2", "3"),
    ("x^2/sqrt((x-1)*(x-2)*(x-3)*(x-4))", "5", "6"),
    ("1/sqrt((x-1)*(x-2)*(x-3)*(x-4))", "4", "oo"),
    ("1/((x+1)*sqrt(x*(1-x)*(2-x)))", "0", "1"),
    ("1/sqrt(x*(1-x)*(2-x))", "1", "0"),
]


def _num(s):
    if s == "oo":
        return mp.inf
    return mp.mpf(mp.mpmathify(eval(s)))


def _bound(s, pool):
    if s == "oo":
        return pool.pos_infinity()
    return parse(s, pool)


@pytest.fixture
def env():
    pool = ak.ExprPool()
    return pool, pool.symbol("x")


@pytest.mark.parametrize(("f", "a", "b"), CASES)
def test_exact_value_matches_quadrature(env, f, a, b):
    pool, x = env
    mp.mp.dps = 30

    def fx(t):
        return eval(f.replace("^", "**").replace("sqrt", "mp.sqrt"), {"x": t, "mp": mp})

    lo, hi = _num(a), _num(b)
    ref = mp.quad(fx, [lo, hi]) if lo < hi else -mp.quad(fx, [hi, lo])
    r = integrate(parse(f, pool), x, _bound(a, pool), _bound(b, pool))
    text = str(r.value)
    assert "Elliptic" in text, text
    assert not _float_lifted(r.value), text
    assert "." not in text, f"float node in {text}"
    got = eval_expr(r.value, {})
    assert abs(got - float(ref)) <= 1e-13 * max(1.0, abs(float(ref))), (text, got, ref)


# Radicands that do not split over ℚ go through the indefinite elliptic route;
# its answer may be a coded refusal, but never a wrong or float-lifted value.
OTHER = [
    ("x/sqrt(x^3+1)", "0", "1"),
    ("x^2/sqrt(1-x^4)", "0", "1/2"),
    ("1/sqrt(x^4+1)", "0", "1"),
    ("1/sqrt((x-1)*(x^2+1))", "2", "3"),
    ("1/sqrt(x*(x^2-2))", "2", "3"),
    ("1/sqrt(x^3-2)", "2", "3"),
]


@pytest.mark.parametrize(("f", "a", "b"), OTHER)
def test_non_split_radicands_are_right_or_refused(env, f, a, b):
    pool, x = env
    mp.mp.dps = 30

    def fx(t):
        return eval(f.replace("^", "**").replace("sqrt", "mp.sqrt"), {"x": t, "mp": mp})

    ref = mp.quad(fx, [_num(a), _num(b)])
    try:
        r = integrate(parse(f, pool), x, parse(a, pool), parse(b, pool))
    except ak.AlkahestError as e:
        code = e.code
        r = None
    if r is None:
        assert code.startswith("E-INT-"), code
        return
    assert not _float_lifted(r.value), str(r.value)
    got = eval_expr(r.value, {})
    assert abs(got - float(ref)) <= 1e-12 * max(1.0, abs(float(ref))), (str(r.value), got, ref)


def test_complete_first_kind_is_k(env):
    pool, x = env
    # ∫_0^1 dx/√(x(1−x)(2−x)) = √2·K(1/2)
    r = integrate(parse("1/sqrt(x*(1-x)*(2-x))", pool), x, parse("0", pool), parse("1", pool))
    assert "EllipticK(1/2)" in str(r.value)
    assert abs(eval_expr(r.value, {}) - float(mp.sqrt(2) * mp.ellipk(0.5))) < 1e-14


def test_pole_inside_interval_is_not_evaluated(env):
    pool, x = env
    f = parse("1/((x-1/2)*sqrt(x*(1-x)*(2-x)))", pool)
    with pytest.raises(Exception):
        integrate(f, x, parse("0", pool), parse("1", pool))


def test_degree_fifteen_rational_log_part(env):
    pool, x = env
    f = parse("(x^7+3*x^2-1)/(x^15+x^4+2*x+1)^2", pool)
    r = integrate(f, x)
    assert "RootSum" in str(r.value)
    d = diff(r.value, x)
    for xv in (0.3, 1.7, -0.4, 2.5):
        want = eval_expr(f, {x: xv})
        got = eval_expr(d.value if hasattr(d, "value") else d, {x: xv})
        assert abs(got - want) <= 1e-8 * max(1.0, abs(want)), (xv, got, want)
