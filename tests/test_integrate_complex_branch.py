"""Integration where the integrand is not real on part of the line (audit A10).

Every numeric gate in the integrator samples where the integrand is a finite
*real* number. For ``sqrt(tan(x)**3)`` that skips every ``x`` with
``tan(x) < 0``, where the integrand is perfectly well defined on the principal
branch, and the substitution routes' sign-losing rewrites
(``(a**3)**(1/2) -> (a**(1/2))**3``) went unseen there:

* ``integrate(3/2*tan**2*(1 + tan**2)/sqrt(tan**3))`` returned
  ``sqrt(tan)**3`` — off by a sign wherever ``tan < 0``;
* ``integrate((1/tan)**(3/2))`` and ``integrate(sqrt(tan**3))`` likewise;
* ``integrate(cot**(3/2), x, 1, 5/2)`` crossed ``pi/2``, where the
  antiderivative jumps on the non-real side, and returned
  ``0.108 + 2.668i`` (Mathematica: ``0.108 - 0.446i``);
* ``integrate(1/x + tanh(sqrt(-1)), x, -1, 2)`` returned a finite value: the
  non-real constant switched every pole scan off.

The antiderivative checks here are independent of the library: ``d/dx F`` is
taken numerically by mpmath on the principal branch, at points on both signs
of ``tan``.
"""

import alkahest as ak
import pytest

mpmath = pytest.importorskip("mpmath")
mp = mpmath.mp

_FUNCS = {
    "sin": mpmath.sin,
    "cos": mpmath.cos,
    "tan": mpmath.tan,
    "exp": mpmath.exp,
    "log": lambda z: mpmath.log(mpmath.mpc(z)),
    "sqrt": lambda z: mpmath.sqrt(mpmath.mpc(z)),
    "atan": mpmath.atan,
    "asin": lambda z: mpmath.asin(mpmath.mpc(z)),
    "acos": lambda z: mpmath.acos(mpmath.mpc(z)),
    "sinh": mpmath.sinh,
    "cosh": mpmath.cosh,
    "tanh": mpmath.tanh,
}


def _to_mp(e, x):
    """An mpmath callable for ``e`` on the principal branch."""
    n = e.node()
    tag = n[0]
    if tag == "symbol":
        if e == x:
            return lambda v: v
        if n[1] == "pi":
            return lambda v: mpmath.pi
        raise ValueError(f"free symbol {n[1]}")
    if tag == "integer":
        c = mpmath.mpf(int(n[1]))
        return lambda v: c
    if tag == "rational":
        c = mpmath.mpf(int(n[1])) / int(n[2])
        return lambda v: c
    if tag == "add":
        fs = [_to_mp(c, x) for c in n[1]]
        return lambda v: mpmath.fsum(f(v) for f in fs)
    if tag == "mul":
        fs = [_to_mp(c, x) for c in n[1]]

        def mul(v):
            r = mpmath.mpf(1)
            for f in fs:
                r *= f(v)
            return r

        return mul
    if tag == "pow":
        base, ex = _to_mp(n[1], x), _to_mp(n[2], x)
        if n[2].node()[0] == "integer":
            k = int(n[2].node()[1])
            return lambda v: base(v) ** k
        return lambda v: mpmath.exp(ex(v) * mpmath.log(mpmath.mpc(base(v))))
    if tag == "func":
        fn = _FUNCS[n[1]]
        arg = _to_mp(n[2][0], x)
        return lambda v: fn(arg(v))
    raise ValueError(f"node {tag}")


# Both signs of tan x, several periods, clear of the poles and zeros.
POINTS = [0.1 + 0.137 * k for k in range(-40, 40)]


def _assert_antiderivative(F, f, x):
    Ff, ff = _to_mp(F, x), _to_mp(f, x)
    checked = 0
    with mp.workdps(30):
        for p in POINTS:
            p = mpmath.mpf(p)
            try:
                d = mpmath.diff(Ff, p)
                want = ff(p)
            except ZeroDivisionError:
                continue
            assert abs(d - want) <= 1e-12 * (1 + abs(want)), (
                f"d/dx F = {complex(d)} but f = {complex(want)} at x = {float(p)}"
            )
            checked += 1
    assert checked > 40


def _tan_cases(p, x):
    t = ak.tan(x)
    return {
        "3/2 tan^2 (1+tan^2)/sqrt(tan^3)": p.rational(3, 2) * t**2 * (1 + t**2) / ak.sqrt(t**3),
        "(1/tan)^(3/2)": (1 / t) ** p.rational(3, 2),
        "sqrt(tan^3)": ak.sqrt(t**3),
        "(cos/sin)^(3/2)": (ak.cos(x) / ak.sin(x)) ** p.rational(3, 2),
        "sqrt(tan)": ak.sqrt(t),
    }


_POOL = ak.ExprPool()


@pytest.mark.parametrize("name", list(_tan_cases(_POOL, _POOL.symbol("x"))))
def test_antiderivative_holds_where_the_integrand_is_not_real(name):
    """Whatever `integrate` returns must differentiate back to the integrand on
    *both* signs of `tan x`; a refusal would also be sound, but these all have
    branch-correct answers and the sign-repaired candidates find them."""
    p = ak.ExprPool()
    x = p.symbol("x")
    f = _tan_cases(p, x)[name]
    F = ak.integrate(f, x).value
    _assert_antiderivative(F, f, x)


def test_the_audit_antiderivative_is_no_longer_sqrt_tan_cubed():
    p = ak.ExprPool()
    x = p.symbol("x")
    t = ak.tan(x)
    F = ak.integrate(p.rational(3, 2) * t**2 * (1 + t**2) / ak.sqrt(t**3), x).value
    assert ak.sqrt(t) ** 3 != F
    assert (t ** p.rational(1, 2)) ** 3 != F


def test_definite_across_the_jump_is_refused_not_wrong():
    """`cot(x)**(3/2)` on `[1, 5/2]`: the value is `0.1085 - 0.4462i`
    (Mathematica `NIntegrate`); the FTC difference across `pi/2` gave
    `0.108 + 2.668i`."""
    p = ak.ExprPool()
    x = p.symbol("x")
    f = (ak.cos(x) / ak.sin(x)) ** p.rational(3, 2)
    try:
        r = ak.integrate(f, x, p.integer(1), p.rational(5, 2))
    except ak.IntegrationError:
        return
    val = _to_mp(r.value, x)(mpmath.mpf(0))
    assert abs(complex(val) - complex(0.10847041546693705, -0.44617383999080299)) < 1e-9


def test_definite_on_the_real_branch_is_still_answered():
    """The control: on `[1/5, 1]` the integrand is real, and the answer is
    `2.1720296687334987` (Mathematica). (Not `[1/5, 6/5]`: the `atan` in the
    antiderivative jumps at `x = atan 2`, and the FTC is refused there.)"""
    p = ak.ExprPool()
    x = p.symbol("x")
    f = (ak.cos(x) / ak.sin(x)) ** p.rational(3, 2)
    r = ak.integrate(f, x, p.rational(1, 5), p.integer(1))
    val = complex(_to_mp(r.value, x)(mpmath.mpf(0)))
    assert abs(val - 2.1720296687334987) < 1e-9


@pytest.mark.parametrize(
    "build",
    [
        lambda p, x: 1 / x + ak.tanh(ak.sqrt(p.integer(-1))),
        lambda p, x: ak.tanh(ak.sqrt(p.integer(-1))) / x,
        lambda p, x: (1 / x + 1) * ak.tanh(ak.sqrt(p.integer(-1))),
    ],
)
def test_a_pole_is_seen_past_a_non_real_constant(build):
    p = ak.ExprPool()
    x = p.symbol("x")
    with pytest.raises(ak.IntegrationError):
        ak.integrate(build(p, x), x, p.integer(-1), p.integer(2))


def test_a_non_real_constant_away_from_the_pole_still_integrates():
    """`int_1^2 (1/x + tanh(i)) dx = log 2 + tanh(i)`."""
    p = ak.ExprPool()
    x = p.symbol("x")
    r = ak.integrate(1 / x + ak.tanh(ak.sqrt(p.integer(-1))), x, p.integer(1), p.integer(2))
    val = complex(_to_mp(r.value, x)(mpmath.mpf(0)))
    want = complex(mpmath.log(2) + mpmath.tanh(1j))
    assert abs(val - want) < 1e-12
