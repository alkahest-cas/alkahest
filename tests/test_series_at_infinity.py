"""``series`` about ``+oo`` / ``-oo`` expands in ``1/x`` (W4, report 10-9).

``series(1/x + 1/(x**2 + 1), x, pool.pos_infinity(), 4)`` used to substitute
the symbol ``∞`` into the Taylor formula and return, as a success, a ``Series``
whose coefficients were ``∞^-1``, ``(x + -1*∞)`` and ``0*∞``.  It now substitutes
``x = ±1/t``, expands about ``t = 0+`` and maps back, so the result is
``1/x + 1/x**2 + O(x**-4)`` — the same terms and the same ``O(x**-order)``
convention as SymPy's ``series(f, x, oo, n)``.  A function with no Laurent
expansion in ``1/x`` (``exp(x)``, ``log(x)``, ``sqrt(x)``) is refused with
``E-SERIES-007``.

The oracle is SymPy, computed inside the test; each case is also checked
numerically: the truncation error at large ``|x|`` must be ``O(|x|**-order)``.
"""

import alkahest as ak
import pytest

sp = pytest.importorskip("sympy")

ORDER = 4


def _cases(ns):
    """``(name, builder)`` pairs; ``ns`` supplies ``x`` and the functions."""
    x = ns["x"]
    sqrt, sin, cos, exp, log, atan = (
        ns["sqrt"],
        ns["sin"],
        ns["cos"],
        ns["exp"],
        ns["log"],
        ns["atan"],
    )
    return [
        ("repro", 1 / x + 1 / (x**2 + 1)),
        ("mobius", (x + 1) / (x - 1)),
        ("pole_at_inf", x**3 / (x**2 + 1)),
        ("rational_deg", (2 * x**2 + 3) / (x**3 - x + 5)),
        ("sqrt_x2p1", sqrt(x**2 + 1)),
        ("sqrt_x2px", sqrt(x**2 + x)),
        ("sqrt_cancel", sqrt(x**4 + 1) - x**2),
        ("x_sin_inv", x * sin(1 / x)),
        ("exp_inv", exp(1 / x)),
        ("x_log1p_inv", x * log(1 + 1 / x)),
        ("atan_inv", atan(1 / x)),
        ("cos_inv_over_x", cos(1 / x) / x),
        ("log_1p_2_x2", log(1 + 2 / x**2)),
        ("x2_expm1_inv", x**2 * (exp(1 / x) - 1)),
        ("poly", 3 * x**2 - x + 7),
    ]


def _ak_cases(pool):
    x = pool.symbol("x")
    ns = {
        "x": x,
        "sqrt": ak.sqrt,
        "sin": ak.sin,
        "cos": ak.cos,
        "exp": ak.exp,
        "log": ak.log,
        "atan": ak.atan,
    }
    return x, dict(_cases(ns))


def _sp_cases():
    x = sp.Symbol("x")
    ns = {
        "x": x,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "exp": sp.exp,
        "log": sp.log,
        "atan": sp.atan,
    }
    return x, dict(_cases(ns))


_X, _SP = _sp_cases()
NAMES = list(_SP)


def _point(pool, sign):
    return pool.pos_infinity() if sign > 0 else pool.neg_infinity()


@pytest.mark.parametrize("sign", [1, -1], ids=["pos_oo", "neg_oo"])
@pytest.mark.parametrize("name", NAMES)
def test_matches_sympy_and_approximates(name, sign):
    pool = ak.ExprPool()
    x, cases = _ak_cases(pool)
    f = cases[name]
    s = ak.series(f, x, _point(pool, sign), ORDER)

    text = str(s)
    assert "∞" not in text, text
    assert f"O(x^-{ORDER})" in text, text

    want = sp.series(_SP[name], _X, sp.oo if sign > 0 else -sp.oo, ORDER).removeO()
    got = s.truncated()

    # Both sides are Laurent polynomials in x with exponents in (-ORDER, 3],
    # so agreement at more points than that span is identity.
    for k in range(9):
        xv = sign * (2.0 + 1.5 * k)
        a = ak.eval_expr(got, {x: xv})
        b = float(want.subs(_X, xv))
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), (name, xv, text, want)

    # The truncation error is O(|x|**-ORDER): its scaled size stays bounded.
    fexact = sp.lambdify(_X, _SP[name], "mpmath")
    for xv in (40.0, 160.0, 640.0):
        xv *= sign
        err = abs(float(fexact(xv)) - ak.eval_expr(got, {x: xv}))
        assert err * abs(xv) ** ORDER < 50.0, (name, xv, err)


def test_repro_terms():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    s = ak.series(1 / x + 1 / (x**2 + 1), x, pool.pos_infinity(), 6)
    got = s.truncated()
    # 1/x + 1/x^2 - 1/x^4 + O(x^-6)
    for xv in (3.0, 7.0, 11.0):
        assert ak.eval_expr(got, {x: xv}) == pytest.approx(1 / xv + 1 / xv**2 - 1 / xv**4)
    assert "O(x^-6)" in str(s)


def test_neg_infinity_is_minus_pos_infinity():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    a = ak.series(ak.sqrt(x**2 + 1), x, pool.neg_infinity(), 4)
    b = ak.series(ak.sqrt(x**2 + 1), x, -pool.pos_infinity(), 4)
    assert str(a) == str(b)
    # sqrt(x^2 + 1) ~ -x - 1/(2x) + ... as x -> -oo
    assert ak.eval_expr(a.truncated(), {x: -100.0}) == pytest.approx(
        (100.0**2 + 1) ** 0.5, rel=1e-9
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda x: ak.exp(x),
        lambda x: ak.log(x),
        lambda x: ak.sqrt(x),
        lambda x: ak.sin(x),
        lambda x: x * ak.exp(-x),
    ],
    ids=["exp", "log", "sqrt", "sin", "x_exp_neg"],
)
def test_no_laurent_expansion_at_infinity_is_refused(build):
    pool = ak.ExprPool()
    x = pool.symbol("x")
    with pytest.raises(ak.SeriesError) as ei:
        ak.series(build(x), x, pool.pos_infinity(), 4)
    assert ei.value.code == "E-SERIES-007"


def test_other_point_mentioning_infinity_is_refused():
    pool = ak.ExprPool()
    x, a = pool.symbol("x"), pool.symbol("a")
    for point in (pool.pos_infinity() + 1, a * pool.pos_infinity()):
        with pytest.raises(ak.SeriesError) as ei:
            ak.series(1 / x, x, point, 3)
        assert ei.value.code == "E-SERIES-007"


def test_infinity_inside_the_expression_is_refused():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    with pytest.raises(ak.SeriesError) as ei:
        ak.series(pool.pos_infinity() * ak.sin(x), x, pool.integer(0), 3)
    assert ei.value.code == "E-SERIES-004"


def test_puiseux_series_refuses_an_infinite_point_up_front():
    from alkahest.experimental import puiseux_series

    pool = ak.ExprPool()
    x = pool.symbol("x")
    for point in (pool.pos_infinity(), pool.neg_infinity()):
        with pytest.raises(ak.SeriesError) as ei:
            puiseux_series(ak.sqrt(x), x, point, 4)
        assert ei.value.code == "E-SERIES-007"


def test_finite_point_unchanged():
    pool = ak.ExprPool()
    x = pool.symbol("x")
    s = ak.series(ak.exp(x), x, pool.integer(0), 3)
    assert ak.eval_expr(s.truncated(), {x: 0.1}) == pytest.approx(1 + 0.1 + 0.005)


def test_asymptotic_expand_returns_every_requested_term():
    """``x*log(1 + 1/x)`` came back with 2-3 of 4 requested terms: the numeric
    gate compared a 2.5e-19 term against f64 rounding noise at x = 1e6."""
    from alkahest.experimental import asymptotic_expand

    pool = ak.ExprPool()
    x = pool.symbol("x")
    terms = asymptotic_expand(x * ak.log(1 + 1 / x), x, 4)
    assert len(terms) == 4, [str(t) for t in terms]
    want = [1.0, -0.5, 1 / 3, -0.25]
    for k, (t, c) in enumerate(zip(terms, want)):
        assert ak.eval_expr(t, {x: 10.0}) == pytest.approx(c / 10.0**k)
