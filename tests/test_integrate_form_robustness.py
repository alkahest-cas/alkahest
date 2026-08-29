"""
`integrate` must not let the *spelling* of an integrand decide the answer.

`a/b`, `a*b**(-1)` and `(a**(-1)*b)**(-1)` denote the same function but build
three different expression trees, and an exponent of `-1` did not always arrive
as the literal `-1`: prefix negation is `(-1) * operand`, so `^(-1)` left an
unevaluated `1 * -1` where `/` gave a literal `-1`.  (The parser now folds that
case, but `Expr.__neg__` and `pool.mul` are public, so a caller building an
expression by hand still can produce it.)  Every structural detector in the
router keyed on tree shape, so the same mathematical object used to get
different verdicts:

    x^(-1)*log(x)^(-1)      ->  log(log(x))
    1/(x*log(x))            ->  E-INT-001        # the same function
    exp(x)*(1+exp(x))^(-1)  ->  log(exp(x)+1)
    exp(x)/(exp(x)+1)       ->  E-INT-001        # the same function

Every case below is verified by differentiating the result back, never by
matching a display string.
"""

import alkahest as ak
import pytest
from alkahest.alkahest import ArbBall, ExprPool, diff, integrate, interval_eval

# Points > 1 so every log in play is real and positive.
_TEST_POINTS = (1.3, 2.1, 3.4, 4.7)


def _parse(src, pool):
    return ak.parse(src, pool)


def _check_antiderivative(pool, x, f, cap, label):
    """d/dx F(x) == f(x) at several real points."""
    d = diff(cap, x).value
    checked = 0
    for pt in _TEST_POINTS:
        bindings = {x: ArbBall(pt)}
        lhs = interval_eval(d, bindings).mid
        rhs = interval_eval(f, bindings).mid
        if lhs != lhs or rhs != rhs:  # NaN at a singularity — skip
            continue
        assert abs(lhs - rhs) < 1e-8, (
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n  F = {cap}\n  f = {f}"
        )
        checked += 1
    assert checked >= 2, f"{label}: too few usable sample points"
    return d


def _integrate_verified(src, pool, x):
    f = _parse(src, pool)
    result = integrate(f, x)
    _check_antiderivative(pool, x, f, result.value, src)
    return result.value


def _spellings(a, b):
    """The three quotient spellings that used to route differently."""
    return [
        f"{a}/({b})",
        f"({a})*({b})^(-1)",
        f"(({a})^(-1)*({b}))^(-1)",
    ]


# Numerator / denominator pairs, each integrated in all three spellings.
_QUOTIENTS = [
    ("1", "x*log(x)"),  # log(log x) — was E-INT-001 in two of three spellings
    ("1", "x*log(x)^2"),
    ("1", "x*log(x)^3"),
    ("2*x", "(x^2+1)*log(x^2+1)"),
    ("log(x)", "x"),
    ("exp(x)", "exp(x)+1"),  # log(exp(x)+1) — was E-INT-001 as written with `/`
    ("exp(x)", "(exp(x)+1)^2"),
    ("1", "exp(x)+1"),
    ("1", "1+exp(-x)"),
    ("x", "x^2+1"),  # controls: these already agreed
    ("1", "x*(x+1)"),
]


@pytest.mark.parametrize(("numer", "denom"), _QUOTIENTS)
def test_every_spelling_of_a_quotient_integrates_to_the_same_thing(numer, denom):
    pool = ExprPool()
    x = pool.symbol("x")
    caps = [_integrate_verified(src, pool, x) for src in _spellings(numer, denom)]

    # Two antiderivatives of one integrand differ by at most a constant.
    base = diff(caps[0], x).value
    for cap, src in zip(caps[1:], _spellings(numer, denom)[1:]):
        other = diff(cap, x).value
        for pt in _TEST_POINTS:
            bindings = {x: ArbBall(pt)}
            lhs = interval_eval(base, bindings).mid
            rhs = interval_eval(other, bindings).mid
            if lhs != lhs or rhs != rhs:
                continue
            assert abs(lhs - rhs) < 1e-8, f"{src}: spellings disagree at x={pt}: {lhs} vs {rhs}"


def test_one_over_x_log_x():
    """∫ dx/(x·log x) = log(log x), however it is written."""
    pool = ExprPool()
    x = pool.symbol("x")
    for src in [
        "1/(x*log(x))",
        "x^(-1)*log(x)^(-1)",
        "(x*log(x))^(-1)",
        "1/x*1/log(x)",
        "1/x/log(x)",
    ]:
        cap = _integrate_verified(src, pool, x)
        assert "log(log" in str(cap), f"∫ {src} dx should be log(log x); got {cap}"


def test_exp_over_exp_plus_one():
    """∫ eˣ/(eˣ+1) dx = log(eˣ+1), however it is written."""
    pool = ExprPool()
    x = pool.symbol("x")
    for src in [
        "exp(x)/(exp(x)+1)",
        "exp(x)*(1+exp(x))^(-1)",
        "((exp(x))^(-1)*(exp(x)+1))^(-1)",
        "exp(x)/(1+exp(x))",
    ]:
        _integrate_verified(src, pool, x)


def test_logistic_is_elementary():
    """∫ dx/(1+e⁻ˣ) = x + log(e⁻ˣ+1) (up to a constant), verified by d/dx."""
    pool = ExprPool()
    x = pool.symbol("x")
    for src in ["1/(1+exp(-x))", "(1+exp(-x))^(-1)", "1/(exp(x)+1)"]:
        _integrate_verified(src, pool, x)


# ---------------------------------------------------------------------------
# The router now falls through on a sub-engine decline.  A *proof* of
# non-elementarity must not be downgraded to the weaker "not implemented" by a
# fallback running out of options.
# ---------------------------------------------------------------------------

_NONELEMENTARY = [
    "exp(x^2)",
    "exp(-x^2)",
    "sin(x)/x",
    "cos(x)/x",
    "exp(x)/x",
    "exp(x)*x^(-1)",  # the same integrand, spelled with `^(-1)`
    "1/log(x)",
]


@pytest.mark.parametrize("src", _NONELEMENTARY)
def test_nonelementary_stays_nonelementary(src):
    pool = ExprPool()
    x = pool.symbol("x")
    f = _parse(src, pool)
    with pytest.raises(ak.IntegrationError) as excinfo:
        integrate(f, x)
    assert excinfo.value.code == "E-INT-004", (
        f"∫ {src} dx should certify NonElementary, got {excinfo.value.code}: {excinfo.value}"
    )


# ---------------------------------------------------------------------------
# Associativity.  `(a*b)*c`, `a*(b*c)` and `a*b*c` are one expression, because
# `ExprPool.mul`/`add` splice nested same-operator children at construction.
# Before that, the parsers' left-associative chains meant every matcher that
# scans the *top-level* arguments of a product or a sum saw two children where
# the user wrote three, and quietly declined to fire on parsed input.
# ---------------------------------------------------------------------------


def _assoc_spellings(a, b, c, op):
    return [
        f"(({a}){op}({b})){op}({c})",
        f"({a}){op}(({b}){op}({c}))",
        f"({a}){op}({b}){op}({c})",
    ]


_ASSOC_PRODUCTS = [
    ("x", "x", "log(x)"),
    ("x", "x", "sin(x)"),
    ("x", "x", "sqrt(x)"),
    ("2", "x", "atan(x)"),
    ("3", "x", "asin(x)"),
    ("x", "exp(x)", "2"),
    ("x^2", "log(x)", "1"),
    ("1/x", "1/log(x)", "1"),
    ("exp(x)", "1/(exp(x)+1)", "1"),
]

_ASSOC_SUMS = [
    ("x", "x^2", "atan(x)"),
    ("exp(x)", "sin(x)", "1/x"),
    ("1/x", "1/(x+1)", "1/(x+2)"),
    ("log(x)", "x", "1"),
    ("sin(x)", "cos(x)", "x^3"),
]


@pytest.mark.parametrize(("a", "b", "c"), _ASSOC_PRODUCTS)
def test_product_associativity_gives_one_expression(a, b, c):
    pool = ExprPool()
    pool.symbol("x")
    spellings = _assoc_spellings(a, b, c, "*")
    first = _parse(spellings[0], pool)
    for src in spellings[1:]:
        assert _parse(src, pool) == first, f"{src} is not the same node as {spellings[0]}"


@pytest.mark.parametrize(("a", "b", "c"), _ASSOC_SUMS)
def test_sum_associativity_gives_one_expression(a, b, c):
    pool = ExprPool()
    pool.symbol("x")
    spellings = _assoc_spellings(a, b, c, "+")
    first = _parse(spellings[0], pool)
    for src in spellings[1:]:
        assert _parse(src, pool) == first, f"{src} is not the same node as {spellings[0]}"


def _verdict(src, pool, x):
    """(code, result-or-None) for ∫ src dx — never raises."""
    f = _parse(src, pool)
    try:
        return "OK", integrate(f, x).value
    except ak.IntegrationError as exc:
        return exc.code, None


# `_TEST_POINTS` is all > 1 because the quotient cases above are full of logs.
# The associativity cases include `asin`/`acos`, which are real only on [-1, 1],
# so verification here samples both sides of 1 and tolerates the points where
# ball evaluation refuses.
_ASSOC_POINTS = (0.11, 0.29, 0.47, 0.63, 0.81, 1.3, 2.1, 3.4, 4.7)


def _check_antiderivative_anywhere(pool, x, f, cap, label):
    """d/dx F == f wherever both sides evaluate; at least two such points."""
    d = diff(cap, x).value
    checked = 0
    for pt in _ASSOC_POINTS:
        bindings = {x: ArbBall(pt)}
        try:
            lhs = interval_eval(d, bindings).mid
            rhs = interval_eval(f, bindings).mid
        except ValueError:
            continue  # outside the real domain of this integrand
        if lhs != lhs or rhs != rhs:
            continue
        assert abs(lhs - rhs) < 1e-8, (
            f"{label}: d/dx F({pt}) = {lhs}, f({pt}) = {rhs} — mismatch\n  F = {cap}\n  f = {f}"
        )
        checked += 1
    assert checked >= 2, f"{label}: too few usable sample points"


@pytest.mark.parametrize(("a", "b", "c"), _ASSOC_PRODUCTS + _ASSOC_SUMS)
def test_every_association_integrates_identically(a, b, c):
    """Same verdict, same antiderivative, whichever way the user brackets it."""
    for op in ("*", "+"):
        pool = ExprPool()
        x = pool.symbol("x")
        spellings = _assoc_spellings(a, b, c, op)
        verdicts = [_verdict(src, pool, x) for src in spellings]
        codes = {v[0] for v in verdicts}
        assert len(codes) == 1, f"{spellings} disagree on verdict: {codes}"
        results = {v[1] for v in verdicts}
        assert len(results) == 1, f"{spellings} disagree on result: {results}"
        # Anything that came back must actually be an antiderivative.
        if verdicts[0][0] == "OK":
            f = _parse(spellings[0], pool)
            _check_antiderivative_anywhere(pool, x, f, verdicts[0][1], spellings[0])


def test_a_three_factor_product_reaches_the_top_level_matchers():
    """`match_inverse_trig_power` and friends scan top-level `Mul` arguments.

    A parsed three-factor product used to arrive as a two-child `Mul` with the
    inverse-trig factor buried in a nested `Mul`, so the scan never saw it.  It
    is now a flat three-child node and the factor is visible.
    """
    pool = ExprPool()
    x = pool.symbol("x")
    f = _parse("2*x*atan(x)", pool)
    tag, children = f.node()
    assert tag == "mul"
    assert len(children) == 3, f"expected a flat 3-factor product, got {f}"
    # The inverse-trig factor is a *top-level* argument, not nested.
    assert any(c.node()[0] == "func" and c.node()[1] == "atan" for c in children)
    cap = integrate(f, x).value
    _check_antiderivative(pool, x, f, cap, "2*x*atan(x)")


def test_association_does_not_change_a_nonelementary_verdict():
    """A certified `E-INT-004` must stay certified, however it is bracketed."""
    pool = ExprPool()
    x = pool.symbol("x")
    for src in _assoc_spellings("exp(x^2)", "1", "1", "*"):
        f = _parse(src, pool)
        with pytest.raises(ak.IntegrationError) as excinfo:
            integrate(f, x)
        assert excinfo.value.code == "E-INT-004", f"{src} lost its certificate"
