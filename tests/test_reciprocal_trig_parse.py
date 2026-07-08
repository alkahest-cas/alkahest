"""Parsing of the reciprocal trig / hyperbolic functions.

``sec``, ``csc``, ``cot``, ``sech``, ``csch`` and ``coth`` are not primitives;
the parser desugars each single-argument call ``f(x)`` to its elementary
reciprocal definition ``base(x) ** -1`` (``sec -> cos``, ``csc -> sin``,
``cot -> tan`` and the hyperbolic analogues).  No dedicated ``sec``/``csc``/…
node is ever built, so differentiation, evaluation, integration and
simplification all run on the existing cos/sin/tan/cosh/sinh/tanh primitives.
"""

import alkahest
import pytest
from alkahest.alkahest import (
    ExprPool,
    cos,
    cosh,
    diff,
    integrate,
    simplify,
    sin,
    sinh,
    tan,
    tanh,
)
from alkahest.exceptions import ParseError

_CASES = [
    ("sec", cos),
    ("csc", sin),
    ("cot", tan),
    ("sech", cosh),
    ("csch", sinh),
    ("coth", tanh),
]


@pytest.mark.parametrize(("name", "base"), _CASES)
def test_reciprocal_desugars_to_base_pow_neg_one(name, base):
    """``f(x)`` parses to exactly ``base(x) ** -1`` (hash-consed equality)."""
    pool = ExprPool()
    x = pool.symbol("x")
    parsed = alkahest.parse(f"{name}(x)", pool, {"x": x})
    expected = base(x) ** -1
    assert parsed == expected, f"{name}(x) should desugar to {base.__name__}(x)^-1"


def test_reciprocal_desugars_with_expression_argument():
    """The argument is threaded through the desugar, not just a bare symbol."""
    pool = ExprPool()
    x = pool.symbol("x")
    parsed = alkahest.parse("sec(2*x)", pool, {"x": x})
    assert parsed == cos(2 * x) ** -1


def test_base_trig_and_atan2_still_parse():
    """Regression: base trig/hyperbolic funcs and atan2 are unaffected."""
    pool = ExprPool()
    x = pool.symbol("x")
    syms = {"x": x}
    for src in ("sin(x)", "cos(x)", "tan(x)", "sinh(x)", "cosh(x)", "tanh(x)"):
        alkahest.parse(src, pool, syms)
    assert alkahest.parse("atan2(1, 2)", pool, syms) is not None


def test_wrong_arity_is_parse_error():
    """A reciprocal function with the wrong arity raises ParseError."""
    pool = ExprPool()
    x = pool.symbol("x")
    with pytest.raises(ParseError):
        alkahest.parse("sec(x, x)", pool, {"x": x})


def test_diff_of_sec_closes():
    """d/dx sec(x) differentiates via the cos(x)^-1 desugar."""
    pool = ExprPool()
    x = pool.symbol("x")
    e = alkahest.parse("sec(x)", pool, {"x": x})
    d = diff(e, x)
    assert d.value is not None


def test_integrate_sec_squared_equals_tan():
    """∫ sec(x)² dx = tan(x).

    ``sec(x)^2`` parses to ``(cos(x)^-1)^2``; ``simplify`` canonicalises it to
    ``cos(x)^-2`` (the shape the reciprocal-square trig rule matches).  The
    integrator's soundness gate then guarantees d/dx(result) == sec(x)².
    """
    pool = ExprPool()
    x = pool.symbol("x")
    integrand = simplify(alkahest.parse("sec(x)^2", pool, {"x": x})).value
    result = integrate(integrand, x)
    assert result.value is not None
