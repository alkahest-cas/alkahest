"""V2-21: Pratt expression parser tests."""
from __future__ import annotations

import pytest
import alkahest
from alkahest import (
    ExprPool,
    parse,
    sin, cos, tan, exp, log, sqrt,
    abs, floor, ceil,
    simplify,
    ParseError,
)


@pytest.fixture
def pool():
    return ExprPool()


@pytest.fixture
def x(pool):
    return pool.symbol("x")


@pytest.fixture
def y(pool):
    return pool.symbol("y")


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------

class TestAtoms:
    def test_integer(self, pool):
        assert parse("0", pool) == pool.integer(0)
        assert parse("42", pool) == pool.integer(42)

    def test_symbol_auto_created(self, pool):
        e = parse("x", pool)
        assert e.node()[0] == "symbol"
        assert e.node()[1] == "x"

    def test_symbol_from_map(self, pool, x):
        e = parse("x", pool, {"x": x})
        assert e == x

    def test_float(self, pool):
        e = parse("3.14", pool)
        assert e is not None
        assert e.node()[0] == "float"

    def test_float_scientific(self, pool):
        e = parse("1e5", pool)
        assert e is not None

    def test_float_leading_dot(self, pool):
        e = parse(".5", pool)
        assert e is not None


# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------

class TestUnary:
    def test_unary_minus_integer(self, pool):
        e = parse("-1", pool)
        # -1 is unary minus applied to 1; simplify should give integer(-1)
        s = simplify(e)
        assert s.value == pool.integer(-1)

    def test_unary_minus_symbol(self, pool, x):
        e = parse("-x", pool, {"x": x})
        assert e is not None
        # Should not be the bare symbol
        assert e != x

    def test_unary_plus_symbol(self, pool, x):
        e = parse("+x", pool, {"x": x})
        assert e == x


# ---------------------------------------------------------------------------
# Binary arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_add(self, pool, x):
        e = parse("x + 1", pool, {"x": x})
        assert e.node()[0] == "add"

    def test_sub(self, pool, x):
        e = parse("x - 1", pool, {"x": x})
        assert e is not None

    def test_mul(self, pool, x):
        e = parse("2 * x", pool, {"x": x})
        assert e.node()[0] == "mul"

    def test_div(self, pool, x):
        e = parse("x / 2", pool, {"x": x})
        assert e is not None

    def test_pow_caret(self, pool, x):
        e = parse("x^2", pool, {"x": x})
        assert e.node()[0] == "pow"
        assert e.node()[2] == pool.integer(2)

    def test_pow_starstar(self, pool, x):
        e = parse("x**2", pool, {"x": x})
        assert e.node()[0] == "pow"
        assert e.node()[2] == pool.integer(2)


# ---------------------------------------------------------------------------
# Precedence and associativity
# ---------------------------------------------------------------------------

class TestPrecedence:
    def test_add_mul_precedence(self, pool, x):
        # x + 2 * x  →  add at top level
        e = parse("x + 2 * x", pool, {"x": x})
        assert e.node()[0] == "add"

    def test_mul_pow_precedence(self, pool, x):
        # 2 * x^3  →  mul at top level
        e = parse("2 * x^3", pool, {"x": x})
        assert e.node()[0] == "mul"

    def test_unary_minus_vs_pow(self, pool, x):
        # -x^2 should be -(x^2), not (-x)^2
        neg_x_sq = parse("-x^2", pool, {"x": x})
        x_sq = parse("x^2", pool, {"x": x})
        neg_x = parse("-x", pool, {"x": x})
        # neg_x_sq should not equal (-x)^2
        assert neg_x_sq != neg_x ** 2
        # It should equal -(x^2)
        assert simplify(neg_x_sq).value == simplify(-x_sq).value

    def test_pow_right_associative(self, pool, x):
        # x^2^3 == x^(2^3) == x^8, not (x^2)^3 == x^6
        e = parse("x^2^3", pool, {"x": x})
        # Right-assoc: exponent is 2^3 = 8
        assert e.node()[0] == "pow"
        exp_node = e.node()[2]
        # Should be a pow node for 2^3, or integer 8 after folding
        exp_s = simplify(exp_node).value
        assert exp_s == pool.integer(8)

    def test_left_assoc_sub(self, pool, x):
        # 5 - 3 - 1  →  (5 - 3) - 1  →  1  (not 5 - (3 - 1) = 3)
        e = parse("5 - 3 - 1", pool)
        s = simplify(e)
        assert s.value == pool.integer(1)

    def test_paren_override(self, pool, x):
        # (x + 2) * x  →  mul at top level
        e = parse("(x + 2) * x", pool, {"x": x})
        assert e.node()[0] == "mul"


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------

class TestFunctions:
    def test_sin(self, pool, x):
        e = parse("sin(x)", pool, {"x": x})
        assert e == sin(x)

    def test_cos(self, pool, x):
        e = parse("cos(x)", pool, {"x": x})
        assert e == cos(x)

    def test_exp(self, pool, x):
        e = parse("exp(x)", pool, {"x": x})
        assert e == exp(x)

    def test_log(self, pool, x):
        e = parse("log(x)", pool, {"x": x})
        assert e == log(x)

    def test_sqrt(self, pool, x):
        e = parse("sqrt(x)", pool, {"x": x})
        assert e == sqrt(x)

    def test_abs(self, pool, x):
        e = parse("abs(x)", pool, {"x": x})
        n = e.node()
        assert n[0] == "func" and n[1] == "abs"

    def test_floor(self, pool, x):
        e = parse("floor(x)", pool, {"x": x})
        n = e.node()
        assert n[0] == "func" and n[1] == "floor"

    def test_ceil(self, pool, x):
        e = parse("ceil(x)", pool, {"x": x})
        n = e.node()
        assert n[0] == "func" and n[1] == "ceil"

    def test_two_arg_atan2(self, pool, x, y):
        e = parse("atan2(x, y)", pool, {"x": x, "y": y})
        n = e.node()
        assert n[0] == "func" and n[1] == "atan2"
        assert len(n[2]) == 2

    def test_nested_function(self, pool, x):
        e = parse("sin(x^2)", pool, {"x": x})
        n = e.node()
        assert n[0] == "func" and n[1] == "sin"
        arg_node = n[2][0].node()
        assert arg_node[0] == "pow"

    def test_function_in_expression(self, pool, x):
        e = parse("2 * sin(x) + 1", pool, {"x": x})
        assert e.node()[0] == "add"


# ---------------------------------------------------------------------------
# Whitespace handling
# ---------------------------------------------------------------------------

class TestWhitespace:
    def test_spaces_around_ops(self, pool, x):
        e1 = parse("x + 1", pool, {"x": x})
        e2 = parse("x+1", pool, {"x": x})
        assert e1 == e2

    def test_tabs_and_newlines(self, pool, x):
        e = parse("x\t+\n1", pool, {"x": x})
        assert e.node()[0] == "add"


# ---------------------------------------------------------------------------
# Symbol map reuse within a call
# ---------------------------------------------------------------------------

class TestSymbolMap:
    def test_same_name_reused(self, pool):
        sym_map: dict = {}
        e = parse("x + x", pool, sym_map)
        # Both x's should be the same interned symbol
        n = e.node()
        assert n[0] == "add"
        children = n[1]
        syms = [c for c in children if c.node()[0] == "symbol"]
        assert len(syms) == 2
        assert syms[0] == syms[1]

    def test_pre_bound_symbol_used(self, pool, x):
        sym_map = {"x": x}
        e = parse("x * x", pool, sym_map)
        n = e.node()
        assert n[0] == "mul"
        for child in n[1]:
            if child.node()[0] == "symbol":
                assert child == x


# ---------------------------------------------------------------------------
# Round-trip equivalence
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_quadratic(self, pool, x):
        built = x ** 2 + pool.integer(2) * x + pool.integer(1)
        parsed = parse("x^2 + 2*x + 1", pool, {"x": x})
        assert simplify(built).value == simplify(parsed).value

    def test_diff_after_parse(self, pool, x):
        from alkahest import diff
        e = parse("x^3 + x", pool, {"x": x})
        r = diff(e, x)
        # d/dx (x^3 + x) = 3x^2 + 1
        from alkahest import UniPoly
        poly = UniPoly.from_symbolic(r.value, x)
        assert poly.coefficients() == [1, 0, 3]

    def test_integrate_after_parse(self, pool, x):
        from alkahest import integrate
        e = parse("exp(x)", pool, {"x": x})
        r = integrate(e, x)
        assert r.value == exp(x)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_empty_string(self, pool):
        with pytest.raises(ParseError):
            parse("", pool)

    def test_whitespace_only(self, pool):
        with pytest.raises(ParseError):
            parse("   ", pool)

    def test_unexpected_char(self, pool):
        with pytest.raises(ParseError):
            parse("x @ y", pool)

    def test_empty_parens(self, pool):
        with pytest.raises(ParseError):
            parse("()", pool)

    def test_unclosed_paren(self, pool):
        with pytest.raises(ParseError):
            parse("(x + 1", pool)

    def test_trailing_operator(self, pool):
        with pytest.raises(ParseError):
            parse("x + ", pool)

    def test_unknown_function(self, pool):
        with pytest.raises(ParseError):
            parse("zeta(x)", pool)

    def test_extra_token(self, pool):
        with pytest.raises(ParseError):
            parse("x y", pool)

    def test_double_star_missing_rhs(self, pool):
        with pytest.raises(ParseError):
            parse("x **", pool)

    def test_parse_error_has_span(self, pool):
        try:
            parse("x @ y", pool)
        except ParseError as e:
            assert e.span is not None
            start, end = e.span
            assert end > start


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_parse_in_module(self):
        assert hasattr(alkahest, "parse")

    def test_parse_in_all(self):
        assert "parse" in alkahest.__all__

    def test_parse_error_in_module(self):
        assert hasattr(alkahest, "ParseError")

    def test_parse_error_in_all(self):
        assert "ParseError" in alkahest.__all__

    def test_parse_returns_expr(self, pool, x):
        e = parse("x + 1", pool, {"x": x})
        assert isinstance(e, alkahest.Expr)
