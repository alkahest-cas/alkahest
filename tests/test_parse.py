"""V2-21: Pratt expression parser tests."""

from __future__ import annotations

import alkahest
import pytest
from alkahest import (
    ExprPool,
    ParseError,
    cos,
    exp,
    log,
    parse,
    simplify,
    sin,
    sqrt,
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
        # The parser folds `-<literal>`, so this is already the literal -1
        # before `simplify` is given a chance to do anything.
        assert e == pool.integer(-1)
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
# Unary minus on a literal is folded at parse time
#
# Prefix `-` is otherwise `(-1) * operand`, which for a literal operand leaves
# an unevaluated product: `x^(-1)` built `x^(1 * -1)` while `1/x` built
# `x^(-1)`.  Same function, two trees — and every structural detector that
# reads an exponent by matching on an integer node saw only the second, so the
# *spelling* of an integrand decided its route through the integrator.
# ---------------------------------------------------------------------------


class TestNegativeLiteralFolding:
    @pytest.mark.parametrize(
        ("src", "want"),
        [
            ("x^(-1)", -1),
            ("x^-1", -1),
            ("x^(-2)", -2),
            ("(x^2+1)^(-1)", -1),
            ("(x*log(x))^(-1)", -1),
        ],
    )
    def test_a_negative_exponent_is_a_literal(self, pool, x, src, want):
        # The exponent node itself: this is what the detectors read.
        node = parse(src, pool, {"x": x}).node()
        assert node[0] == "pow", src
        assert node[2] == pool.integer(want), f"{src!r} exponent is {node[2]}"

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("2*x^(-1)", "2/x"),
            ("log(x)*(x^2+1)^(-1)", "log(x)/(x^2+1)"),
            ("sin(x)*(x*log(x))^(-1)", "sin(x)/(x*log(x))"),
        ],
    )
    def test_a_over_b_and_a_times_b_to_the_minus_one_are_one_node(self, pool, x, a, b):
        ea = parse(a, pool, {"x": x})
        eb = parse(b, pool, {"x": x})
        assert ea == eb, f"{a!r} -> {ea}, {b!r} -> {eb}"

    @pytest.mark.parametrize("sign", ["", "-"])
    def test_a_huge_integer_exponent_stays_exact(self, pool, x, sign):
        # `Expr.__pow__` falls back to an f64 exponent when the integer does not
        # fit in an i64, so the `^` led hands the exponent node straight to
        # `pow_expr`.  Folding `-<literal>` is what made the negative case reach
        # that code path at all.
        n = 10**25
        e = parse(f"x^({sign}{n})", pool, {"x": x})
        assert e.node()[2] == pool.integer(int(f"{sign}{n}"))

    def test_what_the_fold_does_not_claim(self, pool, x):
        # `/` keeps its left operand, so a bare `1/x` is `1 * x^(-1)` and
        # carries a redundant unit factor that `x^(-1)` does not; and `1/x^2` is
        # `(x^2)^(-1)`, not `x^(-2)`.  Both are pre-existing spelling
        # differences *above* the exponent — the simplifier's job, not the
        # parser's.  Pinned so the tests above are not read as claiming more.
        assert parse("1/x", pool, {"x": x}) != parse("x^(-1)", pool, {"x": x})
        assert parse("1/x^2", pool, {"x": x}) != parse("x^(-2)", pool, {"x": x})

    def test_literal_folds(self, pool):
        assert parse("-3", pool) == pool.integer(-3)
        assert parse("-(-3)", pool) == pool.integer(3)
        assert parse("-0", pool) == pool.integer(0)

    @pytest.mark.parametrize("src", ["-x", "-(2+3)", "-sin(x)", "-(-x)", "-x^2", "-2^2"])
    def test_non_literal_operands_are_left_alone(self, pool, x, src):
        # Over-folding would be a silent precedence change (`-2^2` is `-(2^2)`,
        # never `(-2)^2`) and would strip the `(-1) *` prefix that simplify and
        # the display layer key on for symbolic negation.
        assert parse(src, pool, {"x": x}).node()[0] == "mul"

    def test_float_literals_are_deliberately_out_of_scope(self, pool):
        # Negating a float is exact, but folding it would make `-0.0` and
        # `(-1) * 0.0` two different literals rather than two spellings of one,
        # and no detector keys on a float exponent.
        assert parse("-3.5", pool).node()[0] == "mul"

    def test_the_builder_path_is_unchanged(self, pool, x):
        """The parser fold is layer one; the detectors' tolerance is layer two.

        `Expr.__neg__` and `pool.mul` are public, so `(-1) * literal` is still
        reachable without going through the parser at all.  If this stops being
        true the second layer silently stops being exercised.
        """
        assert (-pool.integer(3)).node()[0] == "mul"
        unevaluated = pool.mul([pool.integer(1), pool.integer(-1)])
        assert unevaluated != pool.integer(-1)
        assert x.pow_expr(unevaluated) != x.pow_expr(pool.integer(-1))

    @pytest.mark.parametrize(
        ("src", "text", "latex", "unicode"),
        [
            # Every one of these used to render the unevaluated product; the
            # pre-fold output is in the trailing comment.  `text` is only pinned
            # where the top node is not a `Mul`/`Add` — their child order is the
            # interning order, which is not a property of this change.
            ("-3", "-3", "-3", "-3"),  # (-1 * 3)
            ("-(-3)", "3", "3", "3"),  # (-1 * (-1 * 3)),  "--3"
            ("x^(-1)", "x^-1", r"\frac{1}{x}", "x⁻¹"),  # x^(1 * -1),  "x^(-1)"
            ("x^(-2)", "x^-2", "x^{-2}", "x⁻²"),  # x^(-1 * 2),  "x^(-2)"
            ("-2/3", None, r"-\frac{2}{3}", "-2/3¹"),  # \frac{-2}{3}
            ("2 - -3", None, "2 + 3", "2 + 3"),  # "--3 + 2"
            # Unchanged, because none of these is `-<literal>`.
            ("-x", None, "-x", "-x"),
            ("x - 3", None, "x - 3", "x - 3"),
            ("-x^2", None, "-x^2", "-x²"),
            ("-3.5", None, "-3.5000000000000000", "-3.5000000000000000"),
        ],
    )
    def test_folded_forms_print_at_least_as_well(self, pool, x, src, text, latex, unicode):
        e = parse(src, pool, {"x": x})
        if text is not None:
            assert str(e) == text
        assert e.display_latex() == latex
        assert e.display_unicode() == unicode

    @pytest.mark.parametrize(
        "src",
        [
            "-x",
            "-3",
            "x - 3",
            "2 - -3",
            "-(-x)",
            "-(-3)",
            "x^-1",
            "x^(-1)",
            "-x^2",
            "1/-x",
            "-2/3",
            "-3.5",
            "1/x",
            "(x^2+1)^(-1)",
        ],
    )
    def test_display_round_trips(self, pool, x, src):
        # A representation change the printer cannot spell back is a round-trip
        # bug, not a simplification.
        e = parse(src, pool, {"x": x})
        shown = str(e)
        assert parse(shown, pool, {"x": x}) == e, f"{src!r} displayed as {shown!r}"


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
        assert neg_x_sq != neg_x**2
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
        assert n[0] == "func"
        assert n[1] == "abs"

    def test_floor(self, pool, x):
        e = parse("floor(x)", pool, {"x": x})
        n = e.node()
        assert n[0] == "func"
        assert n[1] == "floor"

    def test_ceil(self, pool, x):
        e = parse("ceil(x)", pool, {"x": x})
        n = e.node()
        assert n[0] == "func"
        assert n[1] == "ceil"

    def test_two_arg_atan2(self, pool, x, y):
        e = parse("atan2(x, y)", pool, {"x": x, "y": y})
        n = e.node()
        assert n[0] == "func"
        assert n[1] == "atan2"
        assert len(n[2]) == 2

    def test_nested_function(self, pool, x):
        e = parse("sin(x^2)", pool, {"x": x})
        n = e.node()
        assert n[0] == "func"
        assert n[1] == "sin"
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
        built = x**2 + pool.integer(2) * x + pool.integer(1)
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
        with pytest.raises(ParseError) as exc_info:
            parse("x @ y", pool)
        assert exc_info.value.span is not None
        start, end = exc_info.value.span
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

    def test_parse_function_call_without_experimental(self):
        """B1: parse must not require importing alkahest.experimental.

        The function table used to eagerly touch ``_ak.experimental.*``, which
        crashes every call like ``sin(x)`` in a fresh interpreter. Run in a
        subprocess so earlier tests that import experimental cannot mask it.
        """
        import subprocess
        import sys

        script = """
import sys
import alkahest as ak
assert "alkahest.experimental" not in sys.modules
e = ak.parse("sin(x)", ak.ExprPool())
assert "sin" in str(e)
e2 = ak.parse("lambert_w(x)", ak.ExprPool())
assert "lambert_w" in str(e2)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_graduated_specials_are_stable(self):
        for name in ("lambert_w", "digamma", "bessel_j0", "bessel_j1", "evaluate", "residue"):
            assert name in alkahest.__all__, name
            assert hasattr(alkahest, name), name


class TestSpecialFunctionOutputBasis:
    """Every name ``integrate`` can emit must parse back to the same node.

    The parsers are two separate hand-maintained implementations
    (``CONTRIBUTING.md`` § "The parser exists twice"), and nothing cross-checks
    them — so this test is the Python half of
    ``parse.rs::the_special_function_output_basis_round_trips``.  Without it,
    ``parse(str(integrate(f)))`` stops being a round trip the moment the
    integrator learns a name the parser does not know, which is how a printed
    result stops being usable input.
    """

    #: Node names, not Python constructor names: ``str(expr)`` prints the node
    #: name, and reading it back is the whole point.
    OUTPUT_BASIS = (
        ("Ei", 1),
        ("li", 1),
        ("Si", 1),
        ("Ci", 1),
        ("Shi", 1),
        ("Chi", 1),
        ("erf", 1),
        ("erfc", 1),
        ("fresnels", 1),
        ("fresnelc", 1),
        ("dilog", 1),
        ("trigamma", 1),
        ("EllipticK", 1),
        # The incomplete elliptic integrals take `(phi, m)`.
        ("EllipticE", 2),
        ("EllipticF", 2),
    )

    @pytest.mark.parametrize(("name", "arity"), OUTPUT_BASIS)
    def test_round_trips(self, name, arity):
        pool = ExprPool()
        src = f"{name}({', '.join(['x', 'y'][:arity])})"
        once = parse(src, pool)
        rendered = str(once)
        assert name in rendered, f"{src} printed as {rendered}"
        twice = parse(rendered, pool)
        assert str(twice) == rendered, f"{src} does not round trip via {rendered}"

    def test_an_emitted_antiderivative_reads_back(self):
        """The end-to-end property: integrate, print, parse, and agree."""
        pool = ExprPool()
        x = pool.symbol("x")
        for src in ("exp(x)/x", "sin(x)/x", "1/log(x)", "exp(-x^2)", "sin(x^2)"):
            f = parse(src, pool)
            printed = str(alkahest.integrate(f, x).value)
            back = parse(printed, pool)
            assert str(back) == printed, f"∫{src} dx = {printed} does not re-parse"

    def test_cbrt_desugars_to_a_third_power(self):
        """``cbrt(u)`` is ``u^(1/3)``; no ``cbrt`` node exists in the pool.

        Compared after ``simplify``: the desugar builds the exponent as a
        folded ``Rational(1, 3)``, while the source spelling ``x^(1/3)`` builds
        ``1 * 3^-1`` — the Python parser does not fold a literal quotient the
        way the Rust one does.  That divergence is real and pre-dates this test;
        what is asserted here is that the two denote the same expression.
        """
        pool = ExprPool()
        rendered = str(parse("cbrt(x)", pool))
        assert "cbrt" not in rendered, rendered
        assert rendered == "x^(1/3)", rendered
        assert str(simplify(parse("cbrt(x)", pool)).value) == str(
            simplify(parse("x^(1/3)", pool)).value
        )
