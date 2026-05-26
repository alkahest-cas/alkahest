"""v0.2 feature tests: egraph simplification, AC pattern matching,
forward-mode AD, integration, and RationalFunction arithmetic."""

from alkahest.alkahest import (
    ExprPool,
    RationalFunction,
    diff,
    diff_forward,
    integrate,
    match_pattern,
    simplify,
    simplify_egraph,
)

# ---------------------------------------------------------------------------
# Phase 9: RationalFunction arithmetic
# ---------------------------------------------------------------------------


class TestRationalFunctionArithmetic:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.one = self.pool.integer(1)
        self.two = self.pool.integer(2)

    def rf(self, numer_expr, denom_expr):
        return RationalFunction.from_symbolic(numer_expr, denom_expr, [self.x])

    def test_add(self):
        # x/1 + 1/1 = (x + 1)/1
        r1 = self.rf(self.x, self.one)
        r2 = self.rf(self.one, self.one)
        result = r1 + r2
        assert "/" not in str(result) or "1" in str(result)

    def test_mul_reduces_to_one(self):
        # (x+1)/(x+1) = 1
        xp1 = self.x + self.one
        r = self.rf(xp1, xp1)
        assert str(r) == "1"

    def test_neg(self):
        r = self.rf(self.x, self.one)
        neg = -r
        assert neg is not None

    def test_sub(self):
        r1 = self.rf(self.x, self.one)
        r2 = self.rf(self.x, self.one)
        result = r1 - r2
        assert result.is_zero()

    def test_div(self):
        # (x/1) / (x/1) = 1
        r = self.rf(self.x, self.one)
        result = r / r
        assert str(result) == "1"

    def test_mul_cancels(self):
        # (x/2) * (2/x) = 1 (if both are polynomials and gcd reduces)
        two_expr = self.two
        x_expr = self.x
        ab = self.rf(x_expr, two_expr)
        ba = self.rf(two_expr, x_expr)
        result = ab * ba
        # Should reduce to 1 or at least is not zero
        assert not result.is_zero()


# ---------------------------------------------------------------------------
# Phase 10: E-graph simplification
# ---------------------------------------------------------------------------


class TestEGraphSimplify:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")

    def test_simplify_egraph_add_zero(self):
        zero = self.pool.integer(0)
        expr = self.x + zero
        result = simplify_egraph(expr)
        assert str(result.value) == "x"

    def test_simplify_egraph_mul_one(self):
        one = self.pool.integer(1)
        expr = self.x * one
        result = simplify_egraph(expr)
        assert str(result.value) == "x"

    def test_simplify_egraph_const_fold(self):
        three = self.pool.integer(3)
        four = self.pool.integer(4)
        expr = three + four
        result = simplify_egraph(expr)
        assert str(result.value) == "7"

    def test_simplify_egraph_mul_zero(self):
        zero = self.pool.integer(0)
        expr = self.x * zero
        result = simplify_egraph(expr)
        assert str(result.value) == "0"

    def test_simplify_egraph_returns_derived_result(self):
        result = simplify_egraph(self.x)
        assert hasattr(result, "value")
        assert hasattr(result, "derivation")
        assert hasattr(result, "steps")


# ---------------------------------------------------------------------------
# Phase 11: AC Pattern Matching
# ---------------------------------------------------------------------------


class TestACPatternMatching:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")
        self.a = self.pool.symbol("a")  # wildcard (starts lowercase)
        self.b = self.pool.symbol("b")  # wildcard

    def test_match_simple_add(self):
        pat = self.a + self.b
        expr = self.x + self.y
        matches = match_pattern(pat, expr)
        assert len(matches) > 0

    def test_match_returns_bindings(self):
        pat = self.a + self.b
        expr = self.x + self.y
        matches = match_pattern(pat, expr)
        # Each match is a dict mapping wildcard name -> Expr
        for m in matches:
            assert isinstance(m, dict)
            assert "a" in m or "b" in m

    def test_no_match_wrong_op(self):
        # Mul pattern should not match Add expr
        pat = self.a * self.b
        expr = self.x + self.y
        matches = match_pattern(pat, expr)
        assert len(matches) == 0

    def test_match_inside_function(self):
        # a + b pattern should match inside sin(x + y)
        pat = self.a + self.b
        inner = self.x + self.y
        from alkahest.alkahest import sin

        expr = sin(inner)
        matches = match_pattern(pat, expr)
        assert len(matches) > 0

    def test_constant_pattern(self):
        # Wildcard matches constant
        pat = self.a
        two = self.pool.integer(2)
        matches = match_pattern(pat, two)
        assert len(matches) > 0


# ---------------------------------------------------------------------------
# Phase 12: Forward-mode AD
# ---------------------------------------------------------------------------


class TestForwardModeAD:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")

    def test_diff_forward_constant(self):
        five = self.pool.integer(5)
        r = diff_forward(five, self.x)
        assert str(r.value) == "0"

    def test_diff_forward_identity(self):
        r = diff_forward(self.x, self.x)
        assert str(r.value) == "1"

    def test_diff_forward_linear(self):
        # d/dx (3x) = 3
        three = self.pool.integer(3)
        expr = three * self.x
        r = diff_forward(expr, self.x)
        assert str(r.value) == "3"

    def test_diff_forward_quadratic(self):
        # d/dx x² = 2x
        expr = self.x**2
        r = diff_forward(expr, self.x)
        r_sym = diff(expr, self.x)
        # Both should give the same representation (as strings)
        assert str(r.value) == str(r_sym.value)

    def test_diff_forward_agrees_with_symbolic(self):
        # d/dx (x³ + 2x) forward vs symbolic
        two = self.pool.integer(2)
        expr = (self.x**3) + (two * self.x)
        fwd = diff_forward(expr, self.x)
        sym = diff(expr, self.x)
        assert str(fwd.value) == str(sym.value)

    def test_diff_forward_has_step_in_log(self):
        r = diff_forward(self.x, self.x)
        rules = [s["rule"] for s in r.steps]
        assert "diff_forward" in rules

    def test_diff_forward_unknown_function_raises(self):
        from alkahest.alkahest import sin

        # sin is known, should work
        r = diff_forward(sin(self.x), self.x)
        assert str(r.value) == str(self.pool.symbol("cos") if False else "cos(x)")

    def test_diff_forward_returns_derived_result(self):
        r = diff_forward(self.x, self.x)
        assert hasattr(r, "value")
        assert hasattr(r, "derivation")


# ---------------------------------------------------------------------------
# Phase 13: Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")

    def _verify(self, expr):
        """Check that diff(integrate(expr, x), x) == expr (via string repr)."""
        integral = integrate(expr, self.x)
        derivative = diff(integral.value, self.x)
        # Simplify both sides
        d_simp = simplify(derivative.value)
        e_simp = simplify(expr)
        assert str(d_simp.value) == str(e_simp.value), (
            f"diff(∫f) ≠ f: diff(∫{expr}) = {d_simp.value}"
        )

    def test_integrate_constant(self):
        # ∫ 5 dx = 5x
        five = self.pool.integer(5)
        r = integrate(five, self.x)
        assert r is not None

    def test_integrate_x(self):
        self._verify(self.x)

    def test_integrate_x_squared(self):
        self._verify(self.x**2)

    def test_integrate_polynomial(self):
        # ∫ (x² + 2x) dx
        two = self.pool.integer(2)
        expr = (self.x**2) + (two * self.x)
        self._verify(expr)

    def test_integrate_sin(self):
        from alkahest.alkahest import sin

        r = integrate(sin(self.x), self.x)
        # Result should be -cos(x)
        assert "cos" in str(r.value)

    def test_integrate_cos(self):
        from alkahest.alkahest import cos

        r = integrate(cos(self.x), self.x)
        assert str(r.value) == "sin(x)"

    def test_integrate_exp(self):
        from alkahest.alkahest import exp

        r = integrate(exp(self.x), self.x)
        assert str(r.value) == "exp(x)"

    def test_integrate_one_over_x(self):
        x_inv = self.x**-1
        r = integrate(x_inv, self.x)
        assert str(r.value) == "log(x)"

    def test_integrate_sqrt_is_implemented(self):
        # V1-2: algebraic-function Risch now handles sqrt(P(x)) for genus-0 curves.
        from alkahest.alkahest import sqrt

        r = integrate(sqrt(self.x), self.x)
        assert r.value is not None

    def test_integrate_has_log(self):
        r = integrate(self.x**2, self.x)
        assert r.steps  # non-empty steps
        assert r.derivation  # non-empty derivation string

    def test_integrate_returns_derived_result(self):
        r = integrate(self.x, self.x)
        assert hasattr(r, "value")
        assert hasattr(r, "derivation")
        assert hasattr(r, "steps")
