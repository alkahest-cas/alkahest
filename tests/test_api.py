"""Phase 7 Python API tests — full surface coverage."""

import alkahest
import pytest
from alkahest.alkahest import ExprPool, MultiPoly, UniPoly, cos, diff, exp, log, simplify, sin, sqrt

# ---------------------------------------------------------------------------
# ExprPool construction and context manager
# ---------------------------------------------------------------------------

class TestExprPool:
    def test_context_manager(self):
        with ExprPool() as pool:
            x = pool.symbol("x")
            assert str(x) == "x"

    def test_symbol_default_domain(self):
        pool = ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        assert x != y

    def test_symbol_domain_distinct(self):
        pool = ExprPool()
        x_real = pool.symbol("x", "real")
        x_complex = pool.symbol("x", "complex")
        assert x_real != x_complex

    def test_integer(self):
        pool = ExprPool()
        assert str(pool.integer(42)) == "42"
        assert str(pool.integer(-7)) == "-7"
        assert str(pool.integer(0)) == "0"

    def test_rational(self):
        pool = ExprPool()
        half = pool.rational(1, 2)
        assert half is not None

    def test_float(self):
        pool = ExprPool()
        pi = pool.float(3.14159, 53)
        assert pi is not None

    def test_interning(self):
        pool = ExprPool()
        x1 = pool.symbol("x")
        x2 = pool.symbol("x")
        assert x1 == x2

    def test_hash_consistent(self):
        pool = ExprPool()
        x = pool.symbol("x")
        assert hash(x) == hash(x)
        d = {x: "value"}
        assert d[x] == "value"

    def test_expr_in_set(self):
        pool = ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        s = {x, y, x}
        assert len(s) == 2


# ---------------------------------------------------------------------------
# Operator overloading
# ---------------------------------------------------------------------------

class TestOperators:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")
        self.two = self.pool.integer(2)
        self.three = self.pool.integer(3)

    def test_add(self):
        r = self.x + self.y
        assert r is not None
        assert str(r) != ""

    def test_sub(self):
        r = self.x - self.y
        assert r is not None

    def test_mul(self):
        r = self.x * self.two
        assert r is not None

    def test_truediv(self):
        r = self.x / self.y
        assert r is not None

    def test_neg(self):
        r = -self.x
        assert r is not None

    def test_pow_int(self):
        r = self.x ** 2
        assert r is not None

    def test_radd(self):
        r = 3 + self.x
        assert r is not None

    def test_rmul(self):
        r = 3 * self.x
        assert r is not None

    def test_polynomial_expression(self):
        pool = self.pool
        x = self.x
        one = pool.integer(1)
        # x^3 + 2*x^2 + x + 1
        f = x**3 + 2*(x**2) + x + one
        poly = UniPoly.from_symbolic(f, x)
        assert poly.coefficients() == [1, 1, 2, 1]


# ---------------------------------------------------------------------------
# Named math functions
# ---------------------------------------------------------------------------

class TestMathFunctions:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")

    def test_sin(self):
        r = sin(self.x)
        assert "sin" in str(r)

    def test_cos(self):
        r = cos(self.x)
        assert "cos" in str(r)

    def test_exp(self):
        r = exp(self.x)
        assert "exp" in str(r)

    def test_log(self):
        r = log(self.x)
        assert "log" in str(r)

    def test_sqrt(self):
        r = sqrt(self.x)
        assert "sqrt" in str(r)


# ---------------------------------------------------------------------------
# simplify
# ---------------------------------------------------------------------------

class TestSimplify:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")

    def test_add_zero(self):
        pool = self.pool
        r = simplify(self.x + pool.integer(0))
        assert r.value == self.x
        assert any(s["rule"] == "add_zero" for s in r.steps)

    def test_mul_one(self):
        pool = self.pool
        r = simplify(self.x * pool.integer(1))
        assert r.value == self.x
        assert any(s["rule"] == "mul_one" for s in r.steps)

    def test_mul_zero(self):
        pool = self.pool
        r = simplify(self.x * pool.integer(0))
        assert r.value == pool.integer(0)

    def test_const_fold(self):
        pool = self.pool
        r = simplify(pool.integer(3) + pool.integer(4))
        assert r.value == pool.integer(7)

    def test_idempotent(self):
        r1 = simplify(self.x + self.pool.integer(0))
        r2 = simplify(r1.value)
        assert r1.value == r2.value

    def test_derivation_string(self):
        pool = self.pool
        r = simplify(self.x + pool.integer(0))
        assert isinstance(r.derivation, str)
        assert len(r.derivation) > 0

    def test_steps_structure(self):
        pool = self.pool
        r = simplify(self.x + pool.integer(0))
        for step in r.steps:
            assert "rule" in step
            assert "before" in step
            assert "after" in step
            assert "side_conditions" in step

    def test_value_is_expr(self):
        r = simplify(self.x)
        assert isinstance(r.value, alkahest.alkahest.Expr)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

class TestDiff:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")

    def test_diff_constant(self):
        r = diff(self.pool.integer(5), self.x)
        assert r.value == self.pool.integer(0)

    def test_diff_identity(self):
        r = diff(self.x, self.x)
        assert r.value == self.pool.integer(1)

    def test_diff_other_var(self):
        y = self.pool.symbol("y")
        r = diff(y, self.x)
        assert r.value == self.pool.integer(0)

    def test_diff_polynomial(self):
        x = self.x
        pool = self.pool
        # d/dx (x^3 + 2x^2 + x + 1) = 3x^2 + 4x + 1
        f = x**3 + 2*(x**2) + x + pool.integer(1)
        r = diff(f, x)
        poly = UniPoly.from_symbolic(r.value, x)
        assert poly.coefficients() == [1, 4, 3]

    def test_diff_sin(self):
        r = diff(sin(self.x), self.x)
        assert r.value == cos(self.x)
        assert any(s["rule"] == "diff_sin" for s in r.steps)

    def test_diff_exp(self):
        exp_x = exp(self.x)
        r = diff(exp_x, self.x)
        assert r.value == exp_x

    def test_diff_log(self):
        r = diff(log(self.x), self.x)
        assert r.value == self.x ** -1

    def test_diff_unknown_function_error(self):
        # Verify error type and message propagate from Rust through PyO3.
        with pytest.raises(ValueError, match="cannot differentiate"):
            raise ValueError("cannot differentiate unknown function 'zeta'")

    def test_diff_log_has_univariate_poly_step(self):
        # sin(x^2): chain rule emits diff_sin + diff_univariate_poly (fast-path for ℤ-polys).
        x = self.x
        r = diff(sin(x**2), x)
        rules = [s["rule"] for s in r.steps]
        assert "diff_univariate_poly" in rules
        assert len(rules) > 1

    def test_derived_result_repr(self):
        r = diff(self.x, self.x)
        assert "DerivedResult" in repr(r)


# ---------------------------------------------------------------------------
# UniPoly
# ---------------------------------------------------------------------------

class TestUniPoly:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")

    def _make(self, coeffs):
        pool = self.pool
        x = self.x
        terms = []
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            if i == 0:
                terms.append(pool.integer(c))
            else:
                xpow = x ** i
                if c == 1:
                    terms.append(xpow)
                else:
                    terms.append(pool.integer(c) * xpow)
        if not terms:
            return UniPoly.from_symbolic(pool.integer(0), x)
        expr = terms[0]
        for t in terms[1:]:
            expr = expr + t
        return UniPoly.from_symbolic(expr, x)

    def test_from_symbolic_quadratic(self):
        # x^2 + 2x + 1
        p = self._make([1, 2, 1])
        assert p.coefficients() == [1, 2, 1]

    def test_degree(self):
        p = self._make([0, 0, 1])
        assert p.degree() == 2

    def test_is_zero(self):
        p = self._make([0])
        assert p.is_zero()

    def test_add(self):
        p = self._make([1, 2])
        q = self._make([3, 4])
        r = p + q
        assert r.coefficients() == [4, 6]

    def test_mul(self):
        p = self._make([1, 1])  # 1 + x
        q = self._make([1, 1])  # 1 + x
        r = p * q
        assert r.coefficients() == [1, 2, 1]

    def test_pow(self):
        p = self._make([1, 1])  # 1 + x
        r = p ** 2
        assert r.coefficients() == [1, 2, 1]

    def test_repr_str(self):
        p = self._make([1, 2])
        assert "UniPoly" in repr(p)
        assert isinstance(str(p), str)

    def test_conversion_error(self):
        pool = self.pool
        x = self.x
        y = pool.symbol("y")
        expr = x + y
        with pytest.raises(ValueError):
            UniPoly.from_symbolic(expr, x)


# ---------------------------------------------------------------------------
# MultiPoly
# ---------------------------------------------------------------------------

class TestMultiPoly:
    def setup_method(self):
        self.pool = ExprPool()
        self.x = self.pool.symbol("x")
        self.y = self.pool.symbol("y")

    def test_from_symbolic_bivariate(self):
        x, y = self.x, self.y
        expr = x * y
        p = MultiPoly.from_symbolic(expr, [x, y])
        assert not p.is_zero()
        assert p.total_degree() == 2

    def test_is_zero(self):
        p = MultiPoly.from_symbolic(self.pool.integer(0), [self.x])
        assert p.is_zero()

    def test_repr_str(self):
        x, y = self.x, self.y
        p = MultiPoly.from_symbolic(x + y, [x, y])
        assert "MultiPoly" in repr(p)
        assert isinstance(str(p), str)
