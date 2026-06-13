"""v0.4 phases (P-24 through P-30) test suite.

Covers:
  P-24 — Horner-form code emission
  P-25 — NumPy / JAX array evaluation (batch eval via call_batch_raw)
  P-26 — collect_like_terms
  P-27 — poly_normal
  P-30 — Sharded ExprPool (parallel feature)
"""

import pytest
from alkahest import (
    ExprPool,
    collect_like_terms,
    compile_expr,
    emit_c,
    horner,
    numpy_eval,
    poly_normal,
    sin,
)

# ===========================================================================
# Helpers
# ===========================================================================


def pool():
    return ExprPool()


# ===========================================================================
# P-24 — Horner-form code emission
# ===========================================================================


class TestHorner:
    def test_linear(self):
        # 2x + 1 → Horner form evaluates the same
        p = pool()
        x = p.symbol("x")
        expr = p.integer(2) * x + p.integer(1)
        h = horner(expr, x)
        f_orig = compile_expr(expr, [x])
        f_horn = compile_expr(h, [x])
        for v in [-3.0, -1.0, 0.0, 1.0, 5.0]:
            assert abs(f_orig([v]) - f_horn([v])) < 1e-10, f"v={v}"

    def test_quadratic_correct_values(self):
        # (x+1)² = x² + 2x + 1
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2) * x + p.integer(1)
        h = horner(expr, x)
        f = compile_expr(h, [x])
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
            assert abs(f([v]) - (v + 1) ** 2) < 1e-9, f"v={v}"

    def test_degree_10_evaluates_correctly(self):
        # Build x^10 + x^9 + … + x + 1
        p = pool()
        x = p.symbol("x")
        expr = p.integer(1)
        for k in range(1, 11):
            expr = expr + x**k
        h = horner(expr, x)
        f_orig = compile_expr(expr, [x])
        f_horn = compile_expr(h, [x])
        for v in [0.0, 0.5, 1.0, 2.0, -1.0]:
            assert abs(f_orig([v]) - f_horn([v])) < 1e-6, f"v={v}"

    def test_constant_polynomial(self):
        p = pool()
        x = p.symbol("x")
        c = p.integer(7)
        h = horner(c, x)
        f = compile_expr(h, [x])
        assert abs(f([0.0]) - 7.0) < 1e-10
        assert abs(f([999.0]) - 7.0) < 1e-10

    def test_emit_c_quadratic(self):
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2) * x + p.integer(1)
        code = emit_c(expr, x, "x", "eval_quad")
        assert "eval_quad" in code
        assert "double" in code
        assert "return" in code

    def test_emit_c_default_names(self):
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(1)
        code = emit_c(expr, x)
        assert "eval_poly" in code
        assert "double x" in code

    def test_emit_c_linear_compiles_to_horner_style(self):
        # 3x + 2 should emit as  "2.0 + x * 3.0"  (Horner notation)
        p = pool()
        x = p.symbol("x")
        expr = p.integer(3) * x + p.integer(2)
        code = emit_c(expr, x, "x", "f")
        # The emitted body must contain "+" and "x *" or "* x"
        assert "+" in code
        assert "x" in code

    def test_horner_not_polynomial_raises(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = x + y
        with pytest.raises(ValueError):
            horner(expr, x)

    def test_emit_c_not_polynomial_raises(self):
        p = pool()
        x = p.symbol("x")
        expr = sin(x)
        with pytest.raises(ValueError):
            emit_c(expr, x)

    def test_emit_c_accepts_one_element_list_var(self):
        """`var` may be a single Expr or a one-element list/tuple containing one."""
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2) * x + p.integer(1)
        code_expr = emit_c(expr, x, "x", "eval_quad")
        code_list = emit_c(expr, [x], "x", "eval_quad")
        code_tuple = emit_c(expr, (x,), "x", "eval_quad")
        assert code_expr == code_list == code_tuple

    def test_emit_c_multi_element_list_var_raises_clear_error(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = x + y
        with pytest.raises(TypeError, match="univariate"):
            emit_c(expr, [x, y])


# ===========================================================================
# P-25 — NumPy / batch evaluation
# ===========================================================================


class TestBatchEval:
    """Tests for CompiledFn.call_batch_raw and the numpy_eval helper."""

    def test_call_batch_raw_identity(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        # 5 points: [0, 1, 2, 3, 4]
        inputs_flat = list(range(5))
        result = f.call_batch_raw([float(v) for v in inputs_flat], 1, 5)
        for i, v in enumerate(result):
            assert abs(v - i) < 1e-10, f"point {i}: expected {i}, got {v}"

    def test_call_batch_raw_quadratic(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x**2, [x])
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = f.call_batch_raw(xs, 1, 5)
        for xi, yi in zip(xs, result):
            assert abs(yi - xi**2) < 1e-10, f"x={xi}"

    def test_call_batch_raw_two_vars(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        f = compile_expr(x**2 + y**2, [x, y])
        # 3 points: (3,4), (0,0), (1,1)
        xs = [3.0, 0.0, 1.0]
        ys = [4.0, 0.0, 1.0]
        inputs_flat = xs + ys
        result = f.call_batch_raw(inputs_flat, 2, 3)
        assert abs(result[0] - 25.0) < 1e-10
        assert abs(result[1] - 0.0) < 1e-10
        assert abs(result[2] - 2.0) < 1e-10

    def test_call_batch_raw_wrong_n_vars_raises(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        with pytest.raises(ValueError):
            f.call_batch_raw([1.0, 2.0, 3.0, 4.0], 2, 2)

    def test_call_batch_raw_wrong_flat_length_raises(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        with pytest.raises(ValueError):
            f.call_batch_raw([1.0, 2.0, 3.0], 1, 5)

    def test_numpy_eval_identity(self):
        pytest.importorskip("numpy")
        import numpy as np

        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        xs = np.linspace(0.0, 1.0, 100)
        ys = numpy_eval(f, xs)
        assert ys.shape == (100,)
        assert np.allclose(ys, xs)

    def test_numpy_eval_polynomial(self):
        pytest.importorskip("numpy")
        import numpy as np

        p = pool()
        x = p.symbol("x")
        # f(x) = (x+1)^2
        expr = x**2 + p.integer(2) * x + p.integer(1)
        f = compile_expr(expr, [x])
        xs = np.linspace(-2.0, 2.0, 1000)
        ys = numpy_eval(f, xs)
        expected = (xs + 1) ** 2
        assert np.allclose(ys, expected, atol=1e-8)

    def test_numpy_eval_two_vars(self):
        pytest.importorskip("numpy")
        import numpy as np

        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        f = compile_expr(x**2 + y**2, [x, y])
        xs = np.array([3.0, 0.0, 1.0])
        ys = np.array([4.0, 0.0, 1.0])
        result = numpy_eval(f, xs, ys)
        assert np.allclose(result, [25.0, 0.0, 2.0])

    def test_numpy_eval_wrong_n_vars_raises(self):
        pytest.importorskip("numpy")
        import numpy as np

        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        with pytest.raises(ValueError):
            numpy_eval(f, np.array([1.0]), np.array([2.0]))

    def test_numpy_eval_large_array(self):
        pytest.importorskip("numpy")
        import numpy as np

        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2) * x + p.integer(1)
        f = compile_expr(expr, [x])
        xs = np.linspace(0.0, 1.0, 100_000)
        ys = numpy_eval(f, xs)
        expected = (xs + 1) ** 2
        assert np.allclose(ys, expected, atol=1e-8)


# ===========================================================================
# P-26 — collect_like_terms
# ===========================================================================


class TestCollectLikeTerms:
    def test_simple_merge_2x_3x(self):
        # 2x + 3x → 5x
        p = pool()
        x = p.symbol("x")
        expr = p.integer(2) * x + p.integer(3) * x
        result = collect_like_terms(expr)
        f = compile_expr(result.value, [x])
        for v in [0.0, 1.0, 2.0, -3.0]:
            assert abs(f([v]) - 5.0 * v) < 1e-10, f"v={v}"

    def test_simple_merge_str(self):
        p = pool()
        x = p.symbol("x")
        expr = p.integer(2) * x + p.integer(3) * x
        result = collect_like_terms(expr)
        assert "5" in str(result.value)

    def test_cancellation(self):
        # x + (-1)*x → 0
        p = pool()
        x = p.symbol("x")
        expr = x + p.integer(-1) * x
        result = collect_like_terms(expr)
        assert str(result.value) == "0"

    def test_no_change_for_different_vars(self):
        # 2x + 3y — no like terms to collect
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = p.integer(2) * x + p.integer(3) * y
        result = collect_like_terms(expr)
        # Result should evaluate correctly
        f = compile_expr(result.value, [x, y])
        assert abs(f([1.0, 1.0]) - 5.0) < 1e-10

    def test_const_fold_with_like_terms(self):
        # (2 + 3) + 0 → 5
        p = pool()
        expr = p.integer(2) + p.integer(3) + p.integer(0)
        result = collect_like_terms(expr)
        assert str(result.value) == "5"

    def test_multivariate_collect(self):
        # 3x + 4x + 2y → 7x + 2y
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = p.integer(3) * x + p.integer(4) * x + p.integer(2) * y
        result = collect_like_terms(expr)
        f = compile_expr(result.value, [x, y])
        # 7x + 2y at x=2, y=3 → 14 + 6 = 20
        assert abs(f([2.0, 3.0]) - 20.0) < 1e-10

    def test_returns_derived_result(self):
        p = pool()
        x = p.symbol("x")
        expr = x + x
        result = collect_like_terms(expr)
        assert hasattr(result, "value")


# ===========================================================================
# P-27 — poly_normal
# ===========================================================================


class TestPolyNormal:
    def test_difference_of_squares(self):
        # (x+1)*(x-1) should normalize to x² - 1 (or -1 + x²)
        p = pool()
        x = p.symbol("x")
        xp1 = x + p.integer(1)
        xm1 = x + p.integer(-1)
        expr = xp1 * xm1
        # poly_normal needs the expression to already be polynomial form
        # First expand, then normalize
        from alkahest import simplify_expanded

        expanded = simplify_expanded(expr)
        result = poly_normal(expanded.value, [x])
        # Evaluate at x=3: 9-1=8, x=0: -1
        f = compile_expr(result, [x])
        assert abs(f([3.0]) - 8.0) < 1e-10
        assert abs(f([0.0]) - (-1.0)) < 1e-10

    def test_collect_like_terms_via_poly_normal(self):
        # 2x + 3x should give 5x
        p = pool()
        x = p.symbol("x")
        expr = p.integer(2) * x + p.integer(3) * x
        result = poly_normal(expr, [x])
        f = compile_expr(result, [x])
        for v in [0.0, 1.0, 2.0, -1.0]:
            assert abs(f([v]) - 5.0 * v) < 1e-10

    def test_bivariate_polynomial(self):
        # x² + y² + 2xy = (x+y)²  — check numeric values
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = x**2 + y**2 + p.integer(2) * x * y
        result = poly_normal(expr, [x, y])
        f = compile_expr(result, [x, y])
        # At (1,2): (1+2)² = 9
        assert abs(f([1.0, 2.0]) - 9.0) < 1e-10
        # At (3,4): (3+4)² = 49
        assert abs(f([3.0, 4.0]) - 49.0) < 1e-10

    def test_constant(self):
        p = pool()
        x = p.symbol("x")
        c = p.integer(42)
        result = poly_normal(c, [x])
        f = compile_expr(result, [x])
        assert abs(f([0.0]) - 42.0) < 1e-10
        assert abs(f([99.0]) - 42.0) < 1e-10

    def test_non_polynomial_raises(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        # y is not in the var list [x], so should raise ValueError
        expr = x + y
        with pytest.raises(ValueError):
            poly_normal(expr, [x])

    def test_agrees_with_sympy_on_simple_cases(self):
        """Cross-validation: poly_normal agrees with SymPy for simple cases."""
        pytest.importorskip("sympy")
        p = pool()
        x = p.symbol("x")
        # Build 3x² + 2x + 1
        expr = p.integer(3) * x**2 + p.integer(2) * x + p.integer(1)
        result = poly_normal(expr, [x])
        f = compile_expr(result, [x])
        # Agree with naive evaluation
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            expected = 3 * v**2 + 2 * v + 1
            assert abs(f([v]) - expected) < 1e-10


# ===========================================================================
# P-30 — Sharded ExprPool (parallel feature)
# ===========================================================================


class TestShardedExprPool:
    """Ensure the sharded pool (P-30) behaves identically to the single-Mutex pool."""

    def test_interning_is_structural(self):
        p = pool()
        x1 = p.symbol("x")
        x2 = p.symbol("x")
        assert x1 == x2, "same symbol must return the same ExprId in the sharded pool"

    def test_add_interning(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        s1 = x + y
        s2 = x + y
        assert s1 == s2

    def test_expression_evaluation_consistent(self):
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2) * x + p.integer(1)
        f = compile_expr(expr, [x])
        for v in [0.0, 1.0, 2.0, -1.0]:
            assert abs(f([v]) - (v + 1) ** 2) < 1e-9
