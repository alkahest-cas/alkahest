"""Phase 21–23 Python tests.

Covers:
  Phase 21 — JIT / interpreter-based compiled evaluation
  Phase 22 — Ball arithmetic (ArbBall, interval_eval)
  Phase 23 — Parallel simplification (simplify_par)
"""

import math

import pytest

from _step_logs import assert_same_step_rules

from alkahest import (
    ArbBall,
    ExprPool,
    compile_expr,
    cos,
    eval_expr,
    exp,
    interval_eval,
    simplify,
    simplify_par,
    sin,
    sqrt,
)

# ===========================================================================
# Helpers
# ===========================================================================

def pool():
    return ExprPool()


# ===========================================================================
# Phase 21 — JIT / compiled evaluation
# ===========================================================================

class TestCompiledFn:
    def test_constant_integer(self):
        p = pool()
        five = p.integer(5)
        f = compile_expr(five, [])
        assert abs(f([]) - 5.0) < 1e-10

    def test_constant_rational(self):
        p = pool()
        half = p.rational(1, 2)
        f = compile_expr(half, [])
        assert abs(f([]) - 0.5) < 1e-10

    def test_identity(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        assert abs(f([3.14]) - 3.14) < 1e-10

    def test_add(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        f = compile_expr(x + y, [x, y])
        assert abs(f([2.0, 3.0]) - 5.0) < 1e-10

    def test_polynomial(self):
        # f(x) = x² + 2x + 1 = (x+1)²
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(2)*x + p.integer(1)
        f = compile_expr(expr, [x])
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]:
            assert abs(f([v]) - (v+1)**2) < 1e-9, f"v={v}"

    def test_sin(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(sin(x), [x])
        assert abs(f([math.pi / 2]) - 1.0) < 1e-10

    def test_cos(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(cos(x), [x])
        assert abs(f([0.0]) - 1.0) < 1e-10

    def test_exp(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(exp(x), [x])
        assert abs(f([0.0]) - 1.0) < 1e-10
        assert abs(f([1.0]) - math.e) < 1e-10

    def test_sqrt(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(sqrt(x), [x])
        assert abs(f([4.0]) - 2.0) < 1e-10
        assert abs(f([9.0]) - 3.0) < 1e-10

    def test_pow_integer(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x ** 3, [x])
        assert abs(f([2.0]) - 8.0) < 1e-10

    def test_pythagorean_triple(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        f = compile_expr(x**2 + y**2, [x, y])
        assert abs(f([3.0, 4.0]) - 25.0) < 1e-10
        assert abs(f([5.0, 12.0]) - 169.0) < 1e-10

    def test_wrong_n_inputs(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        with pytest.raises(ValueError, match="expected 1"):
            f([])

    def test_n_inputs_property(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        f = compile_expr(x + y, [x, y])
        assert f.n_inputs == 2

    def test_repr(self):
        p = pool()
        x = p.symbol("x")
        f = compile_expr(x, [x])
        assert "CompiledFn" in repr(f)
        assert "1" in repr(f)


class TestEvalExpr:
    def test_constant(self):
        p = pool()
        three = p.integer(3)
        assert abs(eval_expr(three, {}) - 3.0) < 1e-10

    def test_symbol(self):
        p = pool()
        x = p.symbol("x")
        assert abs(eval_expr(x, {x: 7.0}) - 7.0) < 1e-10

    def test_add(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        assert abs(eval_expr(x + y, {x: 10.0, y: 5.0}) - 15.0) < 1e-10

    def test_unbound_raises(self):
        p = pool()
        x = p.symbol("x")
        with pytest.raises(ValueError):
            eval_expr(x, {})

    def test_sin_cos_identity(self):
        p = pool()
        x = p.symbol("x")
        pythagorean = sin(x)**2 + cos(x)**2
        for angle in [0.0, 0.5, 1.0, math.pi / 4, math.pi]:
            v = eval_expr(pythagorean, {x: angle})
            assert abs(v - 1.0) < 1e-10, f"angle={angle}"

    def test_float_constant(self):
        p = pool()
        three_f = p.float(3.0)
        assert abs(eval_expr(three_f, {}) - 3.0) < 1e-10


# ===========================================================================
# Phase 22 — Ball arithmetic
# ===========================================================================

class TestArbBall:
    def test_construction(self):
        b = ArbBall(3.0, 0.5)
        assert abs(b.mid - 3.0) < 1e-10
        assert abs(b.rad - 0.5) < 1e-10

    def test_default_exact(self):
        b = ArbBall(2.0)
        assert b.is_exact()

    def test_contains_midpoint(self):
        b = ArbBall(3.0, 0.5)
        assert b.contains(3.0)

    def test_contains_endpoints(self):
        b = ArbBall(3.0, 0.5)
        assert b.contains(2.5)
        assert b.contains(3.5)

    def test_not_contains_outside(self):
        b = ArbBall(3.0, 0.5)
        assert not b.contains(4.0)
        assert not b.contains(2.0)

    def test_lo_hi(self):
        b = ArbBall(3.0, 0.5)
        assert abs(b.lo - 2.5) < 1e-10
        assert abs(b.hi - 3.5) < 1e-10

    def test_add_enclosure(self):
        a = ArbBall(1.0, 0.1)
        b = ArbBall(2.0, 0.2)
        c = a + b
        # True result: [2.7, 3.3]
        assert c.contains(2.7)
        assert c.contains(3.0)
        assert c.contains(3.3)

    def test_sub_enclosure(self):
        a = ArbBall(3.0, 0.1)  # [2.9, 3.1]
        b = ArbBall(1.0, 0.1)  # [0.9, 1.1]
        c = a - b              # [1.8, 2.2]
        assert c.contains(1.82)   # safely inside lower bound
        assert c.contains(2.0)
        assert c.contains(2.18)   # safely inside upper bound

    def test_mul_enclosure(self):
        a = ArbBall(2.0, 0.5)  # [1.5, 2.5]
        b = ArbBall(3.0, 0.5)  # [2.5, 3.5]
        c = a * b              # [3.75, 8.75]
        assert c.contains(4.0)
        assert c.contains(6.0)
        assert c.contains(8.0)

    def test_div_enclosure(self):
        a = ArbBall(6.0, 0.0)  # exact 6
        b = ArbBall(2.0, 0.0)  # exact 2
        c = a / b
        assert c is not None
        assert c.contains(3.0)

    def test_div_by_zero_containing_ball_raises(self):
        a = ArbBall(1.0, 0.0)
        b = ArbBall(0.0, 0.5)  # contains 0
        with pytest.raises(ZeroDivisionError):
            a / b

    def test_neg(self):
        b = ArbBall(3.0, 0.5)
        nb = -b
        assert nb.contains(-3.0)
        assert abs(nb.mid - (-3.0)) < 1e-10
        assert abs(nb.rad - 0.5) < 1e-10

    def test_sin_enclosure(self):
        b = ArbBall(math.pi / 2, 0.01)
        s = b.sin()
        assert s.contains(1.0)

    def test_cos_enclosure(self):
        b = ArbBall(0.0, 0.01)
        c = b.cos()
        assert c.contains(1.0)

    def test_exp_enclosure(self):
        b = ArbBall(0.0, 0.1)  # [-0.1, 0.1]
        e = b.exp()
        assert e.contains(1.0)
        # e^0.1 ≈ 1.10517; use a value safely inside the upper bound
        assert e.contains(1.10)

    def test_log_enclosure(self):
        b = ArbBall(2.0, 0.5)  # [1.5, 2.5]
        lv = b.log()
        assert lv is not None
        assert lv.contains(math.log(2.0))

    def test_log_undefined_raises(self):
        b = ArbBall(0.0, 0.5)  # contains non-positive
        with pytest.raises(ValueError):
            b.log()

    def test_sqrt_enclosure(self):
        b = ArbBall(4.0, 0.5)  # [3.5, 4.5]
        s = b.sqrt()
        assert s is not None
        assert s.contains(2.0)

    def test_sqrt_negative_raises(self):
        b = ArbBall(0.0, 0.5)  # contains negative
        with pytest.raises(ValueError):
            b.sqrt()

    def test_repr(self):
        b = ArbBall(3.14, 0.001)
        r = repr(b)
        assert "ArbBall" in r

    def test_higher_prec_smaller_rad(self):
        # Higher precision should give smaller rounding radius for the same operation
        a32 = ArbBall(1.0, 1e-3, 32)
        a64 = ArbBall(1.0, 1e-3, 64)
        r32 = (a32 + ArbBall(2.0, 1e-3, 32)).rad
        r64 = (a64 + ArbBall(2.0, 1e-3, 64)).rad
        # Both should be within the input radius bounds; 64-bit rad ≤ 32-bit rad
        assert r64 <= r32 + 1e-10


class TestIntervalEval:
    def test_constant(self):
        p = pool()
        five = p.integer(5)
        r = interval_eval(five, {})
        assert r.contains(5.0)

    def test_symbol(self):
        p = pool()
        x = p.symbol("x")
        x_ball = ArbBall(3.0, 0.1)
        r = interval_eval(x, {x: x_ball})
        assert r.contains(3.0)

    def test_polynomial_enclosure(self):
        # x² + 1 at x ∈ [2.9, 3.1] → [9.41, 10.61]
        p = pool()
        x = p.symbol("x")
        expr = x**2 + p.integer(1)
        r = interval_eval(expr, {x: ArbBall(3.0, 0.1)})
        assert r.contains(9.5)
        assert r.contains(10.0)
        assert r.contains(10.5)
        assert not r.contains(9.0)   # 9.0 < 9.41

    def test_multivariate(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        expr = x + y
        r = interval_eval(expr, {x: ArbBall(1.0, 0.1), y: ArbBall(2.0, 0.1)})
        assert r.contains(3.0)

    def test_sin_pythagorean(self):
        # sin²(x) + cos²(x) = 1 for x ∈ [0.5, 0.6]
        p = pool()
        x = p.symbol("x")
        expr = sin(x)**2 + cos(x)**2
        r = interval_eval(expr, {x: ArbBall(0.55, 0.05)})
        assert r.contains(1.0)

    def test_unbound_raises(self):
        p = pool()
        x = p.symbol("x")
        with pytest.raises((ValueError, Exception)):
            interval_eval(x, {})

    def test_rational(self):
        p = pool()
        third = p.rational(1, 3)
        r = interval_eval(third, {})
        # mid should be close to 1/3
        assert abs(r.mid - 1.0/3.0) < 1e-10
        assert r.rad < 1e-30


# ===========================================================================
# Phase 23 — Parallel simplification
# ===========================================================================

class TestSimplifyPar:
    def test_simplify_par_matches_simplify_add_zero(self):
        p = pool()
        x = p.symbol("x")
        expr = x + p.integer(0)
        par_result = simplify_par(expr)
        seq_result = simplify(expr)
        assert str(par_result.value) == str(seq_result.value)
        assert_same_step_rules(
            seq_result.steps, par_result.steps, context="add zero"
        )

    def test_simplify_par_matches_simplify_mul_one(self):
        p = pool()
        x = p.symbol("x")
        expr = x * p.integer(1)
        par_result = simplify_par(expr)
        seq_result = simplify(expr)
        assert str(par_result.value) == str(seq_result.value)
        assert_same_step_rules(
            seq_result.steps, par_result.steps, context="mul one"
        )

    def test_simplify_par_constant_fold(self):
        p = pool()
        args = [p.integer(i) for i in range(1, 21)]
        expr = args[0]
        for a in args[1:]:
            expr = expr + a
        par_result = simplify_par(expr)
        seq_result = simplify(expr)
        assert str(par_result.value) == str(seq_result.value)
        assert_same_step_rules(
            seq_result.steps, par_result.steps, context="constant fold sum"
        )

    def test_simplify_par_large_mul_ones(self):
        p = pool()
        x = p.symbol("x")
        ones = [p.integer(1)] * 15
        factors = [x] + ones
        expr = factors[0]
        for f in factors[1:]:
            expr = expr * f
        par_result = simplify_par(expr)
        seq_result = simplify(expr)
        assert str(par_result.value) == str(seq_result.value)
        assert_same_step_rules(
            seq_result.steps, par_result.steps, context="large mul ones"
        )

    def test_simplify_par_mul_zero(self):
        p = pool()
        x = p.symbol("x")
        expr = p.integer(0) * x
        par_result = simplify_par(expr)
        seq_result = simplify(expr)
        assert str(par_result.value) == str(p.integer(0))
        assert_same_step_rules(
            seq_result.steps, par_result.steps, context="mul zero"
        )

    def test_simplify_par_returns_derived_result(self):
        p = pool()
        x = p.symbol("x")
        expr = x + p.integer(0)
        result = simplify_par(expr)
        # Should be a DerivedResult-like object with a .value attribute
        assert hasattr(result, "value")

    def test_simplify_par_polynomial(self):
        p = pool()
        x = p.symbol("x")
        # (2 + 3) * x + (4 + 0) → 5*x + 4
        expr = (p.integer(2) + p.integer(3)) * x + (p.integer(4) + p.integer(0))
        par_result = simplify_par(expr)
        seq_result = simplify(expr)
        assert str(par_result.value) == str(seq_result.value)
        assert_same_step_rules(
            seq_result.steps, par_result.steps, context="polynomial fold"
        )

    def test_simplify_par_nested_cancellation(self):
        p = pool()
        x = p.symbol("x")
        y = p.symbol("y")
        # x + y + (-1)*x → y
        neg_x = p.integer(-1) * x
        expr = x + y + neg_x
        par_result = simplify_par(expr)
        seq_result = simplify(expr)
        assert str(par_result.value) == str(seq_result.value)
        assert_same_step_rules(
            seq_result.steps, par_result.steps, context="nested cancel"
        )
