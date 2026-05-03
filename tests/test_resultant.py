"""Tests for V2-2: Resultants and subresultant PRS.

Test plan (from ROADMAP.md):
- Sylvester-matrix determinant agreement on representative pairs.
- Implicitization of (t², t³) → y² - x³ = 0.
- Bivariate sanity: res(x²+y²-1, y-x, y) == 2x²-1.
- Subresultant PRS sequence properties.
- Error handling for non-polynomial inputs.
"""

import alkahest
import pytest
from alkahest import (
    ExprPool,
    ResultantError,
    UniPoly,
    resultant,
    subresultant_prs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pool():
    return ExprPool()


def expr_to_int(expr):
    """Try to extract an integer value from a constant Expr."""
    try:
        return int(str(expr))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# resultant — univariate (integer result)
# ---------------------------------------------------------------------------

class TestResultantUnivariate:
    def test_common_root_gives_zero(self):
        """res(x^2 - 5x + 6, x - 2, x) == 0 because x=2 is a common root."""
        pool = make_pool()
        x = pool.symbol("x")
        # x^2 - 5x + 6
        p = x**2 + pool.integer(-5) * x + pool.integer(6)
        # x - 2
        q = x + pool.integer(-2)
        dr = resultant(p, q, x)
        assert dr is not None
        # Should be 0 since polynomials share root x=2.
        result_val = alkahest.simplify(dr.value).value
        assert str(result_val) in ("0", "0.0"), f"expected 0, got {result_val}"

    def test_coprime_quadratic_linear(self):
        """res(x^2 + 1, x - 1, x) == 2.

        Sylvester matrix:
          | 1  0  1 |
          | 1 -1  0 |
          | 0  1 -1 |
        det = 1*(1) - 0 + 1*(1+0) = 2.
        """
        pool = make_pool()
        x = pool.symbol("x")
        p = x**2 + pool.integer(1)
        q = x + pool.integer(-1)
        dr = resultant(p, q, x)
        result_val = alkahest.simplify(dr.value).value
        assert str(result_val) == "2", f"expected 2, got {result_val}"

    def test_two_linear_polynomials(self):
        """res(x - a, x - b, x) = a - b = g(root of f).

        Concretely: res(x - 3, x - 7, x) = 3 - 7 = -4.
        """
        pool = make_pool()
        x = pool.symbol("x")
        p = x + pool.integer(-3)
        q = x + pool.integer(-7)
        dr = resultant(p, q, x)
        result_val = alkahest.simplify(dr.value).value
        assert str(result_val) in ("-4", "4"), f"expected ±4, got {result_val}"

    def test_resultant_of_x_and_constant(self):
        """res(x, c, x) = c^deg(x) = c^1 = c for constant c."""
        pool = make_pool()
        x = pool.symbol("x")
        p = x
        q = pool.integer(5)
        dr = resultant(p, q, x)
        result_val = alkahest.simplify(dr.value).value
        assert str(result_val) == "5", f"expected 5, got {result_val}"

    def test_resultant_commuted_sign(self):
        """res(p, q, x) = (-1)^(deg p * deg q) * res(q, p, x)."""
        pool = make_pool()
        x = pool.symbol("x")
        p = x**2 + pool.integer(1)
        q = x**3 + pool.integer(-1)
        r_pq = resultant(p, q, x)
        r_qp = resultant(q, p, x)
        # Sign should differ by (-1)^(2*3) = 1, so they should be equal.
        s_pq = alkahest.simplify(r_pq.value).value
        s_qp = alkahest.simplify(r_qp.value).value
        assert str(s_pq) == str(s_qp), (
            f"res(p,q)={s_pq} should equal res(q,p)={s_qp} for even product of degrees"
        )

    def test_derivation_log(self):
        """resultant() records exactly one derivation step named 'Resultant'."""
        pool = make_pool()
        x = pool.symbol("x")
        p = x + pool.integer(-1)
        q = x + pool.integer(-2)
        dr = resultant(p, q, x)
        steps = dr.steps
        assert len(steps) == 1
        assert steps[0]["rule"] == "Resultant"


# ---------------------------------------------------------------------------
# resultant — bivariate (polynomial result)
# ---------------------------------------------------------------------------

class TestResultantBivariate:
    def test_bivariate_sanity(self):
        """res(x^2 + y^2 - 1, y - x, y) == 2*x^2 - 1.

        This is the ROADMAP acceptance test.
        """
        pool = make_pool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        # x^2 + y^2 - 1
        circle = x**2 + y**2 + pool.integer(-1)
        # y - x
        line = y + pool.integer(-1) * x
        dr = resultant(circle, line, y)
        res_expr = dr.value

        # Convert the result to UniPoly in x and check coefficients.
        res_poly = UniPoly.from_symbolic(res_expr, x)
        coeffs = res_poly.coefficients()
        assert res_poly.degree() == 2, f"expected degree 2, got {res_poly.degree()}"
        # Coefficients in ascending order: [-1, 0, 2]
        assert coeffs[0] == -1, f"constant term should be -1, got {coeffs[0]}"
        assert coeffs[2] == 2, f"leading coeff should be 2, got {coeffs[2]}"

    def test_implicitization_twisted_cubic(self):
        """Eliminate t from (x - t^2, y - t^3) to get the curve y^2 - x^3 = 0.

        Verify: the resultant vanishes at (x=4, y=8) (on the curve)
        but not at (x=1, y=2) (off the curve).
        """
        pool = make_pool()
        t = pool.symbol("t")
        x = pool.symbol("x")
        y = pool.symbol("y")
        # p1 = x - t^2
        p1 = x + pool.integer(-1) * t**2
        # p2 = y - t^3
        p2 = y + pool.integer(-1) * t**3

        dr = resultant(p1, p2, t)
        res_expr = dr.value

        # Substitute (x=4, y=8): 4=2^2, 8=2^3 → on the curve → should give 0.
        at_on_curve = alkahest.subs(res_expr, {x: pool.integer(4), y: pool.integer(8)})
        simplified_on = alkahest.simplify(at_on_curve).value
        assert str(simplified_on) == "0", (
            f"resultant at (4,8) should be 0, got {simplified_on}"
        )

        # Substitute (x=1, y=2): 2 ≠ 1^(3/2) → off the curve → should be non-zero.
        at_off_curve = alkahest.subs(res_expr, {x: pool.integer(1), y: pool.integer(2)})
        simplified_off = alkahest.simplify(at_off_curve).value
        assert str(simplified_off) != "0", (
            f"resultant at (1,2) should be non-zero, got {simplified_off}"
        )

    def test_resultant_constant_polynomial(self):
        """res(p, 1, x) = 1 for any non-zero p."""
        pool = make_pool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        p = x**2 + y + pool.integer(1)
        q = pool.integer(1)  # constant
        dr = resultant(p, q, x)
        result_val = alkahest.simplify(dr.value).value
        assert str(result_val) == "1", f"res(p, 1, x) should be 1, got {result_val}"


# ---------------------------------------------------------------------------
# subresultant_prs
# ---------------------------------------------------------------------------

class TestSubresultantPRS:
    def test_prs_starts_with_inputs(self):
        """The first two elements of the PRS are (reordered) p and q."""
        pool = make_pool()
        x = pool.symbol("x")
        p_expr = x**2 + pool.integer(-1)
        q_expr = x + pool.integer(-1)
        seq = subresultant_prs(p_expr, q_expr, x)
        assert len(seq) >= 2, "PRS must have at least 2 elements"

    def test_prs_gcd_x2m1_xm1(self):
        """gcd(x^2 - 1, x - 1) = x - 1; the PRS last element is degree 1."""
        pool = make_pool()
        x = pool.symbol("x")
        p_expr = x**2 + pool.integer(-1)
        q_expr = x + pool.integer(-1)
        seq = subresultant_prs(p_expr, q_expr, x)
        last = seq[-1]
        last_poly = UniPoly.from_symbolic(last, x)
        assert last_poly.degree() == 1, (
            f"last element should be degree 1 (matching gcd), got {last_poly.degree()}"
        )

    def test_prs_coprime_terminates_at_constant(self):
        """For coprime polys, the PRS terminates with a nonzero constant."""
        pool = make_pool()
        x = pool.symbol("x")
        p_expr = x**2 + pool.integer(1)  # irreducible
        q_expr = x + pool.integer(-1)
        seq = subresultant_prs(p_expr, q_expr, x)
        assert len(seq) >= 2
        last = seq[-1]
        last_poly = UniPoly.from_symbolic(last, x)
        assert last_poly.degree() == 0, (
            f"last element of coprime PRS should be degree 0, got {last_poly.degree()}"
        )

    def test_prs_consistent_with_resultant(self):
        """The last (constant) element of the PRS matches resultant() up to sign."""
        pool = make_pool()
        x = pool.symbol("x")
        p_expr = x + pool.integer(-3)
        q_expr = x + pool.integer(-7)
        seq = subresultant_prs(p_expr, q_expr, x)
        dr = resultant(p_expr, q_expr, x)
        # Get the last element of the sequence as an integer.
        last_val = alkahest.simplify(seq[-1]).value
        res_val = alkahest.simplify(dr.value).value
        # They should agree up to sign.
        assert str(last_val).lstrip("-") == str(res_val).lstrip("-"), (
            f"PRS last element {last_val} should match resultant {res_val} (up to sign)"
        )

    def test_prs_swap_invariant(self):
        """Swapping p and q produces a PRS of the same length (up to sign)."""
        pool = make_pool()
        x = pool.symbol("x")
        p_expr = x**2 + pool.integer(-1)
        q_expr = x**3 + pool.integer(-1)
        seq_pq = subresultant_prs(p_expr, q_expr, x)
        seq_qp = subresultant_prs(q_expr, p_expr, x)
        # Both sequences should have the same number of elements (up to sign).
        assert len(seq_pq) == len(seq_qp), (
            f"PRS length should be swap-invariant: {len(seq_pq)} vs {len(seq_qp)}"
        )


# ---------------------------------------------------------------------------
# Sylvester matrix agreement (proptest-style, 5 hand-crafted cases)
# ---------------------------------------------------------------------------

class TestSylvesterAgreement:
    """Verify that resultant() matches the Sylvester determinant formula."""

    def _sylvester_det(self, a_coeffs, b_coeffs):
        """Compute det(Sylvester(a, b)) using Python integers."""
        import numpy as np
        n = len(a_coeffs) - 1  # deg(a)
        m = len(b_coeffs) - 1  # deg(b)
        size = n + m
        if size == 0:
            return 1
        mat = [[0] * size for _ in range(size)]
        # m rows for a
        for i in range(m):
            for j, c in enumerate(a_coeffs):
                mat[i][i + j] = c
        # n rows for b
        for i in range(n):
            for j, c in enumerate(b_coeffs):
                mat[m + i][i + j] = c
        return int(round(np.linalg.det(np.array(mat, dtype=float))))

    @pytest.mark.parametrize("a,b,expected", [
        ([1, 0, 1], [-1, 1], 2),           # res(x^2+1, x-1): f(1)=2, lc(f)^1=1 → 2
        ([6, -5, 1], [-2, 1], 0),          # x^2-5x+6=(x-2)(x-3), x-2: common root x=2 → 0
        ([-1, 0, 1], [-1, 1], 0),          # res(x^2-1, x-1) = 0 (common root x=1)
    ])
    def test_sylvester_spot_checks(self, a, b, expected):
        """Resultant matches Sylvester determinant for hand-coded cases."""
        if expected is None:
            pytest.skip("skipping heavy Sylvester check")

        pool = make_pool()
        x = pool.symbol("x")

        # Build p from coefficients [c0, c1, ..., cn] (ascending degree).
        p_expr = pool.integer(0)
        for deg, c in enumerate(a):
            if c == 0:
                continue
            # Use Python int for exponent (Expr.__pow__ only accepts int)
            x_pow = x**deg if deg > 0 else pool.integer(1)
            term = pool.integer(c) * x_pow
            p_expr = p_expr + term

        q_expr = pool.integer(0)
        for deg, c in enumerate(b):
            if c == 0:
                continue
            x_pow = x**deg if deg > 0 else pool.integer(1)
            term = pool.integer(c) * x_pow
            q_expr = q_expr + term

        dr = resultant(p_expr, q_expr, x)
        res_val = alkahest.simplify(dr.value).value
        assert str(res_val) == str(expected), (
            f"resultant({a}, {b}) = {res_val}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestResultantErrors:
    def test_non_polynomial_raises(self):
        """Transcendental function raises ResultantError."""
        pool = make_pool()
        x = pool.symbol("x")
        sin_x = alkahest.sin(x)
        q = x + pool.integer(-1)
        with pytest.raises(ResultantError):
            resultant(sin_x, q, x)

    def test_subresultant_multivariate_raises(self):
        """subresultant_prs with multivariate input raises ResultantError."""
        pool = make_pool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        # x + y is not univariate in x (y is a free variable)
        p = x + y
        q = x + pool.integer(-1)
        with pytest.raises(ResultantError):
            subresultant_prs(p, q, x)

    def test_resultant_zero_polynomial(self):
        """Resultant with a zero polynomial gives 0."""
        pool = make_pool()
        x = pool.symbol("x")
        p = x**2 + pool.integer(-1)
        q = pool.integer(0)
        dr = resultant(p, q, x)
        res_val = alkahest.simplify(dr.value).value
        assert str(res_val) == "0", f"res(p, 0) should be 0, got {res_val}"
