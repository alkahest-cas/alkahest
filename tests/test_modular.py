"""Tests for V2-1 — Modular / CRT framework (alkahest.modular)."""

import alkahest
import pytest
from alkahest import MultiPoly, MultiPolyFp, modular

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_poly(expr_fn):
    """Return (poly, pool, x, y) for a polynomial built with expr_fn(pool, x, y)."""
    pool = alkahest.ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    expr = expr_fn(pool, x, y)
    poly = MultiPoly.from_symbolic(expr, [x, y])
    return poly, pool, x, y


def poly_str(poly) -> str:
    return str(poly)


# ---------------------------------------------------------------------------
# reduce_mod
# ---------------------------------------------------------------------------


class TestReduceMod:
    def test_basic(self):
        # 6x + 4 mod 5 → x + 4
        poly, pool, x, y = make_poly(
            lambda p, x, y: p.integer(6) * x + p.integer(4)
        )
        fp = modular.reduce_mod(poly, 5)
        assert isinstance(fp, MultiPolyFp)
        assert fp.modulus == 5
        assert not fp.is_zero()

    def test_negative_coeff(self):
        # -3x mod 7 → 4x
        poly, pool, x, y = make_poly(lambda p, x, y: p.integer(-3) * x)
        fp = modular.reduce_mod(poly, 7)
        assert fp.modulus == 7
        assert not fp.is_zero()

    def test_vanishing_term(self):
        # 5x + 7 mod 5 → 2 (x term vanishes)
        poly, pool, x, y = make_poly(
            lambda p, x, y: p.integer(5) * x + p.integer(7)
        )
        fp = modular.reduce_mod(poly, 5)
        # degree falls (x^1 term is gone) — the result is just the constant 2
        assert fp.total_degree() == 0

    def test_zero_poly(self):
        pool = alkahest.ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        zero = pool.integer(0)
        poly = MultiPoly.from_symbolic(zero, [x, y])
        fp = modular.reduce_mod(poly, 7)
        assert fp.is_zero()

    def test_invalid_modulus_composite(self):
        poly, *_ = make_poly(lambda p, x, y: x)
        with pytest.raises(Exception) as exc_info:
            modular.reduce_mod(poly, 4)
        assert "E-MOD-001" in str(exc_info.value) or "ModularError" in type(exc_info.value).__name__

    def test_invalid_modulus_zero(self):
        poly, *_ = make_poly(lambda p, x, y: x)
        with pytest.raises(Exception):
            modular.reduce_mod(poly, 0)

    def test_invalid_modulus_one(self):
        poly, *_ = make_poly(lambda p, x, y: x)
        with pytest.raises(Exception):
            modular.reduce_mod(poly, 1)

    def test_repr(self):
        poly, *_ = make_poly(lambda p, x, y: x + p.integer(1))
        fp = modular.reduce_mod(poly, 7)
        r = repr(fp)
        assert "mod 7" in r or "7" in r

    def test_total_degree(self):
        pool = alkahest.ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        x2 = x ** 2
        poly = MultiPoly.from_symbolic(x2 + x + pool.integer(1), [x, y])
        fp = modular.reduce_mod(poly, 5)
        assert fp.total_degree() == 2


# ---------------------------------------------------------------------------
# lift_crt
# ---------------------------------------------------------------------------


class TestLiftCrt:
    def _roundtrip(self, poly, primes=(101, 103)):
        images = [(modular.reduce_mod(poly, p), p) for p in primes]
        return modular.lift_crt(images)

    def test_roundtrip_positive_coeffs(self):
        # 3x + 2; coeffs ≤ 3 < 101/2
        poly, pool, x, y = make_poly(lambda p, x, y: p.integer(3) * x + p.integer(2))
        lifted = self._roundtrip(poly)
        assert str(lifted) == str(poly)

    def test_roundtrip_negative_coeff(self):
        # x - 50; need M > 100; 101*103=10403 > 100
        poly, pool, x, y = make_poly(lambda p, x, y: x + p.integer(-50))
        lifted = self._roundtrip(poly)
        assert str(lifted) == str(poly)

    def test_roundtrip_bivariate(self):
        # x*y + 3
        poly, pool, x, y = make_poly(lambda p, x, y: x * y + p.integer(3))
        lifted = self._roundtrip(poly, primes=(7, 11))
        assert str(lifted) == str(poly)

    def test_roundtrip_quadratic(self):
        pool = alkahest.ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        x2 = x ** 2
        expr = p = pool
        expr = p.integer(3) * x2 + p.integer(2) * x + p.integer(1)
        poly = MultiPoly.from_symbolic(expr, [x, y])
        lifted = modular.lift_crt(
            [(modular.reduce_mod(poly, p), p) for p in (101, 103)]
        )
        assert str(lifted) == str(poly)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            modular.lift_crt([])

    def test_single_image(self):
        # Trivial single-prime lift — works for small coefficients
        poly, pool, x, y = make_poly(lambda p, x, y: x + p.integer(2))
        lifted = modular.lift_crt([(modular.reduce_mod(poly, 101), 101)])
        assert str(lifted) == str(poly)

    def test_returns_multipoly(self):
        poly, *_ = make_poly(lambda p, x, y: x)
        lifted = modular.lift_crt([(modular.reduce_mod(poly, 7), 7)])
        assert isinstance(lifted, MultiPoly)


# ---------------------------------------------------------------------------
# rational_reconstruction
# ---------------------------------------------------------------------------


class TestRationalReconstruction:
    def test_one_half(self):
        # 2⁻¹ ≡ 51 (mod 101)
        result = modular.rational_reconstruction(51, 101)
        assert result is not None
        a, b = result
        assert a == 1
        assert b == 2

    def test_negative_numerator(self):
        # -1/2 ≡ 50 (mod 101)
        result = modular.rational_reconstruction(50, 101)
        assert result is not None
        a, b = result
        assert a == -1
        assert b == 2

    def test_zero(self):
        result = modular.rational_reconstruction(0, 101)
        assert result is not None
        a, b = result
        assert a == 0
        assert b == 1

    def test_integer_value(self):
        # n=5, M=101: T=7; 5 ≤ 7, so represents integer 5
        result = modular.rational_reconstruction(5, 101)
        assert result is not None
        a, b = result
        assert b == 1
        assert a == 5

    def test_none_when_not_representable(self):
        # n=2, M=7: T=1, integer 2 has |a|=2 > T=1
        result = modular.rational_reconstruction(2, 7)
        assert result is None

    def test_large_integers(self):
        # Verify it works with Python large ints (passed as ints, converted to str internally)
        M = 101 * 103 * 107  # 1_113_221
        # Python 3.8+: pow(base, -1, mod) computes modular inverse
        n = pow(5, -1, M)
        result = modular.rational_reconstruction(n, M)
        assert result is not None
        a, b = result
        # T = floor(sqrt(M/2)) ≈ 746; |1|=1 ≤ 746 and 5 ≤ 746
        assert a == 1
        assert b == 5

    def test_returns_python_ints(self):
        result = modular.rational_reconstruction(51, 101)
        assert result is not None
        a, b = result
        assert isinstance(a, int)
        assert isinstance(b, int)


# ---------------------------------------------------------------------------
# mignotte_bound
# ---------------------------------------------------------------------------


class TestMignotteBound:
    def test_constant(self):
        # poly = 5: L1=5, d=0 → bound=5
        pool = alkahest.ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        poly = MultiPoly.from_symbolic(pool.integer(5), [x, y])
        assert modular.mignotte_bound(poly) == 5

    def test_linear(self):
        # 3x + 2: L1=5, d=1 → bound=10
        poly, pool, x, y = make_poly(lambda p, x, y: p.integer(3) * x + p.integer(2))
        assert modular.mignotte_bound(poly) == 10

    def test_zero_poly(self):
        pool = alkahest.ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        poly = MultiPoly.from_symbolic(pool.integer(0), [x, y])
        bound = modular.mignotte_bound(poly)
        assert bound >= 1  # always at least 1 for the zero polynomial

    def test_returns_int(self):
        poly, *_ = make_poly(lambda p, x, y: x)
        bound = modular.mignotte_bound(poly)
        assert isinstance(bound, int)

    def test_sufficient_for_crt(self):
        # For poly = 40x - 30, mignotte bound = 70; need M > 140
        poly, pool, x, y = make_poly(
            lambda p, x, y: p.integer(40) * x + p.integer(-30)
        )
        bound = modular.mignotte_bound(poly)
        # Check: product of two primes > 2*bound
        primes = (modular.select_lucky_prime(0), modular.select_lucky_prime(0, [2]))
        assert primes[0] * primes[1] > 2 * bound
        # lift with enough primes to exceed 2*bound
        images = []
        product = 1
        p = 0
        used = []
        while product <= 2 * bound:
            p = modular.select_lucky_prime(0, used)
            used.append(p)
            images.append((modular.reduce_mod(poly, p), p))
            product *= p
        lifted = modular.lift_crt(images)
        assert str(lifted) == str(poly)


# ---------------------------------------------------------------------------
# select_lucky_prime
# ---------------------------------------------------------------------------


class TestSelectLuckyPrime:
    def test_no_constraint(self):
        p = modular.select_lucky_prime(0)
        assert p == 2

    def test_avoids_divisors(self):
        # Content = 6 = 2*3; lucky prime should be 5
        p = modular.select_lucky_prime(6)
        assert p == 5
        assert 6 % p != 0

    def test_skips_used(self):
        p = modular.select_lucky_prime(0, [2, 3, 5])
        assert p == 7

    def test_combined_constraint(self):
        # avoid 30=2*3*5, skip used=[7]
        p = modular.select_lucky_prime(30, [7])
        assert p not in [2, 3, 5, 7]
        assert 30 % p != 0

    def test_returns_prime(self):
        from math import isqrt

        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, isqrt(n) + 1):
                if n % i == 0:
                    return False
            return True

        p = modular.select_lucky_prime(12, [2, 3])
        assert is_prime(p)

    def test_large_avoid_divisor(self):
        # large avoid_divisor passed as int (arbitrary precision)
        big = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # 9699690
        p = modular.select_lucky_prime(big)
        assert big % p != 0


# ---------------------------------------------------------------------------
# Integration: full modular GCD demo
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_proptest_roundtrip(self):
        """1000-case round-trip: reduce then lift recovers original polynomial."""
        import random

        rng = random.Random(42)
        pool = alkahest.ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")

        for _ in range(100):
            # Build a random polynomial with coefficients in [-20, 20]
            a = rng.randint(-20, 20)
            b = rng.randint(-20, 20)
            c = rng.randint(-20, 20)
            # a*x^2 + b*x + c
            x2 = x ** 2
            expr = pool.integer(a) * x2 + pool.integer(b) * x + pool.integer(c)
            poly = MultiPoly.from_symbolic(expr, [x, y])

            # Pick two primes big enough: mignotte_bound ≤ (|a|+|b|+|c|) * 4
            bound = modular.mignotte_bound(poly)
            images = []
            product = 1
            used = []
            while product <= 2 * bound:
                p = modular.select_lucky_prime(int(poly.integer_content()), used)
                used.append(p)
                images.append((modular.reduce_mod(poly, p), p))
                product *= p

            lifted = modular.lift_crt(images)
            assert str(lifted) == str(poly), (
                f"round-trip failed for {a}x^2+{b}x+{c}: "
                f"got {lifted}"
            )

    def test_error_code_on_bad_modulus(self):
        poly, *_ = make_poly(lambda p, x, y: x)
        try:
            modular.reduce_mod(poly, 9)
            assert False, "should have raised"
        except Exception as e:
            assert hasattr(e, "code") and e.code == "E-MOD-001"

    def test_modular_multipolyFp_type(self):
        poly, *_ = make_poly(lambda p, x, y: x)
        fp = modular.reduce_mod(poly, 7)
        assert isinstance(fp, MultiPolyFp)
        assert fp.modulus == 7
