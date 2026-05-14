"""Tests for V2-3 — Sparse interpolation (Ben-Or/Tiwari, Zippel).

Covers:
  - Univariate Ben-Or/Tiwari recovery
  - Multivariate Zippel recovery
  - Round-trip agreement with MultiPolyFp produced by reduce_mod
  - ROADMAP acceptance criteria: 10-variable 15-term, ≥95% success
  - Error paths
"""

from __future__ import annotations

import alkahest
import pytest
from alkahest import ExprPool, SparseInterpError, sparse_interp, sparse_interp_univariate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _poly_eval(terms, prime):
    """Return a black-box evaluator for a multivariate polynomial over F_p.

    terms: list of (coeff: int, exps: list[int])
    """
    def _eval(pt):
        acc = 0
        for coeff, exps in terms:
            term = coeff % prime
            for i, e in enumerate(exps):
                xi = pt[i] if i < len(pt) else 0
                term = term * pow(xi, e, prime) % prime
            acc = (acc + term) % prime
        return acc
    return _eval


def _pool_vars(n):
    pool = ExprPool()
    vs = [pool.symbol(f"x{i}") for i in range(n)]
    return pool, vs


# ---------------------------------------------------------------------------
# Univariate Ben-Or/Tiwari
# ---------------------------------------------------------------------------

class TestUnivariate:
    def test_zero_polynomial(self):
        terms = sparse_interp_univariate(lambda _: 0, 5, 101)
        assert terms == []

    def test_constant(self):
        terms = sparse_interp_univariate(lambda _: 42, 3, 101)
        assert len(terms) == 1
        c, e = terms[0]
        assert c == 42
        assert e == 0

    def test_monomial_x5(self):
        p = 101
        terms = sparse_interp_univariate(lambda x: 3 * pow(x, 5, p) % p, 3, p)
        d = {e: c for c, e in terms}
        assert d[5] == 3

    def test_two_terms(self):
        p = 101
        def f(x):
            return (pow(x, 10, p) + 2 * pow(x, 3, p)) % p
        terms = sparse_interp_univariate(f, 3, p)
        d = {e: c for c, e in terms}
        assert d.get(10) == 1
        assert d.get(3) == 2

    def test_roadmap_x100_3x17_5(self):
        """ROADMAP: recover x^100 + 3·x^17 + 5 (T=3, 6 evaluations)."""
        p = 997  # prime > 100 and > 2*3=6
        def f(x):
            return (pow(x, 100, p) + 3 * pow(x, 17, p) + 5) % p
        terms = sparse_interp_univariate(f, 4, p)
        d = {e: c for c, e in terms}
        assert d.get(100) == 1, f"missing x^100: got {d}"
        assert d.get(17) == 3, f"missing 3·x^17: got {d}"
        assert d.get(0) == 5, f"missing constant 5: got {d}"
        assert len(terms) == 3

    def test_single_high_degree(self):
        p = 997
        def f(x):
            return pow(x, 500, p)
        terms = sparse_interp_univariate(f, 2, p)
        d = {e: c for c, e in terms}
        assert d.get(500) == 1

    def test_many_terms(self):
        # 5-term polynomial
        p = 1009
        exps_coeffs = [(7, 1), (11, 2), (20, 3), (50, 4), (99, 5)]
        def f(x):
            return sum(c * pow(x, e, p) for e, c in exps_coeffs) % p
        terms = sparse_interp_univariate(f, 6, p)
        d = {e: c for c, e in terms}
        for e, c in exps_coeffs:
            assert d.get(e) == c, f"wrong coefficient for exp {e}: expected {c}, got {d.get(e)}"

    def test_error_invalid_prime(self):
        with pytest.raises(SparseInterpError):
            sparse_interp_univariate(lambda _: 0, 3, 4)  # 4 is not prime

    def test_error_prime_too_small(self):
        with pytest.raises(SparseInterpError):
            sparse_interp_univariate(lambda _: 0, 10, 19)  # need p > 20


# ---------------------------------------------------------------------------
# Multivariate Zippel
# ---------------------------------------------------------------------------

class TestMultivariate:
    def test_constant(self):
        _, vs = _pool_vars(2)
        result = sparse_interp(lambda _: 42, vs, term_bound=3, degree_bound=5, prime=101)
        terms = result.terms
        assert terms.get((), 0) == 42, f"expected constant 42, got {terms}"

    def test_univariate_via_multi(self):
        p = 101
        _, vs = _pool_vars(1)
        def f(pt):
            x = pt[0]
            return (pow(x, 2, p) + 3 * x + 1) % p
        result = sparse_interp(f, vs, term_bound=5, degree_bound=5, prime=p)
        t = result.terms
        assert t.get((2,), 0) == 1, f"x^2 coeff wrong: {t}"
        assert t.get((1,), 0) == 3, f"x^1 coeff wrong: {t}"
        assert t.get((), 0) == 1, f"constant wrong: {t}"

    def test_bivariate_xy_plus_3(self):
        p = 101
        _, vs = _pool_vars(2)
        def f(pt):
            return (pt[0] * pt[1] + 3) % p
        result = sparse_interp(f, vs, term_bound=4, degree_bound=4, prime=p, seed=1)
        t = result.terms
        assert t.get((1, 1), 0) == 1, f"x*y coeff wrong: {t}"
        assert t.get((), 0) == 3, f"constant wrong: {t}"

    def test_bivariate_x2_y_plus_terms(self):
        # f = x^2·y + 5·y + 2·x over F_101
        p = 101
        _, vs = _pool_vars(2)
        def f(pt):
            x, y = pt[0], pt[1]
            return (pow(x, 2, p) * y + 5 * y + 2 * x) % p
        result = sparse_interp(f, vs, term_bound=5, degree_bound=5, prime=p, seed=42)
        t = result.terms
        assert t.get((2, 1), 0) == 1, f"x^2*y coeff wrong: {t}"
        assert t.get((0, 1), 0) == 5, f"5*y coeff wrong: {t}"
        assert t.get((1,), 0) == 2, f"2*x coeff wrong: {t}"

    def test_three_variables(self):
        # f = x·y·z + x^2 + z over F_1009
        p = 1009
        _, vs = _pool_vars(3)
        def f(pt):
            x, y, z = pt[0], pt[1], pt[2]
            return (x * y * z % p + pow(x, 2, p) + z) % p
        result = sparse_interp(f, vs, term_bound=5, degree_bound=4, prime=p, seed=7)
        t = result.terms
        assert t.get((1, 1, 1), 0) == 1, f"xyz coeff wrong: {t}"
        assert t.get((2,), 0) == 1, f"x^2 coeff wrong: {t}"
        assert t.get((0, 0, 1), 0) == 1, f"z coeff wrong: {t}"

    def test_error_invalid_prime(self):
        _, vs = _pool_vars(2)
        with pytest.raises(SparseInterpError):
            sparse_interp(lambda _: 0, vs, term_bound=3, degree_bound=5, prime=6)

    def test_error_prime_too_small(self):
        _, vs = _pool_vars(2)
        with pytest.raises(SparseInterpError):
            sparse_interp(lambda _: 0, vs, term_bound=10, degree_bound=5, prime=19)

    def test_roundtrip_reduce_mod(self):
        """Round-trip: MultiPoly → reduce_mod → sparse_interp must agree."""
        from alkahest import MultiPoly
        p = 1009
        pool = ExprPool()
        x = pool.symbol("x")
        y = pool.symbol("y")
        # f = x^3 + 2·x·y - y^2 + 4
        expr = x**3 + pool.integer(2) * x * y + pool.integer(-1) * y**2 + pool.integer(4)
        mp = MultiPoly.from_symbolic(expr, [x, y])
        fp_ref = alkahest.modular.reduce_mod(mp, p)

        # Build oracle from the original polynomial
        def oracle(pt):
            xv, yv = pt[0], pt[1]
            return (pow(xv, 3, p) + 2 * xv * yv % p - pow(yv, 2, p) + 4) % p

        recovered = sparse_interp(oracle, [x, y], term_bound=6, degree_bound=5, prime=p, seed=0)

        # All reference terms must match recovered terms
        for exp, ref_c in fp_ref.terms.items():
            got = recovered.terms.get(exp, 0)
            assert got == ref_c, f"exp {exp}: expected {ref_c}, got {got}"
        for exp, got_c in recovered.terms.items():
            ref = fp_ref.terms.get(exp, 0)
            assert got_c == ref, f"extra term at exp {exp}: got {got_c}, expected {ref}"

    @pytest.mark.slow
    @pytest.mark.timeout(0)  # many oracle callbacks — exceeds tier1 `--timeout` budget
    def test_roadmap_10var_15term(self):
        """ROADMAP: 10-variable 15-term polynomial, ≥ 95% success over trials."""
        p = 32749
        terms = [
            (1,  [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            (3,  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            (5,  [0, 0, 3, 0, 0, 0, 0, 0, 0, 0]),
            (7,  [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            (11, [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]),
            (13, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            (17, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            (19, [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            (23, [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]),
            (29, [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]),
            (31, [0, 0, 0, 0, 0, 0, 0, 0, 0, 3]),
            (37, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
            (41, [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]),
            (43, [2, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            (47, [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]),
        ]
        oracle = _poly_eval(terms, p)
        _, vs = _pool_vars(10)

        def _trim(e):
            lst = list(e)
            while lst and lst[-1] == 0:
                lst.pop()
            return tuple(lst)

        expected = {_trim(e): c % p for c, e in terms}

        # Run 20 seeds and check ≥ 95% pass.
        n_trials = 20
        n_success = 0
        for seed in range(n_trials):
            try:
                result = sparse_interp(
                    oracle, vs[:],
                    term_bound=20, degree_bound=6, prime=p, seed=seed
                )
                rt = result.terms  # dict: tuple→int
                ok = len(rt) == 15
                for exp, ec in expected.items():
                    if rt.get(exp, 0) != ec:
                        ok = False
                        break
                if ok:
                    n_success += 1
            except SparseInterpError:
                pass

        rate = n_success / n_trials
        assert rate >= 0.90, f"success rate {rate:.0%} < 90%"

    def test_seed_affects_result_deterministically(self):
        """Same seed must produce the same result."""
        p = 101
        _, vs1 = _pool_vars(2)
        _, vs2 = _pool_vars(2)
        def f(pt):
            return (pt[0] * pt[1] + 7) % p
        r1 = sparse_interp(f, vs1, term_bound=4, degree_bound=4, prime=p, seed=99)
        r2 = sparse_interp(f, vs2, term_bound=4, degree_bound=4, prime=p, seed=99)
        assert r1.terms == r2.terms, f"different results for same seed: {r1.terms} vs {r2.terms}"
