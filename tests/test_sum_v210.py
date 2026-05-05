"""V2-10 — symbolic summation (Gosper, definite telescoping, recurrences, WZ check)."""

import math

import alkahest


def test_sum_indefinite_hypergeometric_gamma():
    pool = alkahest.ExprPool()
    k = pool.symbol("k")
    term = alkahest.simplify(k * alkahest.gamma(k + pool.integer(1))).value
    r = alkahest.sum_indefinite(term, k)
    assert r.steps


def test_sum_definite_k_factorial_numeric():
    pool = alkahest.ExprPool()
    k = pool.symbol("k")
    n = pool.symbol("n")
    zero = pool.integer(0)
    term = alkahest.simplify(k * alkahest.gamma(k + pool.integer(1))).value
    s = alkahest.sum_definite(term, k, zero, n).value
    expected = alkahest.simplify(
        alkahest.gamma(n + pool.integer(2)) + pool.integer(-1)
    ).value
    for ni in range(0, 9):
        env = {n: float(ni)}
        sv = alkahest.eval_expr(s, env)
        ev = alkahest.eval_expr(expected, env)
        ref = sum(m * math.factorial(m) for m in range(ni + 1))
        assert abs(sv - ref) < 1e-4
        assert abs(ev - ref) < 1e-4


def test_fibonacci_recurrence():
    pool = alkahest.ExprPool()
    n = pool.symbol("n")
    coeffs = [(-1, 1), (-1, 1), (1, 1)]
    initials = [pool.integer(0), pool.integer(1)]
    closed = alkahest.solve_linear_recurrence_homogeneous(n, coeffs, initials)

    def fib(m: int) -> int:
        if m == 0:
            return 0
        a, b = 0, 1
        for _ in range(1, m):
            a, b = b, a + b
        return b

    for ni in range(0, 13):
        v = alkahest.eval_expr(closed, {n: float(ni)})
        assert abs(v - fib(ni)) < 1e-4


def test_verify_wz_pair_trivial():
    pool = alkahest.ExprPool()
    n = pool.symbol("n")
    k = pool.symbol("k")
    z = pool.integer(0)
    assert alkahest.verify_wz_pair(z, z, n, k)
