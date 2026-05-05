"""V2-18 — difference equations (`rsolve`)."""

import alkahest


def test_rsolve_arithmetic_progression_general():
    pool = alkahest.ExprPool()
    n = pool.symbol("n")
    f = lambda *args: pool.func("f", list(args))
    eq = alkahest.simplify(
        f(n) - f(n + pool.integer(-1)) - pool.integer(1)
    ).value
    sol = alkahest.rsolve(eq, n, "f", None)
    assert "C0" in str(sol)


def test_rsolve_geometric_with_init():
    pool = alkahest.ExprPool()
    n = pool.symbol("n")
    f = lambda *args: pool.func("f", list(args))
    eq = alkahest.simplify(
        f(n) - pool.integer(2) * f(n + pool.integer(-1))
    ).value
    sol = alkahest.rsolve(eq, n, "f", {0: pool.integer(1)})
    env = {n: 5.0}
    v = alkahest.eval_expr(sol, env)
    assert abs(v - 32.0) < 1e-5


def test_rsolve_fibonacci_with_init():
    pool = alkahest.ExprPool()
    n = pool.symbol("n")
    f = lambda *args: pool.func("f", list(args))
    eq = alkahest.simplify(
        f(n)
        - f(n + pool.integer(-1))
        - f(n + pool.integer(-2))
    ).value
    initials = {0: pool.integer(0), 1: pool.integer(1)}
    sol = alkahest.rsolve(eq, n, "f", initials)
    fib = [0, 1]
    for _ in range(2, 15):
        fib.append(fib[-1] + fib[-2])
    for i, expected in enumerate(fib):
        assert abs(alkahest.eval_expr(sol, {n: float(i)}) - expected) < 1e-4
