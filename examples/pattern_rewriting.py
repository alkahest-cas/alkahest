"""examples/pattern_rewriting.py — custom rules via the pattern matcher (R-5).

Demonstrates:
  1. Built-in trig identities (simplify_trig)
  2. Built-in log/exp identities (simplify_log_exp)
  3. User-defined rewrite rules via make_rule + simplify_with
  4. Substitution primitive (subs)

Run after `maturin develop`:
    PYTHONPATH=python python examples/pattern_rewriting.py
"""

from alkahest.alkahest import (
    ExprPool,
    cos,
    exp,
    log,
    make_rule,
    sin,
    simplify,
    simplify_log_exp,
    simplify_trig,
    simplify_with,
    subs,
)


def main():
    pool = ExprPool()
    x = pool.symbol("x")
    y = pool.symbol("y")
    neg_one = pool.integer(-1)

    # ── Built-in: trigonometric identities ────────────────────────────────

    print("=== Trigonometric Identities ===")

    neg_x = neg_one * x

    # sin(-x) → -sin(x)
    r = simplify_trig(sin(neg_x))
    print(f"sin(-x)                 → {r.value}")

    # cos(-x) → cos(x)
    r = simplify_trig(cos(neg_x))
    print(f"cos(-x)                 → {r.value}")

    # sin²(x) + cos²(x) → 1
    sin_sq = sin(x) ** 2
    cos_sq = cos(x) ** 2
    r = simplify_trig(sin_sq + cos_sq)
    print(f"sin²(x) + cos²(x)       → {r.value}")

    # sin²(x) + cos²(x) + x → x + 1
    r = simplify_trig(sin_sq + cos_sq + x)
    print(f"sin²(x) + cos²(x) + x   → {r.value}")

    # ── Built-in: log / exp identities ───────────────────────────────────

    print("\n=== log / exp Identities ===")

    # log(exp(x)) → x  (unconditional on reals)
    r = simplify_log_exp(log(exp(x)))
    print(f"log(exp(x))             → {r.value}")

    # Branch-cut identities need positivity (Assumptions or Domain.Positive)
    from alkahest import Assumptions

    assumptions = Assumptions(pool)
    assumptions.refine(pool.gt(x, pool.integer(0)))
    assumptions.refine(pool.gt(y, pool.integer(0)))

    # exp(log(x)) → x  when x > 0
    r = simplify_log_exp(exp(log(x)), assumptions)
    print(f"exp(log(x)) [x>0]       → {r.value}")

    # log(x)+log(y) → log(x*y) when x,y > 0
    r = simplify_log_exp(log(x) + log(y), assumptions)
    print(f"log(x)+log(y) [x,y>0]   → {r.value}")

    # log(x^3) → 3*log(x) when x > 0
    r = simplify_log_exp(log(x ** 3), assumptions)
    print(f"log(x³) [x>0]           → {r.value}")

    # ── User-defined rules via make_rule ──────────────────────────────────

    print("\n=== User-Defined Rewrite Rules ===")

    # Wildcards: symbols starting with lowercase are wildcards
    a = pool.symbol("a")
    b = pool.symbol("b")

    # Rule: a + a → 2*a  (doubling)
    two = pool.integer(2)
    rule_double = make_rule(a + a, two * a)

    r = simplify_with(x + x, [rule_double])
    print(f"x + x  (doubling rule)  → {r.value}")

    # Rule: exp(a + b) → exp(a) * exp(b)
    rule_exp_sum = make_rule(exp(a + b), exp(a) * exp(b))

    r = simplify_with(exp(x + y), [rule_exp_sum])
    print(f"exp(x+y) (exp sum rule) → {r.value}")

    # Rule: log(a^3) → 3*log(a)  (pattern with concrete exponent)
    three_lit = pool.integer(3)
    rule_logpow = make_rule(log(a ** 3), three_lit * log(a))

    r = simplify_with(log(x ** 3), [rule_logpow])
    print(f"log(x^3) (logpow rule)  → {r.value}")

    # ── Substitution primitive ────────────────────────────────────────────

    print("\n=== Substitution (subs) ===")

    one = pool.integer(1)
    expr_poly = x ** 2 + two * x + one
    print(f"f(x) = {expr_poly}")

    # Substitute x → 3
    three = pool.integer(3)
    r_simp = simplify(subs(expr_poly, {x: three}))
    print(f"f(3) = {r_simp.value}")   # should be 16

    # Substitute x → y + 1
    r_simp2 = simplify(subs(expr_poly, {x: y + one}))
    print(f"f(y+1) = {r_simp2.value}")

    # Substitute in a transcendental expression: sin(x) with x → y^2
    r_subst3 = subs(sin(x), {x: y ** 2})
    print(f"sin(x) with x→y²: {r_subst3}")


if __name__ == "__main__":
    main()
