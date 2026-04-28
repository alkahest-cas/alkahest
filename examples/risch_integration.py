"""examples/risch_integration.py — V1-2 algebraic Risch integration showcase.

Demonstrates ∫ f(x, sqrt(P(x))) dx for the three genus-0 cases:
  - P constant      (sqrt factors out of the integral)
  - P linear        (substitution u = P reduces to rational)
  - P quadratic     (J_0 formula + reduction)

Also demonstrates mixed integrands A(x) + B(x)·sqrt(P(x)) and the
NonElementary guard for elliptic integrands (deg P ≥ 3).

Run after `maturin develop`:
    python examples/risch_integration.py
"""

from alkahest.alkahest import ExprPool, diff, integrate, simplify, sqrt


def display(pool, label, expr):
    print(f"  {label}: {expr}")


def verify(pool, x, f, F, label):
    """Check d/dx F == f numerically at two test points."""
    from alkahest.alkahest import ArbBall, interval_eval
    dF = diff(F, x).value
    ok = True
    for pt in (1.5, 3.7):
        bindings = {x: ArbBall(pt)}
        try:
            lhs = interval_eval(dF, bindings)
            rhs = interval_eval(f, bindings)
            err = abs(lhs.mid - rhs.mid)
            if err > 1e-9:
                print(f"  MISMATCH at x={pt}: dF/dx={lhs.mid}, f={rhs.mid}")
                ok = False
        except Exception as exc:
            print(f"  eval error at x={pt}: {exc}")
            ok = False
    status = "✓" if ok else "✗"
    print(f"  numeric check [{status}]")
    return ok


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def example(pool, label, f, x):
    print(f"\n[{label}]")
    display(pool, "f", f)
    try:
        result = integrate(f, x)
        display(pool, "F", result.value)
        verify(pool, x, f, result.value, label)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    # -----------------------------------------------------------------------
    section("1. P constant — sqrt factors out")
    # -----------------------------------------------------------------------

    # ∫ sqrt(5) dx = x·sqrt(5)
    pool = ExprPool(); x = pool.symbol("x")
    example(pool, "∫ sqrt(5) dx", sqrt(pool.integer(5)), x)

    # ∫ sqrt(3)·x² dx = sqrt(3)·x³/3
    pool = ExprPool(); x = pool.symbol("x")
    example(pool, "∫ sqrt(3)·x² dx", sqrt(pool.integer(3)) * x ** 2, x)

    # -----------------------------------------------------------------------
    section("2. P linear — substitution u = ax+c")
    # -----------------------------------------------------------------------

    # ∫ sqrt(x) dx = (2/3) x^(3/2)
    pool = ExprPool(); x = pool.symbol("x")
    example(pool, "∫ sqrt(x) dx", sqrt(x), x)

    # ∫ x·sqrt(x) dx = (2/5) x^(5/2)
    pool = ExprPool(); x = pool.symbol("x")
    example(pool, "∫ x·sqrt(x) dx", x * sqrt(x), x)

    # ∫ sqrt(2x+1) dx = (1/3)(2x+1)^(3/2)
    pool = ExprPool(); x = pool.symbol("x")
    p = pool.integer(2) * x + pool.integer(1)
    example(pool, "∫ sqrt(2x+1) dx", sqrt(p), x)

    # ∫ 1/sqrt(x) dx = 2·sqrt(x)
    pool = ExprPool(); x = pool.symbol("x")
    sx = sqrt(x)
    inv_sx = sx ** -1
    example(pool, "∫ 1/sqrt(x) dx", inv_sx, x)

    # ∫ x²·sqrt(x+1) dx
    pool = ExprPool(); x = pool.symbol("x")
    p_lp1 = x + pool.integer(1)
    example(pool, "∫ x²·sqrt(x+1) dx", x ** 2 * sqrt(p_lp1), x)

    # -----------------------------------------------------------------------
    section("3. P quadratic — J₀ formula + reduction")
    # -----------------------------------------------------------------------

    # ∫ sqrt(x²+1) dx = x/2·sqrt(x²+1) + 1/2·log(2x + 2·sqrt(x²+1))  [+const]
    pool = ExprPool(); x = pool.symbol("x")
    p_q = x ** 2 + pool.integer(1)
    example(pool, "∫ sqrt(x²+1) dx", sqrt(p_q), x)

    # ∫ 1/sqrt(x²+1) dx = log(x + sqrt(x²+1))
    pool = ExprPool(); x = pool.symbol("x")
    p_q = x ** 2 + pool.integer(1)
    s_q = sqrt(p_q)
    inv_p = p_q ** -1
    example(pool, "∫ 1/sqrt(x²+1) dx", inv_p * s_q, x)

    # ∫ sqrt(x²-1) dx
    pool = ExprPool(); x = pool.symbol("x")
    p_qm = x ** 2 + pool.integer(-1)
    example(pool, "∫ sqrt(x²-1) dx", sqrt(p_qm), x)

    # -----------------------------------------------------------------------
    section("4. Mixed integrand A(x) + B(x)·sqrt(P)")
    # -----------------------------------------------------------------------

    # ∫ (x² + sqrt(x+1)) dx = x³/3 + (2/3)(x+1)^(3/2)
    pool = ExprPool(); x = pool.symbol("x")
    p_lp1 = x + pool.integer(1)
    example(pool, "∫ (x² + sqrt(x+1)) dx", x ** 2 + sqrt(p_lp1), x)

    # ∫ (3x + 2·sqrt(x²+4)) dx
    pool = ExprPool(); x = pool.symbol("x")
    p_q4 = x ** 2 + pool.integer(4)
    example(pool, "∫ (3x + 2·sqrt(x²+4)) dx",
            pool.integer(3) * x + pool.integer(2) * sqrt(p_q4), x)

    # -----------------------------------------------------------------------
    section("5. NonElementary guard — elliptic integrals")
    # -----------------------------------------------------------------------

    pool = ExprPool(); x = pool.symbol("x")
    p_ell = x ** 3 + pool.integer(1)
    s_ell = sqrt(p_ell)
    print(f"\n[∫ sqrt(x³+1) dx  — should raise NonElementary]")
    display(pool, "f", s_ell)
    try:
        integrate(s_ell, x)
        print("  ERROR: should have raised")
    except Exception as e:
        print(f"  Correctly raised: {type(e).__name__}: {e}")

    print("\nAll examples complete.")


if __name__ == "__main__":
    main()
