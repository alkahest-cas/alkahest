"""examples/calculus.py — diff, diff_forward, integrate, and verification.

Run after `maturin develop`:
    python examples/calculus.py
"""

import alkahest as ak


def poly_from_coeffs(pool, x, coeffs):
    """Build a polynomial from a coefficient list (constant first)."""
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        c_id = pool.integer(c)
        if i == 0:
            terms.append(c_id)
        else:
            terms.append(c_id * (x ** i))
    if not terms:
        return pool.integer(0)
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr


def verify_antiderivative(pool, x, expr, label=""):
    """Assert diff(integrate(f, x), x) == f and print the result."""
    integral = ak.integrate(expr, x)
    deriv = ak.diff(integral.value, x)
    simp = ak.simplify(deriv.value)

    try:
        f_poly = ak.UniPoly.from_symbolic(expr, x)
        d_poly = ak.UniPoly.from_symbolic(simp.value, x)
        match = f_poly.coefficients() == d_poly.coefficients()
    except Exception:
        # Fall back to string comparison when rational coefficients are present
        match = str(simp.value) == str(expr)
    status = "✓" if match else "✗"
    print(f"  {status}  {label}: ∫f = {integral.value}")
    return match


def main():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    # ── Symbolic differentiation ──────────────────────────────────────────

    print("=== Symbolic Differentiation ===")

    # Polynomial
    p = poly_from_coeffs(pool, x, [1, 2, 3, 4])   # 1 + 2x + 3x^2 + 4x^3
    dp = ak.diff(p, x)
    print(f"d/dx (1+2x+3x²+4x³) = {dp.value}")
    print(f"   derivation steps  = {len(dp.steps)}")

    # Trigonometric: d/dx sin(x) = cos(x)
    dsin = ak.diff(ak.sin(x), x)
    print(f"d/dx sin(x)          = {dsin.value}")

    # Chain rule: d/dx sin(x^2) = 2*x*cos(x^2)
    sin_x2 = ak.sin(x ** 2)
    dsin_x2 = ak.diff(sin_x2, x)
    print(f"d/dx sin(x²)         = {dsin_x2.value}")

    # Exponential: d/dx exp(x) = exp(x)
    dexp = ak.diff(ak.exp(x), x)
    print(f"d/dx exp(x)          = {dexp.value}")

    # Logarithm: d/dx log(x) = x^(-1)
    dlog = ak.diff(ak.log(x), x)
    print(f"d/dx log(x)          = {dlog.value}")

    # ── Forward-mode AD ───────────────────────────────────────────────────

    print("\n=== Forward-mode Automatic Differentiation ===")

    p2 = poly_from_coeffs(pool, x, [0, 0, 1])  # x^2
    fwd = ak.diff_forward(p2, x)
    sym = ak.diff(p2, x)
    print(f"d/dx x² (forward)    = {fwd.value}")
    print(f"d/dx x² (symbolic)   = {sym.value}")
    print(f"  agree?             = {str(fwd.value) == str(sym.value)}")

    # ── Symbolic integration ──────────────────────────────────────────────

    print("\n=== Symbolic Integration ===")

    # Known functions
    r_sin = ak.integrate(ak.sin(x), x)
    print(f"∫ sin(x) dx          = {r_sin.value}")

    r_cos = ak.integrate(ak.cos(x), x)
    print(f"∫ cos(x) dx          = {r_cos.value}")

    r_exp = ak.integrate(ak.exp(x), x)
    print(f"∫ exp(x) dx          = {r_exp.value}")

    r_inv = ak.integrate(x ** -1, x)
    print(f"∫ 1/x dx             = {r_inv.value}")

    # Verification for polynomials
    print("\nVerification: diff(∫f) == f")
    verify_antiderivative(pool, x, x, "x")
    verify_antiderivative(pool, x, x ** 2, "x²")
    verify_antiderivative(pool, x, poly_from_coeffs(pool, x, [1, 2, 3]), "1+2x+3x²")
    verify_antiderivative(pool, x, poly_from_coeffs(pool, x, [0, 0, 0, 1]), "x³")
    verify_antiderivative(pool, x, poly_from_coeffs(pool, x, [0, 0, 0, 0, 1]), "x⁴")

    # ── Derivation log inspection ─────────────────────────────────────────

    print("\n=== Derivation Log ===")

    result = ak.diff(poly_from_coeffs(pool, x, [0, 0, 1]), x)  # d/dx x^2
    print(f"Expression: x²")
    print(f"Derivative: {result.value}")
    print(f"Steps ({len(result.steps)}):")
    for step in result.steps[:5]:
        print(f"  rule={step['rule']:25s}  {step['before']}  →  {step['after']}")


if __name__ == "__main__":
    main()
