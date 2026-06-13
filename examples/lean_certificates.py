"""examples/lean_certificates.py — Lean 4 proof certificate export.

Every top-level operation returns a ``DerivedResult`` with ``.value``,
``.steps`` (derivation log), and ``.certificate`` (Lean 4 proof term when
available).  Use ``alkahest.to_lean`` on an expression or a ``DerivedResult``.

Run after ``maturin develop``:
    python examples/lean_certificates.py
"""

import alkahest as ak


def main():
    pool = ak.ExprPool()
    x = pool.symbol("x")

    print("=== Lean export from expressions ===")
    lean_expr = ak.to_lean(x**2 + pool.integer(1))
    print(lean_expr[:120] + ("…" if len(lean_expr) > 120 else ""))

    print("\n=== Lean certificate on diff (deriv goals) ===")
    result = ak.diff(x**3, x)
    print(f"d/dx x³ = {result.value}")
    print(f"derivation steps: {len(result.steps)}")

    cert = result.certificate or ak.to_lean(result)
    assert "deriv (fun" in cert, "expected Mathlib deriv goal"
    print(cert[:200] + ("…" if len(cert) > 200 else ""))

    print("\n=== Lean certificate on integrate ===")
    integral = ak.integrate(ak.sin(x), x)
    print(f"∫ sin(x) dx = {integral.value}")
    if integral.certificate:
        print(integral.certificate[:200] + "…")


if __name__ == "__main__":
    main()
