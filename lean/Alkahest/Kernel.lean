/-
Alkahest algebraic kernel: reflective certificate checking.

A certificate is *data* — an `AExpr` claim plus a witness (atom table,
relations, cofactors) — that a verified Boolean checker decides; one soundness
theorem per checker turns `check data = true` into the mathematical statement.

* `Alkahest.Kernel.Expr`  — the AST, real semantics, side conditions
* `Alkahest.Kernel.Poly`  — sparse ℚ-polynomials with kernel-reducible arithmetic
* `Alkahest.Kernel.Norm`  — normal forms in atoms (atom table, inverse handling)
* `Alkahest.Kernel.Sign`  — sign / domain-condition checker
* `Alkahest.Kernel.Deriv` — symbolic derivative `D` and `HasDerivAt` soundness
* `Alkahest.Kernel.Rel`   — atom relations (`sin² + cos² = 1`, `u·u⁻¹ = 1`, …)
* `Alkahest.Kernel.Check` — `checkEq`, `checkIntegral` and their soundness
-/
import Alkahest.Kernel.Check
