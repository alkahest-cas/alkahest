# The reflective certificate kernel

Indefinite-integration certificates are, by default, checked by a small
verified checker in the repository's Lean library
[`Alkahest.Kernel`](https://github.com/alkahest-cas/alkahest/tree/main/lean/Alkahest)
instead of by a tree of Mathlib tactics. This is *proof by reflection*. The
certificate is *data*: an expression AST for the claim plus a witness. A
Boolean checker, written and proved correct in Lean, decides it, and one
soundness theorem, proved once in the library, turns `check data = true` into
the mathematical statement. The same kernel checks `dsolve` solutions
(`alkahest.experimental.dsolve_certificate`, below) and the PDE residuals of
`alkahest.fex.finalize` ([PDE residuals](#pde-residuals-fex-solutions)).

Not every certificate takes this path. Answers the kernel does not encode fall
back to the `HasDerivAt` tactic emitter, and the other certified operations
(derivatives, definite integrals, limits, sums, resultants, rewrites) have
tactic emitters only. `certificate_status["path"]` says which checker a
result's certificate uses.

```lean
import Alkahest.Kernel

open Alkahest.Kernel in
theorem alkahest_integral (x : ℝ) (h1 : x ≠ 0) :
    HasDerivAt (fun (x : ℝ) => Real.log x) ((x)⁻¹) x :=
  integral_sound (H := [(.ne (.var 0))])
    (F := (.app .log (.var 0)))
    (f := (.inv (.var 0)))
    (w := ⟨[⟨(.var 0), .zero⟩, ⟨(.fn .log), (.var 0)⟩, ⟨.inv, (.var 0)⟩], [], []⟩)
    (by decide +kernel) [] x ⟨h1, trivial⟩
```

The statement is written in ordinary notation. `AExpr.eval` unfolds to exactly
that term, so Lean matches it against the soundness theorem's conclusion by
definitional unfolding. A reader checks the statement, not the data.

## What the checker does

`checkIntegral H F f w` (in `Alkahest/Kernel/Check.lean`):

1. computes the symbolic derivative `D 0 F` itself (`Deriv.lean`). The soundness
   theorem `D_hasDerivAt` says that on the domain `dconds F` (denominators,
   `log` arguments `≠ 0`, `sqrt` radicands `> 0`, `arcsin` arguments in
   `(-1, 1)`, `cos ≠ 0` under `tan`), `y ↦ F` has derivative `D 0 F`;
2. decides each of those side conditions from the hypotheses `H` with a
   structural sign checker (`Sign.lean`, the reflective analogue of
   `positivity`): `exp`, `cosh`, `π`, even powers, positive literals, sums and
   products of positive terms, `b < a ⟹ a - b > 0`;
3. checks `D 0 F - f = Σ cᵢ·(Lᵢ - Rᵢ)` as an identity of polynomials over ℚ in
   *atoms* (`Norm.lean`, `Poly.lean`). Each `Lᵢ = Rᵢ` is a relation proved in
   `Rel.lean`: `u·u⁻¹ = 1` (needs `u ≠ 0`), `√u² = u` (needs `0 ≤ u`),
   `sin² + cos² = 1`, `cosh² - sinh² = 1`, the addition formulas for `sin`,
   `cos`, `sinh`, `cosh` and `exp`, `tan = sin/cos`, `tanh = sinh/cosh`,
   `log (exp a) = a`, and a few more. Their side conditions are also decided
   from `H`.

Atoms are given by a table in the witness, but the kernel normalises each
atom's argument against the earlier entries. So `sin (x + 1)` and
`sin (1 + x)`, or `(1 + x²)⁻¹` from `D (atan x)` and `(x² + 1)⁻¹` in the
integrand, are the same atom without any trusted matching. Inverses distribute
over monomials unconditionally (`(c·a²)⁻¹ = c⁻¹·(a⁻¹)²` holds in Lean's ℝ,
where `0⁻¹ = 0`). Facts that need a nonzero denominator enter only through the
relation `u·u⁻¹ = 1`, whose condition the checker decides.

## The trusted base

To accept a kernel certificate you trust:

* Lean's kernel, which type-checks the proof term and evaluates the checker
  when it reduces `decide +kernel`;
* the Mathlib definitions the statement mentions (`HasDerivAt`, `Real.exp`,
  `Real.log`, …);
* the *statement* itself, which you read: the theorem says what it proves in
  ordinary notation.

You do not trust Alkahest, the Rust emitter, the atom table or the cofactors in
the witness: if any of them is wrong, the checker returns `false` and the
certificate does not compile. The checker and its soundness theorems
(`D_hasDerivAt`, `integral_sound`, `ode_sound`, `pde_sound`) are themselves
proved in Lean, and CI rejects any `sorry`, `admit` or `axiom` in
`Alkahest.Kernel`.

The opt-in `kernel_native` mode (below) closes the same certificate with
`native_decide`, which adds Lean's compiler and `Lean.ofReduceBool` to that
list. The default never uses it.

## Kernel reducibility

Every checker is structural recursion over lists and the AST, using `Nat`,
`Int` and `ℚ` arithmetic, which the Lean kernel evaluates with GMP-backed
literals. Merges such as polynomial addition use an inner/outer recursion
instead of well-founded recursion, which the kernel cannot unfold efficiently.
Certificates are closed by `decide +kernel`, which sends the Boolean straight
to the kernel without elaborator-level reduction.

`native_decide` is opt-in: `alkahest.set_lean_certificate_mode("kernel_native")`.
It compiles the checker and adds Lean's compiler and `Lean.ofReduceBool` to the
trusted base. The certificate says so in its header, and
`certificate_status["path"]` reports `"kernel_native"`. CI rejects any default
certificate that uses it.

### Measured cost

Per-certificate check time, from Lean's profiler: every category except
`import`, so loading Mathlib is excluded. Shared corpus: the calculus suites'
163 integrals that both paths certify.

| | kernel | tactic |
|---|---|---|
| median | 0.32 s | 0.58 s |
| 90th percentile | 3.3 s | 29.5 s |
| max | 13.1 s | 107.9 s |
| import (median) | 5.5 s | 7.2 s |

The kernel is faster on every certificate, with a worst-case ratio of 0.82.
What made the difference:

* Witness numbers are raw literals (`nat_lit`, `Int.ofNat`, `Int.negSucc`), so
  the elaborator runs no `OfNat` instance search per number.
* Comparisons are `Nat.blt` matched as `Bool`, not `Decidable` instances.
* The atom lookup uses a structural Boolean equality: the derived `DecidableEq`
  on lists of `ℚ` was the slowest step.
* Products put the shorter operand outermost, and cofactor terms arrive in
  the kernel's own order.

## Fallback and reporting

The Rust emitter (`alkahest-core/src/lean/kernel.rs`) mirrors the Lean checker
exactly: the same `D`, the same normal form and the same sign rules. It
computes the atom table, relations and cofactors the checker will accept, and
re-verifies the identity before emitting. When it cannot encode an answer
(special functions, `RootSum`, roots other than `√`, constants of unproved
sign), `integrate`'s certificate falls back to the `HasDerivAt` tactic emitter,
and `certificate_status["kernel_detail"]` gives the reason
(`withheld_special_function`, `withheld_unsupported_node`, …). Definite
integrals are certified through Mathlib's interval FTC
(`intervalIntegral.integral_eq_sub_of_hasDerivAt`) by the tactic emitter.

### How often each path is taken

Measured on the published 3.15.0 wheel in the default `"auto"` mode,
indefinite integrals only. "Certified" counts the answers that came with a
certificate; the Lean CI job compiles the integration corpus in both modes.

| corpus | integrands | certified | kernel | tactic |
|---|---|---|---|---|
| `tests/lean_integrate_corpus.py` | 50 | 50 | 45 | 5 |
| calculus suite, standard tier | 51 | 44 | 41 | 3 |
| calculus suite, Rubi hard tier | 171 | 136 | 132 | 4 |
| **total** | **272** | **230** | **218 (95%)** | **12** |

Every tactic fallback in these corpora is a special-function answer (`erf`,
`erfi`, `Si`; reason `withheld_special_function`) or an answer with a cube or
other non-square root (`withheld_unsupported_node`). The calculus suites are
the integral tiers of the CAS comparison benchmark (the standard tier, and the
hard tier drawn from the Rubi test suite).

```python
import alkahest

x = alkahest.symbol("x")
r = alkahest.integrate(alkahest.parse("x*exp(x)"), x)
src = r.certificate                       # kernel certificate, else tactic, else None
assert r.certificate_status["path"] == "kernel"   # "kernel" | "kernel_native" | "tactic" | None
r.certificate_status["kernel_detail"]     # why the kernel path withheld, if it did

alkahest.set_lean_certificate_mode("tactic")      # tactic emitter only
assert r.certificate_status["path"] == "tactic"
alkahest.set_lean_certificate_mode("auto")        # back to the default
```

Certificates are built lazily, only when `.certificate` or
`.certificate_status` is read. `integrate` itself does no extra work.

## Checking a kernel certificate yourself

Kernel certificates `import Alkahest.Kernel`, so build the library once from a
checkout of the repository:

```bash
cd lean
lake exe cache get        # Mathlib's prebuilt oleans
lake build Alkahest       # the kernel (about a minute)
lake env lean -DwarningAsError=true -DautoImplicit=false my_certificate.lean
```

A certificate that compiles with no errors is checked. Before trusting it, read
its theorem statement, and check that the file contains no `sorry`, `admit`,
`axiom` or `native_decide` (the last appears only in `kernel_native` mode).

## ODE solutions

`alkahest.experimental.dsolve_certificate(equation, x, y, derivs, y_of_x)`
certifies that a solution returned by `dsolve` satisfies its equation. The
certificate states the derivative chain `S, S', …` of the solution (each link a
`HasDerivAt`, computed by the kernel's `D`) and that the equation with them
substituted is identically zero, through `ode_sound`. Free constants `C1, C2,
…` are universally quantified.

```python
import alkahest as ak
from alkahest import experimental as ex

x, y, yp, ypp = (ak.symbol(n) for n in ("x", "y", "yp", "ypp"))
eq = ak.parse("ypp + y")
sol = ex.dsolve(eq, x, y, [yp, ypp])[0]["y_of_x"]
out = ex.dsolve_certificate(eq, x, y, [yp, ypp], sol)
out["path"]                                 # "kernel"
"ode_sound" in out["certificate"]           # True
```

`native=True` closes it with `native_decide` instead. The Lean CI job compiles
the ODE corpus `tests/lean_ode_corpus.py`.

## PDE residuals (FEX solutions)

`Alkahest.Kernel.Pde` adds one more checker. `checkPde u op n w` takes an
explicit function `u` of `n` coordinates and an operator `op` written over
the variables `0 … n-1` (the coordinates), `n` (`u`), `n+1` (`Δu`) and
`n+2+i` (`∂u/∂xᵢ`). It builds the jet itself with the kernel's derivative
`D` (`Σᵢ D i (D i u)` and `D i u`), substitutes it into `op`
(`AExpr.bind`), and decides that the result is zero with `checkEq`. The
differentiability conditions of `u` and of every first partial must follow
from no hypotheses at all, so they hold on the whole of `ℝⁿ` and in
particular along every coordinate line. `pde_sound` turns acceptance into a
statement about Mathlib's `deriv`:

```lean
theorem pde_sound {u op : AExpr} {n : Nat} {w : Witness} (h : checkPde u op n w = true)
    (ρ : List ℝ) (hn : ρ.length = n) : op.eval (jetR u ρ n) = 0
```

where `jetR u ρ n` is `ρ ++ u(ρ) :: Σᵢ deriv (deriv (y ↦ u(ρ[i ↦ y]))) ρᵢ
:: [deriv (y ↦ u(ρ[i ↦ y])) ρᵢ, …]`. A certificate states the theorem in
ordinary notation about a definition `fexU` of the solution, for example
(cubic Schrödinger, `d = 2`)

```text
theorem fex_pde (x0 x1 : ℝ) :
    ((-((0 : ℝ) + deriv (deriv (fun (y : ℝ) => fexU y x1)) x0
         + deriv (deriv (fun (y : ℝ) => fexU x0 y)) x1)) + (fexU x0 x1) ^ 3 + V * fexU x0 x1) = 0
```

and proves it with `pde_sound _ (by decide +kernel) [x0, x1] rfl`. The
statement matches the kernel's conclusion by definitional unfolding, so the
reader never has to trust Alkahest's derivatives.

`alkahest.fex.finalize` emits these certificates for Poisson, the cubic
Schrödinger equation, the eigenproblem and linear transport whenever the
fitted closed form is exact and its interior PDE is proved, falling back to
the `HasDerivAt` tactic proof for Poisson and transport. The exponential
relations the NLS solutions need (`exp(2a) = exp(a)²`) are the kernel's
`expAdd` family; a constant `exp(−log 3 + a)` is first written `⅓·exp(a)`
(the certificate's `solution_form`). The theorem covers the PDE residual
at every point of `ℝ^d`; boundary conditions are not part of it.

## Next steps

* Definite integrals, derivatives and sums through the kernel; today they use
  the tactic emitters.
* PDE certificates for `verify_pde_solution` through `checkPde`, and for
  symbolic `d` (FEX's `Sum` forms), which needs a kernel notion of an
  `n`-fold sum.
* Polynomial-algebra witnesses (factorisation, Bézout cofactors, Gröbner
  membership) through `checkEq`, and SOS witnesses for domain conditions the
  structural sign checker cannot decide.
