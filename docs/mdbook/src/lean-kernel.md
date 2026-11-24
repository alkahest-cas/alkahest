# The reflective certificate kernel

Indefinite-integration certificates are, by default, checked by a small
verified checker in the repository's Lean library
[`Alkahest.Kernel`](https://github.com/alkahest-cas/alkahest/tree/main/lean/Alkahest)
instead of by a tree of Mathlib tactics. The certificate is *data*: an
expression AST for the claim plus a witness. A Boolean checker decides it, and
one soundness theorem, proved once in the library, turns
`check data = true` into the mathematical statement.

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
sign), `integrate`'s certificate falls back to the `HasDerivAt` tactic emitter.

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

Kernel certificates `import Alkahest.Kernel`, so build the library once:

```bash
cd lean
lake exe cache get        # Mathlib's prebuilt oleans
lake build Alkahest       # the kernel (about a minute)
lake env lean my_certificate.lean
```

## Next steps

* ODE solutions: `deriv_sound` already states `(d/dx) F = G` from the same
  checker. A residual certificate for `dsolve` composes it with `checkEq` on
  the ODE's left-hand side.
* PDE (`verify_pde_solution`) via the partial derivative `D i` and
  `D_hasDerivAt`, and FEX symbolic sums.
* Polynomial-algebra witnesses (factorisation, Bézout cofactors, Gröbner
  membership) through `checkEq`, and SOS witnesses for domain conditions the
  structural sign checker cannot decide.
