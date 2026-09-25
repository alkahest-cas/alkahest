# Error handling

Alkahest uses a structured exception hierarchy. Every error carries a stable diagnostic code, a human-readable message, an optional source span, and an optional remediation hint.

## Exception hierarchy

```
AlkahestError (base)
├── ConversionError   (E-POLY-*)   — expression → polynomial/rational conversion
├── DomainError       (E-DOMAIN-*) — mathematical side conditions violated
├── DiffError         (E-DIFF-*)   — differentiation failed
├── IntegrationError  (E-INT-*)    — integration failed
├── MatrixError       (E-MAT-*)    — matrix shape / singularity / undecidable determinant
│   ├── LinearAlgebraError (E-LINALG-*) — elimination, decompositions, canonical forms
│   └── EigenError         (E-EIGEN-*)  — eigenvalues, eigenvectors, Jordan form
├── CadError          (E-CAD-*)    — real quantifier elimination, see [Positivity](./positivity.md#decide-refuses-rather-than-guessing)
├── OdeError          (E-ODE-*)    — every ODE engine: construction, dsolve, dsolve_system, series_solve, numeric
├── TransformError    (E-TRANSFORM-*) — Laplace / Fourier / Z transforms and their inverses
├── AsymptoticError   (E-ASYMPT-*) — asymptotic expansion at infinity
├── FpsError          (E-FPS-*)    — formal power series
├── DaeError          (E-DAE-*)    — DAE structural analysis
├── SolverError       (E-SOLVE-*)  — polynomial system solving
├── JitError          (E-JIT-*)    — LLVM/JIT codegen
├── CudaError         (E-CUDA-*)   — NVPTX compile, kernel launch, or driver, see [GPU support](./gpu.md)
├── PoolError         (E-POOL-*)   — ExprPool misuse
├── AnsatzError       (E-ANSATZ-*) — ansatz family construction or fitting, see [Ansatz families](./ansatz.md)
├── CrossCheckError   (E-XCHECK-*) — cross-CAS check could not be posed, see [Cross-CAS testing](./crosscheck.md)
├── SmtError          (E-SMT-*)    — SMT-LIB export, solver run, or model lift, see [SMT bridge](./smt.md)
├── VectorError       (E-VEC-*)    — vector calculus over an orthogonal chart, see below
├── QuaternionError   (E-QUAT-*)   — quaternion algebra and rotations, see below
├── ProbabilityError  (E-PROB-*)   — distributions, expectations, generating functions, entropy, see below
├── GroupError        (E-GRP-*)    — permutation groups: orbits, Schreier–Sims, membership
├── FunctionFieldError (E-FFLD-*) — divisors, Pic⁰ and Riemann–Roch on algebraic curves, see below
├── ThetaError        (E-THETA-*)  — Riemann theta, modular and Weierstrass functions as rigorous enclosures
└── BudgetExceededError (E-BUDGET-*) — budget/cancellation trip, see [Budgets](./budgets.md)
```

## Error attributes

Every exception instance exposes:

| Attribute | Type | Description |
|---|---|---|
| `.code` | `str` | Stable diagnostic code, e.g. `"E-POLY-001"` |
| `.remediation` | `str \| None` | What the user should try |
| `.span` | `tuple[int, int] \| None` | Character offset range in source expression |

There is no `.message`. The human-readable description is `str(e)`, which carries the
code, the message and the remediation on separate lines; `e.args[0]` is the same
string. Branch on `.code`, print `str(e)`.

```python
import alkahest
from alkahest import ExprPool, UniPoly, ConversionError

pool = ExprPool()
x = pool.symbol("x")

try:
    # sin(x) cannot be represented as a polynomial
    p = UniPoly.from_symbolic(alkahest.sin(x), x)
except ConversionError as e:
    print(e.code)          # E-POLY-006
    print(str(e))          # "[E-POLY-006] function 'sin' cannot appear in a polynomial
                           #  Remediation: not a polynomial; wrap in the function only
                           #  after rational reduction"
    print(e.remediation)   # "not a polynomial; wrap in the function only after
                           #  rational reduction"
```

## Common errors and remediations

### ConversionError (E-POLY-*)

Raised when an expression cannot be converted to a polynomial or rational function.

| Code | Cause | Remediation |
|---|---|---|
| `E-POLY-001` | A symbol the conversion could not place | Remove it, or declare it as a parameter |
| `E-POLY-002` | A coefficient is not a rational integer | Rationalize, or substitute |
| `E-POLY-003` | A negative or non-integer exponent | Only non-negative integer exponents are supported |
| `E-POLY-004` | Degree ceiling exceeded | Reduce the degree, or use the sparse representation |
| `E-POLY-005` | Symbolic exponent (a variable in the exponent) | Substitute a concrete integer first |
| `E-POLY-006` | A non-polynomial **function** in the input (e.g. `sin`) | Use `Expr` directly, or expand as a series first. This — not `E-POLY-001` — is what `UniPoly.from_symbolic(sin(x), x)` raises |
| `E-POLY-007` | The denominator is zero | Ensure it is non-zero before converting |
| `E-POLY-008` … `E-POLY-010` | `FactorError`, not `ConversionError`: factoring the zero polynomial, or a bad modulus | — |

### DomainError (E-DOMAIN-*)

Raised when a mathematical side condition is violated.

| Code | Cause | Remediation |
|---|---|---|
| `E-DOMAIN-001` | Division by zero | Check denominator before dividing |
| `E-DOMAIN-002` | `log(0)` or `log(negative)` | Ensure argument is positive; use complex domain if needed |
| `E-DOMAIN-003` | `sqrt(negative)` | Use `AcbBall` or declare complex domain |

### IntegrationError (E-INT-*)

| Code | Cause | Remediation |
|---|---|---|
| `E-INT-001` | No integration rule matches | Result may not have an elementary antiderivative |
| `E-INT-002` | Algebraic extension required | Planned for v1.1 (algebraic Risch) |
| `E-INT-003` | Risch gave up (transcendental tower too deep) | Try numerical integration |

### SolverError (E-SOLVE-*)

| Code | Cause | Remediation |
|---|---|---|
| `E-SOLVE-001` | An equation is not polynomial *or rational* in the declared variables — a transcendental function, a symbolic exponent, or a symbol the conversion could not place | Restate it as a ratio of polynomials in the unknowns and parameters. (An *inconsistent* system is not an error: it returns an empty list) |
| `E-SOLVE-002` | High-degree univariate factor (> 2) | Symbolic solution not supported; use numerical solve |
| `E-SOLVE-003` | Gröbner basis did not terminate | Increase node/iteration limits |
| `E-SOLVE-005` | An equation's denominator is identically zero (`1/(x - x)`), so it denotes no function and has no solution set | Check the equation for a subtraction that cancels |

### PslqError (E-PSLQ-*)

Raised by `alkahest.guess_relation`, the augmented-lattice integer-relation heuristic, and
reported without an exception by `alkahest.relation_confidence`.

| Code | Cause | Remediation |
|---|---|---|
| `E-PSLQ-001` | Fewer than two constants supplied | Pass at least two constants that might admit a linear dependence |
| `E-PSLQ-002` | Every constant truncated to zero at the working precision | Use higher precision, or supply the constants as decimal strings |
| `E-PSLQ-003` | Working precision below the engine's 64-bit floor | Allocate at least 64 MPFR bits; ≈664 bits ≈ 200 decimal digits |
| `E-PSLQ-004` | The relation found is **larger than the inputs' precision can justify** — it was purchasable from the available digits and is evidence of nothing | Supply the constants at the precision they were computed to, declare their real accuracy with `digits=`, or pass `check_precision=False` to accept the relation unjudged |
| `E-PSLQ-005` | The constants are exact rationals and the relation is **false for them**: `Σ aᵢ·cᵢ` evaluated in exact arithmetic is not zero | The constants are probably truncations of a numerical computation rather than the values you mean — declare their accuracy with `digits=`, or supply more of them |

`E-PSLQ-004` and `E-PSLQ-005` are raised from Python (`alkahest.guess_relation`), so they
are not in the Rust `REGISTRY`; both are subclasses of `PslqError` and are caught by
`except alkahest.PslqError`. The two are deliberately distinct: `004` is a statement about
how much precision the inputs carry, `005` is a statement about the relation itself and
does not depend on precision at all.

### PrimaryDecompositionError (E-IDEAL-*)

| Code | Cause | Remediation |
|---|---|---|
| `E-IDEAL-001` | No generators supplied | Pass at least one generator |
| `E-IDEAL-002` | Generators disagree on the variable list | Use one variable list for every generator |
| `E-IDEAL-003` | Saturation split exceeded its recursion depth | Simplify the generating set |
| `E-IDEAL-004` | FLINT could not factor a generator | Report the generating set as a minimal failing example |

### Refusals: when Alkahest declines to answer

A refusal is not a malfunction. These codes all mean *"I could not establish this, and
the alternative to saying so is a confident wrong answer"* — the outcome an unattended
loop must record as **undecided**, never as a negative result.

| Code | Class | What it means |
|---|---|---|
| `E-LINALG-010` | `LinearAlgebraError` | An entry's vanishing could be proven neither zero nor non-zero, so `rank` / `rref` / `nullspace` / `eigenvects` / `jordan_form` declined to pick a branch |
| `E-MAT-004` | `MatrixError` | Same, for a determinant: `inverse()` will not divide by something it cannot show is non-zero |
| `E-EIGEN-008` | `EigenError` / `LinearAlgebraError` | The closed-form eigenvalues were produced and then failed their own check: `Π(z − λ)` and `det(zI − A)` disagree at a sampled `z`, so the radicals are not the spectrum on the branches anything reads them on. `eigenvals` / `eigenvects` / `diagonalize` / `jordan_form` / `matrix_exp` refuse rather than return them |
| `E-LINALG-011` | `LinearAlgebraError` | `matrix_exp` will not stand behind this `e^A`. Either nothing confirmed the eigenvalue list is the spectrum — Putzer's expansion is built entirely on it, and a wrong list yields a plausible matrix rather than an error — or the candidate was produced and then disagreed with `e^A` recomputed by scaling and squaring. An eigenvalue gap `λᵢ − λⱼ` that could not be settled is *not* this: it is divided by and **reported**, as `NonZero` side conditions |
| `E-CAD-001` | `CadError` | `decide` is outside its fragment, or the only candidate solutions lie at an irrational boundary point it cannot test exactly |
| `E-SOS-002` | `SosError` | No positivity certificate of this shape at this degree — a statement about the search, not a proof that none exists. **Record it as `unknown`, never as "not SOS" or "the inequality is false":** `p` may be SOS outside the LP subcone searched, SOS at a higher `basis_degree`, or non-negative without being SOS (Motzkin). `E-SOS-003`, which carries a witness point, is the only SOS *refutation*. The message carries a `what the search actually did:` trace; lines marked `NOT SEARCHED` are budgets that fired, not searches that came up empty, and mean the corresponding basis or multiplier power was never looked at. See [Positivity certificates](./positivity.md#three-outcomes-deliberately-kept-apart) |
| `E-IDEAL-005` | `IdealRefusal` | `radical` cannot certify `√I` for this ideal. Only monomial, principal and zero-dimensional ideals — and anything whose primary decomposition is certified — are answered; the alternative is asserting `√I = I` with nothing behind it |
| `E-IDEAL-006` | `IdealRefusal` | `primary_decomposition` reached a component it cannot show is primary, so it will not report the ideal itself with an unjustified `associated_prime` |
| `E-SOLVE-004` | `TriangularizeRefusal` | `triangularize` extracted a chain that does not generate an ideal containing the input, i.e. one that cuts out a larger variety than the system. Splitting on the initials (Lazard–Kalkbrener) is not implemented |
| `E-SERIES-003` | `SeriesError` | `series` ran past its work ceiling (or an active `Budget`) before reaching the requested order. Coefficients are formed by repeated differentiation without re-simplifying, so a nested radical's derivatives grow by a constant factor each time; a *shorter* series would carry an `O(h^order)` label nothing bounded |
| `E-SERIES-004` | `SeriesError` | A `series` coefficient came out as an indeterminate form (`0/0`, `1/0`, `log(0)`) rather than a number, so the expansion point is a singularity this engine cannot resolve — a branch point (`√x`), an essential singularity (`e^{1/x}`), or a removable one it could not cancel. Returning the `Series` anyway would report success and hand back coefficients that evaluate to `NaN` |
| `E-PSLQ-004` | `PslqError` | `guess_relation` found an integer relation the inputs' precision cannot justify — pinning down `n` coefficients bounded by `H` costs about `n·log10(2H+1)` digits of agreement, and the inputs do not carry that many. **Record it as `undecided`, not as "no relation exists":** the same constants at higher precision may well admit one. `relation_confidence` reports the same judgement as data, including a three-valued `credible` whose `None` means *the inputs' precision is not knowable*, never a pass |
| `E-PSLQ-005` | `PslqError` | The constants are exact rationals and `Σ aᵢ·cᵢ` is not zero in exact arithmetic. **This one is a verdict, not a refusal** — the relation is refuted for the numbers supplied |
| `E-SERIES-005` | `SeriesError` | `experimental.puiseux_series` found no Puiseux expansion at the point at all: a logarithm of something vanishing (`log x`, `√x·log x` — a Puiseux–*log* / transseries term, which this engine has no representation for), an essential singularity (`e^{1/x}`), or a ramification index past the range an expansion can be *checked* at. Truncating `√x·log x` to `x^{1/2}` would be a wrong answer rather than a coarse one |
| `E-SERIES-006` | `SeriesError` | `experimental.puiseux_series` **computed** an expansion and then withheld it, because its verifier could not confirm it — the truncation residual did not decay at the claimed rate, or nothing near the point could be evaluated (a branch that is not real on the side sampled, a head with no numeric kernel). Distinct from `E-SERIES-005` on purpose: that one says there is nothing to return, this one says there was something and it is not trustworthy |
| `E-INT-004` | `IntegrationError` | Proven non-elementary. **This one is a verdict, not a refusal** — keep it apart from the rest |
| `E-BUDGET-001..005` | `BudgetExceededError` | Ran out of the time, steps or memory it was given, was cancelled, or is about to exhaust the process address-space limit |
| `E-TRANSFORM-004` | `TransformError` | The unilateral transform's causality hypothesis is **refuted**. `L{θ(t+1)}` is `1/s`; the shift rule would emit `e^{s}/s`, because the edge at `a = −1` lies outside the range `∫₀^∞` sees. On the inverse, an advance factor `e^{+as}` is the transform of no causal function. **A verdict, not a refusal** — there is nothing a wider table would find |
| `E-TRANSFORM-013` | `TransformError` | A Fourier table entry's decay hypothesis is **refuted**: at a non-positive rate the defining integral diverges, and a negative Lorentzian amplitude has a transform whose sign *and* direction of growth are opposite to the tabulated one. Also a verdict |
| `E-ODE-011` | `OdeError` | `dsolve` produced a candidate closed form and then withheld it, because substituting it back into the equation did not verify. Distinct from `E-ODE-010` on purpose: that one says no class matched, this one says something was found and is not trustworthy |
| `E-ODE-044` | `OdeError` | The same for `series_solve`: a candidate Frobenius series that failed the exact-residual gate |
| `E-ASYMPT-004` | `AsymptoticError` | `asymptotic_expand` computed an expansion and then withheld it, because the numeric `o()`-gate rejected every candidate term at large `x`. The function may have an oscillatory or non-power-scale tail |
| `E-PROB-005` | `ProbabilityError` | A moment, CDF, quantile, characteristic function or entropy was **computed** and then withheld, because quadrature of its own defining integral could not confirm it. The same shape as `E-ODE-011` and `E-SERIES-006`: something was found and it is not trustworthy |
| `E-PROB-003` | `ProbabilityError` | The reduction integral was built and the symbolic integrator declined it. A fact about **this integrator**, so a wider integration table would close it — the message names the integral, so it can be handed to quadrature instead |
| `E-PROB-006` | `ProbabilityError` | The defining integral **diverges**, so there is no value. **A verdict, not a refusal** — `E[e^{X²}]` under a standard normal, `E[e^{tX}]` for a log-normal at any `t > 0`, `D(P‖Q)` where `P` charges a set `Q` gives probability zero. Keep it apart from `E-PROB-003`: retrying with a wider integration table is the wrong next step. The convergence gate is *sufficient, not complete* — a divergence it cannot see still surfaces as `E-PROB-003` |

`E-SERIES-003` and `E-SERIES-004` travel out of band for the same reason (`SeriesError` is
exhaustive) but *are* wired into the bindings: `series` returns `SeriesError::InvalidOrder`
with `calculus::series::take_series_refusal()` pending, and the Python layer raises
`SeriesError` with `.code == "E-SERIES-003"` / `"E-SERIES-004"` — or `BudgetExceededError`
when a budget was what stopped it. `SeriesRefusal::cause()` distinguishes the two in Rust.

`E-SERIES-005` and `E-SERIES-006` come from a different type — `calculus::puiseux::PuiseuxError`,
which is `#[non_exhaustive]` from birth and so needs no out-of-band channel — but they raise the
same Python `SeriesError`, because `puiseux_series` is the series engine widened rather than a
new subsystem, and `except SeriesError` should keep covering both entry points. A budget trip
inside `puiseux_series` raises `BudgetExceededError`, not `E-SERIES-003`: "raise your budget"
and "this expansion does not close" are different problems.

A **removable** singularity is not in that list, because it is no longer refused: `series`
puts the expression over a common denominator and divides the two power series rather than
substituting the expansion point into a quotient, so `sin(x)/x` gives `1 − x²/6 + O(x⁴)` and
`1/sin(x)` gives `x⁻¹ + x/6 + O(x)`. `E-SERIES-004` is what is left when that does not
apply.

`E-EIGEN-008` travels out of band on the same arrangement: `EigenError` and
`LinearAlgebraError` are exhaustive, so the refusal is returned inside
`UnsupportedIrreducibleDegree` — whose text now states the disjunction it covers — with
`matrix::take_spectrum_refusal()` pending, and the Python bindings raise `EigenError` /
`LinearAlgebraError` carrying `.code == "E-EIGEN-008"` and the offending `λ` list.

`E-IDEAL-005`, `E-IDEAL-006` and `E-SOLVE-004` are new in 3.8 and travel **out of band**:
`PrimaryDecompositionError` and `SolverError` are public exhaustive enums that cannot gain
a variant in a patch release, so the refusal is returned inside an existing variant and the
real code is available from `ideal::take_ideal_refusal()` /
`solver::regular_chains::take_triangularize_refusal()`. The Python bindings consult both, so
`radical` and `primary_decomposition` raise `AlkahestError` with `.code == "E-IDEAL-005"` /
`"E-IDEAL-006"`, and `triangularize` raises `SolverError` with `.code == "E-SOLVE-004"`.
`AlkahestError` subclasses `ValueError`, so code that catches `ValueError` is unaffected.

`E-SOLVE-005` uses the same channel for the same reason: `solve` returns
`SolverError::NotPolynomial` with `solver::take_undefined_equation()` pending, and the
Python binding raises `SolverError` with `.code == "E-SOLVE-005"`. It means the equation's
denominator vanishes identically, so it is not a rational function of anything and there is
no solution set to report — distinct from `E-SOLVE-001`, which means the equation is a
perfectly good function that is merely not polynomial or rational in the declared unknowns.

The takers are *consuming*, which is what keeps the carrier variant honest: a genuinely
non-polynomial equation still reports `E-SOLVE-001`, because no refusal is pending for it.
Both readings of the shared variant stay distinguishable.

The three-valued zero test behind `E-LINALG-010` / `E-MAT-004` is new in 3.8. Before it,
"could not prove `det ≠ 0`" was silently read as "`det = 0`", and `Matrix.nullspace()`
returned a confident wrong basis for any 2×2 with a symbolic determinant.

```python
import alkahest as ak

pool = ak.ExprPool()
a = pool.symbol("a")
zero, one = pool.integer(0), pool.integer(1)

# `mystery` has no evaluation rule, so its vanishing is genuinely undecidable.
opaque = pool.func("mystery", [a])
m = ak.Matrix([[opaque, zero], [zero, one]])

try:
    m.inverse()
except ak.MatrixError as e:
    print(e.code)          # E-MAT-004
    print(e.remediation)   # substitute concrete values for the parameters

try:
    ak.Matrix([[opaque, zero], [zero, zero]]).nullspace()
except ak.LinearAlgebraError as e:
    print(e.code)          # E-LINALG-010
```

`LinearAlgebraError` and `EigenError` are both subclasses of `MatrixError`, so
`except ak.MatrixError` catches all three families; catch the subclass when you want to
distinguish them. Note that `eigenvects()` raises `EigenError` — with code
`E-LINALG-010`, because the code identifies *what could not be decided*, not which
wrapper it surfaced through.

## Catching errors by code

For programmatic error handling:

```python
try:
    result = alkahest.integrate(expr, x)
except alkahest.AlkahestError as e:
    if e.code.startswith("E-INT-"):
        print(f"Integration failed: {e.remediation}")
    else:
        raise
```

## Error taxonomy

Every error is classified on two independent axes: **subsystem** (determines the code prefix and exception class) and **cause** (informs the remediation hint).

### Subsystem axis

| Prefix | Class | Scope |
|---|---|---|
| `E-POLY-*` | `ConversionError` | Expression → polynomial/rational-function conversion |
| `E-DOMAIN-*` | `DomainError` | Side-condition violations (div-by-zero, log of 0, `sqrt` of negative) |
| `E-DIFF-*` | `DiffError` | Forward/reverse differentiation, unknown derivatives |
| `E-INT-*` | `IntegrationError` | Symbolic integration (Risch, heuristic, table) |
| `E-MAT-*` | `MatrixError` | Matrix shape, proven-singular, non-invertible, and (`E-MAT-004`) an undecidable determinant |
| `E-LINALG-*` | `LinearAlgebraError` *(subclass of `MatrixError`)* | Elimination, decompositions, canonical forms; `E-LINALG-010` is the undecidable-entry refusal |
| `E-EIGEN-*` | `EigenError` *(subclass of `MatrixError`)* | Eigenvalues, eigenvectors, Jordan form, diagonalisation |
| `E-CAD-*` | `CadError` | Real quantifier elimination — outside the fragment, or an untestable irrational boundary point |
| `E-ODE-*` | `OdeError` | Every ODE engine, one class per prefix: construction and lowering (`001`–`003`), `dsolve` (`010`–`014`), the numeric integrators (`020`–`026`), `dsolve_system` (`030`–`034`), and `series_solve` (`040`–`045`, moved off `020`–`025` in 3.10 where they collided with the numeric block) |
| `E-TRANSFORM-*` | `TransformError` | Laplace (`00x`), Fourier (`01x`) and Z (`10x`) transform tables and their inverses. `E-TRANSFORM-004` and `E-TRANSFORM-013` are **refuted hypotheses**, not table gaps — see below |
| `E-ASYMPT-*` | `AsymptoticError` | `experimental.asymptotic_expand`; `E-ASYMPT-004` is an expansion the numeric `o()`-gate could not confirm, withheld rather than returned |
| `E-FPS-*` | `FpsError` | Formal power series (`experimental.Fps`) — a pole at the origin (`001`/`002`), a non-rational coefficient (`003`), or one of the constant-term hypotheses `f(0) = 0` / `1` / `≠ 0` that make composition, `log` and the inverse well defined (`004`–`006`) |
| `E-DAE-*` | `DaeError` | DAE structural analysis (Pantelides, index reduction) |
| `E-SOLVE-*` | `SolverError` | Polynomial system solving, Gröbner basis |
| `E-LAT-*` | `LatticeError` (`001`–`004`), `LatticeGeometryError` (`005`–`014`) | Lattices over ℤ. `001`–`004` are LLL reduction (empty/ragged basis, `δ` outside `(¼, 1)`, iteration guard) and stay on the stable `LatticeError`. The experimental lattice toolkit raises `LatticeGeometryError`, which is `#[non_exhaustive]` and **wraps** a `LatticeError` when a reduction under it fails, so `001`–`004` reach a caller unchanged; in Python it is a *subclass* of `LatticeError`, so one `except` still catches both. `005`–`007` reject a Gram matrix that is not square, not symmetric or not positive definite; `008`/`009` are the exact-enumeration refusals (rank above the ceiling, node budget exhausted) and are **not** an invitation to approximate — there is no heuristic SVP/CVP here; `010` needs an integral Gram matrix for a theta series, `011` is a vector of the wrong length, `012` an out-of-range constructor parameter, `013` an operation needing ambient coordinates on a Gram-only lattice, and `014` an internal invariant, reported rather than panicked because these run under a PyO3 boundary |
| `E-PSLQ-*` | `PslqError` | Integer-relation search (`guess_relation`); `E-PSLQ-004` is the input-precision refusal and `E-PSLQ-005` the exact refutation |
| `E-JIT-*` | `JitError` | LLVM/Cranelift codegen and linking |
| `E-CUDA-*` | `CudaError` | NVPTX compile, kernel launch, driver/runtime failures |
| `E-POOL-*` | `PoolError` | `ExprPool` misuse (closed, cross-pool, persisted-handle mismatch) |
| `E-PARSE-*` | `ParseError` *(reserved)* | Parser integration — owns `span()` by default |
| `E-IO-*` | `IoError` *(reserved)* | Checkpoint/serde paths (`PoolPersistError`) |
| `E-CERT-*` | `CertificateUnavailableError` | A Lean certificate was required but withheld |
| `E-BUDGET-*` | `BudgetExceededError` | Budget/cancellation trip — see [Budgets, cancellation, and determinism](./budgets.md) |
| `E-ANSATZ-*` | `AnsatzError` | Ansatz family construction and fitting — see [Ansatz families](./ansatz.md) |
| `E-XCHECK-*` | `CrossCheckError` | Cross-CAS differential testing — see [Cross-CAS testing](./crosscheck.md) |
| `E-SMT-*` | `SmtError` | SMT-LIB export, solver invocation, model lift — see [SMT bridge](./smt.md) |
| `E-RESIDUE-*` | `AlkahestError` | `residue` — not a rational function, zero denominator, pole order out of range, or (`E-RESIDUE-005`) a point that is not an exact constant in ℚ(i) |
| `E-VEC-*` | `VectorError` | Vector calculus over an orthogonal chart — a non-differentiable component, a repeated or non-symbol coordinate, a chart that could not be *proven* orthogonal (`E-VEC-004`), or a degenerate scale factor (`E-VEC-005`). See [Vector calculus and quaternions](#vector-calculus-and-quaternions) |
| `E-QUAT-*` | `QuaternionError` | Quaternion algebra and rotations — a zero or undecided norm (`E-QUAT-001`), the axis of the identity rotation, which does not exist (`E-QUAT-002`), or a matrix that could not be checked to be a proper rotation (`E-QUAT-003`) |
| `E-GFQ-*` | `FiniteFieldError` | Linear algebra over the finite fields GF(q), q = p^k (`alkahest.experimental.FiniteField` / `GfMatrix`). `001` a non-prime characteristic and `002` one past a machine word — both refusals, because ℤ/nℤ for composite n has zero divisors and no well-defined rank; `004` a reducible defining polynomial; `009` a singular matrix; `010` a linear system with no solution, refused rather than approximated |
| `E-CODE-*` | `CodingError` | Classical linear codes over GF(q), weight enumerators, MacWilliams, Krawtchouk polynomials and the Delsarte LP bound (`alkahest.experimental.LinearCode` / `delsarte_lp_bound`). Two matter most. `E-CODE-004` refuses a codeword enumeration past its hard cap rather than truncating it — the minimum weight of *some* codewords is an upper bound on `d` wearing `d`'s name. `E-CODE-007` **withholds** an LP bound whose dual certificate failed its own exact feasibility check, because an upper bound that came out too small "rules out" codes that exist. The rest: `001` a zero-length code, `002` a distance outside `1..=n`, `003` a GF(q) refusal passed through with its own `E-GFQ` code kept visible in the message, `005` a length past the LP cap, `006` a vector that is not the weight distribution of a linear code, `008` an unusable alphabet size. See [Certified coding bounds](#certified-coding-bounds) |
| `E-NUMF-*` | `NumberFieldError` | Algebraic number fields ℚ[x]/(f) and the cyclotomic fields ℚ(ζ_n) (`alkahest.experimental.NumberField`). `E-NUMF-003` is the one that matters: the defining polynomial is **checked** for irreducibility and a reducible one is refused, because ℚ[x]/(f) for reducible f is a ring with zero divisors in which `inverse` has no answer and `norm` is not multiplicative — the error names a proper factor. `005` zero has no inverse; `006` two fields with different canonical defining polynomials, which are not interchangeable even when isomorphic; `008` ℚ(ζ_n) with φ(n) past the degree cap. Note `polynomial_discriminant` is the discriminant of the **defining polynomial**, not the field discriminant — the ring of integers is not computed |
| `E-GRP-*` | `GroupError` | Permutation groups (`alkahest.experimental`) — an images array that is not a bijection (`E-GRP-001`), a degree mismatch, which is never repaired by padding with fixed points (`E-GRP-002`), a point outside `0..degree` — points are **0-based** here (`E-GRP-003`), a group too large to list element by element, whose order is still exact (`E-GRP-004`), a degree above the Schreier–Sims memory limit (`E-GRP-005`), or a standard family asked for below the `n` where its degree-`n` action is faithful, e.g. `dihedral(2)` (`E-GRP-006`) |
| `E-MATGRP-*` | `MatGroupError` | Matrix groups over GF(q) built from **arbitrary generators** (`alkahest.experimental.MatGroup`) — exact order and membership from a Schreier–Sims base and strong generating set on the action on vectors, orbits on vectors and on projective points, random elements by product replacement, derived subgroup, centre and normal closure. `#[non_exhaustive]`, and it **wraps** `FiniteFieldError` and `GroupError`, so an `E-GFQ-*` or `E-GRP-*` raised underneath reaches the caller with its own code rather than a relabelled one; in Python it is a subclass of `AlkahestError` like the rest. `E-MATGRP-007` is the one to read twice: Schreier–Sims that exhausts its work budget **refuses**, because a chain missing a level reports the product of the orbits it did build — a proper *divisor* of `|G|`, indistinguishable from the right answer for a smaller group. `E-MATGRP-006` refuses an orbit past its cap for the same reason, and because an orbit on vectors is bounded by `q^d − 1` that cap is on `q^d` and **not** on `|G|`: `|Sp(12,3)| ≈ 10^40` is fine while `GL(2, 4096)` is not. `E-MATGRP-008` withholds the element *list* while `order()` stays exact. The rest: `001` a non-square matrix, `002` a degree mismatch — vectors are `1 × d` **rows**, because the action is `v ↦ v·M`, `003` a matrix over a different field, `004` a singular generator, `005` a degree above the ceiling, `009` a `q` too large for the operations that enumerate GF(q) itself, `010` a family with no constructor here (`GU`, `SO` and the twisted types; build them with `MatGroup` from your own generators), `011` a centralizing algebra too large to enumerate for the centre, `012` the zero vector, which spans no projective point, `013` an internal invariant, reported rather than panicked because these run under a PyO3 boundary |
| `E-FPGRP-*` | `FpGroupError` | Finitely presented groups `⟨X \| R⟩`, Todd–Coxeter coset enumeration, Reidemeister–Schreier and low-degree group cohomology (`alkahest.experimental.FpGroup`). **`E-FPGRP-004` and `E-FPGRP-005` are different facts and must not be conflated.** `004` means coset enumeration hit its cap: the word problem for finitely presented groups is undecidable, so it says *only* "I did not finish" — it is **not** a claim that the group is infinite, nor that it is finite. `005` means the group **is** infinite, proved, because its abelianisation has an infinite cyclic factor, which is a terminating Smith normal form. `⟨a, b \| a², b³, (ab)⁷⟩` is the (2,3,7) triangle group — infinite, with a *trivial* abelianisation — and therefore lands on `004`, which is the honest answer. The rest: `001`–`003` a word letter outside the rank (letters are signed and **1-based**), a malformed alphabet, and relator syntax; `006` a coset cap above what the build will allocate and `007` a coset outside `0..index` (cosets are **0-based**, and coset 0 is the subgroup `H`); `008` a cohomological degree above 2 and `009` a group or module above the cohomology's size limits, refused rather than run for an hour; `010` a malformed module; `011` an "action" that is not one, because some relator does not act as the identity on `M` — refused rather than silently used; `012` an internal invariant, reported rather than panicked because these run under a PyO3 boundary |
| `E-PROB-*` | `ProbabilityError` | `alkahest.experimental`'s distribution surface — laws, expectations, moments, characteristic and generating functions, entropy and KL divergence. `E-PROB-005` and `E-PROB-006` are the two to branch on: one is a closed form that was **withheld**, the other says the quantity **does not exist**. See [Probability: four ways not to answer](#probability-four-ways-not-to-answer) |
| `E-FFLD-*` | `FunctionFieldError` | `alkahest.experimental`'s function-field surface — divisors, the divisor class group `Pic⁰`, torsion order and Riemann–Roch. The implemented class is the **imaginary hyperelliptic** model `y² = a(x)` with `a` squarefree of **odd** degree and **ℚ-rational places**, because that is what the Mumford/Cantor machinery reused from the algebraic integrator is scoped to; every boundary outside it is one of these codes rather than a guess. Three to keep apart: `E-FFLD-007` is a **verdict** (the class has infinite order), `E-FFLD-006` is the matching **undecided**, and `E-FFLD-011` is an answer that was computed and then **withheld** for failing its own check. The most common one in practice is `E-FFLD-003`: the divisor has a place of degree ≥ 2, which cannot be represented. See [Function fields: what is modelled](#function-fields-what-is-modelled) |
| `E-STAB-*` | `StabilizerError` | `alkahest.experimental`'s symplectic / stabilizer surface — the `(x \| z)` binary symplectic form, Pauli operators, stabilizer and CSS codes, and the classical matrix groups `GL`/`SL`/`Sp` over GF(q). `E-STAB-004`, `-005` and `-006` are **structural**: the generators do not define a stabilizer code at all (they anticommute, fail `H_X · H_Zᵀ = 0`, or multiply to `−I`). `E-STAB-008` is the exhaustive minimum-distance search past its cap — minimum distance is NP-hard and nothing here approximates it; the answer on offer instead is a `Distance` with `exact == False`. `E-STAB-013` is a result computed, failed against its own invariant, and withheld. A failure raised inside the GF(q) layer keeps its own `E-GFQ-*` code rather than being relabelled. See [Stabilizer codes: the `(x \| z)` convention](#stabilizer-codes-the-x--z-convention) |
| `E-THETA-*` | `ThetaError` | `alkahest.experimental`'s theta surface — genus-`g` Riemann theta `θ[a;b](z, τ)`, the classical modular functions `η`, `j`, `λ`, `Δ`, the Jacobi theta functions and the Weierstrass family, each returned as a **ball** carrying its own error bound. `E-THETA-010` is the one to branch on: a value was computed and then **withheld** because its enclosure was too wide to read, and its `achieved_bits` field separates "needs more precision" from "is exactly zero, and so has no relative accuracy at any precision" (`j(ρ)`, `θ₁(0, τ)`) — use `Precision.bits(...)` and `ComplexBall.contains_zero()` for those. `E-THETA-007` / `E-THETA-009` mean `Im(τ)` could not be **proved** positive (definite), never that it was disproved. `E-THETA-001` is a FLINT too old to carry Arb (< 3.1) or `acb_theta` (< 3.2) |
| `E-LIMIT-*` | `LimitError` | `limit` could not be established; `E-LIMIT-006` is a limit that turns on the sign of a free parameter nothing states — assume it, or declare the symbol `Domain.Positive` |
| `E-SERIES-*` | `SeriesError` | `series` and `experimental.puiseux_series`. `003` a work ceiling, `004` an indeterminate coefficient, `005` no Puiseux expansion exists, `006` one computed and withheld |
| `E-SUM-*` | `SumError` | Symbolic summation (`sum_indefinite`, `sum_definite`) — not hypergeometric, or not Gosper-summable |
| `E-PROD-*` | `ProductError` | Symbolic discrete products (`product_indefinite`, `product_definite`) |
| `E-REC-*` | `LinearRecurrenceError` | `solve_linear_recurrence_homogeneous` |
| `E-RSOLVE-*` | `RsolveError` | Difference equations (`rsolve`) |
| `E-HOLO-*` | `HolonomicError` | One prefix, five engines: `001`–`008` single-index `zeilberger` plus modular / `p`-adic evaluation, `020`–`024` `q_zeilberger`, `040`–`042` `experimental.telescope2d` / `telescope_md`, `060`–`064` the continuous (Almkvist–Zeilberger) engine, which has no Python entry point yet. See [Creative telescoping](./telescoping.md) |
| `E-VALIDATED-*` | `ValidatedError` | Rigorous Taylor-model bounds. **Every variant is a refusal, never a guess** — see [Rigorous global bounds](./validated-bounds.md) |
| `E-NT-*` | `NumberTheoryError` (`001`–`005`), `ArithmeticError` (`006`) | FLINT-backed integer number theory (`alkahest.number_theory`), including the classical arithmetic functions, which are exposed under `alkahest.experimental` — the partition function, Bernoulli, Euler, Stirling and harmonic numbers, Möbius μ, σ_k and sums of squares. Those refuse with `ArithmeticError`, which is `#[non_exhaustive]` and **wraps** a `NumberTheoryError` when the argument is out of domain, so `001`/`002` reach a caller unchanged; in Python it is a *subclass* of `NumberTheoryError`, so one `except` still catches both. `E-NT-006` is a **work cap**, not a claim that the value does not exist: `p(10^9)` is a perfectly good integer, this module simply will not spend unbounded time on a call that looked cheap. Note the Bernoulli convention is `B₁ = −1/2` |
| `E-MOD-*` | `ModularError` | Modular / CRT reconstruction (`alkahest.modular`) |
| `E-DIOPH-*` | `DiophantineError` | Integer Diophantine solving — linear and quadratic patterns |
| `E-ROOT-*` | `RealRootError` | Real root isolation (VAS) |
| `E-RES-*` | `ResultantError` | Resultants and the subresultant PRS |
| `E-INTERP-*` | `SparseInterpError` (`001`–`004`), `SparseGcdError` (`010`–`012`) | Sparse multivariate interpolation and sparse modular GCD |
| `E-HOMOTOPY-*` | `HomotopyError` | Numerical polynomial continuation (`solve(..., method="homotopy")`) |
| `E-PARAMGB-*` | `ParamGroebnerError` | Gröbner bases over `Q(params)` — `GroebnerBasis.compute(..., params=[...])` and `experimental.ParametricGroebnerBasis` |
| `E-SIMPLIFY-*` | `AssumptionError` | An explicit simplification assumption contradicted the active context |
| `E-DEPTH-*` | `DepthLimitError` | The expression nesting ceiling. A refusal rather than letting a recursive walk overflow the native stack, which would be a `SIGSEGV` and not an exception |

Four registry prefixes are **Rust-side only** and have no Python entry point that
raises them today: `E-LOGIC-*` (`LogicError`), `E-NFM-*` (`NormalFormError`),
`E-DIFFALG-*` (`DiffAlgError`) and `E-HOLO-060…064` (`DiffTelescopingError`).
`E-EVAL-*` has no class of its own — the bindings surface it as `DomainError`,
`E-EVAL-009` being "undefined at this point", which is a **verdict**. `E-POLY-008`
… `E-POLY-010` belong to `FactorError` rather than `ConversionError`, which shares
the prefix.

`scripts/check_error_codes.py` checks that this table names every prefix in the Rust
`REGISTRY`, so a new subsystem cannot land undocumented.

### Transforms: a table gap is not a refuted hypothesis

`E-TRANSFORM-*` covers three tables under one prefix and one class — Laplace
(`00x`), Fourier (`01x`), Z (`10x`) — the way `E-ODE-*` covers five ODE engines
under `OdeError`. The number says which table and which failure; a caller who
wants only one of them filters on the code.

The split that matters is not between the tables but *inside* each of them:

| Code | Reading |
|---|---|
| `E-TRANSFORM-001` / `011` / `101` | No forward rule matched. A fact about **this implementation** — these are table-driven, and a wider table would close it |
| `E-TRANSFORM-002` / `102` | The inverse table does not reach this form. Same kind of fact |
| `E-TRANSFORM-003` / `012` / `103` | The two variables passed are the same symbol |
| `E-TRANSFORM-004` | The unilateral (causality) hypothesis is **refuted** |
| `E-TRANSFORM-013` | A Fourier entry's decay hypothesis is **refuted** |

"Not in the table yet" and "no such transform exists" are opposite instructions.
The first says rewrite the input or wait for a wider table; the second says stop
looking. Before 3.10 all of these arrived as a bare `ValueError` with no `.code`
at all, so telling them apart meant matching English prose.

A **symbolic** parameter is reported as neither. `θ(t−a)` with a symbolic `a`
cannot be decided, so the hypothesis `a ≥ 0` is recorded rather than assumed or
refused. It is **not** a field on the returned value — these functions return a
bare `Expr`, and there is nowhere in band to hang a hypothesis — so read it from
the thread-local `experimental.transform_side_conditions()`, which describes the
most recent `laplace_transform`, `inverse_laplace_transform`,
`fourier_transform`, `inverse_fourier_transform` or `inverse_z_transform` on this
thread whether it returned or raised:

```python
import alkahest as ak
from alkahest import experimental as ex

pool = ak.ExprPool()
t, s = pool.symbol("t"), pool.symbol("s")

# A literal negative edge: refuted, and nothing to find.
theta = pool.func("heaviside", [t + pool.integer(1)])
try:
    ex.laplace_transform(theta, t, s)
except ak.TransformError as e:
    print(e.code)          # E-TRANSFORM-004
    print(e.remediation)   # "the unilateral transform integrates over t ≥ 0 only, ..."

# The control: shift it the other way and the rule applies.
ex.laplace_transform(pool.func("heaviside", [t - pool.integer(1)]), t, s)

# Undecidable: answered, with the hypothesis stated out of band.
a = pool.symbol("a")
ex.laplace_transform(pool.func("heaviside", [t - a]), t, s)   # e^{-a·s}/s
print(ex.transform_side_conditions())                          # ['a ∈ NonNegative']
```

An empty list means every branch taken was forced by the input, not that none was
taken.

### Probability: four ways not to answer

`E-PROB-*` is one prefix over one class, and the six numbers exist because
"there is no closed form", "*this* integrator could not find one", "the
quantity does not exist" and "one was found and is not trustworthy" call for
four different next steps. Collapsing them would send a caller looking for a
better integrator when the integral diverges.

| Code | Reading | What to do next |
|---|---|---|
| `E-PROB-001` | A parameter is a **number** outside the law's own constraint (`sigma <= 0`, `a >= b`, `p` outside `[0, 1]`) | Fix the parameter. A *symbolic* parameter is never reported here — it cannot be decided, so it is carried on `Distribution.constraints()` for the caller to discharge |
| `E-PROB-002` | Outside the modelled class: a product of two variates (there are no joint laws here), a `probability_generating_function` for a law that is not on the non-negative integers, a cross-family KL divergence | Restate the query, or do the joint-law step yourself |
| `E-PROB-003` | The reduction integral was built and the symbolic integrator declined it. A fact about **this implementation** | The message names the integral — hand it to quadrature |
| `E-PROB-004` | No closed form exists inside this library's primitive set: the `Gamma` CDF at non-integer shape (incomplete gamma), the `Beta` CDF (incomplete beta), the normal quantile (`erf⁻¹`), the log-normal characteristic function (none at all) | Use an integer shape parameter where that collapses to a finite sum, or go numeric |
| `E-PROB-005` | A closed form **was computed** and is being **withheld**, because quadrature of its own defining integral did not confirm it | Record it as undecided and report the query. Never a warning: a caller cannot tell a checked value from an unchecked one once it is in hand |
| `E-PROB-006` | The quantity **does not exist** | A verdict. Stop looking |

Two of these are the reason the surface exists at all. `moment_generating_function`
for a `LogNormal` raises `E-PROB-006`: `E[e^{tX}]` is `+inf` for every `t > 0`, and a
CAS that completes the square anyway hands back a clean closed form that is the value
of no integral. And `M_X(t) = lambda/(lambda − t)` for an `Exponential` holds *only*
on `t < lambda` — outside that strip the expression is still finite, still plausible
and still wrong — so a decidable argument outside the strip raises `E-PROB-006` and an
undecidable one publishes the condition on `experimental.prob_side_conditions()`.

```python
import alkahest as ak
from alkahest import experimental as ex

pool = ak.ExprPool()
t, lam = pool.symbol("t"), pool.symbol("lam")

ex.Exponential(lam).moment_generating_function(t)   # lam/(lam - t)
print(ex.prob_side_conditions())
# the convergence strip and the parameter constraint: ['lam - t > 0', 'lam > 0']

try:
    ex.LogNormal(pool.integer(0), pool.integer(1)).moment_generating_function(t)
except ak.ProbabilityError as e:
    print(e.code)        # E-PROB-006
```

`prob_side_conditions()` is the same out-of-band channel as
`transform_side_conditions()`, for the same reason: the return value is an `Expr`,
so a conditional answer and a theorem look identical at the call site unless the
hypotheses are published somewhere. A law whose MGF is entire leaves the list
empty — so empty means *checked*, not *unexamined*.

### Vector calculus and quaternions

`alkahest.experimental`'s rigid-body layer refuses in three places where a clean,
plausible number is available, and that is the point of each of them.

| Code | Raised by | Why a refusal rather than an answer |
|---|---|---|
| `E-VEC-004` | `Coordinates.from_embedding` | Every `grad`/`div`/`curl`/`∇²` formula in the module is derived for an **orthogonal** frame. On a skew chart they still evaluate, to an expression that looks like a divergence and is not one. So the tangent inner products must be *proven* to vanish; undecided is a refusal, not an assumption |
| `E-VEC-005` | `Coordinates.from_embedding` | A scale factor that is identically zero — or whose non-vanishing could not be established — is divided by in every operator |
| `E-QUAT-002` | `Quaternion.to_axis_angle` | The identity rotation has no axis: *every* unit vector is one. The conventional stand-in `(0, 0, 1)` is a stated answer to a question with no answer, and nothing downstream can tell it from a real axis |
| `E-QUAT-003` | `Quaternion.from_rotation_matrix` | Shepperd's method returns a perfectly ordinary unit quaternion for a reflection, a scaled matrix or a shear — representing some *other*, proper rotation. So `RᵀR = I` and `det R = +1` are checked, the recovered quaternion is required to reproduce the matrix, and a **symbolic** matrix refuses outright, because the branch selection is a comparison between entries and there is none to make on a symbol |

The vector Laplacian is a fourth trap without an error code, because there is a right
answer: `experimental.vector_laplacian` is `∇(∇·F) − ∇×(∇×F)`, which equals the
componentwise scalar Laplacian in Cartesian coordinates **only**. Applying
`experimental.laplacian` to each physical component of a cylindrical or spherical field
silently drops the terms that come from the basis turning — `∇²(φ̂)` is `−φ̂/ρ²`, not `0`.

`E-RESIDUE-005` is raised only at the Python boundary — the Rust `residue` takes an
already-parsed point and cannot reach that state — so it is deliberately absent from
`alkahest-core`'s `REGISTRY`, on the same footing as `E-SMT-001`/`003`/`004` in
`alkahest/smt.py` and `E-BATCH-001` in `alkahest/_batch.py`. It exists because
`residue(f, z, a)` with a symbolic `a` reads perfectly well and used to escape as a
bare `AttributeError` naming an attribute of the implementation, which is not an
`AlkahestError` and so was invisible to `except ak.AlkahestError`.

Three of these describe outcomes that are **results rather than malfunctions**, and
the wording of each is deliberate. `E-ANSATZ-003` means *no member of this family
satisfies the constraints* — for a search loop that is a closed branch worth
recording, not a failure. `E-XCHECK-002` means no oracle is installed, and exists
so that a missing oracle can never be mistaken for agreement. `E-SMT-003` refuses
a model containing an algebraic number that cannot be lifted exactly, rather than
truncating it to a float — a float witness recorded as an exact one is precisely
the silent-error shape these subsystems exist to prevent.

### `E-CERT-*` — certificate policy

| Code | Meaning | Remediation |
|---|---|---|
| `E-CERT-001` | A result was required to carry a Lean certificate and none was available | Pick a certifiable route — see [Certificate coverage](./certificate-coverage.md) and `alkahest.certifiable()` — or drop the requirement |

This one is unusual: the computation *succeeded*. What is missing is the
machine-checkable evidence, so it is a policy failure rather than a
mathematical one. It is raised only when you ask for it, by
`alkahest.require_certificate(result)` or ambiently inside
`with alkahest.context(require_certificate=True):`. The remediation names the
blocking rewrite rules where they can be identified.

```python
import alkahest as ak

p = ak.ExprPool()
x = p.symbol("x")

with ak.context(require_certificate=True):
    ak.diff(ak.sin(x), x)              # fine — certifies
    ak.diff(ak.log(ak.sin(x)), x)      # raises E-CERT-001
```

### Cause axis

1. **User-input** — the expression or argument is outside the supported fragment. Always has a `remediation`; carries a `span` once parsing lands.
2. **Domain** — input is syntactically fine but violates a mathematical side condition. Remediation is "substitute a different value," not "reformulate."
3. **Unsupported** — the operation is not implemented for this case. Must name the missing capability so users can file a feature request.
4. **Resource/environment** — CUDA device absent, out-of-memory, JIT target mismatch, pool closed. Typically no `span`; remediation references the environment, not the expression.
5. **Internal invariant** — a bug. Should never reach users in release; in debug it carries a backtrace. Use `E-INTERNAL-001`.

### Adding a new error code

1. Does it fit an existing subsystem? Add a variant and a code one higher than the current max for that prefix.
2. Does it name a new subsystem? Add a prefix, a class, and an entry in `REGISTRY` in the same PR. Do not reuse prefixes across unrelated subsystems.
2b. **One code, one meaning.** Two `AlkahestError` impls under the same prefix must not return the same number. `REGISTRY`'s `no_duplicate_codes` test cannot catch this — it only sees the registry, and the colliding impl is typically the one that never got registered. `scripts/check_error_codes.py` compares the impls themselves and fails on any code claimed by two of them; a genuine alias (the same fact reported by two types that share a Python class) goes in its `DELIBERATE_ALIASES` with the reason.
3. Write the `remediation` before the message — if you cannot say what the user should do, the taxonomy is telling you this is an internal bug, not a user error.

Users match on subsystem (the exception class); triagers filter on cause (the code suffix and remediation text).

## Function fields: what is modelled

## Stabilizer codes: the `(x | z)` convention

`E-STAB-*` has thirteen numbers and only one of them is about running out of
room. The rest exist because the commonest failure in this area is not a
resource limit — it is a set of generators that looks like a stabilizer code,
type-checks like a stabilizer code, and is not one.

Everything in this layer writes a Pauli operator as `2n` bits in **`(x | z)`
layout**: the first `n` are the `X` exponents, the last `n` the `Z` exponents,
with

```text
    ⟨(x₁ | z₁), (x₂ | z₂)⟩  =  x₁·z₂ + z₁·x₂     (mod 2)
```

and two Paulis commuting exactly when that vanishes. The `(z | x)` layout is
also in use in the literature. A check matrix transcribed from a paper that uses
it produces generators that commute, have the right rank, and encode a different
code — and no error at all. There is nothing this library can do about that
except say so loudly, which is why the convention is repeated at the top of
every module in the layer. If a code comes out with the right `n` and `k` and
the wrong distance, suspect the layout before suspecting the search.

Three of the codes deserve to be read apart from the rest:

| Code | What it means | What to do |
|---|---|---|
| `E-STAB-004` | Two proposed generators **anticommute** | A stabilizer group is abelian; these two share no `+1` eigenspace. Check the `(x \| z)` layout, then the transcription |
| `E-STAB-006` | A product of the generators is `−I` | The stabilized subspace is `{0}`: the code encodes nothing. Only reachable from a *dependent* generating list. Negate one generator, or drop the dependent one |
| `E-STAB-008` | The exhaustive distance search is past its cap | There is no cleverer algorithm behind the cap — minimum distance is NP-hard and the only method here enumerates `2^(n+k)` centralizer elements. Take `distance_upper_bound()`, which returns a `Distance` with `exact == False`, and report it **as a bound** |

The last one is the reason `minimum_distance()` does not return an `int`. A
`Distance` carries `.value` and `.exact`, and the upper-bound path sets `.exact`
to `False`. A caller that stores `.value` in a field named `d` has thrown away
the one fact that made the number safe to publish.


`E-FFLD-*` is one prefix over one class, and the eleven numbers exist because
the honest answer to most function-field questions, in most of the space of
possible inputs, is *not this implementation*.

The implemented class is

```text
    K = ℚ(x)[y] / (y² − a(x)),    a squarefree,  deg a = 2g + 1 odd,
```

with divisors supported on **ℚ-rational places** — the degree-one points
`(α, β)` with `α, β ∈ ℚ`, plus the single place at infinity. That is not a
convenience boundary. The divisor class group here is Cantor arithmetic on
Mumford pairs, reached through the code the algebraic integrator already uses
(`integrate::algebraic::jacobian_torsion`, `coates`), and that code measures
every class against **one rational place at infinity**. An even-degree ("real")
model has two, and `n > 2` has no Mumford representation at all.

| Code | What it means | What to do |
|---|---|---|
| `E-FFLD-001` | `deg_y f ≠ 2`, a non-constant `y²` coefficient, or a discriminant that is zero or constant | Restate as `c₂y² + c₁(x)y + c₀(x)` with `c₂` a non-zero rational constant. Superelliptic `yⁿ = a(x)` with `n > 2` is genuinely not implemented |
| `E-FFLD-002` | The **real** (even-degree) model. Two places above `x = ∞` | The **genus is still returned** — it does not depend on the model. Only divisors, `Pic⁰` and Riemann–Roch refuse. Sending a rational root of `a` to infinity moves the model to odd degree |
| `E-FFLD-003` | A place of degree ≥ 2 appears — a conjugate pair `(α, ±√c)` with `c` a non-square, or a place over an irrational `α` | Record it as *not representable*, **never as absent**: dropping the place would silently change `deg D`. The rational-root search is capped, so this never proves irreducibility. `div(y)` on `y² = x⁵ + 1` lands here, because `x⁵ + 1` has one rational root |
| `E-FFLD-004` | A place `(α, β)` with `β² ≠ a(α)` | Coordinates are in the **normalised** model — check `FunctionField::curve()` and `normalisation()` before assuming they are the ones you wrote |
| `E-FFLD-005` | A class-group operation on a divisor of non-zero degree | `Pic⁰` is the group modelled; subtract `deg(D)·∞` |
| `E-FFLD-006` | The torsion order was **not decided**: too few good primes, or a candidate past the exact-confirmation cap | Record as undecided. **Not** a non-torsion certificate |
| `E-FFLD-007` | The class has **infinite order**. A verdict | Stop looking for a principal multiple. Reduction mod good primes is injective on prime-to-`p` torsion, so disagreeing orders refute torsion outright |
| `E-FFLD-008` | `div(0)` | The zero function has no divisor |
| `E-FFLD-009` | Two operands from different function fields | Rebuild both over one `FunctionField` |
| `E-FFLD-010` | A multiplicity, order or linear system past the work cap | Reduce the divisor |
| `E-FFLD-011` | A result was **computed and then withheld** for failing its own check — `div(u)`'s pole order at infinity disagreeing with the degrees of `p` and `q`, or a Riemann–Roch dimension violating Riemann's inequality | A bug in this module, not in the input. Report it; the point is that the wrong answer was not returned |

The last row is the one worth dwelling on. `div(u)` derives the multiplicity at
infinity from `deg div(u) = 0` and then checks it against
`v_∞(p + qy) = min(−2 deg p, −deg a − 2 deg q)`, which is exact on the
odd-degree model because the two candidates differ in parity. Riemann–Roch
checks its dimension against `dim ≥ deg D + 1 − g` and, above the canonical
degree, against the equality. Neither check can fire on valid input; both exist
so that a wrong answer arrives as `E-FFLD-011` rather than as a divisor.

## Certified coding bounds

`delsarte_lp_bound(n, d, q)` returns an upper bound on `A_q(n, d)` — the largest
possible size of *any* code (linear or not) of length `n` over an alphabet of
size `q` with minimum distance at least `d`. Every step is exact rational
arithmetic, but exactness alone is not what makes the number trustworthy: a
solver that stopped early would report a bound that is too *small*, and a bound
that is too small silently rules out codes that exist.

So the number is read off the **dual** programme and comes with the multipliers
that prove it. `DelsarteBound.certificate()` is a vector `y ≥ 0` satisfying
`Σ_k y_k K_k(i) ≤ −1` for every `i` in `d..=n`; from that alone, with no
reference to this implementation, `|C| ≤ 1 + Σ_k y_k K_k(0)` for every such
code. `DelsarteBound.verify_certificate()` re-checks it from scratch, and the
constructor refuses with `E-CODE-007` rather than returning a bound whose
certificate did not pass. A caller who does not trust this crate can re-verify
the certificate with nothing but a Krawtchouk evaluator.

This is the plain Delsarte programme. Schrijver's semidefinite strengthening and
the extra inequalities that beat it on specific `(n, d)` are not implemented:
what is returned is always a valid upper bound, not always the tightest one
known.

