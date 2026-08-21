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
├── OdeError          (E-ODE-*)    — ODE construction or lowering
├── DaeError          (E-DAE-*)    — DAE structural analysis
├── SolverError       (E-SOLVE-*)  — polynomial system solving
├── JitError          (E-JIT-*)    — LLVM/JIT codegen
├── CudaError         (E-CUDA-*)   — NVPTX compile, kernel launch, or driver, see [GPU support](./gpu.md)
├── PoolError         (E-POOL-*)   — ExprPool misuse
├── AnsatzError       (E-ANSATZ-*) — ansatz family construction or fitting, see [Ansatz families](./ansatz.md)
├── CrossCheckError   (E-XCHECK-*) — cross-CAS check could not be posed, see [Cross-CAS testing](./crosscheck.md)
├── SmtError          (E-SMT-*)    — SMT-LIB export, solver run, or model lift, see [SMT bridge](./smt.md)
└── BudgetExceededError (E-BUDGET-*) — budget/cancellation trip, see [Budgets](./budgets.md)
```

## Error attributes

Every exception instance exposes:

| Attribute | Type | Description |
|---|---|---|
| `.code` | `str` | Stable diagnostic code, e.g. `"E-POLY-001"` |
| `.message` | `str` | Human-readable description |
| `.remediation` | `str \| None` | What the user should try |
| `.span` | `tuple[int, int] \| None` | Character offset range in source expression |

```python
import alkahest
from alkahest import ExprPool, UniPoly, ConversionError

pool = ExprPool()
x = pool.symbol("x")

try:
    # sin(x) cannot be represented as a polynomial
    p = UniPoly.from_symbolic(alkahest.sin(x), x)
except ConversionError as e:
    print(e.code)          # E-POLY-001
    print(e.message)       # "expression contains non-polynomial term: sin(x)"
    print(e.remediation)   # "Use Expr directly, or expand sin(x) as a series first"
```

## Common errors and remediations

### ConversionError (E-POLY-*)

Raised when an expression cannot be converted to a polynomial or rational function.

| Code | Cause | Remediation |
|---|---|---|
| `E-POLY-001` | Non-polynomial term (e.g. `sin`) | Use `Expr` directly; or expand as series |
| `E-POLY-002` | Non-integer exponent | Algebraic extension not yet supported |
| `E-POLY-003` | Symbolic exponent (variable in exponent) | Use `Expr.pow`, not `UniPoly` |

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
| `E-SOLVE-001` | System is inconsistent | No solutions exist |
| `E-SOLVE-002` | High-degree univariate factor (> 2) | Symbolic solution not supported; use numerical solve |
| `E-SOLVE-003` | Gröbner basis did not terminate | Increase node/iteration limits |

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
| `E-CAD-001` | `CadError` | `decide` is outside its fragment, or the only candidate solutions lie at an irrational boundary point it cannot test exactly |
| `E-SOS-002` | `SosError` | No positivity certificate of this shape at this degree — a statement about the search, not a proof that none exists. **Record it as `unknown`, never as "not SOS" or "the inequality is false":** `p` may be SOS outside the LP subcone searched, SOS at a higher `basis_degree`, or non-negative without being SOS (Motzkin). `E-SOS-003`, which carries a witness point, is the only SOS *refutation*. The message carries a `what the search actually did:` trace; lines marked `NOT SEARCHED` are budgets that fired, not searches that came up empty, and mean the corresponding basis or multiplier power was never looked at. See [Positivity certificates](./positivity.md#three-outcomes-deliberately-kept-apart) |
| `E-IDEAL-005` | `IdealRefusal` | `radical` cannot certify `√I` for this ideal. Only monomial, principal and zero-dimensional ideals — and anything whose primary decomposition is certified — are answered; the alternative is asserting `√I = I` with nothing behind it |
| `E-IDEAL-006` | `IdealRefusal` | `primary_decomposition` reached a component it cannot show is primary, so it will not report the ideal itself with an unjustified `associated_prime` |
| `E-SOLVE-004` | `TriangularizeRefusal` | `triangularize` extracted a chain that does not generate an ideal containing the input, i.e. one that cuts out a larger variety than the system. Splitting on the initials (Lazard–Kalkbrener) is not implemented |
| `E-SERIES-003` | `SeriesError` | `series` ran past its work ceiling (or an active `Budget`) before reaching the requested order. Coefficients are formed by repeated differentiation without re-simplifying, so a nested radical's derivatives grow by a constant factor each time; a *shorter* series would carry an `O(h^order)` label nothing bounded |
| `E-PSLQ-004` | `PslqError` | `guess_relation` found an integer relation the inputs' precision cannot justify — pinning down `n` coefficients bounded by `H` costs about `n·log10(2H+1)` digits of agreement, and the inputs do not carry that many. **Record it as `undecided`, not as "no relation exists":** the same constants at higher precision may well admit one. `relation_confidence` reports the same judgement as data, including a three-valued `credible` whose `None` means *the inputs' precision is not knowable*, never a pass |
| `E-PSLQ-005` | `PslqError` | The constants are exact rationals and `Σ aᵢ·cᵢ` is not zero in exact arithmetic. **This one is a verdict, not a refusal** — the relation is refuted for the numbers supplied |
| `E-INT-004` | `IntegrationError` | Proven non-elementary. **This one is a verdict, not a refusal** — keep it apart from the rest |
| `E-BUDGET-001..005` | `BudgetExceededError` | Ran out of the time, steps or memory it was given, was cancelled, or is about to exhaust the process address-space limit |

`E-SERIES-003` travels out of band for the same reason (`SeriesError` is exhaustive) but *is*
wired into the bindings: `series` returns `SeriesError::InvalidOrder` with
`calculus::series::take_series_refusal()` pending, and the Python layer raises `SeriesError`
with `.code == "E-SERIES-003"` — or `BudgetExceededError` when a budget was what stopped it.

`E-IDEAL-005`, `E-IDEAL-006` and `E-SOLVE-004` are new in 3.8 and travel **out of band**:
`PrimaryDecompositionError` and `SolverError` are public exhaustive enums that cannot gain
a variant in a patch release, so the refusal is returned inside an existing variant and the
real code is available from `ideal::take_ideal_refusal()` /
`solver::regular_chains::take_triangularize_refusal()`. The Python bindings consult both, so
`radical` and `primary_decomposition` raise `AlkahestError` with `.code == "E-IDEAL-005"` /
`"E-IDEAL-006"`, and `triangularize` raises `SolverError` with `.code == "E-SOLVE-004"`.
`AlkahestError` subclasses `ValueError`, so code that catches `ValueError` is unaffected.

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
| `E-ODE-*` | `OdeError` | ODE construction, lowering, event handling |
| `E-DAE-*` | `DaeError` | DAE structural analysis (Pantelides, index reduction) |
| `E-SOLVE-*` | `SolverError` | Polynomial system solving, Gröbner basis |
| `E-LAT-*` | `LatticeError` | Exact LLL lattice reduction over ℤ |
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
    ak.diff(ak.sin(x), x)        # fine — certifies
    ak.integrate(ak.log(x), x)   # raises E-CERT-001
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
3. Write the `remediation` before the message — if you cannot say what the user should do, the taxonomy is telling you this is an internal bug, not a user error.

Users match on subsystem (the exception class); triagers filter on cause (the code suffix and remediation text).
