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
├── CudaError         (E-CUDA-*)   — CUDA kernel launch or driver
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

### Refusals: when Alkahest declines to answer

A refusal is not a malfunction. These codes all mean *"I could not establish this, and
the alternative to saying so is a confident wrong answer"* — the outcome an unattended
loop must record as **undecided**, never as a negative result.

| Code | Class | What it means |
|---|---|---|
| `E-LINALG-010` | `LinearAlgebraError` | An entry's vanishing could be proven neither zero nor non-zero, so `rank` / `rref` / `nullspace` / `eigenvects` / `jordan_form` declined to pick a branch |
| `E-MAT-004` | `MatrixError` | Same, for a determinant: `inverse()` will not divide by something it cannot show is non-zero |
| `E-CAD-001` | `CadError` | `decide` is outside its fragment, or the only candidate solutions lie at an irrational boundary point it cannot test exactly |
| `E-SOS-002` | `SosError` | No positivity certificate of this shape at this degree — a statement about the search, not a proof that none exists |
| `E-INT-004` | `IntegrationError` | Proven non-elementary. **This one is a verdict, not a refusal** — keep it apart from the rest |
| `E-BUDGET-001..003` | `BudgetExceededError` | Ran out of the time/steps it was given, or was cancelled |

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
