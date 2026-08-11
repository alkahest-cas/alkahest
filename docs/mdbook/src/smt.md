# SMT/SAT bridge

Discrete and mixed integer/real/boolean subproblems are **not Alkahest's problem class**,
and the fastest way to make Alkahest worse would be to pretend otherwise. What a search
loop needs is not an in-tree SAT solver but a way to *hand the subproblem off* — and,
crucially, a way to bring the answer back in a form the rest of the toolchain can trust.

That is what this bridge is. `alkahest.to_smtlib` emits standard SMT-LIB 2 text; an
external solver (z3, cvc5, …) consumes it; `alkahest.smt.solve` reads the answer back,
lifts the model into exact rationals, and **checks it**. It is the same shape as
[Lean certificates](./lean-certs.md): Alkahest emits a standard artifact, and an
independently maintained tool it does not control does the hard part.

```python
import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
n = pool.symbol("n", "integer")

# x is real, n is an integer, x sits strictly between n and sqrt(10) — mixed
# integer/real, which is exactly what neither CAD nor `diophantine` handles.
f = ak.And(pool.gt(x, n), ak.And(pool.lt(x * x, pool.integer(10)), pool.gt(n, pool.integer(1))))

with ak.context(pool=pool):
    result = ak.smt.solve(f, budget=ak.Budget(wall_ms=5000))

print(result.status)        # 'sat'
print(result.model)         # {'x': Fraction(3, 1), 'n': Fraction(2, 1)}
print(result.engine)        # 'z3 Z3 version 4.13.0 - 64 bit'
print(result.badge)         # 'the solver's model was substituted back and ...'
```

## The two asymmetries

Everything else on this page follows from these.

### 1. `sat` and `unsat` have different trust stories

A `sat` model is checkable inside Alkahest **for free**: substitute it back and evaluate
the formula exactly. `solve` always does this. There is no flag to turn it off, and a
model that fails raises `SmtError` `E-SMT-004` rather than warning — a failure there means
the emitter mistranslated the formula or the solver returned an unsound model, and neither
belongs in a log line.

`unsat` is a different matter: checking it means consuming an unsat proof, which is a large
project against unstable formats. So an `unsat` result carries the status
**`externally_asserted`**, whose badge reads:

> an external solver asserted this; NO proof was checked and nothing in Alkahest verified it

and which is deliberately **absent** from `alkahest.research.MACHINE_CHECKED_STATUSES`.
That set means *a checker actually ran in this process*. Quietly widening it to include
"z3 said so" would erode the one guarantee [`research.py`](./claim-graphs.md) makes.

| Solver says | `verification["status"]` | Counted as machine-checked? |
| --- | --- | --- |
| `sat` | `exactly_verified` | **yes** — the model was substituted back in-process |
| `unsat` | `externally_asserted` | **no** — nothing checked it |
| `unknown` | `unverified` | no |

### 2. Exactness is where a model reader breaks

Rationals lift cleanly. `(/ 25.0 4.0)` becomes `Fraction(25, 4)`; `0.1` becomes
`Fraction(1, 10)` — parsed from the **string**, so it is the exact decimal rational the
solver meant and never the nearest binary double.

Algebraic numbers do not lift. Ask z3 for a witness to `x² = 2 ∧ x > 0` and it answers
`(root-obj (+ (^ x 2) (- 2)) 2)`. The tempting move is to evaluate that to `1.41421356…`
and carry on. **A float witness recorded as an exact one is precisely the silent error this
bridge exists to prevent** — a loop would go on to build a hundred derived results on a
value that does not actually satisfy the constraints, and its own consistency checks would
happily confirm them. So `root-obj` is refused with `E-SMT-003`, and lifting it into the
existing real-algebraic machinery (`RootInterval` / `refine_root`) is future work.

## Planning ahead: `supported()`

`supported` is to this module what [`certifiable`](./certificate-coverage.md) is to Lean
export: a loop must be able to choose a route *before* it commits.

```python
support = ak.smt.supported(f)
bool(support)            # would solve() run?
support.exportable       # would to_smtlib() succeed? (independent of solver install)
support.logic            # 'QF_NIRA'
support.reason           # 'ok' | 'outside_fragment' | 'quantified' |
                         # 'not_exactly_checkable' | 'no_solver'
support.recommendation   # 'smt' | 'prefer_in_tree'
support.script           # the emitted script, so you don't pay for it twice
```

`recommendation` is the part worth reading, and it cuts against the usual instinct:

- **`prefer_in_tree`** for real arithmetic with no integer variables (`QF_LRA` / `QF_NRA`).
  [`prove_nonneg` / `sos_decompose`](./positivity.md) return a `PositivityCertificate` that
  composes with `to_lean`; `decide` is complete. z3's `nlsat` returns an answer and **no
  artifact**. Reach for SMT here as a *fallback* when the in-tree route refuses or exceeds
  its budget.
- **`smt`** for anything with integer variables — mixed integer/real/boolean is the
  genuinely new capability, and neither CAD nor `diophantine` covers it.

## What gets emitted

`to_smtlib` produces a complete, runnable script:

```python
print(ak.to_smtlib(ak.And(pool.gt(x, pool.integer(0)), pool.lt(x, pool.integer(3)))))
```

```smt2
; alkahest SMT-LIB 2 export
(set-logic QF_LRA)
(set-option :produce-models true)
(declare-fun x () Real)
(assert (and (> x 0) (< x 3)))
(check-sat)
(get-model)
```

The emitter lives in Rust (`alkahest-core/src/logic/smtlib.rs`) next to `Formula`, for the
same reason the Lean emitter lives in `alkahest-core/src/lean/`: it must be **exhaustive**
over `Formula` and `PredicateKind`, and `rustc`'s match-exhaustiveness check is the
enforcement mechanism. There is no `_ =>` arm anywhere in that file, and `tests/test_smt.py`
asserts that from the outside as well, so a node added to the kernel later fails to compile
rather than silently emitting plausible-but-wrong SMT-LIB.

### Logic selection

`logic="auto"` (the default) infers the weakest logic that fits:

| Formula | Logic |
| --- | --- |
| linear, reals only | `QF_LRA` |
| nonlinear, reals only | `QF_NRA` |
| linear, integers only | `QF_LIA` |
| nonlinear, integers only | `QF_NIA` |
| mixed int/real | `QF_LIRA` / `QF_NIRA` |
| any of the above with `Forall`/`Exists` | same name without the `QF_` prefix |

You may name a logic explicitly (`ak.to_smtlib(f, "QF_NRA")`), and one that is **too weak**
for the formula is an error, not a silent downgrade — sending a nonlinear problem under
`QF_LRA` would ask a different question than the one you have. The names accepted are
`QF_LIA`, `QF_NIA`, `QF_LRA`, `QF_NRA`, `QF_LIRA`, `QF_NIRA`, their quantified forms
without the `QF_` prefix, `AUFLIRA`, `AUFNIRA`, and `ALL`.

#### The mixed names are solver-facing, not catalog-standard

`QF_LIRA` / `QF_NIRA` / `LIRA` / `NIRA` are **not** in the official SMT-LIB 2.7 logic
catalog, which stops at `AUFLIRA` / `AUFNIRA` for mixed `Int`/`Real`. Alkahest emits them
anyway, and that is a deliberate contract rather than an oversight:

- they are the names the solvers this bridge drives actually use for the mixed fragment.
  z3 accepts them silently (an unrecognised name draws `ignoring unsupported logic`), Yices
  documents `QF_LIRA` / `QF_NIRA` among the names it recognises beyond the official set,
  and SMT-COMP runs `QF_LIRA` / `QF_NIRA` divisions over SMT-LIB benchmarks;
- the catalog alternatives are strictly worse for what is emitted here. `AUFLIRA` /
  `AUFNIRA` are *quantified* logics that additionally carry arrays and free function
  symbols, so naming one for a quantifier-free mixed formula discards the `QF_` hint that
  decides which solver core runs and claims a far larger fragment than is in use.

If you are feeding a consumer that accepts catalog names only, ask for one explicitly:
`ak.to_smtlib(f, "AUFLIRA")` (linear) or `ak.to_smtlib(f, "AUFNIRA")` (nonlinear) — both are
sound supersets of everything the emitter produces — or `ALL`. `tests/test_smt.py` pins that
the installed solver accepts the names alkahest emits, so a regression in that contract
fails the suite rather than the user's pipeline.

Getting this right is load-bearing, not cosmetic: under `(set-logic QF_NRA)` z3 has no
`Int` sort at all, and under a `Reals_Ints` logic an integer numeral in a real position
needs an explicit `to_real`. The emitter tracks sorts and inserts exactly the coercions the
chosen theory requires — and none where they would not parse.

### Translation table

| Alkahest | SMT-LIB 2 |
| --- | --- |
| `Symbol` (Real / Positive / NonNegative / NonZero) | `(declare-fun x () Real)` |
| `Symbol` (Integer) | `(declare-fun n () Int)` |
| `Positive` / `NonNegative` / `NonZero` refinement | an extra `(assert (> x 0))` / `(>= x 0)` / `(not (= x 0))` |
| `Integer(-3)` | `(- 3)` |
| `Rational(-1, 3)` | `(/ (- 1) 3)` |
| `Add` / `Mul` | `(+ …)` / `(* …)` |
| `Pow(x, 3)` | `(* x x x)` — SMT-LIB 2 has no portable `^` |
| `Pow(x, -1)` | `(/ 1 x)` for a `Real` base; `(/ (to_real 1) (to_real n))` for an `Int` base |
| `Piecewise` | `(ite c v …)` |
| `Lt/Le/Gt/Ge/Eq` | `(< …)` / `(<= …)` / `(> …)` / `(>= …)` / `(= …)` |
| `Ne` | `(not (= …))` |
| `And` / `Or` / `Not` | `(and …)` / `(or …)` / `(not …)` |
| `Forall` / `Exists` | `(forall ((x Real)) …)` / `(exists ((x Real)) …)` |

A refined domain travels with its binder under a quantifier: `∀ x:Positive . P` becomes
`(forall ((x Real)) (=> (> x 0) P))` and the existential takes `(and …)`. Getting that
backwards would be a soundness bug, so both are written out explicitly in the emitter.

A negative power over an `Int` base is coerced rather than emitted as `(/ 1 n)`, and the
reason is semantic rather than cosmetic: in SMT-LIB `/` is **real** division and `div` is
integer division, so `n^-1` for an integer `n` is the real reciprocal the kernel means, not
integer division. The emitter therefore lifts both operands with `to_real` — which is also
why such a formula selects a `Reals_Ints` logic even though only one sort appears in the
source expression.

### What is refused

The emitter is **total-or-refuse**. Everything below raises `SmtError` `E-SMT-002`:

- **Float literals.** `0.1` is a binary double, not the exact question it looks like.
  Write `pool.rational(1, 10)`; exporting the dyadic expansion would silently change what
  is being asked.
- **Complex-domain symbols** — SMT-LIB arithmetic is ordered and real.
- **Transcendental function heads** (`sin`, `exp`, `log`, …). Only `abs` has an exact
  SMT-LIB rendering.
- **Non-integer or very large exponents.** Powers become products, so the expansion is
  *bounded* (`MAX_POW_EXPANSION`, 128) — a loop must never be able to ask for a megabyte
  of `x`.
- **`BigO`, `RootSum`, predicates in term position**, and symbol names containing `|`.
- **Two symbols sharing a name but not a domain** — SMT-LIB has one namespace.

## Driving a solver

Solver discovery, subprocess management, and model parsing live in Python
(`python/alkahest/smt.py`), where they iterate faster. This mirrors the Lean split
exactly: emission in Rust, harness in Python.

```python
ak.smt.solvers()
# {'z3': 'Z3 version 4.13.0 - 64 bit', 'cvc5': None}
```

Absence is reported **negatively and explicitly**, so an agent can tell before it plans
that the hand-off is unavailable. `PATH` is searched, plus the running interpreter's script
directory — a `pip install z3-solver` into a venv that was never `activate`d still gets
found.

If no solver is installed, `solve` raises `E-SMT-001`. It does **not** fall back to
`alkahest.satisfiable`, the interval heuristic: that would answer `Unknown` and look for
all the world like a solver had run and found nothing.

### `SmtResult`

| Field | Meaning |
| --- | --- |
| `status` | `"sat"` / `"unsat"` / `"unknown"` |
| `model` | `dict[str, Fraction]` — the exact witness, and exactly what was verified |
| `model_exprs` | the same values interned into a pool, when one was passed or is active |
| `engine` | which solver answered, and at what version |
| `logic` | the logic that was sent |
| `smtlib` / `certificate` | the script that was sent — an artifact, not a checked proof |
| `verification` | `DerivedResult`-shaped, so `ResearchSession.record` takes it unchanged |
| `badge` | the honest one-line rendering of the status |
| `machine_checked` | `True` only when a checker ran **in this process** |
| `reason_unknown` | the solver's own explanation, on `unknown` |
| `elapsed_ms` | wall-clock time in the solver process |

`SmtResult` has no `__bool__` on purpose: `unsat` and `unknown` would both be falsy, and a
loop writing `if not result:` would conflate "proved impossible" with "gave up". Branch on
`.status`.

`model` is keyed by symbol name and holds `Fraction`s rather than `Expr`s because an `Expr`
carries no reference to its pool; `model_exprs` is populated when `solve(..., pool=…)` is
given one or `alkahest.context(pool=…)` is active. The `Fraction` map is always present and
always the thing that was checked, so nothing about the guarantee depends on ambient
context.

### How a `sat` model is checked

Two independent exact checks, both mandatory:

1. Substitute the model into the **original Alkahest formula** and evaluate it exactly
   through the kernel. This is the invariant the bridge rests on, and
   `tests/test_smt.py` asserts it as a property test over generated formulas rather than
   a handful of examples.
2. Evaluate every assertion in the **script that was actually sent**, over exact
   rationals. A mistranslation in the emitter would have to fool both to slip through, and
   this pass also re-checks the refined-domain side conditions, which are separate
   assertions.

A model missing a "don't care" variable is completed with `0` and then checked, so the
witness a caller receives is always total and always substitutable.

Because the guarantee is unconditional, `solve` refuses **up front** — before the solver
runs — for a formula the kernel cannot evaluate exactly (`E-SMT-002`, reason
`not_exactly_checkable`). `abs` is the case you will actually hit: it exports fine, so
`to_smtlib` handles it, but `evaluate(..., mode="exact")` does not, and a refusal that
arrives only after paying for the solver run reads like a bug in the solver.

For the same reason `solve` takes **quantifier-free formulas only**. `to_smtlib` exports
quantified ones happily — export it and drive the solver yourself, or use
`alkahest.decide` for real quantifier elimination (see [Positivity certificates](./positivity.md)).

## Budgets

`solve(..., budget=ak.Budget(wall_ms=…))` passes the limit to the solver's own timeout flag
*and* enforces a parent-side deadline as a backstop. A trip raises `BudgetExceededError`
(`E-BUDGET-001`), not a bare `unknown`, so the loop gets the structured
"hard, not hung" distinction [budgets](./budgets.md) were built for:

```python
try:
    result = ak.smt.solve(candidate, budget=ak.Budget(wall_ms=250))
except ak.BudgetExceededError:
    log.info("deprioritising: hard, not hung")
```

Without a budget, a solver `unknown` is returned as a result with `reason_unknown` set —
a real answer ("I could not decide this"), distinct from a resource verdict.

## Recording into a claim graph

`SmtResult` quacks like a `DerivedResult`, so it records unchanged:

```python
with ak.research.session(title="mixed feasibility", pool=pool) as s:
    result = ak.smt.solve(f, budget=ak.Budget(wall_ms=5000))
    claim = s.record(result, statement="the system is feasible", method="smt.solve")

claim.status           # 'exactly_verified' for sat, 'externally_asserted' for unsat
claim.machine_checked  # True only for the checked sat case
```

## What is *not* here, and why

- **No vendored solver.** No libz3 in the Rust build: it keeps the wheel small, the
  licensing simple, and — the real reason — an in-tree solver would become a soundness
  liability the project has to defend, in a problem class that is not Alkahest's.
- **No unsat-proof checking.** See the first asymmetry above; the status vocabulary is
  honest about it rather than papering over it.
- **`dpll_sat` is not wired in.** `alkahest_core::logic::dpll_sat` remains a standalone CNF
  utility, sound and complete for the propositional problem it is *handed*, and it is
  deliberately not an engine behind this bridge. The only route from a `Formula` to it is
  to abstract each arithmetic atom to a fresh proposition, and that abstraction is sound in
  one direction only: it can confirm `unsat`, but it calls `x > 0 ∧ x < 0` *sat* and hands
  back a meaningless model. Under a bridge whose whole premise is that every `sat` model is
  checked exactly, that is the silent error the design is built to exclude — so a missing
  solver is a refusal (`E-SMT-001`), not a degradation.

## Error codes

| Code | Raised by | Meaning |
| --- | --- | --- |
| `E-SMT-001` | Python driver | No solver binary found. A refusal, never a fallback. |
| `E-SMT-002` | Rust emitter + driver | Formula outside the supported fragment (or an unusable logic, a quantified formula passed to `solve`, or one the kernel cannot check exactly). |
| `E-SMT-003` | Python driver | A model value (`root-obj`) cannot be lifted exactly. Refused, not rounded. |
| `E-SMT-004` | Python driver | A model failed back-substitution. Always raised, never warned. |
| `E-BUDGET-001` | Python driver | The solver hit `Budget.wall_ms`. |

Only `E-SMT-002` appears in `alkahest_core::errors::codes::REGISTRY`, because it is the only
one Rust raises; `scripts/check_error_codes.py` requires the registry and the Rust
`AlkahestError` impls to agree exactly, so codes raised only from Python stay out of it
(`E-BATCH-001` in `alkahest/_batch.py` is the same precedent).

## See also

- [Autoresearch / agent loops](./search-plumbing.md)
- [Budgets, cancellation, and determinism](./budgets.md)
- [Claim graphs](./claim-graphs.md)
- [Lean certificates](./lean-certs.md)
- [Error handling](./errors.md)
