# Ansatz families and conjecture generation

Stage 1 of a search loop is *generate*: propose a structured family of candidates — every
polynomial of degree ≤ 3 in `x` and `y`, a Padé approximant of type (2, 2), a quadratic
Lyapunov candidate — and then either sweep it numerically or solve for the coefficients
that make a residual vanish. Agents hand-roll this constantly, and the hand-rolled version
is usually wrong in one of three specific ways: it loses the distinction between an
unknown coefficient and an independent variable; it assumes the first *m* sample points
give *m* independent equations; and it never substitutes the answer back, so a fit that
only satisfies the sampled constraints is reported as if it satisfied the identity.

`alkahest.ansatz` is that plumbing, done once.

```python
import alkahest as ak
from alkahest.ansatz import polynomial, fit

pool = ak.ExprPool()
x = pool.symbol("x")

A = polynomial(pool, [x], degree=2)      # c_0 + c_1*x + c_2*x^2
target = x**2 - pool.integer(3) * x + pool.integer(2)

sol = fit(A, A.expr - target)
sol.expr          # (2 + x^2 + (x * -3))
sol.rank, sol.free
sol.status        # 'exactly_verified'
```

Everything here is **pure Python** composed from primitives that are already fast in Rust
(`Matrix.rref`, `simplify`, `subs`), so it works in a build without the `groebner`
feature. The one path that needs Gröbner — a residual genuinely nonlinear in the unknowns
— refuses with `E-ANSATZ-004` rather than degrading silently.

## Honesty invariants

**Solving may be heuristic; checking is exact.** The linear system is built by
*collocation* — evaluating the residual at sample points — which proves identical vanishing
only for polynomial residuals of bounded degree. So the fit is never trusted on its own:
`fit(..., certify="residual")` (the default) substitutes it back and normalises.

| Outcome | `verification["status"]` | What it means |
| --- | --- | --- |
| The residual normalises to `0` | `exactly_verified` | A symbolic proof. The claim is machine-checked. |
| It does not, but samples are small | `numerically_checked` | Evidence, not a proof. The surviving normal form is in `verification["residual"]`. |
| `certify="none"` | `unverified` | Nothing was checked. There is deliberately no way to call this "solved". |

Those are the existing [`research.STATUS_BADGES`](./claim-graphs.md) strings, so a fitted
ansatz lands in a claim graph correctly labelled with no new vocabulary.

**Inconsistent is a result, not a malfunction.** When no member of the family can satisfy
the constraints, `fit` raises `AnsatzError` `E-ANSATZ-003`. For a loop that is a *closed
branch* — a positive finding worth recording — in the same spirit as a non-elementarity
verdict from `integrate`.

**Underdetermined is also a result.** When the rank is below the number of unknowns, the
members that work form a positive-dimensional family. `AnsatzSolution.free` returns the
free parameters rather than picking an arbitrary member:

```python
B = polynomial(pool, [x], degree=3, name="d")
under = fit(B, ak.diff(B.expr, x).value)      # d/dx of the family vanishes identically
under.rank                                    # 3
under.free                                    # (d_0,)  — still symbolic in .expr
```

## The families

| Constructor | Family | Typical use |
| --- | --- | --- |
| `polynomial(pool, vars, degree)` | `Σ c_α · x^α`, total degree ≤ `degree` | Undetermined coefficients, invariants |
| `rational(pool, vars, num_degree, den_degree)` | `p / q` | Padé, rational-function reconstruction |
| `linear_combination(pool, basis)` | `Σ cᵢ · basisᵢ` | The escape hatch: any basis you can write down |
| `exponential_polynomial(pool, var, rates, degree=…)` | `Σ pᵢ(x)·e^{λᵢ x}` | ODE / recurrence ansätze with known characteristic roots |
| `quadratic_form(pool, vars)` | `Σ_{i ≤ j} q_ij · xᵢ xⱼ` | Lyapunov candidates |

Every constructor takes `name=` (the coefficient prefix), `max_terms=` (a hard bound), and
`reserved=` (extra symbols the coefficients must not collide with). `reserved=` reserves
*names* only — it never adds an independent variable, so `linear_combination(pool, basis,
reserved=[y])` is still a family in the basis's own symbols. Pass `vars=` to say otherwise.

### Predictable coefficient names

Names are `c_0, c_1, …` for one variable and `c_0_0, c_1_0, c_0_1, …` (graded, then
lexicographic with the first variable heaviest) for several. They are **never gensym-ed**:
an agent that cannot predict the symbol names cannot write the follow-up call.

If a generated name collides with a symbol already in play, the constructor raises
`E-ANSATZ-001` instead of quietly fitting the wrong thing:

```python
c0 = pool.symbol("c_0")
polynomial(pool, [x], degree=1, reserved=[c0])   # AnsatzError E-ANSATZ-001
polynomial(pool, [x], degree=1, name="k", reserved=[c0])   # fine: k_0, k_1
```

The collision check sees the family's own variables, every free symbol of the expressions
handed to the constructor, and anything passed as `reserved=`. An `ExprPool` exposes no
symbol listing, so a symbol that appears in *none* of those is not detected at construction
— `fit`'s back-substitution check is the backstop.

### Bounds are mandatory, not advisory

`C(n + d, d)` is a combinatorial explosion, so every constructor is bounded and the bound
is checked from the *count* before anything is materialised:

```python
polynomial(pool, [x, y], degree=40, max_terms=32)
# AnsatzError E-ANSATZ-002: … needs 861 unknown coefficients, which exceeds max_terms=32
```

### `rational` keeps the Padé case linear

`p/q ≈ f` is *not* linear in the coefficients of `p` and `q`, but `p − f·q = 0` is linear
in them jointly. `rational()` records the numerator/denominator split so that transform can
be applied, and `fit` applies it automatically when it sees an unknown-bearing denominator:

```python
A = rational(pool, [x], num_degree=1, den_degree=1)
A.expr                       # ((a_0 + (x * a_1)) * (1 + (x * b_1))^-1)
A.residual(target)           # a_0 + a_1*x - target*(1 + b_1*x)   — affine in the unknowns

sol = fit(A, A.expr - pool.integer(1) / (pool.integer(1) + x))
[s["rule"] for s in sol.steps]        # includes 'ansatz_clear_denominator'
sol.status                            # 'exactly_verified'
```

The denominator's constant term is fixed to `1` by default (`monic_denominator=True`),
because `p/q` and `(λp)/(λq)` are the same function — without a normalisation **every**
rational fit would report a spurious extra free parameter.

A Padé approximant is a *local* match, not an identity, so it wants the exact-system route
with an explicit degree bound:

```python
A = rational(pool, [x], num_degree=2, den_degree=2, name="u", den_name="v")
sol = fit(A, A.residual(ak.exp(x)), certify="exact", degree_bound=4)
# u_0=1, u_1=1/2, u_2=1/12, v_1=-1/2, v_2=1/12  — the (2,2) Padé of exp
sol.status                   # 'numerically_checked': an approximant is not an identity,
sol.verification["residual"] # and the check says so rather than claiming otherwise
```

Asking for the same thing as an *identity* (`fit(A, A.residual(ak.exp(x)))`, default
certify) is correctly answered with `E-ANSATZ-003`: no rational function equals `exp`.

## `fit`

```python
def fit(ansatz, residual, *, certify="residual", seed=None, oversample=None,
        max_points=None, degree_bound=None, tolerance=1e-8, samples=5) -> AnsatzSolution: ...
```

`residual` is the expression that must vanish identically in `ansatz.vars` — usually
`ansatz.expr - target`.

**How a residual becomes a finite system.** Because the residual is affine in the unknowns,
each row is built by *probing*: evaluate it with every unknown set to `0` for the constant
column, then with unknown *j* set to `1` for column *j*. That is `subs` and nothing else —
no coefficient collection over a symbolic ring is required.

`certify` selects how the system is built as well as how it is graded:

- `"residual"` (default) — collocation at sample points, then exact back-substitution.
- `"exact"` — Taylor-coefficient extraction (`∂^α R / α!`), so the *system itself* is exact
  for polynomial residuals up to `degree_bound`; still back-substituted. Every multi-index of
  total degree ≤ `degree_bound` contributes an equation, and the bound that was reached is
  written into the derivation log — the system is never quietly cut short. `max_points` caps
  sample-point *draws* and therefore applies to collocation only; `degree_bound` is the knob
  that sizes this one.
- `"none"` — no check at all; `status="unverified"` and no re-verification recipe. For hot
  loops that verify downstream, never for anything recorded as a result.

**Rank is read off the reduction, never assumed.** `fit` draws strictly more equations than
there are unknowns (`oversample`, default `max(4, len(ansatz))`) and takes the rank from the
reduced row echelon form. Assuming the first *m* points are independent is the specific bug
in every hand-rolled version of this.

**Points where the residual is undefined are skipped.** A vanishing denominator is detected
exactly (not by catching a float `inf`), the point is resampled, and the count of skipped
points appears in the derivation log.

### Determinism

Sample points come from a deterministic generator seeded from `budget_seed()`, so a fit is
reproducible across runs and machines — see [Budgets](./budgets.md):

```python
with ak.context(budget=ak.Budget(seed=7)):
    sol = fit(A, A.expr - target)
sol.points          # the exact points used, as rational strings
```

With no budget active the seed is `ansatz.DEFAULT_SEED`, a fixed constant, so two machines
still agree. `fit(..., seed=…)` overrides both.

## `AnsatzSolution`

| Field | Meaning |
| --- | --- |
| `expr` / `value` | The fitted member (`value` is the alias `ResearchSession` reads) |
| `assignment` | `{unknown: Expr}` for the determined coefficients |
| `free` | Unknowns the constraints do not pin down |
| `rank` | Rank of the system; `rank == len(ansatz)` iff the fit is unique |
| `status` | Mirrors `verification["status"]` |
| `verification` | `{"status", "evidence", "method", "residual", "max_abs_residual", …}` |
| `steps` | Derivation log in the `STEP_FIELDS` schema |
| `check` | Re-verification recipe for `ClaimGraph.verify()` |
| `points` | The sample points used, enough to reproduce the system |
| `certificate` | Always `None` — this module emits no Lean certificate |

It has the same shape as a `DerivedResult` where it matters, so it records unchanged:

```python
with ak.research.session(title="ansatz", pool=pool) as s:
    sol = fit(A, A.expr - target)
    s.record(sol, method="ansatz.fit", check=sol.check)

s.graph.summary()          # {'exactly_verified': 1}
```

## `enumerate_family` — stage 2 material

Enumeration and fitting stay separate. `enumerate_family` feeds the *falsify* stage
(generate candidates, hammer them with [`compile_expr`](./codegen.md) or
[`batch_map`](./batch.md)); `fit` is the *discover* stage. Fusing them produces an API that
does neither well.

```python
from alkahest.ansatz import enumerate_family

A = polynomial(pool, [x], degree=1)
[str(m) for m in enumerate_family(A, [0, 1])]     # ['0', 'x', '1', '(x + 1)']
```

Enumeration is lazy **and** bounded: `len(coeffs) ** len(ansatz)` is checked against
`max_members` before the first member is built, and exceeding it raises `E-ANSATZ-002`.

## Positivity: hand off, don't reimplement

The module's job ends when it has produced the candidate. `certify_nonneg` is a one-line
adapter onto [`prove_nonneg` / `sos_decompose`](./positivity.md), and every outcome — a
`PositivityCertificate`, an `E-SOS-003` refutation with a witness point, an `E-SOS-002`
"no certificate of this shape at this degree" — comes back unmodified:

```python
from alkahest.ansatz import quadratic_form, certify_nonneg

V = quadratic_form(pool, [x, y], name="q")
sol = fit(V, V.expr - lyapunov_residual)
cert = certify_nonneg(sol)          # PositivityCertificate, or the SosError as raised
```

A solution that still carries free parameters is refused (`E-ANSATZ-003`): an
undetermined form is a family, not a candidate. Instantiate first.

## Error codes

| Code | Meaning |
| --- | --- |
| `E-ANSATZ-001` | A coefficient name collides with a symbol already in play (or an unknown name was passed to `instantiate`). You called it wrong. |
| `E-ANSATZ-002` | The requested family exceeds `max_terms` / `max_members`. Refused before anything was materialised. |
| `E-ANSATZ-003` | **No member of this family satisfies the constraints.** A *result*: a closed branch a loop should record. Also raised when no system could be built at all — the residual, or a derivative of it, is undefined everywhere the sampler looked — in which case the message says so instead of claiming anything about the family. |
| `E-ANSATZ-004` | The residual is nonlinear in the unknowns and escalating to `solve` needs a `groebner` build, which this is not. |

Only the first two mean "you called it wrong"; the other two are findings. See
[Error handling](./errors.md).

## Known limits

- **Collocation is not a proof of identical vanishing** except for polynomial residuals of
  bounded degree. That is exactly why the back-substitution check exists and why a fit that
  does not normalise to zero is never reported as verified.
- **Transcendental bases are only as exact as the simplifier's zero test.** When the
  collocation matrix is not over ℚ (an exponential family sampled at rational points),
  `fit` first tries Taylor extraction, which usually recovers an exact rational system; if
  it cannot, the reduction falls back to a symbolic elimination whose zero test is
  best-effort. In that case an *apparent* inconsistency is corroborated numerically before
  `E-ANSATZ-003` is claimed — a confident "no member of this family works" is exactly the
  kind of wrong answer this package exists to avoid.
- **The pool exposes no symbol listing**, so the `E-ANSATZ-001` collision check covers the
  variables, the constructor's inputs, and `reserved=` — not the whole pool.

## See also

- [Autoresearch / agent loops](./search-plumbing.md)
- [Claim graphs](./claim-graphs.md) — where a fitted ansatz gets recorded
- [Batch and streaming evaluation](./batch.md) — sweeping an enumerated family
- [Budgets, cancellation, and determinism](./budgets.md) — where the sample seed comes from
- [Positivity certificates](./positivity.md) — the hand-off target for `quadratic_form`
