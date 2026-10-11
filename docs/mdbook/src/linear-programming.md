# Exact linear and integer programming

`alkahest.linprog`, `alkahest.milp` and `alkahest.linprog_expr` solve linear and
mixed-integer programmes **exactly over the rationals** and return every status
with a certificate that proves it. Nothing in the solver is a float.

```python
import alkahest as ak
from fractions import Fraction

# max 3a + 5b  s.t.  a ≤ 4,  2b ≤ 12,  3a + 2b ≤ 18,  a, b ≥ 0
r = ak.linprog([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], sense="max")
r.status          # 'optimal'
r.x, r.fun        # ([2, 6], 36)
r.ineqlin         # [0, Fraction(3, 2), 1]   — shadow prices of the three rows
r.verify()        # True (re-checks the certificate exactly)
```

The calling convention is `scipy.optimize.linprog`'s: minimise `c @ x` subject to
`A_ub @ x <= b_ub`, `A_eq @ x == b_eq` and `bounds` (one `(lower, upper)` pair for
all variables or one per variable, `None` or `±inf` for "no bound", default
`(0, None)`), with `sense="max"` to maximise. Coefficients must be exact — `int`,
`fractions.Fraction`, `"p/q"` strings or rational `Expr`s. A float is refused with
`E-LP-002`: a certificate for its nearest rational would certify a different
programme.

## Certificates

| status | certificate | what it proves |
|---|---|---|
| `optimal` | `x`, `fun`, and multipliers `ineqlin`, `eqlin`, `lower`, `upper` | `c = A_ubᵀ·ineqlin + A_eqᵀ·eqlin + lower + upper` with the sign conditions of the sense (minimising: `ineqlin ≤ 0`, `lower ≥ 0`, `upper ≤ 0`; reversed when maximising — SciPy's `marginals` convention), complementary slackness, and dual objective = `fun`. Weak duality then makes `x` optimal. |
| `infeasible` | `farkas` — weights `ineqlin ≥ 0`, `eqlin`, `lower ≥ 0`, `upper ≥ 0` | the weighted sum of the constraints reads `0·x ≤ (negative number)`. |
| `unbounded` | `x` (feasible) and `ray` | `A_ub·ray ≤ 0`, `A_eq·ray = 0`, the ray respects every finite bound, and `c·ray < 0` (`> 0` when maximising). |

The solver checks its own certificate exactly before returning; one that fails
is withheld as `E-LP-006` rather than returned.

The algorithm is a dense two-phase simplex over `rug::Rational` with **Bland's
rule**, which cannot cycle (Beale's classical cycling example terminates). The
active `ak.context(budget=ak.Budget(...))` is checked at every pivot and a trip
raises `BudgetExceededError`.

## Integer programmes

```python
# 0/1 knapsack
r = ak.milp([10, 13, 7, 8], A_ub=[[4, 6, 3, 5]], b_ub=[12], bounds=(0, 1), sense="max")
r.x, r.fun, r.gap   # ([1, 0, 1, 1], 25, 0)
```

`integrality` takes one flag per variable (`1` integer, `0` continuous — SciPy's
convention) or a single bool; the default makes every variable an integer.
Branch and bound runs depth-first until it has an incumbent, then best-bound,
solving every node LP exactly with its certificate checked. The outcomes are:

* `optimal` — the search tree closed, so `gap == 0` and `bound == fun`;
* `infeasible` — every leaf was LP-infeasible (each with a verified Farkas vector);
* `unbounded` — `x` is integer-feasible and `ray` is integral, so every
  `x + t·ray` (`t ∈ ℕ`) is integer-feasible with unboundedly better objective
  (Meyer's theorem makes this sound for rational data).

The search stops at `max_nodes` node LPs (default 100 000) with **`E-LP-005`**,
and at a budget trip with `BudgetExceededError`. Neither is an answer: both
exceptions carry `incumbent` (the best integer point found, *unproven*),
`incumbent_fun`, and `bound` (the best open bound), which bracket the optimum.

## Programmes from expressions

```python
p = ak.ExprPool(); x, y = p.symbol("x"), p.symbol("y")
r = ak.linprog_expr(3*x + 2*y,
                    [p.le(x + y, p.integer(4)), p.le(x + 3*y, p.integer(6)),
                     p.ge(x, p.integer(0)), p.ge(y, p.integer(0))],
                    [x, y], sense="max")
r.solution        # {x: 4, y: 0}
r.program()       # the rows the relations became
```

The objective and both sides of every relation must be affine in the listed
variables with rational coefficients. Variables are **free** unless a
constraint bounds them. Each `≤`/`≥` becomes one `A_ub` row and each `==` one
`A_eq` row, in order, so the multipliers line up with the constraints.
`integer=True` (or a list of variables) solves it as a MILP instead. Refusals:
`E-LP-003` for a non-affine term or a symbol that is not a decision variable,
`E-LP-002` for a float or irrational coefficient, `E-LP-004` for a strict
inequality (which in general has no optimum) or a non-relation.

## Error codes

| Code | Meaning |
|---|---|
| `E-LP-001` | Malformed shapes (row length, right-hand side count, bounds, integrality, sense) |
| `E-LP-002` | An inexact coefficient (float, irrational constant) |
| `E-LP-003` | Not affine in the decision variables |
| `E-LP-004` | A constraint that is not `<=`, `>=` or `==` |
| `E-LP-005` | Branch and bound reached `max_nodes`; incumbent and bound attached |
| `E-LP-006` | A certificate failed its exact check; the result was withheld (a bug) |

## Rust

The same API lives in `alkahest_cas::linprog`: `linprog(&LinearProgram) ->
LpOutcome` (`Optimal(OptimalCertificate)`, `Infeasible(FarkasCertificate)`,
`Unbounded(UnboundedCertificate)`, each with an exact `verify`),
`milp(&LinearProgram, &MilpOptions) -> MilpOutcome`, and
`program_from_exprs` / `linear_form` for the expression front end.
