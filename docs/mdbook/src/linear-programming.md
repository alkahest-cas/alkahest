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

Two exact simplex methods are available through `method=`:

* `"dense"` — a two-phase tableau over `rug::Rational` with **Bland's rule**,
  which cannot cycle (Beale's classical cycling example terminates);
* `"revised"` — a revised simplex that keeps `A` sparse and holds `B⁻¹` as an
  exact product-form (eta-file) factorisation, rebuilt periodically. It prices
  by Dantzig's rule and falls back to Bland's rule after a run of degenerate
  pivots, so it cannot cycle either. It starts from a slack crash basis;
* `"auto"` (default) — dense for at most 12 rows, revised otherwise.

The active `ak.context(budget=ak.Budget(...))` is checked at every pivot of
either method and a trip raises `BudgetExceededError`.

On random sparse programmes (5 non-zeros per row, `0 ≤ x ≤ 10`, maximising;
release build, one core) the revised method is what makes medium sizes
practical:

| rows × columns | `≤` rows only: dense | revised | half `≥` rows (phase 1): dense | revised |
|---|---|---|---|---|
| 25 × 25 | 0.10 s | 0.001 s | 0.04 s | 0.01 s |
| 50 × 50 | 1.7 s | 0.006 s | 1.0 s | 0.04 s |
| 100 × 100 | 47.5 s | 0.013 s | 21.4 s | 0.97 s |
| 200 × 200 | — | 0.17 s | — | 6.2 s |
| 300 × 300 | — | 0.35 s | — | 52 s |
| 500 × 500 | — | 3.4 s | — | — |

(The dense tableau updates every entry at every pivot and its entries grow;
the revised method touches only the pivot column and the eta file. Run with
`cargo test -p alkahest-cas --release --lib bench_revised_vs_dense -- --ignored --nocapture`.)

### Warm starts

Every `LPResult` carries its final `basis` — a list of `(kind, index)` pairs
(`("x", j)`, `("x-", j)` for the negative part of a free variable,
`("slack", i)`, `("upper", j)`), named in the caller's terms so it carries over
to a modified programme. Pass it back with `basis=` (or pass the result
itself):

```python
c, A, b = [3, 5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18]   # the example above
cut, rhs = [1, 1], 7                                     # add a + b ≤ 7
r1 = ak.linprog(c, A_ub=A, b_ub=b, sense="max")
r2 = ak.linprog(c, A_ub=A + [cut], b_ub=b + [rhs], sense="max", basis=r1)
r2.warm_started   # True — the old basis was dual feasible, so a dual simplex ran
```

A basis that is still primal feasible goes straight to phase 2; one that is
still dual feasible (after a right-hand-side change, a tightened bound or an
added row) runs the **dual simplex**, which proves infeasibility with a Farkas
vector read off a row of `B⁻¹`; anything else falls back to a cold start.
Labels that do not exist in the new programme are ignored. The certificates
are the same, and verified the same way, whichever path ran.

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

### Gomory cuts (cut-and-branch)

Before branching, `milp` rounds the bounds of integer variables inward and
then runs up to `cut_rounds` (default 8) rounds of **Gomory mixed-integer
cuts**, at most `max_cuts` (default 200) in all: each round solves the root
relaxation with the revised simplex, reads a GMI cut off each tableau row
whose basic variable is an integer at a fractional value, appends the cuts as
`A_ub` rows and re-solves from the previous basis. Exact rationals make every
cut valid — a column is treated as integer only when it is an integer at
every integer-feasible point (an integer variable with an integer bound
shift, the slack of a row with integer data over integer variables), anything
else as continuous, which only weakens a cut — so **no cut removes an
integer-feasible point**. When every variable in a cut is an integer it is
also scaled to coprime integer coefficients and its right-hand side rounded
down (a Chvátal–Gomory step). The cuts are returned in `MILPResult.cuts` as
`(row, rhs)` pairs, so they can be checked; `cut_rounds=0` turns cutting off.

```python
r = ak.milp([1, 1], A_ub=[[2, 2]], b_ub=[3], sense="max")
r.fun, r.cuts, r.nodes   # (1, [([1, 1], 1)], 1) — the cut closes the gap at the root
```

Node LPs are warm-started from their parent's basis (`warm_start=True`): a
branching bound leaves the parent basis dual feasible, so each child is a few
dual-simplex pivots.

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

## Parametric programmes

`linprog_parametric` solves a programme whose right-hand side **or** objective
is affine in one parameter `θ`, for every `θ` at once:

```python
# max x + y  s.t.  x + 2y ≤ 4 + θ,  3x + y ≤ 6,  x, y ≥ 0
r = ak.linprog_parametric([1, 1], A_ub=[[1, 2], [3, 1]], b_ub=[4, 6],
                          b_ub_theta=[1, 0], sense="max")
r.breakpoints                        # [-4, -2, 8]
[g["status"] for g in r.regions]     # ['infeasible', 'optimal', 'optimal', 'optimal']
[g["value"] for g in r.regions[1:]]
# [(4, 1), (Fraction(14, 5), Fraction(2, 5)), (6, 0)]   i.e. 4 + θ, (14 + 2θ)/5, 6
r.value_at(0)                        # Fraction(14, 5)
```

Pass `c_theta` for an objective `(c + θ·c_theta) @ x`, or `b_ub_theta` /
`b_eq_theta` for right-hand sides `b + θ·δb` — not both (`E-LP-007`).
`theta_range=(lo, hi)` restricts `θ` (`None` for an infinite end). Each region
is a dict with `theta`, `status`, and for optimal regions the affine `value`,
`x`, multipliers (`ineqlin`, `eqlin`, `lower`, `upper`) as pairs `(a, b)`
meaning `a + θ·b`, and the optimal `basis`. Optimal regions are closed;
infeasible (RHS case) and unbounded (objective case) regions are open at the
end they share with an optimal one.

The interval of `θ` with an optimum comes from two auxiliary LPs (feasibility
in `(x, θ)`, or dual feasibility in `(y, z, θ)`); inside it the solver walks
from basis to basis, each breakpoint an exact ratio, and re-optimises at a
breakpoint by Bland pivots on the programme perturbed to `θ + ε`, so every
region has positive length. **Each optimal region's certificate is verified
for every `θ` in it**, not sampled: every certificate condition is affine in
`θ`, so it is checked at both ends, and for an infinite end its slope is
checked as the optimality certificate of a derived programme.

With expressions, `linprog_expr(..., parameter=t, parameter_range=(lo, hi))`
returns the same result; `t` may appear in the objective's coefficients
(`(2 + t)*x`) or in the constraints' constant terms (`x + y <= 4 + t`).
`E-LP-007` refuses what this does not handle: `t` multiplying a variable
inside a constraint (a moving constraint matrix), `t` entering non-linearly
(`t**2`, `sin(t)`), `t` in both the objective and the right-hand side, or a
parameter together with `integer=`.

## Error codes

| Code | Meaning |
|---|---|
| `E-LP-001` | Malformed shapes (row length, right-hand side count, bounds, integrality, sense) |
| `E-LP-002` | An inexact coefficient (float, irrational constant) |
| `E-LP-003` | Not affine in the decision variables |
| `E-LP-004` | A constraint that is not `<=`, `>=` or `==` |
| `E-LP-005` | Branch and bound reached `max_nodes`; incumbent and bound attached |
| `E-LP-006` | A certificate failed its exact check; the result was withheld (a bug) |
| `E-LP-007` | An unsupported parametric form (parameter in the constraint matrix, non-linear, in both objective and right-hand side, or with integers) |

## Rust

The same API lives in `alkahest_cas::linprog`: `linprog(&LinearProgram) ->
LpOutcome` (`Optimal(OptimalCertificate)`, `Infeasible(FarkasCertificate)`,
`Unbounded(UnboundedCertificate)`, each with an exact `verify`),
`linprog_with(&LinearProgram, &LpOptions) -> LpSolution` (method choice,
warm-start `LpBasis`, final basis), `milp(&LinearProgram, &MilpOptions) ->
MilpOutcome` (default cuts and warm starts), `milp_with(&lp, &MilpOptions,
&CutOptions) -> MilpReport` (`cut_rounds`, `max_cuts`, `warm_start`; returns
the outcome and the root cuts),
`linprog_parametric(&lp, &ParametricDelta, lower, upper) ->
ParametricSolution`, and `program_from_exprs` / `parametric_from_exprs` /
`linear_form` for the expression front end.
