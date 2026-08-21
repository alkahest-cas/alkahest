# Polynomial system solving

Alkahest solves systems of polynomial equations symbolically using Gröbner bases.

## solve

`solve` finds the solutions of a system of polynomial equations in a list of variables. It uses the `groebner` Cargo feature, which is **included in all PyPI wheels** (default feature since 2.3.1) and in all source builds.

```python
from alkahest import ExprPool, solve, sqrt

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

# Linear system
solutions = solve([x + y - pool.integer(1), x - y], [x, y])
# → [{x: 1/2, y: 1/2}]

# Circle intersected with a line: irrational solutions
solutions = solve(
    [x**2 + y**2 - pool.integer(1), y - x],
    [x, y]
)
# → [{x: sqrt(2)/2, y: sqrt(2)/2}, {x: -sqrt(2)/2, y: -sqrt(2)/2}]

# Parametric solve: free symbols omitted from `vars` stay as parameters
solutions = solve([x**2 - y], [x])
# → [{x: sqrt(y)}, {x: -sqrt(y)}]
```

Solutions are symbolic: irrational roots are returned as `Expr` trees (e.g. `sqrt(2)/2`) rather than floats. Quadratic elimination produces exact symbolic answers. Free symbols that appear in the equations but are not listed in `vars` are treated as parameters, so solutions may depend on those symbols.

### Solution types

The return value is a list of dicts mapping `Expr` variable → `Expr` solution:

```python
for sol in solutions:
    for var, val in sol.items():
        print(f"{var} = {val}")
        # Evaluate numerically if needed
        from alkahest import eval_expr
        numeric = eval_expr(val, {})
```

`solve` returns an empty list for inconsistent systems and a `GroebnerBasis` handle for parametric families (infinite solution sets).

Pass `numeric=True` to return float values directly: `solve(eqs, vars, numeric=True)`.

## GroebnerBasis

A `GroebnerBasis` can be constructed directly for ideal-theoretic operations:

```python
from alkahest import GroebnerBasis

# Compute a Gröbner basis (lex by default)
polys = [x**2 + y**2 - pool.integer(1), x - y]
gb = GroebnerBasis.compute(polys, [x, y])

# Check ideal membership
print(gb.contains(x - pool.rational(1, 2)))  # False

# Reduce a polynomial modulo the ideal — the remainder is a GbPoly
reduced = gb.reduce(x**3 + y**3)
print(reduced.to_expr())                     # y
```

### Reading a basis

A `GroebnerBasis` is a sequence of `GbPoly`, and each `GbPoly` converts back to an `Expr`. This is how you read an elimination result — the generators of a `Lex` basis that are free of the eliminated variables *are* the eliminated relations:

```python
len(gb)                       # 2 — number of generators
gb.order                      # "lex"
gb.variables()                # [x, y] — what exponent slots 0, 1 refer to

for g in gb:
    print(g.to_expr(), "= 0")
# (y^2 + -1/2) = 0
# (x + (y * -1)) = 0

gb.to_exprs()                 # the same list in one call
gb[0].terms()                 # [((0, 0), Fraction(-1, 2)), ((0, 2), 1)]
```

`terms()` gives `(exponent tuple, coefficient)` pairs with the coefficient as an exact Python `int` or `fractions.Fraction`; the exponent tuple is parallel to `variables()`.

The conversion runs the other way with `expr_to_gbpoly`, which is what `reduce` and `contains` accept alongside plain `Expr`:

```python
from alkahest import expr_to_gbpoly

p = expr_to_gbpoly(x**2 + y**2 - pool.integer(1), [x, y])
p.n_terms                     # 3
gb.contains(p)                # True
GroebnerBasis.compute_raw([p])
```

A `GbPoly` stores exponent vectors, not names, so converting one back needs the variable list its slots refer to. Every basis Alkahest hands out carries that list — including the ones from `solve` (solve variables followed by the free parameters), `triangularize` and `rosenfeld_groebner` — so `to_expr()` normally takes no arguments. Naming too few variables raises `ValueError` rather than quietly misreading the exponent slots.


### Monomial orders

Supported orders: `Lex` (lexicographic), `GrLex` (graded lexicographic), `GRevLex` (graded reverse lexicographic). `GRevLex` is generally fastest for basis computation; `Lex` is required for elimination.

### Parallel F4

With `--features "groebner parallel"`, Gröbner basis computation uses Rayon for parallel S-polynomial reduction via the F4 algorithm.

### GPU-accelerated Macaulay matrix (groebner-cuda) — Rust only, not wired into the solver

`--features "groebner-cuda"` compiles a CUDA kernel for the mod-p row reduction of the Macaulay matrix, with multi-prime CRT lifts reconstructing rational coefficients, and falls back to pure-Rust row reduction when no CUDA device is present.

**It does not accelerate anything on this page.** `GroebnerBasis.compute`, `solve` and `triangularize` run Buchberger/F4 on the CPU regardless; the GPU routine is reachable only as the Rust function `alkahest_cas::poly::groebner::compute_groebner_basis_gpu`, and production dispatch deliberately does not prefer it until the [benchmark harness](https://github.com/alkahest-cas/alkahest/blob/main/docs/symbolic-gpu-benchmarks.md) says it wins. There is correspondingly **no `capabilities()["features"]["groebner_cuda"]` bit** — it used to exist and report that the kernel had been compiled in, which no Python observation could confirm or refute, so 3.8 removed it. See [GPU support](./gpu.md#groebner-cuda-is-not-reachable-from-python).

Because the Rust entry point falls back to CPU row reduction when no device is present, it returns a `GpuBackendReport` alongside the basis: `let (basis, backend) = compute_groebner_basis_gpu(gens, order, Some(0))?;` and `backend.ran_on_gpu()` is the only way to tell a real GPU run from a fallback, since the basis is identical either way.

## Elimination ideals

`GroebnerBasis.eliminate` computes the elimination ideal `I ∩ k[remaining vars]` by dropping every generator whose support mentions one of the given variables. Under a `lex` basis with the eliminated variables ordered **first**, what is left is a Gröbner basis for that ideal:

```python
# Implicitize the parametric curve (t, t**2): eliminate the parameter t.
gb = GroebnerBasis.compute([x - t, y - t**2], [t, x, y])
gb.to_exprs()                      # [(t + (x * -1)), ((y * -1) + x^2)]

implicit = gb.eliminate([t])
implicit.to_exprs()                # [((y * -1) + x^2)]  —  y = x**2
```

Note the variable order passed to `compute`: `t` comes first, so `lex` eliminates it. `eliminate` requires the basis to know its variables (`gb.variables()`), and rejects a variable it is not written over.

## Coefficient fields: `Q(params)` instead of `Q[vars, params]`

`GroebnerBasis.compute(polys, vars, params=[...])` moves the listed symbols into the **coefficient field** `Q(params)` instead of the polynomial ring. They never enter the monomial order and never generate S-pairs, which is the difference between eliminating states from `Q[states, Y, params]` and from `Q(params)[states, Y]` — the parameter count no longer inflates the staircase.

```python
from alkahest import GroebnerBasis

# a lives in the coefficient field Q(a), not the ring Q[x, y, a]
gb = GroebnerBasis.compute([a*x - y, x + y - one], [x, y], params=[a])
type(gb)                      # ParametricGroebnerBasis
[g.to_expr() for g in gb]     # coefficients are rational functions of a
```

Measured on a catenary compartmental ODE model (a linear chain of `n` states, output the first compartment, eliminating the states from the jet equations down to the input–output relation): at `n = 4` states / 7 rate constants the parametric route runs in 0.27s against 4.2s putting the rate constants in the ring (`Lex`, `--release`, ~15×) and leaves 5 total basis generators against 25; at `n = 5` states / 9 rate constants the parametric route finishes in 6.9s while the direct computation had not finished after 240s. These are wall-clock numbers on one machine, illustrating the shape of the difference (S-pairs among the parameters are exactly what the ring route pays for and the coefficient-field route never generates) rather than a promised ratio.

**The result is generic.** A leading coefficient in `Q(params)` can be a non-zero rational function of the parameters and still vanish at a specific parameter point, and there the basis this computation built is not the one the same algorithm would build over ℚ at that point:

```python
gb.conditions()               # [a + 1] — the basis says nothing at a = -1
gb.is_regular_at([3])         # True
gb.is_regular_at([-1])        # False

gb.specialize([3])            # an ordinary GroebnerBasis over Q
gb.specialize([-1])           # raises ParamGroebnerError, code "E-PARAMGB-004"
```

`conditions()` lists the hypersurfaces the computation assumed non-vanishing — every leading-coefficient inversion contributes its numerator and denominator, every input coefficient contributes its denominator — factored into irreducible, primitive pieces so the report is a list of conditions rather than one opaque polynomial in many parameters. The list is **sufficient, not necessary**: it can flag a point that turns out fine (a removable coincidence the bookkeeping cannot see), but it never misses a point where the generic basis is genuinely wrong. `specialize` refuses on the flagged locus with `ParamGroebnerError` (`E-PARAMGB-004`) rather than returning something that is not a basis; check `is_regular_at` first if a degenerate point is a normal outcome for your caller.

### `specialize(values, verify=True)`

Most of what `conditions()` lists are leading coefficients that were inverted somewhere inside the Buchberger loop and then cancelled, so a large share of the points it excludes are ordinary. Measured over eight parametric systems on the small-integer box `{-2..2}`: 646 refusals, of which 239 are genuine poles, 73 are necessary, and **334 are unnecessary**. The fraction is box-dependent and dominated by parameters equal to exactly 0 — `{-2..2}` gives 52%, `{-2,-1,1,2}` gives 25%, `{-3..3}` gives 58% — so the honest headline is **a quarter to a half of refusals on small-integer grids**, not a single number.

```python
gb = GroebnerBasis.compute([a*x + b*y - one, c*x + d*y - one], [x, y],
                           params=[a, b, c, d])
gb.conditions()                          # [c, a*d - b*c, a]

gb.specialize([0, 1, 1, 1])              # E-PARAMGB-004: a = 0 is on the locus
gb.specialize([0, 1, 1, 1], verify=True) # [x, y - 1] — the refusal was unnecessary
gb.specialize([1, 1, 2, 2], verify=True) # still E-PARAMGB-004: a*d - b*c = 0
```

`verify=True` re-solves the specialised system over ℚ and compares it with the specialised generic basis, instead of deciding on the recorded conditions alone. It is strictly more complete and never less sound: the refusals were separately checked to be *sound* (1,930 regular points across three sweeps, zero disagreements between `specialize` and the basis computed directly at the point), so this closes a completeness gap, not a correctness one. It is off by default because on the locus it pays for a second Gröbner basis over ℚ.

### Feeding a parametric basis back in

`reduce` and `contains` accept input that is **rational in the parameters**. A `den**-1` factor is an ordinary element of `Q(params)`, not a non-polynomial, so a basis accepts its own `to_exprs()` output — only a denominator in a *ring variable* is refused:

```python
gb = GroebnerBasis.compute([a*x - one], [x], params=[a])
gb.to_exprs()                            # ['(x + (-1 * a^-1))']
gb.contains(gb.to_exprs()[0])            # True
```

`equals_ideal` answers the question that needs: do two parametric bases generate the same ideal of `Q(params)[vars]`?

```python
g1 = GroebnerBasis.compute([a*x - one], [x], params=[a])
g2 = GroebnerBasis.compute([a*a*x - a], [x], params=[a])
g1.equals_ideal(g2)                      # True
g1.contains_ideal(g2)                    # True  — one direction only
```

This is exact, not generic: reduction over `Q(params)` only ever divides by non-zero field elements, so neither basis's `conditions()` enters the answer. Those conditions still bound what each basis says about a *specialised* parameter point — equal ideals over the fraction field can specialise differently on the locus. Bases over different numbers of variables or parameters compare `False`; there is no shared ring.

The read path matches `GroebnerBasis`: the object is a sequence of `ParametricGbPoly`, each with `to_expr()` / `terms()`, and the basis itself has `to_exprs()`, `eliminate(vars)` (same `Lex`-with-eliminated-variables-first contract, refuses to eliminate a coefficient-field parameter since there is nothing to eliminate), `reduce`, `contains`, `contains_ideal` and `equals_ideal`. `GroebnerBasis.compute(..., params=None)` or `params=[]` is the unmodified `Q[vars]` path; `ParametricGroebnerBasis.compute(polys, vars, params, order=None)` is the equivalent direct constructor in `alkahest.experimental`.

### Differential elimination with the parameters in `Q(params)`

`rosenfeld_groebner(dae, params=[...])` runs the prolongation loop of [Rosenfeld–Gröbner](./ode-dae.md) over `Q(params)` instead of putting the model parameters in the ring, and returns a `ParametricRosenfeldGroebnerResult` whose `final_basis()` is a `ParametricGroebnerBasis`. That is the input–output relation of an ODE model computed end to end, rather than prolonged by hand:

```python
# SIR:  S' = -b*S*I,  I' = b*S*I - g*I,  R' = g*I,  y0 = I
dae = DAE.new([dS + b*S*I, dI - b*S*I + g*I, dR - g*I, y0 - I],
              [S, I, R, y0], [dS, dI, dR, dy0], t)

r = rosenfeld_groebner(dae, params=[b, g], eliminate=[S, I, R],
                       minimal=True, order="lex")
r.minimal_prolongation_rounds            # 2
states = ("S", "I", "R")
io = r.final_basis().eliminate([v for v in r.variables()
                                if str(v).lstrip("d").split("/")[0] in states])
io.to_exprs()                            # y0*y0'' - y0'^2 + b*y0^2*y0' + b*g*y0^3, over Q(b,g)
```

`eliminate=[...]` names the variables the caller intends to eliminate — typically the unobserved states. Their whole jet chain goes with them (`x` also names `dx/dt`, `d2x/dt2`, …), and the ranking is built with those jets first so the `Lex` elimination contract holds.

**One prolongation too many is expensive out of all proportion.** On the SIR model above, stopping at the first informative round costs 0.03 s and gives one 4-term relation; prolonging one round further does not finish in ten minutes. The over-supplied answer is *correct*, just enormously more expensive, so nothing else signals it. `minimal=True` stops at the first informative round; without it, a run that prolonged past one emits a `UserWarning` naming the round it could have stopped at, and `minimal_prolongation_rounds` records it either way.

**Scope of "minimal".** `minimal_prolongation_rounds` is "the first round at which eliminating those variables leaves a generator", not a theorem about the differential ideal. For a single-output model it coincides with the jet order the input–output relation needs; **for multi-output models the criterion is known to be wrong**, because one output can become informative several rounds before the others and the truncated basis then misses their relations. Treat it as a cost signal, not a certificate — and note that a `minimal=True` result is always flagged `truncated`.

### Which structural identifiability this decides

Where this pipeline is used for structural identifiability, it answers **multi-experiment** identifiability, not single-experiment: by Ovchinnikov–Pillay–Pogudin–Scanlon, [*Computing all identifiable functions of parameters for ODE models*](https://arxiv.org/abs/2004.07774), Theorem 19, a function of the parameters is multi-experiment identifiable iff it is input–output identifiable, for general rational systems. That is why an IO-elimination route can call a model globally identifiable where a single-experiment tool such as SIAN reports "locally, not globally" — both verdicts are right, about different questions.

**Hypothesis, not a guarantee.** Theorem 19 requires the input–output equations to be the *characteristic presentation* of `I_Σ ∩ C(θ){y,u}`. An algebraic lex elimination at a hand-picked finite jet order — what the surface above does — is not guaranteed to produce one. So "this is a multi-experiment tool" holds under that standard; where it fails, the computed field can be a proper subfield of the identifiable one.

This surface is experimental (`alkahest.experimental.ParametricGroebnerBasis` / `ParametricGbPoly` / `ParametricRosenfeldGroebnerResult`) and requires `--features groebner`.

## Performance

On the `solve_circle_line` benchmark (2-variable quadratic system), Alkahest is approximately **40× faster** than SymPy due to the FLINT-backed polynomial arithmetic and the compiled F4 core.

**Upcoming (v2.0):** F5 / signature-based Gröbner basis, real root isolation, primary decomposition, and other advanced algorithms.
