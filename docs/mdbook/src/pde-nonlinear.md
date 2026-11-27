# Nonlinear PDEs: symmetries, conservation laws, exact solutions

*Experimental* (`alkahest.experimental`, Rust `alkahest_cas::pde`): milestone
M6 of the PDE plan. It covers one scalar equation in one unknown `u(x, t)`
(symmetries and conservation laws work in any number of independent
variables; the reductions need two).

As everywhere in the PDE layer, **nothing is returned unverified**:

| object | check |
|---|---|
| symmetry generator `X` | `pr X(F) = 0` on solutions, recomputed independently |
| conservation law | `D_t T + D_x X = Λ·F` as an identity in the jets |
| solution | substitution into the equation, `verify_pde_solution` |
| transformation | the transformed equation as a jet identity |

A candidate that fails its check is discarded, and a refusal is an
`E-PDE-300 … E-PDE-306` error (see [Error handling](./errors.md)).

## Lie point symmetries

```python
import alkahest as ak
import alkahest.experimental as akx
from alkahest.experimental import D, Function

p = ak.ExprPool()
x, t = p.symbol("x"), p.symbol("t")
u = Function("u")(x, t)

kdv = D(u, t) + u * D(u, x) + D(u, x, 3)
S = akx.pde_symmetries(kdv, u)
print(len(S.generators), S.complete)  # → 4 True
for g in S.generators:
    print(g.operator)  # ∂_t, ∂_x, t∂_x + ∂_u, x∂_x + 3t∂_t − 2u∂_u
```

`PointSymmetries` holds a verified basis of the finite-dimensional part,
`structure_constants` (`[X_a, X_b] = Σ C_abc X_c`), `.lie_algebra()` (an
`alkahest.LieAlgebra` on noncommutative `X1, X2, …`), the
infinite-dimensional part in `infinite` (`"superposition"` for linear
equations, `"passive"` for a completed linear system such as the conformal
group of the wave equation), and `complete`: whether this is the whole
point-symmetry algebra.

```python
heat = D(u, t) - D(u, x, 2)
H = akx.pde_symmetries(heat, u)
print(len(H.generators), H.infinite["kind"])  # → 6 superposition
alg = H.lie_algebra()
```

The determining equations (`determining_equations`) come from the
prolongation formula on the jet space and are solved by splitting,
elimination, `dsolve` on linear ODEs and a linear differential-Gröbner
completion. With `cases=True`, the equation's parameters are classified:

```python
m = p.symbol("m")
cases = akx.pde_symmetries(D(u, t) - D(u**m * D(u, x), x), u, cases=True)
for c in cases:
    print(c.case, len(c.generators))  # generic m: 4; m = -4/3: 5; m = 0: 6
```

## Conservation laws

`conservation_laws(eq, u, order)` finds the multipliers `Λ` with
`E_u(Λ·F) ≡ 0` up to the given order (`euler_operator` is the variational
derivative), then recovers the density and flux and checks them. Expressions
are in jet symbols (`u`, `u_x`, `u_xx`, …).

```python
for law in akx.conservation_laws(kdv, u, order=2):
    print(law.multiplier, "|", law.density, "|", law.verified)
```

For KdV this gives the four classical laws: mass, momentum, energy and the
Galilean law with multiplier `t·u − x`.

## Exact solutions by reduction

- `traveling_wave_solutions`: `u = F(x − ct)`. The reduced ODE is integrated
  as far as it is exact, then passed to `dsolve`; next come the separatrices
  of autonomous second-order ODEs (solitons, kinks) and the tanh / sech / tan
  expansion methods.
- `ansatz_solutions(eq, u, method="tanh")`: the expansion method alone (pole
  order by balancing, coefficient system by Gröbner bases).
- `invariant_solutions(eq, u, generator)`, `similarity_solutions`,
  `reduce_by_generator`: reductions by Lie point symmetries, which give the
  heat-source and Barenblatt solutions.

```python
for s in akx.traveling_wave_solutions(D(u, t) + 6 * u * D(u, x) + D(u, x, 3), u):
    print(s.method, s.value, s.verified)
```

`pdsolve` uses these through its `cole_hopf`, `traveling_wave` and
`similarity` tiers for nonlinear equations in two variables. Their answers
have `kind == "special"`: verified families of exact solutions, not general
solutions.

## Linearising transformations

```python
burgers = D(u, t) + 2 * u * D(u, x) - 3 * D(u, x, 2)
ch = akx.cole_hopf(burgers, u)
print(ch["substitution"], ch["verified"])
```

`miura` maps mKdV to KdV, `hodograph` turns a 2×2 homogeneous quasilinear
system into a linear one for `x(u, v)`, `t(u, v)`, and `hopf_lax` returns
the Legendre dual and the minimisation formula for `u_t + H(u_x) = 0`.

## Limits

- One scalar equation in one unknown; systems are refused (`E-PDE-300`).
- Point symmetries only (no contact, generalized or nonclassical
  symmetries); multipliers of order ≤ 6.
- The expansion methods are tanh, sech and tan; the (G′/G) method is not
  implemented separately.
