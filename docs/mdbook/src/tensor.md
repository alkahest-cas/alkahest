# Tensor calculus

`alkahest.tensor` computes curvature from a metric on a coordinate chart and
offers an abstract-index layer with Einstein summation.

```python
import alkahest as ak
from alkahest import sin, cos

p = ak.ExprPool()
t, r, th, ph, M, a = (p.symbol(s) for s in ("t", "r", "theta", "phi", "M", "a"))

# Kerr, Boyer–Lindquist
sigma = r**2 + a**2 * cos(th)**2
delta = r**2 - 2*M*r + a**2
gtt = -(1 - 2*M*r/sigma)
gtp = -2*M*r*a*sin(th)**2/sigma
gpp = (r**2 + a**2 + 2*M*r*a**2*sin(th)**2/sigma) * sin(th)**2
g = [[gtt, 0, 0, gtp], [0, sigma/delta, 0, 0], [0, 0, sigma, 0], [gtp, 0, 0, gpp]]

m = ak.Metric(g, [t, r, th, ph])
m.is_ricci_flat()      # True — proved, not sampled
m.timings()            # seconds per stage: convert, inverse, christoffel, riemann, ricci
```

## The metric layer

`Metric(g, coords)` takes an `n x n` symmetric list of lists (or a `Matrix`)
and `n` distinct coordinate symbols. Entries may contain parameters,
`sin`/`cos`/`tan`/…, `sinh`/`cosh`/…, `exp`, `log`, radicals and **undefined
functions** of the coordinates — the FLRW scale factor is just
`p.func("a", [t])`, and its derivatives appear as `Derivative(a(t), t)`.

| Method | Result |
|---|---|
| `as_tensor()`, `inverse()`, `determinant()` | `g_ab` (`"dd"`), `g^ab` (`"uu"`), `det g` |
| `christoffel()` | nested list `[a][b][c] = Γ^a_bc` |
| `christoffel_first_kind()` | `Γ_abc = g_ad Γ^d_bc` |
| `riemann()`, `riemann_lower()` | `R^a_bcd` (`"uddd"`), `R_abcd` (`"dddd"`) |
| `ricci()`, `ricci_scalar()` | `R_ab = R^c_acb`, `R = g^ab R_ab` |
| `einstein()` | `G_ab = R_ab − ½ R g_ab` |
| `weyl()` | `C_abcd` (`n ≥ 3`) |
| `kretschmann()` | `R_abcd R^abcd` |
| `is_ricci_flat()` | `True` iff every `R_ab` is exactly zero |
| `geodesic_equations(v)` | `d²x^a/dτ² = −Γ^a_bc v^b v^c` |
| `simplify(e)`, `is_zero(e)`, `diff(e, x)` | the exact field's canonical form, zero test, partial derivative |

Sign conventions are MTW / Carroll / Wald:
`R^a_bcd = ∂_c Γ^a_db − ∂_d Γ^a_cb + Γ^a_ce Γ^e_db − Γ^a_de Γ^e_cb`,
so the round 2-sphere of radius ρ has `R = 2/ρ²` and Schwarzschild has
`K = 48 M²/r⁶`.

### Why the zeros are exact

Every component lives in one exact field: a quotient of FLINT integer
polynomials over the coordinates, the parameters and one generator per
transcendental atom. The relations `sin² = 1 − cos²`, `cosh² − sinh² = 1`,
`i² = −1` and `(√b)² = b` are applied after every operation, and each
denominator is multiplied by its conjugate until it is free of `sin`, radicals
and `i`. Partial derivatives act on the generators by their own rules
(`∂ sin v = cos v · ∂v`, `∂ √b = √b · ∂b / 2b`, `∂ a(t) = Derivative(a(t), t)`),
which respect the relations. The resulting normal form is **canonical**: two
components are equal exactly when their forms are identical, so a component
that vanishes is the literal `0` — no sampling, no floats. That is what lets
`is_ricci_flat()` prove the Kerr metric is a vacuum solution, which the
`cancel` + `simplify_trig` route could not.

Results are printed factored over ℤ, with `(cos θ − 1)(cos θ + 1)` folded back
to `−sin² θ` (so `Γ^φ_θφ` reads `cos θ · sin θ⁻¹`).

Only independent components are computed (`Γ_abc` for `b ≤ c`, `R_abcd` for
`a < b`, `c < d`, `(a,b) ≤ (c,d)`, `R_ab` for `a ≤ b`); the rest are filled in
by symmetry. Each stage is cached.

## Abstract indices

```python
a, b, c, d = ak.tensor_indices("a b c d")
Ric, g = m.ricci(), m.as_tensor()
R = Ric[a, -a]                         # raise one slot with g^ab, contract: the Ricci scalar
G = Ric[-a, -b] - g[-a, -b] * R / 2    # same components as m.einstein()
m.riemann()[c, -a, -c, -b]             # R^c_acb = R_ab
Rl = m.riemann_lower()[-a, -b, -c, -d]
Rl.antisymmetrize(b, c, d).is_zero()   # first Bianchi identity: True
```

`-a` is a lower index and `+a` (or `a`) an upper one. Indexing a `Tensor`
in the other variance than it is stored raises or lowers that slot with the
metric; a repeated index — once up, once down — is summed. `IndexedTensor`
supports `+`, `-`, `*` (outer product plus contraction), scalar `*` and `/`,
`symmetrize`, `antisymmetrize`, `is_zero()`, `components` and `tensor()`.
Components are kept in the metric's canonical form.

## Refusals

`TensorError` with a stable code:

| Code | Meaning |
|---|---|
| `E-TENSOR-001` | the metric is not `n x n` for `n` distinct coordinate symbols |
| `E-TENSOR-002` | the metric is not symmetric |
| `E-TENSOR-003` | the determinant is identically zero |
| `E-TENSOR-004` | a floating-point entry (the layer is exact) |
| `E-TENSOR-005` | a node the field cannot hold (`Piecewise`, a predicate) |
| `E-TENSOR-006` | a partial derivative is unavailable |
| `E-TENSOR-007` | a denominator vanishes identically |
| `E-TENSOR-008` | the Weyl tensor below dimension 3 |
| `E-TENSOR-009` | index misuse: an index three times, twice up or twice down, mismatched free indices |
| `E-TENSOR-010` | an internal invariant failed |

An active `Budget` is checked once per component; a trip raises
`BudgetExceededError`.
